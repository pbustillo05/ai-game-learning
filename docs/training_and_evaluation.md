# Training, evaluation, and configuration

## Installation

The project requires Python 3.10+, NumPy, PyYAML, pytest, and PyTorch 2.2 or
newer. Poker development also requires the sibling evaluator, installed
separately in editable mode. On the shared server, use
`/home/pablob/.venvs/poker`; do not install into system Python.

```bash
/home/pablob/.venvs/poker/bin/python -m pip install -e .
/home/pablob/.venvs/poker/bin/python -m pip install -e ../poker-gpu-evaluator
```

The second command is required only for poker evaluator paths and their real
integration tests.

## Configuration schema

`load_config` converts YAML recursively into typed top-level dataclasses.
Unknown keys, missing required keys, wrong scalar/container types, invalid
enumerations, and invalid beta values raise `ConfigError` with a dotted field
path. Game-owned `synthetic`, `poker`, and `model` mappings are validated by
their factories.

Top-level sections are:

- `game`: selects `synthetic` or `poker` and supplies its owned mapping.
- `rollout`: tree count per rank, optional branched-only public-stage `beta`
  map, per-tree node budget, and reach storage mode.
- `completion`: samples per leaf (`null` in normal mode) and evaluator backend.
- `model`: game-owned network dimensions.
- `loss`: a registered mode-compatible objective (`ppo` for normal and
  `rb_ppo` for the current RB paths), lambda, policy/value clipping, and
  regularization. `value_loss` explicitly selects `mse` or `clipped`.
- `training`: `normal`, `rao_blackwell`, or `branched_rao_blackwell`, plus
  iterations, gradient steps, minibatch size, Adam settings, AMP, seed,
  checkpoint cadence, output directory, and the `strict`, `seeded`, or
  `throughput` reproducibility profile.

Synthetic game and model mappings are validated against their owned typed
schemas during `load_config`, as are poker mappings. Synthetic public-state
embedding capacity is derived from complete public-tree enumeration; supplied
capacities smaller than the reachable tree are rejected.
Synthetic poker uses `model.architecture: mha` by default; the former MLP is
available only as `mlp_baseline` for controlled ablations.

Every branched `rollout.beta` map requires `default`, and every intensity must
be finite and at least one. Non-branched modes require `beta: null`.
`reach_storage` is `full`, `decisions_only`, or `recompute`.

## Shipped presets

- `configs/kuhn_smoke.yaml`: small sequential Kuhn training and checkpoint
  smoke run.
- `configs/leduc.yaml`: fixed branching on Leduc, with more branching after
  public chance.
- `configs/synthetic_dial_example.yaml`: two-player configurable game using
  decision-only reach storage.
- `configs/train/nlhe6.yaml`: hardware-independent six-player 100bb NLHE
  settings, combined with `configs/hardware/local_3x2080ti.yaml` or the
  conservative, explicitly selected `quest_slurm.yaml` execution profile.

## Running training

Long-running training uses separate scientific and hardware inputs:

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti
```

Resume into the same run directory with its existing checkpoint, run identity,
and gap-free metrics prefix:

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti \
  --resume runs/nlhe6/ckpt_0000100.pt
```

Omitting `--hardware-profile` defaults to `local_3x2080ti`. Quest access is
available, but cluster execution is deliberately deferred until the local
readiness gate in `HARDWARE.md` passes and always requires an explicit
`--hardware-profile quest_slurm` selection.

For multiple ranks, launch the same entry code with `torchrun`. Sequential
trajectory counts and minibatch sizes remain per rank; they are not divided by
world size. Branched RB always samples exactly one tree per rank/GPU.

Before any CUDA run on the shared server, inspect `nvidia-smi`, select only an
idle GPU explicitly, begin with a tiny smoke configuration, and verify that
memory/process state returns to idle afterward. The NLHE preset should be
profiled before increasing trajectories (sequential modes), the single-tree
node budget (branched RB), completions, model size, or
minibatch size.

## Comparative performance profiles

`pbg-benchmark` profiles any scientific config under all three training modes,
with mode-appropriate completion and branching settings. It emits normalized
throughput, retained bytes per decision, realized topology, phase timings, and
CUDA phase peaks. Save a same-device baseline and gate later runs with:

```bash
pbg-benchmark --config configs/kuhn_smoke.yaml --warmup 2 --repeats 7 \
  --output benchmarks/results/kuhn_cpu.json
pbg-benchmark --config configs/kuhn_smoke.yaml --warmup 2 --repeats 7 \
  --baseline benchmarks/results/kuhn_cpu.json
```

See `benchmarks/README.md` and `benchmarks/workloads.yaml` for tolerance,
environment matching, GPU hygiene, and the minimum workload matrix.

## Metrics and checkpoints

Rank zero appends one JSON object per iteration to `metrics.jsonl`. Reported
fields include total loss, policy/value/entropy components, approximate KL,
clip fraction, exact policy entropy, explained variance, and maximum active PPO
ratio. Every record carries the stable identity in `run.json`; starting a new
run refuses to append over existing run artifacts.

Checkpoints are written through a temporary file and atomic replace. They
contain:

- unwrapped model state;
- optimizer and gradient-scaler state;
- completed iteration and serialized configuration;
- explicit Torch and NumPy generator state for every DDP rank;
- process Torch CPU and all CUDA RNG states for every rank; and
- the stable run identity.

`train(..., resume_from=...)` validates the scientific configuration, run
identity, DDP world size, iteration schedule, checkpoint location, and metrics
prefix before continuing at the stored iteration. Strict-mode interrupted and
uninterrupted runs are bitwise-equal in the end-to-end regression test.

## Synthetic exploitability

`exploitability_of_checkpoint` rebuilds the synthetic bundle, loads the model,
exports exact policy tables, and evaluates exact best responses. It is read
only. `exploitability_curve` expands a checkpoint glob, reads stored iteration
numbers, sorts by iteration rather than filename, evaluates each checkpoint,
and writes rows identical to `exploitability.jsonl`.

Exact evaluation is restricted by the synthetic solver's profile/tree guard.
Large dial games are intentionally rejected rather than approximated under the
same API.

## Poker duplicate-deal arena

`arena` compares two checkpoints with the independent CPU NLHE engine. It
samples exact joint profiles through the sibling evaluator, pairs complement
runouts, and replays the same deal/history uniforms with the model assignment
swapped. Duplicate replays rotate the singled-out button-relative seat to
reduce positional bias.

Fold rewards are public and local. Every showdown, including fixed stipulated
profiles, crosses the evaluator facade for per-stratum shares. The report uses
pair-level payoff differences to produce mean bb/100, a paired Student-t
confidence interval, per-seat results, and the number of physical hands.

## Tests

Run the full suite from the repository root:

```bash
/home/pablob/.venvs/poker/bin/python -m pytest -q
```

Tests fall into four groups:

- contract tests for strict configuration, import direction, shape/dtype/device
  rules, and forbidden local poker math;
- exact/statistical tests for synthetic posteriors, terminal laws, branching,
  and exploitability;
- CPU reference and parity tests for poker rules, model adapters, ranges,
  completion, and GPU-style tensor transitions; and
- integration/device tests for the real sibling facade, CUDA residency, DDP,
  checkpointing, and short training runs.

Optional CUDA or sibling-dependent tests skip when their prerequisites are not
available. GPU tests must follow the shared-server hygiene described above.
