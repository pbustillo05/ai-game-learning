# Training, evaluation, and configuration

## Installation

The project requires Python 3.11+, NumPy, PyYAML, pytest, and PyTorch 2.2 or
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
- `rollout`: tree count per rank, `sequential`/`fixed` beta mode, public-stage
  beta map, per-tree node budget, and reach storage mode.
- `completion`: samples per leaf and backend selection.
- `model`: game-owned network dimensions.
- `loss`: `rb_ppo`, lambda, PPO/value clipping, value/entropy coefficients,
  entropy schedule, and optional suit-consistency flag.
- `train`: iterations, gradient steps, minibatch size, Adam settings, AMP,
  seed, checkpoint cadence, and output directory.
- `debug_asserts`: enables expensive validation where supported.

Every `beta_fix` map requires `default`, and every intensity must be finite and
at least one. `reach_storage` is `full`, `decisions_only`, or `recompute`.

## Shipped presets

- `configs/kuhn_smoke.yaml`: small sequential Kuhn training and checkpoint
  smoke run.
- `configs/leduc.yaml`: fixed branching on Leduc, with more branching after
  public chance.
- `configs/synthetic_dial_example.yaml`: three-player configurable game using
  decision-only reach storage.
- `configs/nlhe6_h100.yaml`: six-player 100bb NLHE CUDA preset sized as a
  conservative first H100 profile, with recomputed reaches and AMP. Its marked
  values are tuning starting points, not throughput guarantees.

## Running training

There is currently no console-script wrapper. Load a preset and call the public
training function:

```bash
/home/pablob/.venvs/poker/bin/python - <<'PY'
from pbg.core.config import load_config
from pbg.core.training import train

train(load_config("configs/kuhn_smoke.yaml"))
PY
```

For multiple ranks, launch the same entry code with `torchrun`. Tree counts and
minibatch sizes remain per rank; they are not divided by world size.

Before any CUDA run on the shared server, inspect `nvidia-smi`, select only an
idle GPU explicitly, begin with a tiny smoke configuration, and verify that
memory/process state returns to idle afterward. The NLHE preset should be
profiled before increasing trees, node budget, completions, model size, or
minibatch size.

## Metrics and checkpoints

Rank zero appends one JSON object per iteration to `metrics.jsonl`. Reported
fields include total loss, policy/value/entropy components, approximate KL,
clip fraction, exact policy entropy, explained variance, and maximum active PPO
ratio.

Checkpoints are written through a temporary file and atomic replace. They
contain:

- unwrapped model state;
- optimizer and gradient-scaler state;
- completed iteration and serialized configuration;
- explicit Torch and NumPy generator state; and
- process Torch CPU and all CUDA RNG states.

`load_checkpoint` can restore all training state or only the model. Exact RNG
restoration is covered by tests.

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
