# Training, evaluation, and configuration

## Environment

Use Python 3.10+, PyTorch, NumPy, PyYAML, and the separately installed sibling
poker evaluator. On the shared server use `/home/pablob/.venvs/poker`; never
install into system Python. `tools/poker-python` selects the checked CUDA
toolkit and verifies the sibling contract-v1 API before running Python.

## Strict NLHE configuration

`pbg.config.load_config` is the public loader. It recursively validates the
typed schema and reports unknown, missing, mistyped, or invalid fields with a
dotted path. There is no game name or implementation selector.

```yaml
game:
  n_players: 6
  stack_bb: 10.0
  rank_count: 13
rollout:
  sequential_roots_per_gpu_initial: 1536
  sequential_roots_per_gpu_minimum: 512
  sequential_roots_per_gpu_maximum: 2048
  beta: null
  node_budget: 64
  family_batch_rows: 128
  max_active_trajectories: 1536
  arena_chunk_nodes: 4096
  execution_memory_bytes: null
```

`game` is equal-stack NLHE with $P$, $S$, and $R$ subject to the repository
contracts. `training.mode` is `normal`, `rao_blackwell`, or
`branched_rao_blackwell`. Normal mode requires `ppo`, `completion.n_samples:
null`, and `rollout.beta: null`. Sequential RB requires `rb_ppo`, positive
completion samples, and `beta: null`. Branched RB uses `rb_ppo`, positive
completion samples, all three sequential-root fields set to `1`, and a beta mapping containing a
finite `default >= 1`. It must also set `rollout.branching_mode` to `fixed` or
`dynamic`, enable `model.use_branching_heads`, and use positive `loss.c_t` and
`loss.c_c`. Fixed mode follows the beta map for the whole run. Dynamic mode
requires `rollout.adaptive_kkt`, follows the beta map during warmup, and enables
T/C-driven KKT rollout control only afterward. The legacy
`adaptive_kkt.enabled` switch is rejected.

In sequential modes, the controller selects one common per-GPU root width from
the phase-wise memory budget before any current-iteration draw. Unsafe evidence
decreases it immediately; successful evidence widens it only after the configured
delay. Branched mode always creates one distinguished root per rank. Completion
and backward reduction stream inside a fixed rollout; removed
keys such as `executor`, `n_trees`, `trajectory_chunk_size`, and
`queue_packing` are rejected rather than ignored.

Hardware profiles are separate execution envelopes. They control attention,
precision, token budgets, VRAM reserve, and device limits without changing the
checkpoint's scientific identity.

## Training and resume

For every shared-server run, use the repository launcher. It preserves stdout
and stderr, records start/end GPU snapshots and exit status, and can run in a
detached `tmux` session:

```bash
tools/run-training.sh \
  --config configs/train/nlhe_hu_32bb_sequential_rb_candidate.yaml \
  --hardware-profile configs/hardware/local_2x2080ti_12h.yaml \
  --run-dir runs/nlhe_hu_32bb_sequential_rb \
  --gpus 0,1 --resume runs/nlhe_hu_32bb_sequential_rb/ckpt_0000010.pt \
  --tmux
```

The launcher writes timestamped stdout, stderr, combined-output, and command
files, plus `status.json`, `gpu_start.csv`/`gpu_end.csv`, and sampled
`gpu_telemetry_*.csv` (temperature, utilization, memory, and power) in the run
directory. This preserves separate evidence for every resume attempt. The
status records the final exit code and any signal forwarded to `torchrun`.
Check a detached run after reconnecting with:

```bash
tools/run-status.sh runs/nlhe_hu_32bb_sequential_rb
tmux attach -t pbg-nlhe_hu_32bb_sequential_rb
```

The candidate sequential-RB run checkpoints every 5 iterations. The v3
checkpoint includes storage/controller schema metadata, distributed logical
RNG state, optimizer/reference schedule, loss scale, and memory-root controller
state needed for deterministic resume. A recent unversioned v2 checkpoint is
migrated as metadata only; incompatible or partial state is rejected on CPU
before GPU allocation. If a selected checkpoint predates metrics already flushed after
it, resume trims that uncheckpointed suffix and preserves it as
`metrics.jsonl.before_resume_*.bak`.

## Training diagnostics

Every training iteration records major phase wall time and CUDA peak memory
for rollout, seal, completion, backward estimator, and the
full optimization phase. The optimization phase also records host-side time
for minibatch sampling, forward/loss computation, backward propagation,
gradient synchronization, and the optimizer update. These subphase timings
are intentionally marked `*_host_seconds`: CUDA launches are asynchronous, so
making each subphase GPU-exact would require synchronization at every
minibatch and would materially distort throughput.

In Rao--Blackwell modes, completion is streamed inside the backward-estimator
phase. `profiled_iteration_seconds` therefore uses top-level phases only and
does not double-count completion;
`phase_backward_estimator_seconds_excluding_completion` provides the derived
attribution.

For poker rollouts, `decision_nodes_{preflop,flop,turn,river}` and the matching
`decision_fraction_*` fields describe the decision-node mix. In distributed
runs, use the `*_global` fractions and `*_rank_sum` counts. Existing
`execution_*` fields report evaluator calls, transfers, collective bytes and
time, queue pressure, synchronization, padded inference work, and execution
memory. `diagnostic_metric_summary_sync_*` additionally covers the small
cross-rank all-reduces used to produce global diagnostic summaries. Together
with the launcher GPU telemetry, these are intended to
separate game-tree mix, estimator quality, model cost, communication cost, and
hardware saturation.

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti
```

Resume into the same run directory and gap-free metrics history:

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti \
  --resume runs/nlhe6/ckpt_0000100.pt
```

For multiple ranks, launch the same CLI with `torchrun`. Every rank must receive
at least one sequential start. The learner globally sums decision mass and one
flat gradient buffer, so unequal rank work is expected and supported.

Rank zero writes `run.json`, append-only `metrics.jsonl`, and atomic
checkpoints. A versioned checkpoint includes the model, optimizer, scaler, completed
iteration, scientific configuration, run id, explicit RNG state for every
rank, and reference-policy state when KL is active. Resume checks configuration,
world size, run identity, location, iteration schedule, and exact metrics
prefix before continuing.

## GPU hygiene

Before CUDA work:

1. inspect `nvidia-smi`;
2. select only idle GPUs explicitly;
3. use the shared poker virtualenv/toolkit;
4. start with a tiny smoke and retain the hardware-profile reserve;
5. increase starts, node budget, completion samples, or minibatch size only
   after observing memory; and
6. confirm device memory and processes return to idle afterward.

On the local server, two-rank acceptance uses idle RTX 2080 Ti GPUs 1 and 2.

## Performance profiles

`pbg-benchmark` runs the same production implementation in each requested mode
and reports topology, throughput, retained bytes, phase timings, synchronization,
and CUDA peaks.

```bash
pbg-benchmark --config configs/benchmarks/nlhe_hu_short.yaml \
  --hardware-profile local_3x2080ti \
  --warmup 2 --repeats 7 --output /tmp/nlhe_profile.json
```

`--checkpoint PATH` initializes every mode from one read-only model state and
records its identity. Baseline comparison now compares versions of the same
production implementation; it does not select or compare rollout executors.

## Duplicate-deal evaluation

`pbg.eval.arena` compares checkpoints with the independent CPU NLHE engine. It
samples collision-free profiles through the sibling evaluator, pairs complement
runouts, swaps model assignments, and rotates the singled-out button-relative
seat. Fold chip rewards are public, while folded-card posteriors and every
showdown remain evaluator-owned. Reports include paired bb/100 confidence
intervals and per-seat results.

## Sequential best-response probes

An optional top-level `probe` block turns sequential Rao--Blackwell training
into a shared-seat approximate best response against one frozen checkpoint:

```yaml
probe:
  target_checkpoint: runs/target/ckpt_0000100.pt
  target_sha256: null       # optional pin; resolved and stored at launch
  initialization: random    # or target for identical architectures
```

The target must have identical `(n_players, stack_bb, rank_count)`. Its model
size and factorized action-head size may differ. Probe checkpoints record the
resolved target SHA-256; resume and evaluation reject changed target contents,
while allowing the target file to be moved.

Evaluate one or more probe checkpoints with paired GPU rollouts (heads-up
example):

```bash
pbg-br-eval --probe runs/probe/ckpt_0000100.pt \
  --hardware-profile local_3x2080ti --hands 10000 --batch-size 100
```

`--hands` and `--batch-size` must be multiples of $P^2$ so exploiter seats and
button positions are exactly balanced. The headline mean unilateral gain is
an approximate exploitability lower bound; the report also includes its paired
confidence interval, per-seat gains, and approximate NashConv.
It also records paired evaluation work and wall time plus the probe
checkpoint's cumulative logical trajectories, completion samples, minibatches,
and optimizer steps. If omitted, `--batch-size` is chosen automatically as a
multiple of $P^2$ no larger than 256.

## Validation

Run the CPU suite from the repository root:

```bash
/home/pablob/.venvs/poker/bin/python -m pytest -q
```

CUDA and distributed tests skip without prerequisites. Required coverage
includes strict retired-key rejection, all three modes, topology and beta,
logical RNG invariance, completion/evaluator integration, old-policy identity,
unequal global mass, finite reduced gradients, checkpoint/resume, and memory
preflight. Follow the staged GPU procedure above for device tests.
