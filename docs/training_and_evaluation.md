# Training, evaluation, and configuration

## Environment

Use Python 3.10+, PyTorch, NumPy, PyYAML, and the separately installed sibling
poker evaluator. On the shared server use `/home/pablob/.venvs/poker`; never
install into system Python. `tools/poker-python` selects the checked CUDA
toolkit and verifies the sibling `pbg-v3` API before running Python.

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
  sequential_starts: 1536
  beta: null
  node_budget: 64
  reach_storage: recompute
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
completion samples, `sequential_starts: 1`, and a beta mapping containing a
finite `default >= 1`.

In sequential modes, `sequential_starts` is a global count partitioned into
contiguous rank quotas. Branched mode always creates one distinguished root per
rank. Completion and backward reduction stream inside a fixed rollout; removed
keys such as `executor`, `n_trees`, `trajectory_chunk_size`, and
`queue_packing` are rejected rather than ignored.

Hardware profiles are separate execution envelopes. They control attention,
precision, token budgets, VRAM reserve, and device limits without changing the
checkpoint's scientific identity.

## Training and resume

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
checkpoints. A checkpoint includes the model, optimizer, scaler, completed
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
