# Queued Phase 10 validation and audit

Date: 2026-07-28

Status: passed.

Clean phase start:
`2aaaab17eeac6e53f12459a7fb9ff057350a1276`.

Final executable candidate:
`944d097e221b90a075c5c813de9f604bfa38dfae`.

Candidate tree:
`92b4298d3177b8d403c9d22f40321a84aed0a24a`.

The candidate worktree was clean before and after independent validation and
the final read-only audit. This report, the runtime/consumer contract, and the
ledger update were added only after both gates passed.

## Delivered contract

Phase 10 migrates every accepted public product to one strict queued
configuration and `TrainingSession` composition. It covers ordinary,
distributed, and replica training; best-response training/evaluation;
CUDA/MPS inference gates; strict model-only export; variance; matched-work
benchmarks; strict checkpoint/resume/failure/preemption; observability; and
operational launchers.

The validation/audit cycle additionally restored actual evaluator `n_eff`,
the preserved pre-iteration interval reference cadence, effective-hand
overshoot/resume semantics, exact per-rank/global logical work, JSON-stable
matched-work identity, strict nested `RunState` validation, exact operational
checkpoint selection, and rank-common optimizer-call scheduling for disjoint
distributed work.

## CUDA environment and isolation

Independent validation used a read-only archive of the exact candidate:

```text
/tmp/phase10-validation-944d097-source.ZjnuF0
```

All imports were pinned to that archive. The environment was:

```text
venv                 /home/pablob/.venvs/poker
Python               3.10.12
PyTorch              2.12.1+cu130
CUDA/driver           13.0 / 580.173.02
GPU                   NVIDIA GeForce RTX 2080 Ti, capability 7.5
evaluator             0.6.0, contract v2
evaluator revision    9d35191b396cca5bf07518a07ba2976b11971a5a
logical RNG           counter_hash_v4
```

The primary fixture was HU, 8 bb, four ranks, self-play/sequential/sampled,
FP32 AdamW with strict reproducibility, root count 1, and seed `104830`.
Distributed rank 0/1 used logical bases `0` and `65970697666560` and explicit
generator seeds `104830` and `1104833`.

GPU 0 was explicitly selected for single-device gates; GPUs 0 and 1 were
explicitly selected for NCCL gates. Before and after validation, GPUs 0/1 were
at their 6 MiB display baseline, GPU 2 was at its 16 MiB display baseline, all
showed 0% utilization, and no compute process remained.

## Test evidence

Focused consumers and inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs \
  tests/test_q10_consumers.py \
  tests/test_inference_api.py
# 19 passed, 1 unavailable-MPS skip in 65.10s
```

Cumulative queued phases:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py \
  tests/test_q03_layered_targets.py \
  tests/test_q04_rollout_transactions.py \
  tests/test_q05_sequential_sampled_training.py \
  tests/test_q06_sequential_modes.py \
  tests/test_q07_branched_sampled_modes.py \
  tests/test_q08_branched_rb_modes.py \
  tests/test_q09_replica_identity.py \
  tests/test_q10_consumers.py
# 246 passed, 1 separately covered two-GPU skip in 223.17s
```

Dedicated distributed gate:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q -rs \
  tests/test_q09_replica_identity.py::\
test_distributed_nccl_sum_and_asymmetric_failure_agreement
# 1 passed in 5.40s
```

All four sequential and all four branched mode variants used by the
effective-hand gate passed:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_q06_sequential_modes.py::\
test_all_four_sequential_modes_complete_an_optimizer_step \
  tests/test_q08_branched_rb_modes.py::\
test_four_branched_rb_modes_complete_optimizer_step
# 8 passed in 6.82s
```

RB rows returned positive actual posterior effective hands no greater than
`leaves * n_samples`; sampled rows returned no completion exposure.

## Real distributed execution

Fresh and resumed two-rank production runs used:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/phase10_distributed_smoke.py \
  --out /tmp/phase10-dist-944d097 --iterations 1
# PASS, 6.381s

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/phase10_distributed_smoke.py \
  --out /tmp/phase10-dist-944d097 --iterations 2 \
  --resume /tmp/phase10-dist-944d097/checkpoint_00000001.pt
# PASS, 6.202s
```

Rank-local first-iteration player nodes were 5 and 3, proving disjoint work.
All 174 model tensors, optimizer state, and scaler state were exactly equal
across ranks; explicit RNG states differed. Iterations 1 and 2 published
strict world-size-two manifests and content-addressed rank files.

Iteration-zero per-rank logical work was
`{trajectories: 1, completion_samples: 0, minibatches: 2,
optimizer_steps: 2}` on each rank. Global work was
`{trajectories: 2, completion_samples: 0, minibatches: 4,
optimizer_steps: 2}`. At iteration one, cumulative per-rank work was
`{2, 0, 3, 3}` and global work was `{4, 0, 6, 3}` in key order. Rank files
reported identical aggregate fields.

## Cadence, persistence, and artifacts

The two-lane effective-hand fixture used a threshold of `0.5` and proved no
refresh before the first iteration, refresh after completion, positive
overshoot, reset to zero, and next reference iteration 1. Resume preserved
deliberately distinct lane states `(7, 13.5)` and `(9, 21.5)`. Distributed
interval cadence refreshed before iteration 0 and not before iteration 1.

Strict rank-checkpoint probes rejected empty, negative, and extra-field
logical-work mappings. Distributed manifests rejected modified rank content.
Inference export accepted only queued checkpoints/manifests and explicit
replica lanes; offline legacy inference conversion emitted source/result
hashes while runtime loading rejected legacy/permissive artifacts.

The operational launcher selected only `checkpoint_00000001.pt` and
`failure_checkpoint_00000003.pt` from a fixture also containing rank shards.

## Benchmark and memory evidence

The production matched-work command was:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pbg.cli.benchmark \
  --config configs/train/nlhe_hu_8bb_normal_queued_smoke.yaml \
  --warmup 1 --repeats 2 \
  --output /tmp/phase10-benchmark-944d097.json
```

The report parsed as JSON and compared equal with its JSON round trip. Both
samples moved from logical work `{0,0,0,0}` to `{1,0,2,2}` and included the
required structure and telemetry. Median time was `0.351853` seconds for five
player nodes (`14.210479` nodes/s), with 201,216 incremental allocated bytes,
30,920,192 peak allocated bytes, and 52,428,800 peak reserved bytes. A combined
performance, JSON, reachability, launcher, and lifecycle selection passed 16
tests in 2.61 seconds.

Seven construct/step/close/delete/GC/synchronize cycles showed no continuing
allocation or workspace growth. After warmup, cycles 1–6 were byte-identical:
26,601,472 bytes before construction, 26,802,688 after deletion, 30,920,192
peak allocated, 52,428,800 peak reserved, and 25,559,040 bytes released by
each session.

## Independent audit

The mandatory final audit used the immutable archive:

```text
/tmp/phase10-final-audit-oXiwxb
```

It reported PASS for exact SHA
`944d097e221b90a075c5c813de9f604bfa38dfae` and tree
`92b4298d3177b8d403c9d22f40321a84aed0a24a`.

The auditor traced actual `Completion.n_eff`, both reference cadence laws,
rank/lane/global work, distributed step budgets and failure agreement, strict
persistence and inference artifacts, BR/variance/benchmark consumers,
accelerator gates, operational checkpoint discovery, and last-consumer
release. Accepted-root reachability had no missing root and no path to
`pbg.core`; migrated-body similarity returned no violation. No critical/high
ownership, memory, preservation, scheduling, artifact, or consumer finding
remained.
