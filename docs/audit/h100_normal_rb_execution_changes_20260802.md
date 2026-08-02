# H100 Normal/RB execution changes, 2026-08-02

## Purpose and rollback boundary

This log records the implementation-only changes made before benchmarking HU
32bb Normal and RB on RunPod H100s and then Quest H100s. It intentionally does
not change the mathematical objective, MHA architecture, or model dimensions.

- Pre-change snapshot: `be74898` (`Snapshot queued runtime cleanup before H100 tuning`)
- Implementation commit: `ef22ee2` (`Fix sequential sampling and prepare K1 H100 runs`)
- Audit-log commit: the commit containing this file

To revert the complete implementation while retaining this audit record:

```bash
git revert ef22ee2
```

The sections below also list file groups that can be restored selectively from
`be74898` and committed as an ordinary forward-moving revert.

The selective commands are file-level. `pbg/cli/h100_pilot.py` participates in
both the execution-profile and progress-report changes, so restoring that whole
file reverts both. For a narrower rollback, start with
`git revert --no-commit ef22ee2`, retain only the unwanted hunks in the working
tree, and commit the result after rerunning the affected tests.

## 1. Sequential decision sampling correctness

### Previous behavior

For fixed-step sequential learning, `IterationEngine` automatically constructed
a cost-aware family allocation. With representative 512-root retained rows, it
allocated approximately 239/17 Normal rows and 238/18 RB rows to pre/post,
instead of their approximately 76/24 and 74/26 retained decision masses. The
learner weighted those rows by allocated batch fraction without correcting for
the different stratum selection probabilities. This changed the effective
decision objective. The no-allocation fallback alternated whole-family batches
and was not pooled-uniform either.

### New behavior

- Default fixed-step sequential sampling draws `batch_size` indices from the
  complete pooled retained decision extent with replacement.
- Draws are grouped by input family only for model execution. Every grouped
  family batch retains one common logical optimizer-step identity.
- Physical microbatch splitting weights every slice by its exact fraction of
  the pooled logical batch.
- Explicit cost-stratified allocation remains available, but its family batches
  now carry normalized target-mass weights. Physical splitting preserves those
  weights.
- The automatic token-cost allocation was removed from sequential fixed-step
  iteration preparation. Branched allocation behavior was not changed.

Files:

- `pbg/core/learner.py`
- `pbg/core/iteration.py`
- `pbg/core/training.py`
- `tests/test_sequential_sampled_training.py`

Selective rollback:

```bash
git restore --source=be74898 -- \
  pbg/core/learner.py \
  pbg/core/iteration.py \
  pbg/core/training.py \
  tests/test_sequential_sampled_training.py
git commit -m "Revert pooled sequential sampler"
```

### Known trade-off

Pooled draws are IID-with-replacement, so a logical batch may contain duplicate
retained decisions or omit a small family. This is intentional and unbiased for
the pooled decision law. Explicit cost-stratified weighting assumes the supplied
`mass` vector describes the target stratum mass; callers must not substitute a
proposal distribution for that value.

## 2. K=1 H100 execution profile

The scientific batch and optimizer schedule remain 256 logical rows and 116
steps. Only physical execution capacities changed.

Production K=1 candidate:

| Setting | Old | New |
|---|---:|---:|
| rollout rows | 84 | 256 |
| completion rows | 256 | 512 |
| learner rows | 84 | 256 |
| immutable reference outputs | off | auto |
| measured repeats | 2 | 3 |

The H100 arm now represents rollout, completion, and learner packs as three
independent fields and CLI flags. Local smoke scales them down independently.
The checked-in BF16 smoke uses rollout/completion packs 4/16 and still exercises
the 256-row learner admission. A short batch naturally uses fewer physical rows.

The automatic pilot now benchmarks K=1 only. K=2/K=4/K=8 types and selection
utilities remain available for deliberate experiments, but serial learner lanes
are no longer part of the default paid pilot. Reference caching is automatically
enabled only for K=1 because the current cache budget is not replica-aggregated.

Files:

- `pbg/h100_pilot.py`
- `pbg/cli/h100_pilot.py`
- `configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_normal_h100_bf16_smoke.yaml`
- `configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_rb_h100_bf16_smoke.yaml`
- `configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_normal_h100_replica_k1.yaml`
- `configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_rb_h100_replica_k1.yaml`
- `tests/test_h100_pilot.py`
- `tests/test_h100_launch_blockers.py`

Selective rollback:

```bash
git restore --source=be74898 -- \
  pbg/h100_pilot.py \
  pbg/cli/h100_pilot.py \
  configs/experiments/nlhe_hu_32bb_h100_pilot \
  tests/test_h100_pilot.py \
  tests/test_h100_launch_blockers.py
git commit -m "Revert K1 H100 execution profile"
```

### Known trade-off

The 256/512/256 profile is an H100 benchmark candidate, not a claim that these
are final optima. RunPod must measure real peak allocated/reserved memory,
per-phase time, and Flash SDPA dispatch before Quest production. In particular,
`reference_cache: auto` has CUDA equivalence coverage but has not yet completed
a full 116-step iteration on an H100 in this commit.

## 3. K=1 launch and SM90 extension isolation

The RunPod selection CLI, replica-training CLI, and Slurm wrapper now accept one
positive lane instead of requiring at least two. This lets the same strict
manifest path launch K=1 Normal and RB jobs.

The RunPod and Slurm shell wrappers default `TORCH_CUDA_ARCH_LIST` to `9.0` and
place `TORCH_EXTENSIONS_DIR` in an SM90-specific temporary directory. Both can
still be overridden by an explicit environment value. This avoids stale
cross-architecture extension reuse and shared compile-directory collisions.

Files:

- `pbg/cli/runpod_replica_selection.py`
- `pbg/cli/train_replicas.py`
- `tools/run-runpod-replica-selection.sh`
- `tools/run-replica-training-slurm.sh`
- `tests/test_runpod_replica_selection.py`

Selective rollback:

```bash
git restore --source=be74898 -- \
  pbg/cli/runpod_replica_selection.py \
  pbg/cli/train_replicas.py \
  tools/run-runpod-replica-selection.sh \
  tools/run-replica-training-slurm.sh \
  tests/test_runpod_replica_selection.py
git commit -m "Revert K1 H100 launcher support"
```

## 4. Atomic benchmark progress reports

`benchmark_training` accepts an optional progress path. It atomically reports
the active warmup/repeat, completed counts, measured samples, failure details,
and final medians. H100 arm execution writes this next to the final report as
`*.progress.json`. A killed subprocess therefore retains completed repeat data
even if the final arm report was not produced.

Files:

- `pbg/benchmark.py`
- `pbg/cli/h100_pilot.py`
- `tests/test_consumers.py`

Selective rollback:

```bash
git restore --source=be74898 -- \
  pbg/benchmark.py \
  pbg/cli/h100_pilot.py \
  tests/test_consumers.py
git commit -m "Revert atomic H100 benchmark progress"
```

### Known limitation

Progress is updated at warmup/repeat boundaries. It is not a mid-iteration phase
heartbeat. A process killed inside one long iteration can lose that active
iteration's partial phase data, while previously completed samples remain safe.

## Verification evidence

Shared-server hygiene:

- GPU state checked before CUDA work.
- Tests explicitly targeted idle GPU 0.
- Final GPU state returned to the display-only idle baseline: 6 MiB allocated,
  0% utilization, no test process remaining.

Commands and results:

```bash
POKER_GPU=0 ./tools/poker-python -m pytest \
  tests/test_sequential_sampled_training.py \
  tests/test_h100_pilot.py \
  tests/test_h100_launch_blockers.py \
  tests/test_runpod_replica_selection.py \
  tests/test_consumers.py -q
# 56 passed

POKER_GPU=0 ./tools/poker-python -m pytest -q
# 357 passed, 1 skipped

python -m ruff check <changed Python files>
# All checks passed

bash -n tools/run-replica-training-slurm.sh \
  tools/run-runpod-replica-selection.sh
# passed

git diff --check
# passed
```

The CUDA suite ran on a local RTX 2080 Ti and verifies CUDA behavior and
contracts. It does not replace the required RunPod H100 BF16, memory, Flash
SDPA, checkpoint/resume, and throughput gates.

## Deliberately deferred work

The implementation commit does not include:

- core math, loss-definition, MHA, or model-dimension changes;
- dense Adam CUDA fusion or CUDA-graph capture;
- tighter sequential topology/partition capacity proofs;
- an automatic root-count change from 512 to 4096;
- a change to 1024 RB completion samples or reference-refresh thresholds;
- concurrent K>1 learner execution;
- a runtime proof that the H100 SDPA call selected Flash kernels.

Those should be evaluated from RunPod measurements rather than folded into the
correctness and launchability rollback unit.
