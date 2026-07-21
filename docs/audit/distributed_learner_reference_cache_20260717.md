# Bounded distributed learner and reference-cache acceptance — 2026-07-17

> Historical evidence: these measurements predate the 2026-07-21 sole-path
> storage restoration. The learner features remain supported, but any rollout
> ownership, phase-memory, or readiness claim below describes the retired
> compatibility pipeline and is not current acceptance evidence.

This note records phase-2 implementation and validation evidence for
`docs/todo/refactor/02_distributed_learner_and_reference_cache.md`.

## Implemented contracts

- `pbg/core/learner.py` defines `DistributedFixedRolloutUpdate` with a strict
  `TrainingStorage` boundary, common-order asynchronous fp64 log-mass exchange,
  exact global stratum coefficients, uniform/Walker index-only sampling, and
  arbitrary positive statistical counts independent of physical shapes.
- `ModeLoss` implements only production-v1 clipped PPO, actor value regression
  with optional clipping, entropy, and legal-action
  `KL(current || reference)`. Normal derives return/advantage at loss time;
  sequential RB dereferences one exact posterior row; branched RB forms the
  mean child posterior, evaluates node terms once, and averages sampled-action
  policy terms. Natural outputs are mapped by the fixed `phi`; preflop-sized
  outputs cannot be expanded silently.
- `PhysicalPlanner` consumes only an int32 index vector, preserves repeats and
  order, uses a finite warmed shape set plus an exact tail, and rejects an
  infeasible singleton before learner randomness. Every subspan is gathered,
  forwarded, reduced, backwarded, and released before the next subspan.
- Reference-cache selection searches every finite subset, always including the
  empty set. Feasibility charges final resident destinations and the maximum of
  measured build/training workspaces. Selected destinations are all reserved
  before build forwards; an exact byte-contract/build failure clears every
  partial cache and returns to uncached recomputation without consuming learner
  randomness.
- `pbg/core/optimization.py` owns one fp32 `FlatGradientBuffer`; every trainable
  `.grad` aliases a fixed view. Backward accumulates into that buffer directly,
  one SUM collective runs with no world-size division, and the reduced buffer
  supplies the common finite decision before one unscale/clip/optimizer
  transition. Nonfinite skips zero but do not destroy gradient views.
- Metrics expose statistical rows per partition, physical shape histograms and
  tails, current/reference rows, cache build rows/time/bytes and peaks,
  activation-checkpoint use, exact objective mass, flat-gradient bytes, and
  training allocated/reserved peaks.

The existing end-to-end rollout harness now also uses the direct flat-gradient
owner, so its one- and two-GPU smoke updates exercise the no-copy SUM/finite
gate. Final removal of the older rollout/minibatch compatibility modules is a
phase-3 migration item; they are not accepted by `DistributedFixedRolloutUpdate`.

## CPU and distributed validation

Environment: `/home/pablob/.venvs/poker`.

```text
CUDA_VISIBLE_DEVICES='' /home/pablob/.venvs/poker/bin/python -m pytest -q
  418 passed, 21 skipped in 48.52s
```

The total includes local two-process Gloo tests for unequal rank mass, exact
flat SUM, and identical nonfinite skips. Focused phase-2 tests cover log-domain
stability and empty strata, Walker frequencies, non-quantized variance/time
allocation, repeated indices and one-row tails, cache build-workspace
feasibility, all three loss modes, value clipping choices, zero posterior
entries, cache/recompute equality, split/unsplit objectives and gradients,
immutable-boundary rejection, and direct gradient aliasing.

## CUDA validation

`nvidia-smi` showed GPUs 0 and 1 idle at 6 MiB/0%; GPU 2 initially had another
user's 310 MiB process and was not used. All evaluator-backed commands used the
checked `./tools/poker-python` wrapper and the sibling evaluator remained
read-only.

```text
CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py -k 'tiny_gpu_training_modes and normal'
  1 passed in 3.91s

CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py -k 'tiny_gpu_training_modes and rao_blackwell'
  2 passed in 4.65s

CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_distributed_learner.py -k tiny_cuda
  3 passed (normal, sequential RB, branched RB cache/recompute equality)

CUDA_VISIBLE_DEVICES=0,1 ./tools/poker-python -m pytest -q -s \
  tests/test_distributed_learner.py -k two_cuda_unequal
  1 passed; local masses 0.25/0.75 and statistical counts 3/4

CUDA_VISIBLE_DEVICES=0,1 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py -k tiny_two_gpu
  1 passed in 7.56s
```

Measured real evaluator-backed end-to-end peaks:

| Mode | Roots | Retained targets | Rollout reserved peak | Iteration allocated peak | Iteration reserved peak |
|---|---:|---:|---:|---:|---:|
| normal | 2 | 63,888 | 48,234,496 | 43,776,512 | 71,303,168 |
| sequential RB | 2 | 74,330 | 27,262,976 | 24,259,072 | 33,554,432 |
| branched RB | 1 | 642,246 | 88,080,384 | 43,657,728 | 186,646,528 |

The evaluator-independent hard-cache cases used five statistical rows split as
2/2/1. Each retained reference cache was 72 bytes, the allocator-reserved cache
build peak was 2,097,152 bytes, and training allocated peaks were 47,104 bytes
(normal), 46,592 bytes (sequential RB), and 52,224 bytes (branched RB). The
flat-gradient owner was four bytes in this one-parameter acceptance model.

The final `nvidia-smi` showed GPUs 0 and 1 at 6 MiB and 0% utilization, with no
remaining compute process.

## Concrete phase-3 items

- Remove the quarantined legacy `_IterationData`, reach-minibatch, and optional
  `q_hat` compatibility surfaces after all rollout producers construct the
  phase-1 sealed store directly in the runtime path.
- Route production configuration/calibration into `DistributedLearnerConfig`
  and persist its warmed physical-shape/cache measurements across resumes.
- Run the longer fixed-seed full-deck soak/throughput matrix and tune only
  physical execution knobs; promote its stable peak envelopes into hardware
  profiles.
