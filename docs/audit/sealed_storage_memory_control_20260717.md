# Sealed storage and memory-root control acceptance — 2026-07-17

This note records phase-1 implementation and validation evidence for
`docs/todo/refactor/01_rollout_storage_targets_memory_control.md`.

## Implemented contracts

- `pbg/core/partition.py` defines immutable partition/completion boundaries,
  exact/model domains, fixed quotient maps, routers, forwards, and byte/work
  contracts without game-name parsing in core.
- `pbg/core/storage.py` uses one immutable tensor schema for allocation,
  capacity bytes, validation, diagnostics, and exact sealing. Chunked arenas
  expose stable logical row ids and reserve source/destination/scratch bytes
  transactionally. Sealing allocates each exact destination once and uses one
  bounded reusable copy slab.
- `pbg/core/topology.py` supplies chronological decision-only sequential CSR
  and the six-column contiguous-child branched span.
- `pbg/core/targets.py` implements normal, sequential-RB, and bounded branched
  target construction plus final immutable sampling metadata. Sequential
  posterior rows are extent-planned and stored once per trajectory/player;
  branched v1 retains no `Q`, `W`, `S`, or parent-posterior target.
- Poker partitions enforce `K=R**2` natural preflop rows and
  `H=choose(4R,2)` exact/postflop rows. Full-deck boundary tests enforce 169
  versus 1326 and the `H -> K` lookup.
- Sequential rollout retains only its live per-root reach, commits decision and
  stopped-leaf reaches at their last consumer, and has no reach replay or
  selectable reach-storage mode. Rollout uses the live policy synchronously in
  evaluation mode; it does not allocate a second behavior-policy snapshot.
- Root width is a common per-GPU value selected only by four phase-specific
  conservative memory forecasts. The v2 checkpoint state records phase
  observations, one-sided envelopes, limiting phase/field, guard, predicted
  and actual peaks, successful streak, and current width. Sampled iterations
  have no retry nonce, decision-target width control, or reroll/backoff loop.

## CPU and distributed validation

Environment: `/home/pablob/.venvs/poker`.

```text
python -m compileall -q pbg tests
python -m pytest -q tests --ignore=tests/test_p13b_ddp.py -m 'not gpu'
  402 passed, 17 skipped in 42.83s

python -m pytest -q tests/test_p13b_ddp.py
  4 passed in 9.01s
```

Focused storage/target/controller/integration runs were included in the full
totals. They cover accounting equality and rollback, multi-chunk seal order and
slack, both topology formats, lambda endpoints, zero posterior mass,
posterior deduplication, fixed-phi lookup, bounded branched frontiers, root
multiplicity, phase binary search/rank minimum/hysteresis/resume/no-reroll
failure, logical-RNG schedule invariance, and real sibling-evaluator execution.

## CUDA validation

`nvidia-smi` before work showed GPUs 0 and 1 idle at 6 MiB and 0% utilization.
All commands used `CUDA_VISIBLE_DEVICES` and `./tools/poker-python`; no package
was installed and the sibling evaluator repository was not modified.

```text
CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q \
  tests/test_p13a_gpu_rollout.py::test_poker_actual_cost_keeps_physical_pack_indices_on_device \
  tests/test_p13a_gpu_rollout.py::test_real_sibling_execution_poker_tiny_cuda \
  tests/test_p13a_gpu_rollout.py::test_cuda_legal_event_conditioning_survives_illegal_type_underflow \
  tests/test_p13a_gpu_rollout.py::test_cuda_outputs_and_sibling_inputs_remain_on_original_device
  4 passed in 2.93s

CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py
  one-GPU mode cases: 3 passed

CUDA_VISIBLE_DEVICES=0,1 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py::test_tiny_two_gpu_common_width_and_collective_update
  1 passed in 7.34s
```

Measured one-GPU learner peaks:

| Mode | Roots | Retained target bytes | Rollout peak reserved | Iteration peak allocated | Iteration peak reserved |
|---|---:|---:|---:|---:|---:|
| normal | 2 | 63,888 | 48,234,496 | 43,737,600 | 71,303,168 |
| sequential RB | 2 | 74,330 | 27,262,976 | 24,185,344 | 33,554,432 |
| branched RB | 1 | 642,246 | 88,080,384 | 43,618,816 | 186,646,528 |

The final `nvidia-smi` showed GPUs 0 and 1 at 6 MiB and 0% utilization and GPU
2 at 16 MiB/0%, with no remaining compute process. The pre-existing process on
GPU 2 was never selected or touched by these tests and ended independently.

## Phase-2 handoff

The partition ids, sealed row ids, `TrainingStorage`, per-partition sampling
metadata, and checkpointed common root width are stable inputs for the final
distributed learner. Phase 2 should consume these strata directly and may add
an optional reference cache only through a separate transactional admission;
the uncached optimization phase remains the root-width safety baseline.
