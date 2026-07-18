# Migration cleanup and GPU acceptance — 2026-07-17

## Verdict and revision

The memory-ownership and correctness migration is accepted. Long-run release
approval remains **blocked on throughput**: the final same-machine fixed-width
benchmark is slower than parent `86756b8` by 13.4% in normal mode and 20.5% in
sequential RB. This result is reported rather than hidden behind the roughly
2x retained-target reduction. General adaptive branching beta also remains a
research-only deferred item.

The tested source is the worktree based on `86756b8`; the final commit hash is
reported in the implementation handoff because a commit cannot contain its own
hash. Hardware was two idle RTX 2080 Ti cards:

- GPU 0: `GPU-76849e3b-2611-2c0d-759c-dbaa98a48f86`
- GPU 1: `GPU-5d940064-e460-977c-c144-4649036d59e8`

GPU 2 had another user's 512 MiB process and was excluded from every command.

## Implemented boundary

The production learner now accepts only versioned immutable
`TrainingStorage`. Sealing writes natural preflop inputs at 169 rows and dense
postflop inputs at 1326 rows, exact RB posterior/return/advantage rows remain
1326, sequential parent posteriors are deduplicated by trajectory/player, and
branched actions use CSR plus fp64 Walker metadata. The rollout tree, completion
rows, and decision reach owner are released before learner construction.

The unreachable legacy optimizer and its extra decision-mass collective were
deleted. Production has no reach replay API, reroll/retry loop, retained Q
target, padded maximum-action target axis, target-node root control, or dead
target-reuse schedule. Normal retains one lambda return rather than separate
advantage/value-target copies. The learner uses direct flat-gradient SUM, one
common finite gate, device-resident stratum masses, AMP, bounded physical
subspans, and a model-derived nonzero transient-byte contract.

Checkpoint schema `pbg-checkpoint-v3` records the training-storage and
controller versions. Recent v2 payloads migrate metadata deterministically;
partial RNG/controller state and unknown versions are rejected by a CPU-only
inspection before bundle/GPU allocation. Model, optimizer, scaler, reference
schedule, logical work, per-rank RNG, and controller state round-trip.

## Correctness evidence

CPU-only final suite:

```text
CUDA_VISIBLE_DEVICES='' /home/pablob/.venvs/poker/bin/python -m pytest -q
420 passed, 21 skipped in 48.71s
```

The focused CUDA matrix used the checked `./tools/poker-python` wrapper:

```text
CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py::test_tiny_gpu_training_modes_report_bounded_phase_memory
3 passed

CUDA_VISIBLE_DEVICES=0 ./tools/poker-python -m pytest -q -s \
  tests/test_distributed_learner.py::test_tiny_cuda_bounded_update_cache_and_recompute \
  tests/test_p13a_gpu_rollout.py::test_poker_actual_cost_keeps_physical_pack_indices_on_device \
  tests/test_p13a_gpu_rollout.py::test_real_sibling_execution_poker_tiny_cuda \
  tests/test_p13a_gpu_rollout.py::test_cuda_legal_event_conditioning_survives_illegal_type_underflow \
  tests/test_p13a_gpu_rollout.py::test_cuda_outputs_and_sibling_inputs_remain_on_original_device
7 passed

CUDA_VISIBLE_DEVICES=0,1 ./tools/poker-python -m pytest -q -s \
  tests/test_refactor_gpu_acceptance.py::test_tiny_two_gpu_common_width_and_collective_update \
  tests/test_distributed_learner.py::test_two_cuda_unequal_strata_exact_sum
2 passed
```

Two-rank full-deck production smokes also passed: sequential RB ran two
iterations with four global trajectories and 11 decisions each iteration,
117,440,512 rank-max reserved bytes, and identical collective optimizer
decisions; fixed-beta branched RB ran exactly one tree/rank with four global
decisions and 31,457,280 rank-max reserved bytes. The branched root remains row
zero and uses ordinary randomized-rounded multiplicity. Cache and uncached
learner paths, repeated indices, exact tails, unequal/empty strata, injected
nonfinite gradients, and SUM semantics are covered by the suites above.

Full-deck and compact-deck tests cover $R=4,\ldots,13$. They assert the 169
full-deck preflop quotient, canonical 1326 postflop/exact domain, impossible-hand
zeros, multiple active-player counts, immutable folded-summary anchors, and no
anchor for `post_P`. Completion dtype/layout tests continue to call only the
read-only sibling evaluator API.

## Four-phase ledger and soak

A final full-deck sequential-RB ledger used four roots, eight completion
samples, a 32-wide one-layer model, and the local 2080 Ti hardware profile. The
second warm iteration reported:

| phase | predicted | allocated peak | reserved peak | hard-ceiling headroom |
|---|---:|---:|---:|---:|
| rollout | 25.28 MiB | 60.76 MiB | 76.00 MiB | 8,696 MiB |
| seal | 74.00 MiB | 30.12 MiB | 76.00 MiB | 8,696 MiB |
| targets | 14.54 MiB | 30.63 MiB | 76.00 MiB | 8,696 MiB |
| optimization | 42.00 MiB | 100.45 MiB | 148.00 MiB | 8,624 MiB |

The hard operating ceiling was 8,772 MiB with a separate 2,048 MiB guard.
Iteration end was 29.40 MiB allocated and 148.00 MiB reserved. External NVML
was checked before, between, and after jobs; GPU 0/1 returned to the 6 MiB idle
baseline. No polling sampler was used to claim a finer NVML high-water value;
the process-specific allocated/reserved peaks above are the authoritative peak
measurements.

The retained training store was 203,304 bytes: preflop 64,912 bytes (8 rows,
10,848 input bytes), postflop 106,504 bytes (4 rows, 42,448 input bytes), and
31,888 bytes of deduplicated posterior/sampling metadata. Calibrated physical
shapes were two rows per active partition; the declared transient bounds were
18,683,736 bytes preflop and 581,382,576 bytes postflop. There were two physical
subspans, zero tails, and four model rows.

Six-iteration one-GPU normal and sequential-RB soaks checkpointed at iterations
3 and 6. Sequential reserved memory stabilized at 155,189,248 bytes after warmup
and end allocation stabilized near 30.8 MiB. The controller widened 4 -> 5 roots
after its delay. Resume from checkpoint 3 repeated normal decision counts
`5/16/5` at widths `4/4/4` and sequential counts `7/17/12` at widths `4/5/5`.
Throughput FP16 kernels changed only low-order loss digits; logical addresses,
counts, and next widths matched. Controller evidence now uses allocated phase
peaks so allocator-bin history cannot change a resumed width; reserved/NVML
remain hard-guard diagnostics. No real OOM or `empty_cache()` ownership scheme
was used.

## Throughput comparison

The benchmark used GPU 0, full deck, HU 8bb, 32 roots/GPU, fixed width, 8-row
family packs, a 32-wide one-layer model, minibatch 16, identical precision and
completion budgets, one warmup plus three timed iterations. `86756b8` was run
from a clean `git archive` on the same machine. Because learner index draws
changed during the migration, later fixed-seed rollout objects are distribution-
equivalent rather than address-identical; aggregate counts are shown explicitly.

| mode | revision | decisions | timed seconds | decisions/s | max retained targets |
|---|---|---:|---:|---:|---:|
| normal | `86756b8` | 216 | 1.984 | 108.90 | 915,728 B |
| normal | candidate | 253 | 2.682 | 94.34 | 529,888 B |
| sequential RB | `86756b8` | 270 | 2.681 | 100.69 | 3,642,170 B |
| sequential RB | candidate | 262 | 3.272 | 80.06 | 1,766,688 B |

For normal, candidate timed totals were rollout 1.809 s, targets 0.063 s,
seal 0.035 s, and learner 0.775 s; the parent learner took 0.355 s. For
sequential RB they were rollout 1.874 s, backward/completion 0.360/0.134 s,
seal 0.075 s, and learner 0.963 s; the parent learner took 0.512 s. The normal
loss hot path was improved by avoiding a needless 1326 expansion and duplicate
log-probability call, and per-partition alpha host synchronizations were removed,
but the remaining learner regression is material. This is the sole release
blocker found by phase 3 beyond adaptive-beta research.
