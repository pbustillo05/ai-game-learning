# Queued Phase 6 validation and audit

Date: 2026-07-27

Status: passed.

Executable candidate:
`df25d020284c16f4600ae90603ba3de594d7e98c`.

The candidate worktree was clean before and after independent validation and
the final read-only audit. This report, the sequential-mode contract, and the
ledger update were added only after the executable candidate passed both
gates.

## Delivered contract

Phase 6 completes all four sequential mode rows through one generated
composition:

- sampled and Rao–Blackwell self play;
- sampled and Rao–Blackwell best response;
- K-wide preflop and H-wide postflop RB learning;
- exact evaluator-v2 dense/selected completion;
- compact trained/frozen inference and headless frozen storage;
- retained signed folded summaries with raw mass, U/C semantics, event refresh,
  ordinary-transition copy, seal, and hot-owner release;
- selected-seat synthetic probe cutoff and paired mixed/baseline evaluation;
- exact probe target, initialization, logical-work, and report identity;
- approved one-time sequential-RB/probe conversion; and
- explicit FP32 posterior, loss-control, gradient, and optimizer arithmetic.

## CUDA environment

Validation used:

```text
venv                 /home/pablob/.venvs/poker
CUDA toolkit         nvidia/cu13 from that venv
CUDA_VISIBLE_DEVICES 0
GPU                   NVIDIA GeForce RTX 2080 Ti
driver/toolkit        driver 580.173.02 / CUDA 13.0
```

GPU 0 was idle before testing and returned to its 6 MiB, 0% utilization
baseline with no compute process afterward.

## Test evidence

Focused:

```bash
pytest -q tests/test_q06_sequential_modes.py
# 31 passed in 6.35s
```

Cumulative:

```bash
pytest -q \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py \
  tests/test_q03_layered_targets.py \
  tests/test_q04_rollout_transactions.py \
  tests/test_q05_sequential_sampled_training.py \
  tests/test_q06_sequential_modes.py
# 152 passed in 19.04s
```

The targeted authority, native-profiler, workspace, all-four-mode, and
multiway matrix passed 22 tests in 8.38 seconds. The warmed Phase 5 regression
passed 30 tests in 6.10 seconds.

Coverage includes:

- optimizer steps for all four sequential modes;
- sampled and RB three-player best response after folds;
- exact `complete_selected == complete[:, selected]`;
- K/H partitions and deterministic full-deck H-to-K;
- one-, two-, and three-folder signed folded oracles;
- raw folded mass, tight/loose classification, retained-summary validity,
  refresh/copy/release, and learner reuse;
- trained/frozen compaction and absence of frozen learner work;
- selected-fold cutoff, paired trajectory identity, complete seat/button
  blocks, per-seat gains, lower bound, and Student-$t$ interval;
- converter identity preservation, disagreement rejection, and one-time
  destination behavior; and
- strict lowercase SHA-256 configuration.

## Memory and timing

Fresh-process and maximum-geometry probes reported:

| Probe | Result |
|---|---|
| P6/R13 folded, batch 8 | 25,980,416 B peak ≤ 29,532,288 B isolated bound |
| additive family bound | 34,321,280 B for batch 8 |
| full-deck H-to-K, batch 8 | 9,427,456 B peak ≤ 9,444,224 B |
| H-to-K determinism | bit-identical |
| native postflop scalar events | none |

H-to-K admission includes the resident 896,376-byte mask, the 8 MiB GEMM
workspace, per-row posterior/moment scratch, and 64 KiB allocator alignment.
Weak-reference phase instrumentation proved both the RB completion binding and
its CUDA mask dead by align-and-seal and still dead at final backward for HU
self-play RB and three-player BR RB.

## Independent audit

The audit iterated until every critical/high finding was closed. The final
read-only audit of `df25d020284c16f4600ae90603ba3de594d7e98c`
reported PASS with no critical or high-severity blockers.

It independently confirmed:

- compact folded summaries are retained, refreshed only at their event
  boundary, copied otherwise, and consumed after hot-owner release without
  recomputation;
- raw mass and tight-wing mathematics match authority;
- multiway sampled/RB and self-play/BR use the shared engine and learner;
- frozen rows are headless;
- evaluator selected ownership and probe cutoff are exact;
- complete $P^2$ pairing and finite-sample Student-$t$ reporting are enforced;
- conversion identities are reconciled and preserved; and
- all concurrent, resident, first-use, and last-consumer memory is admitted.
