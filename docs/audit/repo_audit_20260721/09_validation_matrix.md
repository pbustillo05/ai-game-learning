# Validation matrix

## Verdict

All 457 collected non-CUDA tests pass when the three sandbox-blocked Gloo tests
are rerun with network permission. Tiny GPU evidence supports three training
semantics, KKT/T/C, queue transactions, the sibling boundary, and two-rank
exact-mass/SUM. It does **not** establish five-mode NVIDIA acceptance.

| Validation | Command/result | Disposition |
|---|---|---|
| Collection | `env CUDA_VISIBLE_DEVICES= ... -m pytest --collect-only -q`: 489 collected | 29 CUDA and 3 MPS cases skipped in CPU-hidden run |
| Broad non-CUDA | `env CUDA_VISIBLE_DEVICES= ... -m pytest -q -ra`: 454 passed, 32 skipped, 3 sandbox Gloo failures in 109.70 s | Exact three rerun outside network sandbox: 3 passed in 8.20 s; therefore 457/457 non-CUDA passed |
| Tiny four-training-mode CUDA acceptance | `CUDA_VISIBLE_DEVICES=0 ... test_tiny_gpu_training_modes_report_bounded_phase_memory`: 3 passed, 1 failed in 4.82 s | Sequential 22 MiB; fixed 24 MiB; KKT 24 MiB. Normal environment-blocked |
| Normal rerun with Ninja on `PATH` | one parametrized test failed in 3.48 s | Sibling JIT requires missing `CUDA_HOME`/toolkit; no numerical conclusion |
| Queue/KKT/safety/sibling smoke | four selected tests: 4 passed, 1 failed in 6.63 s | Queue transaction, T/C Gamma loss, dynamic safeguard, sibling boundary passed; fixed-refill environment-blocked at same JIT path |
| Tiny two-GPU learner | `CUDA_VISIBLE_DEVICES=0,1 ... test_two_cuda_unequal_strata_exact_sum[normal]`: 1 passed in 6.46 s | Bit-identical update and exact masses 0.25/0.75; does not test C02 asymmetric failure |

All Python commands used `/home/pablob/.venvs/poker/bin/python`. The exact
multiline commands are preserved in
[batch 12](batches/12_validation_evidence.md).

## GPU hygiene

Initial state: three idle RTX 2080 Ti GPUs. GPU 0 and GPU 1 each showed 6 MiB,
0% utilization; GPU 2 showed 16 MiB, 0%, with display-only processes. GPU 0 was
used for single-GPU tests and GPUs 0/1 for the tiny DDP test. State was checked
after every stage. Final state at 18:09:24 was identical—6 MiB/0%, 6 MiB/0%,
16 MiB/0%—with no compute processes or unexplained growth.

## Open validation gaps

- Normal CUDA update and fixed-refill soak require the sibling evaluator's
  undeclared local CUDA-toolkit/JIT dependency.
- No BR CUDA gate exists; heads-up CPU tests do not exercise C04.
- No asymmetric mandatory-failure NCCL test covers C02.
- Tiny successes cannot clear direct-seal, complete-ledger, child-wave, queue,
  folded-summary, or production-scale risks.
- No historical-size target, learner, fixed, or KKT benchmark was run.
- Evaluator v1 remains accepted statically despite the passing v2-capable
  sibling boundary.

No large benchmark, soak, or production-size job ran, and every GPU returned to
idle.
