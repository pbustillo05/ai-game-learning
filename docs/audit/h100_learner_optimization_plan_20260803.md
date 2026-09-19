# H100 learner optimization plan, 2026-08-03

Status: proposal only. This document makes no production-code or mathematical
change.

## Scope

Improve sequential Normal/RB execution without changing the objective, MHA
mathematics, model dimensions, or sampling law. Most work can be implemented
and CUDA-tested on the local server; an H100 is required only for final SM90
dispatch, memory, and throughput proof.

## Planned change sequence

1. **Runtime gate and reproducibility.** Reject unsupported PyTorch/evaluator
   combinations, record the complete CUDA/cuDNN/extension environment, and use
   a supported `torch>=2.11` stack for comparisons.
2. **Measurement first.** Add opt-in NVTX/PyTorch phase ranges for learner
   gather, forward, loss, backward, gradient checks, optimizer, and allocation.
   Add an actual-input SDPA dispatch probe that proves Flash/efficient/math
   selection without changing attention semantics.
3. **K=1 learner overhead.** Based on the profile, remove avoidable per-step
   device-to-host synchronizations and temporary FP32 allocations; evaluate
   fused/capturable optimizer operations, `torch.compile`, and CUDA Graphs with
   strict output/gradient/update parity gates.
4. **Memory and packing.** Reduce unnecessary copies, casts, padding, and
   allocator churn while preserving the exact logical batches and model inputs.
5. **Replica execution.** Replace serial learner-lane execution and the SDPA
   `vmap` fallback with a genuinely batched or concurrent implementation. Do
   this after K=1 is efficient so replica work does not conceal the base
   bottleneck.

## Commit and rollback policy

Each numbered item will be a separate commit with its own CUDA tests and audit
entry. Revert an unsuccessful optimization with `git revert <commit>`; do not
mix performance changes with scientific/configuration changes.

## Acceptance gates

- Local: CUDA correctness, deterministic replay parity, finite gradients,
  checkpoint/resume parity, and no CPU production fallback.
- H100: forced/default SDPA dispatch evidence, Nsight/DCGM Tensor and memory
  counters, stable peak memory, and matched before/after learner replay plus
  full-iteration throughput.
- No speed claim is accepted from `nvidia-smi` utilization alone.

