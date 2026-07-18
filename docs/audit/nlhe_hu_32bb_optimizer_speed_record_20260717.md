# HU 32 bb optimizer speed record — 2026-07-17

## Outcome

The full-deck heads-up 32 bb sequential-RB production geometry now completes
232 optimizer steps in 69.427 seconds on two RTX 2080 Ti GPUs, including the
one-time frozen-reference cache build. This is 0.2993 seconds per step.

Relative to the sustained refactor baseline reproduced in this work, the
optimizer is 10.99x faster. Relative to the 6.232-second hot step documented in
`nlhe_hu_32bb_optimization_regression_20260717.md`, it is 20.83x faster.

The complete fixed-2,048-root profiled iteration took 93.13 seconds:

| Phase | Seconds |
|---|---:|
| rollout | 19.514 |
| targets | 4.147 |
| optimization, 232 steps | 69.427 |
| complete profiled iteration | 93.133 |

Artifact:
`runs/experiments/nlhe_hu_32bb_optimizer_final_widecache_232step/metrics.jsonl`.

## Benchmark progression

Every sustained comparison used two ranks on GPUs 0 and 1, 2,048 sequential
roots per GPU, full-deck HU 32 bb, batch 64 per rank, AMP, and initialization
from `runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0000020.pt`.

| Implementation | Steps | Seconds/step | Speedup vs 3.290 baseline |
|---|---:|---:|---:|
| refactor baseline, equal 32/32, Flex row 3 | 20 | 3.290 | 1.00x |
| cost-aware 53/11 allocation, Flex row 3 | 20 | 1.417 | 2.32x |
| calibrated Flex row 10 | 20 | 1.267 | 2.60x |
| sampled reference cache, Flex row 10 | 60 | 1.159 | 2.84x |
| fused policy terms, Flex row 10 | 60 | 1.144 | 2.88x |
| fused/cache Flex row 11 | 60 | 1.129 | 2.91x |
| fused/cache SDPA row 11 | 60 | 0.322 | 10.21x |
| final fused/cache SDPA row 11 | 232 | 0.2993 | 10.99x |

The final result includes 8.794 seconds to build two caches containing 4,669
unique reference rows. Without caching, the reference model would evaluate
14,848 sampled rows per rank. The cache occupies 80.5 MB. Separate no-grad
build shapes reduced cache construction to 123 forwards (64-row preflop and
22-row postflop caps) without changing the training shapes.

## Implemented changes

1. Restored unbiased cost-aware statistical allocation. Each positive-mass
   stratum receives a positive integer count, apportioned by
   $\alpha_m\sqrt{V_m/t_m}$. The learner continues to weight every draw by the
   exact coefficient $\alpha_m/B_m$, so unequal 53/11 counts change variance
   and cost, not the expected objective.
2. Corrected the phase-boundary allocator measurement by releasing inactive
   rollout cache blocks before querying learner headroom.
3. Replaced the unconditional dense-attention transient estimate with a model
   of the configured bounded-memory backend plus a 2x envelope and the separate
   hardware reserve. The validated physical plan is one 53-row preflop span and
   one 11-row postflop span per step, down from 14 spans.
4. Added exact sampled-row reference caching. The learner pre-draws the same
   index calls in the original step/partition order, caches only sorted unique
   rows, and reuses the stored plan on rollback. Cached and uncached updates and
   final RNG state are bitwise equal in the focused test.
5. Fused poker legal-event probability construction for selected log-prob,
   entropy, and KL, avoiding three current-policy compositions per row.
6. Added native biased SDPA. The fp32 reach/key bias is supplied as an additive
   mask, after the same learned temperature and content scale as FlexAttention.
7. Added aggregate CUDA-event telemetry for gather, reference lookup, current
   forward, loss, backward, gradient synchronization, and optimizer update.

## Final execution and memory

The final 232-step update used 53 preflop and 11 postflop samples per rank per
step and exactly two physical subspans per step (464 total rather than 3,248 at
the previous 14-way split).

| CUDA bucket | Total seconds | Seconds/step |
|---|---:|---:|
| reference-cache build | 8.794 | 0.0379 |
| current forward | 18.828 | 0.0812 |
| backward | 33.873 | 0.1460 |
| gradient synchronization | 3.196 | 0.0138 |
| loss | 2.328 | 0.0100 |
| optimizer update | 1.989 | 0.0086 |
| gather + cache lookup | 0.069 | 0.0003 |

Peak learner allocation/reservation was 3.64/3.88 GB. Peak whole-iteration
allocation/reservation was 4.45/6.80 GB. The production hardware profile keeps
its 0.5 GiB reserve and 0.945 fractional ceiling. No OOM occurred in any staged
row-3, row-6, row-10, or row-11 probe.

## Correctness and validation

- 397 CPU/Gloo tests passed and 21 were skipped. Twenty-seven tests were
  deselected only because their acceptance YAML fixtures were already deleted
  in the incoming dirty worktree; those files were not restored or overwritten.
- Focused allocation, cache, adapter, config, and attention suites passed.
- One-GPU bounded cache/recompute learner tests passed for normal, sequential
  RB, and branched RB.
- The CUDA no-host-sync biased-attention test passed.
- The two-GPU unequal-mass distributed learner test passed.
- Biased SDPA matches explicit forward/backward at fp32 test precision. Under
  fp16 throughput execution, isolated Flex/SDPA mean differences were
  $1.45\times10^{-4}$ in outputs and $3.28\times10^{-8}$ in input gradients.
- Flex and SDPA production probes realized identical nodes, decisions,
  statistical counts, and exact objective mass. Throughput-mode model updates
  can differ at low order because the attention kernels have different fp16
  reduction orders.

GPUs 0 and 1 returned to their 6 MiB display-only footprint after the final
run. GPU 2 retained another user's process throughout and was not used.

## Production handoff

`configs/hardware/local_2x2080ti.yaml` now selects `attention_backend: sdpa`
and the validated 53/11 physical envelope. The scientific config writes to a
fresh `_optimized` output directory and should be initialized, not resumed:

```bash
./tools/run-training.sh \
  --config configs/train/nlhe_hu_32bb_rb.yaml \
  --hardware-profile configs/hardware/local_2x2080ti.yaml \
  --run-dir runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_optimized \
  --gpus 0,1 --nproc-per-node 2 \
  --init-checkpoint runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0000020.pt \
  --tmux
```

The 93.13-second fixed-root iteration projects to about 2.33 hours for 90
iterations before controller growth and checkpoint overhead. A practical
planning range is 2.5--3 hours. Treat that as an estimate, not a production
completion guarantee.
