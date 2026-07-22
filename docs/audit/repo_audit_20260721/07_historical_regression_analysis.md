# Historical regression analysis

## Verdict

The named improvements after `537700f` are still identifiable, but only KKT,
refill primitives, and parts of learner behavior have focused validation.
There is no comparable current run proving end-to-end efficiency preservation.

| Historical responsibility | Current disposition |
|---|---|
| `537700f`: cost-aware counts, sampled cache, fused policy terms, biased SDPA, bounded shapes, flat direct backward, telemetry | Strong static/focused-test preservation; no equal-work current performance gate (C12) |
| `b94941c`: remove scalar CUDA validation loops | Vectorized tree validator survives, but new target loops create C11 static regression risk |
| `79a354b`: fixed/dynamic KKT, T/C, controller/checkpoint | Strong static and focused-test preservation; tiny KKT CUDA update passed |
| `1cde6de`: protected-root unbiased refill and controller correction | Strong static preservation; CPU formula tests pass, end-to-end CUDA refill environment-blocked |
| storage refactor commits through `dce75ff` | Page transactions survive, but C01/C05/C06/C07 show incomplete authoritative migration |
| large-tree/OOM safeguards | Partly preserved; fixed/warmup complete-closure safeguard remains open |

## Artifact-backed historical baselines

- `runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl`: 1,110 two-rank
  iterations; means were 143.340 s profiled iteration, 44.255 s rollout,
  7.642 s targets, 91.386 s optimization, 23.371 s learner forward, 42.750 s
  backward, 14.095 s reference-cache build, 6.953 s gradient synchronization,
  and 11,188.8 decisions.
- `runs/experiments/nlhe_hu_32bb_sequential_value_vectorized_bench/metrics.jsonl`:
  one GPU, 2,048 roots, 5,051 decisions, 1.055 s target, 18.866 s iteration.
- `...sequential_value_vectorized_bench_2560/metrics.jsonl`: one GPU, 2,560
  roots, 6,343 decisions, 1.313 s target, 23.718 s iteration.
- Normalized-loss smoke/pilot artifacts report roughly 0.24–0.27 s target
  phases near 2,100 decisions, but changed scale/geometry prevents comparison.

These are historical absolute observations. The target probes do not embed a
Git hash, and no post-`dce75ff` workload matches hardware, geometry, samples,
steps, and cache cadence.

## Documentary-only baseline

`docs/audit/nlhe_hu_32bb_optimizer_speed_record_20260717.md` reports a
two-RTX-2080-Ti, 232-step update of 69.427 s (0.2993 s/step): 18.828 s forward,
33.873 s backward, 3.196 s synchronization, 2.328 s loss, 1.989 s update,
8.794 s reference build, and 3.64/3.88 GB learner allocated/reserved peaks.
Its named raw run directory is absent, so this is precise documentary evidence,
not an independently re-aggregatable baseline.

## Honest regression statement

There is no measured current-HEAD regression. C11, C01, C06, C07, allocator
flushes at `pbg/core/training.py:2360,2808`, and worst-width execution are
static risks/opportunities. The historical algorithms' presence does not prove
their original throughput. Equal-work gates in the performance document are
required before claiming preservation or regression magnitude.
