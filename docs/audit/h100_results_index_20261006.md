# H100 results index — 2026-10-06

Start here for offline continuation. This file summarizes the final state;
the two longer reports retain chronology, identities and failed attempts.

## What is established

- Primary optimization target: HU/P=2, R=13, 32bb. Queue designs and full
  private-hand domains are unchanged. The rejected postflop compact bound
  was not attempted. Protected theory documents were not modified.
- Integrated fixes cover strict reproducibility, branched capacity, per-lane
  topology admission, uneven family-tail scheduling, retaining compilation
  after graph-budget rejection, and scheduled reference-cache reuse.
- The integrated production tree passed572 CUDA tests/2 skips before the
  final capture-stream repair. Eighteen full-width attention checks and36
  full-model consumer iterations add native BF16/FP16, HU/6-max, RB/sampled,
  sequential/fixed/KKT and refresh coverage. See the detailed validation.
- Matched source8236a61 and work, HU RB K=1: median152.107s with packs32/8
  and cacheoff;43.132s with packs256/128 and cacheoff;32.086s with
  packs256/128 and1GiB reference cache. Gains are3.53× from packs,1.34×
  from caching,4.74× combined. Each uses one warmup/three repetitions,
  29,696 presentations and116 optimizer steps; all completion work remains.
- Separate Normal cache pair on an equivalent integrated production tree:
  41.952→30.681s,1.37×. Normal batch512 candidate26.963s changes116
  updates to58 and does not establish matched learning quality.
- K=2 with smaller per-lane packs does not beat the tuned K=1 aggregate
  throughput. K=16 full eager warmup exceeded15min: no valid rate.
- Actual learner GEMM+attention summed kernel time:28.41% pre,46.50% post.
  There are3,466/3,981 kernels. Elementwise/copy/cast/reduction costs explain
  substantial time outside Tensor work. GPM Tensor activity is9–10%,
  while a large BF16 GEMM calibration reaches92.9% and680.667TFLOP/s.
  Activity is not peak-FLOP utilization. Nsight Compute counters were denied
  by driver policy; GPM and CUPTI traces work without changing that policy.
- BF16 activation GEMMs and Hopper WGMMA/cuDNN attention are observed.
  FP32 master weights, gradients, optimizer state and numeric islands remain.
  TF32 is still a candidate; the earlier model probe gained only~1%, so no
  claim of a large training gain or quality equivalence is made.
- Exact zero working padding343→344 selects Hopper GEMMs and gives~4%
  model-only postflop improvement. CUDA output/gradient/storage oracles pass.
  It is experimental, not adopted, and is unrelated to hand/queue compaction.

## Capture correctness and remaining gates

Original asynchronous full Normal K=2 capture failed with illegal memory
access. A blocking diagnostic passed but did not clear the failure. An
asynchronous Compute Sanitizer run then found a concrete AccumulateGrad/
stream-capture mismatch (CUDA906/901), rather than an invalid-global-memory
report. Persistent-stream repair96a7654 passed21 focused CUDA tests.
See the closing outcome below for the full asynchronous public smoke.
The broad572-test suite predates that repair; additional RB/full-production
capture gates, repeated async/memcheck checks and warmed performance remain.

## Recommended next work

1. Finish capture correctness gates before adopting captured throughput.
2. Tune actual occupancy of batches/packs/K, with matched statistical work;
   first use the measured strong K=1 packs256/128/cache configuration.
3. Fuse/compile elementwise and activation movement; avoid needless FP32
   activation banks. Test alignment/low-precision normalization separately.
4. Architecture ablations:3×192 or2×256 vs6×128 shared trunk. Later compare
   full-domain64-latent encoder/decoder and dense range-context MLP.
   Preserve I/O, reach/card/folded features and shared pre/post weights.
5. Compare learning quality with several seeds and equal-time/equal-work
   budgets before declaring architecture or schedule changes improvements.

## Files to preserve

- `h100_replica_validation_20261006.md`: detailed experiments and integrated fixes.
- `h100_bounded_final_handoff_20261006.md`: final series, artifacts and caveats.
- `h100_tensorcore_architecture_brainstorm_20261006.md`: ranked alternatives,
  invariants, analytic costs and ablations; no architecture implementation.
- `h100_validation_agent_handoff_20261006.md`: previous conversation's handoff.
- `runs/experiments/h100_offline_analysis_20261006.tar.gz`: raw reports,
  telemetry, traces, logs, experiment scripts, successful configs and source
  diff against629b2a8. Runtime/checkpoint directories are excluded.
- `tools/h100-gpm-monitor.py`: reusable read-only NVML GPM probe.

The archive is deliberately tracked despite runs/ being ignored, so pushing
these commits preserves the evidence. Successful experiment configurations
are also committed. No pod shutdown or remote push was performed.


## Closing outcome, 23:59 UTC — capture candidate rejected

The full public asynchronous Normal K=2 smoke on96a7654 **failed again** with
CUDA illegal memory access, surfaced at owner.fatal.clone, followed by
rollback/context failures and allocator abort(exit134). No valid report or
throughput was produced. The235s time limit was not reached. The persistent
stream candidate therefore does not resolve the production failure, despite
21 passing focused tests. It was **reverted**; the final production tree again
matches the previously572-test-validated tree. This experiment is a useful
negative result, not a successful repair. Its patch is retained as
`reports/capture_stream_candidate_96a7654.patch`; result metadata and full log
are `reports/capture_stream_async_public_result.json` and
`reports/capture_stream_async_public_cuda.log` under the validation directory.
Failed runtime/config scratch was removed. No GPU compute owner remains.

Captured K=2 safety remains an unresolved correctness gate. Do not recommend
captured mode or claim this fix works. Next diagnostics should reproduce under
memcheck while holding autograd streams consistent, check compiled backward
buffer lifetimes/replay aliases, and isolate family/lane/shape transitions.
The original sanitizer906/901 stream failure and the asynchronous illegal
address are distinct observations; their causal relationship is unproven.
No additional GPU tests were launched. The latest archive contains this
rejected candidate, its tests/failure, architecture proposals and all earlier
raw evidence. Eager training improvements and their identities are unchanged.
