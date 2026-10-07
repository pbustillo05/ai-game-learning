# Bounded capture triage extension — 2026-10-07

At00:03 UTC the user explicitly selected up to30 more minutes of H100
debugging after the earlier closing archive. The new hard stop is00:33:35 UTC;
reserve the final minutes for cleanup and committing refreshed evidence.
Production remains at1b69b5b while diagnostic monkeypatches run. Do not use
these instrumentation timings as warmed performance.

## Established before the extension

- Original public Normal K=2 asynchronous capture failed at owner.fatal.clone
  with an illegal CUDA address. That is the reporting boundary, not the
  established first failing kernel.
- CUDA_LAUNCH_BLOCKING=1 passed five captures/seven replays; this changes
  ordering and does not establish asynchronous safety.
- Compute Sanitizer on original code stopped at cudaStreamWaitEvent906 and
  capture-invalidated901, with a retained AccumulateGrad stream warning.
  It did not report an invalid-global-memory access.
- Candidate96a7654 kept all runner autograd on one persistent stream. It
  passed21 focused CUDA checks but still failed the asynchronous full public
  K=2 smoke(exit134). The candidate was reverted by26757a9. This negative
  result prevents claiming stream ownership alone repairs the illegal address.
- The cohort key contains model owner/method, family, batch/action size,
  estimator/precision/policy/reference schema, lane run and input signatures.
  No simple wrong-family/shape cache collision was found in source review.
- Repository CUDA extensions use getCurrentCUDAStream; no use of a default
  launch stream was found in these kernels. This does not audit library kernels.

## Added instrumentation and saved compiler sources

`runs/experiments/nlhe_hu_32bb_normal_h100_validation/scripts/h100_capture_events.py`
wraps the unchanged public training harness. It records identity, runner/key,
input shapes/strides/pointers, mutable pointers, stream and capture status,
invocation entry/return/error, and replay counters. Default instrumentation has
no explicit CUDA synchronization. `--sync-after-program` is a separate diagnostic
option and cannot establish asynchronous correctness. Only syntax was checked
before the first launch; CUDA behavior is being exercised in the trial below.

`--persistent-stream-candidate` loads the exact execution_program.py from
candidate96a7654 into the module for that process only. The event identity
records this override and wrapper/training hashes. Disk source and HEAD remain
unchanged. The ordinary benchmark source-consistency field verifies only disk
identity; it does not account for this explicitly recorded runtime override.

Four full six-layer K=2 compiler forward/backward source files are saved under
`runs/experiments/nlhe_hu_32bb_normal_h100_validation/compiler_sources/`.
Their manifest records exact hashes, original paths, 6,248,394 total bytes,
master lane stride6,889,987, six cuDNN forward or backward calls per file,
and pre535/post2854 token shapes. These are shape-matched cache files, not proof
that the failed run executed them. No runtime tensor values/master weights
were extracted. Their original mtimes are before the last candidate trial,
consistent with cache reuse; loaded paths were not logged.

## First trial

Directory: `runs/experiments/nlhe_hu_32bb_normal_h100_captured_k2_stream_memcheck/`.
Full public HU Normal K=2, roots4, batch8/steps2, learner caps2/2, R=13,
six-layer width128 shared model, BF16/IEEE; no warmups/one diagnostic repetition.
Compute Sanitizer memcheck, error exit86, CUDA_LAUNCH_BLOCKING unset,
ATEN-only compile search/four threads, persistent-stream candidate enabled,
no explicit sync-after-program; timeout570s. Event/console/memcheck logs are
preserved. Intent: get past the earlier AccumulateGrad capture error so the
checker can report the actual illegal access, if it reproduces. The outcomes
and any follow-up trials are appended below before re-archiving.

An upstream issue reports Hopper attention/capture fragility on the same
Torch/CUDA generation, but uses FlashAttention in vLLM rather than our cuDNN
biased attention. It is a background lead, not a diagnosis or excuse to change
the locked environment: https://github.com/pytorch/pytorch/issues/188733 .


### First trial outcome at00:13:39 UTC

Timeboxed at570s(exit124). Eight learner calls returned; preflop K=2 admitted
one captured entry and replayed it three times. No906/901 error was reported
with the consistent-stream candidate. The last event was call9's uncaptured
compiled warmup entry for a preflop tail shape. No complete sanitizer error
summary or full public report exists. No invalid-access report appeared before
the limit; this is **incomplete evidence**, not a successful capture gate.
Memory was released and the GPU had no compute owner after process exit.
Failed runtime/config scratch was removed. This trial did not justify another
production fix. Following the user's question about capture effort, no further
capture run is planned in this extension.


## Final follow-up, 2026-10-07: ready for offline continuation

The user authorized a30-minute extension at00:03 UTC, then asked to finish
at00:24 UTC. No additional experiment was started after that request.
Production pbg/tools/tests remain identical to the previously validated source;
neither the persistent-stream candidate nor BF16 normalization was adopted.

New findings, all saved with identities and raw evidence:

- Learner-only BF16 normalization:30.39571→29.18032s (**1.04165×**), peak
  allocated20.408→16.458GB (**19.36% less**). All presentations, updates,
  fresh nodes, per-family work/cache counts and non-timing structures match.
  Four full-model CUDA output/gradient/storage oracles passed. The attempted
  full production suite timed out after75% progress, without reported assertions;
  its gate remains incomplete and the production patch was reverted.
- Model-only selected-output architecture probes:3×192 gave1.42× pre/1.52×
  post;2×256 gave1.58×/1.62× vs6×128. Three warmups/21 repeats, actual
 176/80 rows, finite gradients. Parameters grow1.52×/2.29×. No Tensor-activity,
  full-training or learning-quality claim follows. No architecture adopted.
- Consistent-stream capture under memcheck progressed through eight calls
  including preflop capture/replays, then timed out at570s in another compiled
  shape. No complete sanitizer summary exists. This did not locate or clear
  the original illegal address. No more capture experiments followed.
- Four shape-matched K=2 compiler sources and exact wrapper versions were
  preserved before pod shutdown, with hashes and provenance limitations.

Read `h100_bf16_normalization_validation_20261007.md` and
`h100_capture_triage_extension_20261007.md` for exact commands, outcomes,
source/precision caveats and future gates. The architecture report's follow-up
section now includes measured timings and actual parameter counts.
The refreshed tracked offline archive includes all these files and raw reports.
The goal is being paused at the user's request to call it a day; unresolved
capture correctness and future optimization adoption are not marked complete.
