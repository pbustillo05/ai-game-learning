# Bounded H100 final run and offline analysis — 2026-10-06

The user requested a final progress assessment after roughly three hours and
no more than two additional hours on the Runpod. We chose a detached final
series of approximately one hour. Do not expand or restart this matrix after
its deadline. Analysis of the saved results can happen without the GPU.

## Ownership, deadline, and commands

The active tmux session is `h100-final`. Its controller is
`runs/experiments/nlhe_hu_32bb_normal_h100_validation/scripts/h100_bounded_finish.py`.
It started around 23:05 UTC and stops launching/running benchmark work by
**2026-10-06 23:59:48 UTC**, reserving teardown/archive time until the hard
deadline **2026-10-07 00:01:48 UTC**. It does not stop or bill-manage the pod.

```bash
tmux attach -t h100-final
cat runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/bounded_final_status.json
tail -f runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/bounded_final_console.log
```

The final archive will be
`runs/experiments/h100_offline_analysis_20261006.tar.gz`. Download it and the
working checkout/source changes before releasing the pod. The archive includes
H100 audit documents, report JSON, phase/GPM/NVML samples, logs, experiment
scripts, successful configs and source-identity/diff records, excluding runtime
checkpoint directories. The archive's embedded status precedes archive
creation; the live status file adds the archive path and size afterward.

To start offline analysis, extract it with `tar -xzf`, read this file and
`h100_replica_validation_20261006.md`, then inspect
`bounded_final_status.json`. Accept a report only if source consistency and
GPU isolation passed. Timeboxed captures with missing reports are incomplete
evidence, not automatically code defects. Do not turn cold compilation times
into warmed throughput claims.

## Final series, in priority order

Priority update at 23:11 UTC: the Normal K=2 capture ended with **CUDA illegal
memory access**, before any valid report. This is an observed correctness
failure, not merely a capture timebox. The retained log is
`runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/nlhe_hu_32bb_normal_h100_captured_k2_p2_q2_s104830_8236a61_smoke.queue.log`.
The status originally used a generic missing-report reason; this log supersedes
that wording. The controller was replaced at the independently grouped batch
candidate's boundary, leaving its timing intact. The revised priority is:
actual pre=256/post=128 learner CUPTI profiles, then the K=2 captured smoke with
`CUDA_LAUNCH_BLOCKING=1` to locate the illegal access, then the remaining arms
below. This keeps the same absolute deadline. The original adopted capture
is already over and must not be described as still running.

1. Finish the already-running native Normal K=2 captured smoke. Its original
   launcher was SIGSTOP'd before starting the bounded controller; only the
   launcher was stopped, and the active GPU child continues. This smoke has
   a 15-minute total timebox ending 23:15:31 UTC.
2. Normal K=1, physical packs 512/256, batch 512 and 58 optimizer steps,
   reference cache 1 GiB, one warmup and three repetitions. This changes the
   optimizer schedule; even with the same 29,696 presentations it is not a
   matched learning-quality comparison against batch 256 / 116 steps.
3. Normal and RB K=1, packs 256/128, reference cache 1 GiB, independent seed
   205991, one warmup and three repetitions each.
4. Native captured RB K=1 and K=2 smoke, at most 15 minutes each. Zero warmups
   and one repeat: compatibility evidence only, not steady-state performance.
5. Normal K=16, packs 32/8, cache off, one warmup and two repeats if time allows.

All jobs run serially to keep timings uncontaminated. Before and after every
job the controller checks explicit GPU UUID ownership. Future children have
separate process groups, and deadline cleanup kills only the controller's own
jobs. Failed/timeboxed runtime and config scratch is removed; diagnostics
remain. The final status records completed, failed/timeboxed, and deferred
arms and final GPU owners. It refuses new work if another GPU owner appears.

Source remains frozen at `8236a61fb23da5bde426a2f082c7f3f3f4646265`; do not
commit or edit production `pbg/` or tracked `tools/` until the series ends.
The harness SHA is recorded in status and each result. Compiler experiments
use ATEN-only GEMM search and four compile threads, explicitly recorded; this
is not a claim about the default full autotuning search.

## Verified progress before the final series

- Six repairs are integrated: strict reproducibility, branched capacities,
  per-replica topology admission, uneven family-tail cohort scheduling, retaining
  compiled execution after graph residency rejection, and scheduled reuse in
  reference-cache admission. Protected theory and queue-layout designs were
  not changed. The unwanted postflop private-hand compact bound was not attempted.
- The integrated CUDA suite passed **572 tests**, with 2 skips. Eighteen
  full-width native attention checks and **36 full-model consumer iterations**
  cover HU/6-max, sequential/fixed/KKT, sampled/RB, finite gradients and refresh.
- RB same-source matched-work pack/cache results:

| HU RB K=1 configuration | Median iteration | Gain versus small uncached |
| --- | --- | --- |
| Packs 32/8, cache off | 152.107 s | 1.00x |
| Packs 256/128, cache off | 43.132 s | 3.53x |
| Packs 256/128, cache budget 1 GiB | 32.086 s | 4.74x |

These preserve 169 fresh nodes, 29,696 learner presentations, 116 optimizer
steps and 16,384 completions. Caching adds 1.34x over large packs. They measure
iteration throughput, not a long-run learning-quality improvement.

- Normal pack tuning separately gave 3.99x; a matched Normal cache pair gave
  1.37x. These pairs span different revisions and must not be multiplied into
  a same-source total. Normal K=2 at packs 128/64 achieved 665.927 presentations/s,
  below large-pack uncached K=1's 707.857/s. Increasing K alone is not a win.
- RB phase attribution: total 43.282 s, learner 42.033 s (97.1%), rollout
  1.122 s, completion 0.077 s, targets 0.041 s. Duration-weighted learner GPM
  samples: SM activity 82.75%, occupancy 45.17%, Tensor/HMMA activity 9.40%,
  DRAM bandwidth 38.61%. Rollout SM activity 6.21% but only 2.6% of total wall.
  Shorter-than-sampling phases have no attribution, not zero usage. Raw evidence:
  `runs/experiments/nlhe_hu_32bb_rb_h100_eager_k1_p256_q128_s104830_8236a61_phase/phase_gpm_summary.json`.

## Precision, metrics, and remaining optimization ideas

The main runs use BF16 autocast for eligible matrix/attention operations, with
FP32 master weights, gradients/Adam and sensitive reductions. BF16 Tensor Core
GEMMs and Hopper cuDNN fused attention were observed in kernel traces. Native
attention retains its learned FP32 additive key bias and padded key stride.
It does not append a prior feature to each head.

The user's priority is speed over immaterial numeric differences. TF32 remains
an experimental option; the earlier exact FP32 test failures are not evidence
that its training quality is unacceptable. The isolated model probes measured
only roughly 1% benefit, and short finite-loss checks are not long-run quality
evidence. Existing precision tests/tolerances were not silently weakened.

`tools/h100-gpm-monitor.py` samples NVML GPM every 250 ms without a CUDA context.
Tensor activity is active-time percentage, not percentage of peak Tensor FLOPs;
SM activity is not per-core instruction issue or exact individual-core idle
cycles. Occupancy is active warps relative to the maximum. DRAM reports bandwidth
relative to peak. Near-zero non-tensor FP16 activity does not contradict BF16
Tensor Core use. Detailed Nsight Compute counters are permission-denied on
this pod; shared driver policy/capabilities were not modified. No MFU claim can
be derived from these percentages alone.

Optimize the learner first: packs/batches/K, repeated reference work, launch
and cast overhead, then warmed compilation/capture with an explicit cold-start
cost. The low-precision LayerNorm candidate was prepared but not tested or
adopted. Full production warmed-capture sweeps, extra model compile sweeps,
and extensive K/seed sweeps are deferred to honor the deadline. No queue
partition changes or compact private-hand bounds are authorized. Multi-GPU
NCCL, MPS and other CUDA architectures require other hardware; this is one H100.

The original optimization goal remains unfinished until the final results are
analysed. This file supplies the continuation state for that offline analysis.

## Latest evidence and scope update at 23:30 UTC

The user raised low Tensor Core activity as the primary concern. Actual cached
K=1 learner programs were profiled with configured pre/post caps 256/128.
The statistical batch splits into **176 pre and 80 post rows**, so an initial
profiler hook that waited for exactly 256/128 rows captured nothing. Its failed
wrapper report is not profiling evidence. The hook was corrected to profile
actual packets after an unprofiled warmup; both families completed. Reports:
`runs/experiments/nlhe_hu_32bb_normal_h100_learner_profile_actual_8236a61/`.

| Profiled GPU kernel time | Pre 176 rows | Post 80 rows |
| --- | --- | --- |
| Matrix multiplication | 12.17% | 15.14% |
| Attention and associated cuDNN kernels | 16.24% | 31.36% |
| Elementwise | 36.77% | 27.41% |
| Copy/cast/concatenation | 14.98% | 13.78% |
| Normalization | 7.55% | 5.60% |
| Reductions | 12.01% | 6.57% |

There are 3,466 / 3,981 kernels in these single actual learner programs.
Shape recording/profiling affects execution; these are kernel-duration sums,
not unprofiled wall-time or SM instruction accounting. Forward/backward
attention traces explicitly show BF16 Q/K/V and Hopper WGMMA cuDNN kernels.

A separate eight-second **8192-square BF16 GEMM calibration** achieved
**92.90% Tensor/HMMA activity**, SM activity 95.70%, occupancy 14.15%, and
approximately **680.67 arithmetic TFLOP/s**. This validates that this H100 and
GPM probe can report high Tensor activity. It is a forward-only dense matmul
calibration, not training MFU or model throughput. Raw samples and calculation
are `dense_bf16_gemm_gpm.jsonl` and `dense_bf16_gemm_calibration.json` in that
directory. No source precision tests or tolerances were weakened.

The trace identified the 343-wide post static FFN projection as using older
`cutlass_75...align1` Tensor Core kernels. An ignored experiment adds one zero
working feature and one zero working-weight column, preserving the stored
parameter width, checkpoint/gradient layout, all queue layouts and all hands.
Matched same-source eager model probes (3 warmups, 7 repeats; profiling separate):

| Post model-only objective | Original 343 | Working pad 344 | Speedup |
| --- | --- | --- | --- |
| K=1, rows=80 | 184.119 ms | 177.143 ms | 1.039x |
| K=16, rows=8/lane | 281.856 ms | 270.491 ms | 1.042x |

The padded GEMMs select Hopper `nvjet` kernels. Three CUDA oracle checks at
K=1/2/4 passed for outputs, input/master gradients, strided zero-copy gradient
addresses and no writes outside the parameter slices. This candidate is
**not adopted in production**, and no full-training speedup is claimed.
Both pairs are under `runs/experiments/...h100_model_post_k{1,16}_r{80,8}_`
`{baseline343,padded344}_8236a61/`. K=1 metadata predates adding the oracle
invocation; K=16 executed it. The padding operation itself is unchanged.

Independent seed205991 cached large-pack iterations completed:
Normal **32.477 s**, RB **33.566 s** (one warmup, three repetitions each).
The larger-batch Normal candidate completed **26.963 s**, but changes
116 optimizer steps at batch256 to 58 at batch512. Neither seed variation
nor a changed schedule establishes a learning-quality result.

The synchronous Normal K=2 captured diagnostic completed **5 captured entries
and 7 replays**, with finite public training results. Its wall time was 536.863 s
including cold compilation. `CUDA_LAUNCH_BLOCKING=1` was explicitly supplied;
this pass **does not fix or clear** the original asynchronous illegal memory
access. Stream/buffer lifetime is a hypothesis, not an established diagnosis.
The original failure remains a correctness limitation. Additional RB captures
were deferred to prioritize this investigation and the user's time budget.

The final remaining GPU arm is Normal K=16 eager, caps32/8, one warmup and two
repetitions, with a 15-minute stage timebox. The tmux controller then checks
GPU ownership is empty and creates the offline archive; it may finish earlier
than 00:01:48 UTC. Do not restart captures or extend the matrix afterward.

## Final capture diagnostic queued at 23:35 UTC

One explicit correctness diagnostic is queued after the verified live benchmark
controller (PID165191/start495141690) exits: tmux **`h100-capture-check`**, script
`scripts/h100_capture_memcheck.py`, status
`reports/capture_memcheck_status.json`. It owns no CUDA context while waiting.
It launches only after the selected GPU has no compute owner and runs one
Normal K=2 full public captured smoke under **Compute Sanitizer memcheck**.
CUDA_LAUNCH_BLOCKING is unset and force-blocking-launches is not enabled.
The diagnostic stops by **2026-10-07 00:00:00 UTC**, then tears down its own
process group, confirms GPU ownership is empty, and refreshes the archive.
No new matrix or GPU work is permitted afterward under this Runpod budget.

Compute Sanitizer version2025.1.0.0 is installed at
`/usr/local/cuda/bin/compute-sanitizer`. It checks Hopper warpgroup MMA and
reports invalid accesses; this does not require Nsight performance-counter
permissions. CUDA13/older-tool compatibility remains to be exercised. All its
timing is instrumented diagnostic data. A clean run alone cannot clear an
intermittent asynchronous failure. Its report/log directory is
`runs/experiments/nlhe_hu_32bb_normal_h100_captured_k2_memcheck_8236a61/`.

**The first archive produced by h100-final is provisional.** Wait until the
capture diagnostic's status is terminal and its final_gpu_owners is empty
before downloading the refreshed archive and releasing the pod. If the
diagnostic is deferred, its status records why and the first archive remains
the available evidence. Inspect both controller status files, not only the
earlier final-series status.

The archive also contains `reports/h100_changes_from_629b2a8.diff`, a binary-safe
Git diff of the committed fixes against baseline629b2a8. Download the working
checkout too if offline analysis needs complete source history. Production
source and tracked tools remain unchanged during these measurements.

## Initial Tensor Core investigation

Existing model-only K=16 probes were reanalysed using host-operator External-ID
correlation and kernel names, including cuBLAS `nvjet` kernels rather than
counting only names containing GEMM. A generic `nocast` template name is not
counted as a dtype conversion. Profiled summed kernel-duration proportions:

| Category | Pre, 32 rows/lane | Post, 8 rows/lane |
| --- | --- | --- |
| Matrix multiplication | 13.24% | 16.28% |
| Attention, including associated cuDNN kernels | 18.24% | 32.05% |
| Elementwise | 39.05% | 28.24% |
| Copy/cast/concatenation | 15.72% | 13.95% |
| Normalization | 8.39% | 5.62% |
| Reductions | 5.31% | 3.80% |

There are 2,192 / 2,726 kernels per forward/backward model probe, before the
production PPO loss/Adam orchestration. These are model-only, value-objective,
profiled sums, not full-pipeline wall percentages, TFLOPs or GPM Tensor-active
percentages. They indicate a real cost from unfused elementwise work and
memory movement; they do not justify concluding Tensor Cores are disabled.
Representative actual large-pack learner profiles are next, with shape/type
metadata and CPU/kernel correlation saved for offline analysis. Matrix FLOPs
dominating arithmetic does not imply matrix kernels dominate elapsed time.


## Closing update, 23:56 UTC: concrete capture diagnosis and repair

This update supersedes the earlier prospective controller/capture descriptions.
The main bounded series ended at 23:43 UTC. K=16 eager reached its 15-minute
limit before completing its first warmup, so it has **no valid throughput
result**. A partial last-minute warmup window had SM57.44%, occupancy30.17%,
Tensor6.82%, DRAM25.07%; its phase is unknown, so this is not a whole-learner
attribution. All memory and owned GPU processes were released.

The asynchronous Compute Sanitizer diagnostic ended at 23:46:15 UTC with
exit86 and four CUDA API errors. It did **not** report an invalid-global-memory
access. The first error was cudaErrorStreamCaptureImplicit(906) at
cudaStreamWaitEvent: a legacy stream would depend on a capturing blocking
stream; subsequent errors included invalidated capture(901). PyTorch also
warned that a retained AccumulateGrad node was on a different stream from
its incoming gradient. Its artifacts are:
`runs/experiments/nlhe_hu_32bb_normal_h100_captured_k2_memcheck_8236a61/`.
This establishes a capture stream defect; its relationship to the original
asynchronous illegal-memory-access failure still requires validation.

Commit **96a7654** gives each captured TensorProgramRunner one persistent
non-default CUDA stream, used from the first eager admission through compiled
warmup, captures, fallbacks and replays. Explicit caller/runner stream waits
preserve producer/consumer ordering. Inputs and mutable storage are recorded
on that stream. This keeps retained autograd gradient-accumulation nodes on
the same stream when new exact shapes are admitted. Eager/compiled modes retain
their existing stream behavior. No queues, private-hand domains or architecture
contracts were changed.

Focused CUDA verification passed **21 tests / 15 warnings in49.19s**, including
native BF16 model forward/backward/replay, stable gradients/live weights, graph
budget rejection, and a new multiple-shape persistent-stream regression.
Log: `runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/capture_stream_focused_cuda.log`.
The previous572-test suite predates this final stream repair. A full public
Normal K=2 asynchronous capture smoke is running with CUDA_LAUNCH_BLOCKING
unset, ATEN-only compile search, four compile threads and a235s limit. Its
outcome is recorded in the closing result below. No warmed capture speedup
has been demonstrated. RB capture coverage and a broad post-repair suite
remain future verification requirements.

The architecture brainstorming agent completed a read-only investigation.
See `docs/audit/h100_tensorcore_architecture_brainstorm_20261006.md` for
ranked proposals, invariants, analytic cost estimates, risks and future tests.
First compare a3×192 or2×256 shared trunk against6×128; then consider a
64-latent full-domain encoder/decoder or dense range-context MLP. Every
proposal retains the original queues, full private IDs, external I/O,
reach/card/folded inputs and shared pre/post trunk weights. No architecture
candidate was implemented, and the protected theory/design files were not
modified. Performance and learning-quality claims require future ablations.


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
