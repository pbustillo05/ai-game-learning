# H100 replica validation and performance evidence

Status: integrated CUDA suite and eager consumer gates passed; full-pipeline
tuning is bounded by the Runpod deadline. A newly observed asynchronous K=2
capture illegal memory access remains unresolved despite a synchronous pass.
See the latest-evidence/bounded-handoff sections before using captured execution.

The active validation workstream is Codex conversation
`01a112e3-cf04-72c3-aa47-8a7c4f333c52`. The earlier conversation detected these
shared edits and stopped its pilot. The pilot prepared at `629b2a8` must not
resume against changed production code. Its correctness log remains evidence
for that revision; its interrupted autotuning does not establish throughput.
Commit `3a2d9ac` binds the following experiments. A fresh default pilot plan is
prepared under `runs/experiments/nlhe_hu_32bb_normal_h100_pilot_strictfix/`.
The old pilot's two interrupted reset snapshots and failed configs were removed;
its reports, plan and run state were retained.

## Hardware and recovered validation

- NVIDIA H100 80GB HBM3, UUID `GPU-c0a3cd37-65a1-2be4-42ea-79c44f710069`.
- Reported memory: 85,017,493,504 bytes; production reserve: 10 GiB.
- Python 3.10.21, Torch 2.12.1+cu130, CUDA 13.0, cuDNN 9.20.0;
  pinned evaluator `caeeb976045a17454d12293811469863e5e1ce61`.
- Native BF16 fused attention forward/backward preflight passed; `pip check`
  reported no broken requirements.
- At `629b2a8`, the full suite passed 541 tests, with two hardware-dependent skips.
  Evidence: `runs/experiments/nlhe_hu_32bb_normal_h100_pilot_629b2a8/reports/correctness.log`.
- Prior committed repairs include fused head-feature alignment and the clean
  CUDA wheel linker name (`5ba5653`), and current shared-model consumer/replay
  gates (`629b2a8`).

## Confirmed defect and repair

The public `strict` reproducibility setting selected learner ordering but never
configured deterministic CUDA algorithms. Full six-layer BF16 K=16 checkpoint
replay changed a small number of attention key-projection biases, despite
restoring logical work, weights, optimizer state, references and RNG. The
throughput profile permits nondeterministic reductions; declaring strict did
not remove them. Enforcing deterministic CUDA dispatch externally produced
bitwise-identical replay across all 110,239,792 stored parameter elements.

The runtime now applies and records the chosen CUDA backend policy before model
execution. Strict uses deterministic operators, deterministic cuDNN, IEEE FP32,
and validated cuBLAS workspace configuration. Seeded disables backend
benchmarking; throughput permits autotuning. All profiles retain IEEE FP32:
enabling TF32 changed six existing FP32 incidence/folded/model parity gates.
This establishes numerical differences, not damaged training. Following user
clarification, TF32 remains a performance candidate to evaluate separately
against IEEE timing, output/gradient error, and short training behavior.
No tolerance has been widened to hide the differences. All profiles disable
reduced-precision GEMM reductions/FP16 accumulation so native FP16/BF16 compute
retains FP32 accumulation. Runtime metrics and benchmark environment metadata
record the actual settings.

`tests/test_h100_full_width.py` retains output/gradient parity, lane isolation,
and full-model checkpoint/reference regressions. Its 18 CUDA tests passed,
including bitwise K=2/K=16 replay after effective-hand reference refresh.
After restoring IEEE defaults, the complete existing CUDA suite passed
541 tests with two skips (160.20 seconds). A separate CUDA run passed all
82 affected numerical/model tests, including the 18 new production-width
regressions (66.82 seconds). Together these cover all 559 passing tests.

## Untimed concurrent CUDA stress

Further public native-model consumer checks exposed two branched defects:

- The compiler sized chance/frontier packets linearly in the branch degree,
  although an action can branch through flop, turn and river without another
  player decision. With degree bound 3 and rollout pack 2, a packet held only
  8 rows while the runtime required at least 120 for one maximal expansion.
  The compiler now covers three chance generations and the terminal frontier;
  capacity derivation version is `nlhe_capacity_v3`.
- `LayerPlan.assert_admitted` compared all replicas' concatenated nodes/edges
  against one replica's limits. It now validates node and edge counts per lane,
  retaining rejection of an individual overflowing forest. A two-lane
  metadata regression tests both admission and individual overflow rejection.

After these repairs, the full CUDA suite passed **562 tests**, with two skips
and 15 warnings, in 234.50 seconds. Evidence:
`reports/full_branched_capacity_cuda.log` under the validation run directory.
All twelve public six-layer BF16 consumer variants passed: HU/6max,
sequential/fixed/KKT branching, sampled/Rao–Blackwell, with three iterations
and reference refresh per variant (36 iterations total). Finite losses,
gradients, masters, applied Adam steps and zero-copy ownership were checked.
The six-player fixed/RB check produced 545 aggregate player nodes against
a 512-node per-replica cap, exercising the admission repair. Evidence:
`reports/native_consumers_fixed.log` and `reports/native_consumers.json`.
The failed candidate's logs remain diagnostic evidence, not successful results.

The earlier conversation's isolated, CUDA-tested scheduler and capture repairs
were reviewed and integrated as `5024269` and `b14850e`:

- Uneven preflop tails previously let low-numbered lanes run through postflop
  before their peers reached it, destroying otherwise compatible cohorts.
  Scheduling now drains earlier FIFO families and selects the largest exact
  contiguous cohort, preserving per-lane samples, ordering and updates.
- When measured graph residency exceeds the cache budget, the entire graph pool
  is released while its already-warmed compiled function remains cached. This
  preserves live mutable addresses and avoids reverting recurring shapes to
  eager execution. Staging/pre-admission failures still execute eagerly.

After integration, all 12 full-depth native BF16 variants passed another 36
iterations at clean `b14850e`. The integrated full suite passed 566 tests and
found one stale cross-replica launch-count assertion; scheduling changed the
healthy lane's calls before injected failure from two to one. That count is
not a rollback contract. The test now requires healthy work and exactly one
injected failure, then checks unchanged masters, buffers, Adam state, scaler,
RNG, versions and logical counters. All 38 affected CUDA tests passed with one
hardware-dependent skip (43.08 seconds). No numerical tolerance changed.

These checks ran alongside compilation or other correctness checks. They
establish numerical and memory facts, not isolated timing or throughput.

| Cohort | Physical rows per lane | Largest allocated model peak | Outcome |
| --- | --- | --- | --- |
| K=16 | pre=32, post=8 | 28,704,442,880 bytes | Finite gradients; warmed peaks stable |
| K=32 | pre=16, post=4 | 29,806,562,816 bytes | Finite gradients; warmed peaks stable |

The peaks include native model forward/backward and model ownership. They
exclude production queues, targets, reference forwards, optimizer transitions,
and graph pools. They are not whole-pipeline memory ceilings.

Production storage/owner admission accepts default K=1/8/16 and rejects default
K=32 before allocation. For K=32, the Normal and RB learner storage plans alone
require 80,588,599,296 and 84,999,761,920 bytes, respectively, exceeding the
74,280,075,264-byte usable budget. Smaller physical packs are a tuning candidate
supported by the model stress checks; whole-pipeline admission and timing remain
to be verified.

Evidence lives under
`runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/`, including
`pack_stress_k16.json`, `pack_stress_k32.json`,
`production_storage_admission.json`, `full_width_regressions_cuda.log`, and
`strict_fixed.log`.

The untimed K=16/post=8 graph admission diagnostic completed with finite outputs
and gradients, but the 8 GiB graph cache rejected capture: zero retained graphs,
zero replay calls, one denied shape. Allocated peak was 29,144,750,080 bytes;
reserved peak was 50,774,147,072. This diagnostic started under the initial TF32
candidate and ATEN-only compile search, so it does not certify the final default
policy or throughput. Large graph residency merits further measured tuning.

At `3a2d9ac`, a full six-layer native BF16 Normal K=1 captured pipeline smoke
completed with four retained graphs, six replay calls, and no denied shapes.
Retained graph bytes were 641,752,556; allocated/reserved peaks were
910,909,440/1,495,269,376 bytes. The 697.6-second cold iteration includes four
compilations under ATEN-only GEMM search, so it is capture compatibility evidence
and cannot establish steady-state throughput. Untimed consumer checks overlapped
this cold smoke. Its report is under
`runs/experiments/nlhe_hu_32bb_normal_h100_captured_smoke_k1_3a2d9ac/`.

Nsight Compute 2025.1.1 is installed, but its per-kernel hardware counter access
fails with `ERR_NVGPUCTRPERM`. Shared driver policy is unchanged. A subsequent
read-only Hopper NVML GPM probe succeeded: device-wide SM activity, achieved warp
occupancy, Tensor/HMMA activity, integer/FP32/FP16 activity, and DRAM bandwidth
percentages can be sampled without enabling shared profiler permissions.
The retained [`tools/h100-gpm-monitor.py`](../../tools/h100-gpm-monitor.py) writes full decimal values every
250 ms. Both API status and finite, in-range percent values are required.
One concurrent `nvidia-smi dmon` probe returned a spurious graphics-activity
value above 100%; raw values are retained and invalid percentages excluded.
Concurrent probe numbers are instrumentation checks, not workload attribution.
GPM stream-state queries returned disabled before and after monitoring; no
setter was called. Precise per-kernel stalls/issue metrics still require Nsight.

## Remaining work

1. Complete native six-layer captured Normal/RB K=1/2 smoke.
2. Measure compiled/captured model gains and matched full-pipeline K=16/RB arms.
3. Test smaller-pack K=32 or admitted K=8 where useful; validate selected settings
   across additional seeds and reference refreshes.
4. Profile phase/model costs separately from isolated unprofiled measurements;
   report aggregate throughput and per-run latency with explicit denominators.
5. Tune measured launch, layout, capture, or memory bottlenecks and rerun affected
   CUDA gates. One H100 cannot validate multi-rank NCCL; MPS/SM75 validation also
   requires the corresponding hardware.

The reference-cache reuse estimate was subsequently repaired as `a7a20a1`.
It had counted chunks of fresh rollout rows and ignored scheduled learner
presentations, so the 116-step workload was wrongly denied cache admission.
It now reads sampler metadata without advancing RNG, and rejects known
unprofitable caches before launching a reference forward. The prior conversation
passed all 20 affected CUDA tests (50.78 seconds) and then the full suite:
**572 passed, 2 skipped** in 273.60 seconds. That full run used `fdb58ce`,
whose production source and tests are byte-identical to `a7a20a1`
(`git diff --exit-code ... -- pbg tests` passed). This thread's duplicate full
run was stopped while waiting on shared CUDA extension compilation; its log
does not establish a completed suite. The retained monitor was also exercised
against the H100 with explicit NVML argument types and unchanged stream state.

## Isolated performance evidence at `3a2d9ac`

These measurements use the same six-layer, d=128 BF16 model and IEEE FP32
matmuls. Model probes are value-objective forward/backward only; they exclude
rollout, PPO loss, references, Adam and scheduling. Seven unprofiled repetitions
follow three warmups. CUDA traces are collected separately and include profiler
overhead. Model and whole-pipeline gains must not be interchanged.

| Model probe | K=1 latency | K=16 latency | K=16 aggregate gain |
| --- | --- | --- | --- |
| Pre, 32 rows per lane | 43.03 ms | 131.98 ms | 5.22x |
| Post, 8 rows per lane | 48.33 ms | 281.62 ms | 2.75x |

K=32/pre=16/post=4 matches K=16's total physical rows and delivered almost the
same aggregate model throughput. Together processing K=16 lanes was 6.38x/3.18x
faster than sequentially processing those 16 lanes for pre/post respectively.

The whole-pipeline Normal comparison retains roots=64, batch=256, steps=116,
seed=104830, reference cache off, eager execution and the same initialized model.
Each experiment has two warmups and three isolated checkpoint-reset samples.
Both execute 169 fresh player nodes and 29,696 learner presentations per sample.

| Physical learner caps | Sample seconds | Median seconds | Presentations/s |
| --- | --- | --- | --- |
| pre=32, post=8 | 165.014, 163.468, 164.894 | 164.894 | 180.091 |
| pre=256, post=128 | 41.360, 41.358, 41.729 | 41.360 | 717.993 |

Larger physical packs provide a **3.99x full-pipeline gain**, with statistical
batch and optimizer work preserved. Peak allocated memory is about 1.90 GiB
versus 19.00 GiB; the larger run's maximum reserved peak is about 42.28 GiB,
within the H100 admission budget. This is a configuration gain over the current
default eager execution, not a speedup over the legacy architecture or over the
still-unmeasured captured default. Further seeds, RB and capture remain pending.

Evidence:
`runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_3a2d9ac/report.json` and
`runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_large_isolated_3a2d9ac/report.json`.
The earlier `...eager_k1_large_3a2d9ac` timing is marked `TIMING_INVALID` because
an untimed diagnostic crossed warmup into its first timed sample. Its timing is
excluded entirely; its interrupted runtime/config scratch was removed.

## Precision candidates and utilization

BF16 Tensor Core dispatch is visible in `tensorop` BF16 GEMM and Hopper `wgmma`
fused-attention kernel names. BF16 model GEMMs use FP32 accumulation. Master
weights, stored gradients, optimizer state and loss/reach reductions stay FP32;
some intermediate activations also remain FP32. TF32 applies to eligible FP32
matrix operations, complementing BF16 rather than accelerating BF16 itself.

TF32's K=16 pre/post model medians were 130.81/279.37 ms, approximately 0.9%/0.8%
faster in this sweep. Such small changes need more evidence to establish a gain.
Untimed three-iteration BF16 training in Normal/RB with two seeds retained finite
losses, gradients and weights for both IEEE and TF32. Normal update directions
had cosine similarity above 0.9995; RB above 0.944. These are stability facts,
not long-run quality evidence or a reason to require bitwise training parity.
Detailed losses and gradient norms: `reports/tf32_training_stability.json`.

The K=16 pre probe's GPU activity union covered 92.9% of its profiled CPU range;
NVML sampled GPU busy time around 97-100%. This is temporal activity, not spatial
core occupancy or Tensor Core utilization. Around 2,192 kernels were launched.
The CUPTI trace also reports grids, registers and shared memory. Its occupancy
estimator reports zero for some kernels that actually execute, so those estimates
are not used as achieved occupancy evidence.

The offline Nsight metric catalog confirms these GH100 counters are available
when permissions permit collection:

- Tensor activity: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`.
- SM activity: `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`.
- Achieved occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active`.
- Scheduler issuing: `smsp__issue_active.avg.pct_of_peak_sustained_active`.
- Memory bandwidth: `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

The current driver has `RmProfilingAdminOnly: 1`; the container lacks both
CAP_SYS_ADMIN and CAP_PERFMON. Provider/host profiling permission is required
for those Nsight measurements. No driver policy or container permissions were
changed. GPM interval measurements are a separate permitted path, with metric
definitions documented by [NVIDIA](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlGpmEnums.html).

An interactive integer-rounded view can be started with:

```bash
nvidia-smi dmon --id GPU-c0a3cd37-65a1-2be4-42ea-79c44f710069 \
  --gpm-metrics 1,2,3,5,7,10,12,13 --format csv -o DT -d 1
```

For full decimals, run the retained monitor with `--uuid` selecting this GPU,
`--interval 0.25`, and `--output` naming a JSONL evidence file. GPM reports
device-wide interval averages: low SM activity can coexist with high graphics
activity when small kernels run frequently. SM activity's complement describes
inactive SM time averaged over the chip, not every CUDA core's issue-idle cycles.
Occupancy measures resident active warps, not arithmetic throughput; Tensor
activity likewise is not achieved TFLOP/s. Tensor pipe activity, instruction
issue, memory traffic and end-to-end throughput must be interpreted together.

## Measured optimization priorities

User constraint: preserve `docs/queued/design` and the existing queue design and
layouts. Do not pursue postflop private-hand compact-bound changes or further
queue partitioning. No such change has been implemented. Optimization work
focuses on batch size, physical packs, replica count K, and measured pipeline
underutilization within the current design.
Heads-up ($P=2$) is the primary optimization workload. Multi-way games remain
correctness/compatibility gates; the performance and tuning matrix uses HU.

1. Tune physical packs before increasing K: the default K=1 physical caps leave
   much of this H100 unused, and larger caps already yielded a whole-pipeline gain.
2. Measure existing pointwise fusion and graph replay. Thousands of eager
   launches and FP32 elementwise traffic remain visible; large captured packs
   currently lose graph admission under the 8 GiB limit. Compare retaining
   compiled execution after a graph denial if compiled timing supports it.
3. Evaluate device reference-output caching after matched cache-off baselines.
   The same frozen reference is currently recomputed for reused learner rows.
4. Attribute GPM SM activity, warp occupancy, Tensor/HMMA activity and DRAM
   bandwidth to existing rollout, completion, target, reference and learner
   stages. Prioritize stages with low activity and material wall time, then
   tune their existing packs and reduce launch overhead without changing queues.
5. Compare existing fused attention backends while preserving the learned
   additive key prior. Per-kernel Tensor Core counters would improve attribution;
   permitted GPM interval counters already support whole-pipeline comparisons.


## Current-source HU pack and hardware-counter confirmation

The isolated matched cache-off baseline at `fdb58ce` (production code/tests
equal to `a7a20a1`) measured 41.952 seconds median with pre=256/post=128,
707.857 presentations/s, 169 fresh nodes and 116 optimizer steps. The earlier
`83e9f55` cache-off baseline measured 42.552 seconds median. Raising only the
postflop physical cap to 192 at `83e9f55` measured 42.664 seconds median
(42.608, 42.664, 42.771 seconds). There is no demonstrated improvement: both
post caps already execute the same 116 post and 116 pre learner cohort calls
per iteration for this seed. Physical caps must change executed work to help.

The post=192 run's three isolated repeated samples gave consistent read-only
GPM interval averages: SM active 75.95/75.97/75.70%, warp occupancy
40.46/40.50/40.34%, Tensor/HMMA active 9.00/8.98/8.97%, DRAM bandwidth
active 34.69/34.73/34.66%, and FP32 active 4.33/4.33/4.32%. These are aggregate
hardware counters, not inferred utilization from kernel names or model FLOP
estimates. Approximate timed windows use progress observation timestamps plus
recorded reset durations, trimming 300 ms from both ends; they do not provide
per-kernel attribution. Raw status/value records and window boundaries are in
`runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_p256_q192_83e9f55/`
(`gpm.jsonl`, `gpm_summary.json`, `progress_observations.jsonl`).


## Matched reference-cache result

At frozen `fdb58ce` (production source/tests equal to `a7a20a1`), HU Normal
K=1, pre=256/post=128, batch=256, steps=116, roots=64, IEEE/BF16 eager,
seed=104830, two warmups and three isolated repetitions:

| Reference cache | Repeated seconds | Median seconds | Presentations/s |
| --- | --- | --- | --- |
| Off | 41.952, 41.952, 41.950 | 41.952 | 707.857 |
| 1 GiB budget | 30.681, 30.694, 30.582 | 30.681 | 967.905 |

Caching provides **1.37x** further whole-pipeline throughput with unchanged
169 fresh player nodes, 29,696 learner presentations and 116 optimizer steps.
Per-sample runner counter differences show 29,696 reference cache presentations
and zero repeated reference forward presentations. Allocated peak changes from
20,403,443,200 to 20,408,236,544 bytes (+4,793,344 bytes); these peaks are process
allocator measurements, not cache-payload accounting. The cache budget is not
the actual occupied bytes. Reserved peak is 44,077,940,736 bytes with caching.
There were no competing CUDA jobs. This is a same-source matched comparison,
independent of the older 3.99x physical-pack comparison.

Raw reports: `runs/experiments/nlhe_hu_32bb_normal_h100_cache_off_fdb58ce/report.json`
and `runs/experiments/nlhe_hu_32bb_normal_h100_cache_on_fdb58ce/report.json`.


## HU phase attribution at `8236a61`

Separate instrumented Normal K=1, pre=256/post=128, cache off, one warmup
and one repeated iteration measured 41.442 seconds. Synchronized phase windows
were aligned with GPM using only intervals wholly contained in a phase; means
are weighted by actual interval duration. This run is for attribution, not a
replacement for the uninstrumented repeated medians.

| Phase | Wall seconds | SM active | Warp occupancy | Tensor active | DRAM bandwidth |
| --- | --- | --- | --- | --- | --- |
| Rollout | 1.079 | 5.77% | 2.37% | 0.50% | 1.11% |
| Learner | 40.290 | 80.29% | 42.79% | 9.52% | 36.65% |

Learner accounts for 97.2% of iteration wall time. Rollout is poorly saturated,
but its entire removal would save only about 2.6% on this workload. Completion
(0.054 seconds), targets (0.012 seconds), queue drain, alignment/seal and final
backward bookkeeping have no whole 250 ms GPM interval, so their activity is
unmeasured rather than assumed zero. Use a faster permitted sampling interval
or a separately repeated fixed stage to measure them; neither requires changing
queues. Raw evidence and interval coverage: `phase_windows.jsonl`, `gpm.jsonl`,
`phase_gpm_summary.json`, and `report.json` under
`runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_p256_q128_s104830_8236a61_phase/`.


With the 1 GiB reference-cache budget, the separate instrumented HU iteration
measured 30.544 seconds, with learner 29.380 seconds, rollout 1.090 seconds.
Learner GPM: SM active 78.85%, occupancy 40.81%, Tensor/HMMA active 9.93%,
DRAM bandwidth 35.91%. Reference caching reduces wall time while aggregate
SM activity slightly falls; optimizing for a utilization percentage alone
would miss this throughput improvement. The reference build/gather remains
inside learner timing; the existing phase schema does not separately time
reference and learner forwards. Model kernel traces and runner presentation
counters provide the complementary attribution. Evidence is the corresponding
`...phase_cache/phase_gpm_summary.json`.


Data-type interpretation: BF16 has FP32's exponent range with fewer mantissa
bits; BF16 GEMMs and attention feed Tensor Cores while accumulating matrix
products in FP32. FP32 master parameters and Adam state do not imply FP32
model matmuls: autocast supplies BF16 operands to eligible operators. FP32
non-tensor reductions, losses, normalization and residual intermediates remain
visible separately. NVML HMMA activity includes BF16/FP16 Tensor Core work;
near-zero scalar FP16 activity is compatible with active BF16 Tensor Cores.
GPM is a hardware interval measurement, but it does not expose per-CUDA-core
issue-idle or stall reasons. SM activity's complement measures average inactive
SM cycles across the device, not each arithmetic lane's idle cycles. Nsight
SM scheduler issue/stall counters are the appropriate next attribution if the
provider grants read-only counter access.


## First HU replica/pack tuning result

Normal K=2, pre=128/post=64, cache off, batch=256 and steps=116 per lane,
roots=64 per lane, one warmup and three exclusive samples at `8236a61`:
89.770, 89.136, 89.187 seconds; median **89.187 seconds**, **665.927 learner
presentations/s**, 59,392 presentations and 232 optimizer steps in aggregate.
Fresh nodes are 169/144 by lane (313 total); post/pre mixture differs between
lanes, so this is statistical-work scaling, not an equal-token model benchmark.
Peak allocated/reserved bytes: 13,894,959,616 / 16,577,986,560. Source identity
stayed fixed and no competing jobs were detected.

Per-sample physical work counts: pre 44,851 presentations / 348 cohort calls;
post 14,541 / 340 calls. Two large-pack K=1 executions make 464 total calls,
versus 688 here. Unequal family tails prevent enough exact cohorts at these
caps. The observed aggregate rate is below large-pack uncached K=1's 707.857/s;
this K=2 setting is not recommended on throughput grounds. No conclusion
about other K/pack combinations follows. Evidence:
`runs/experiments/nlhe_hu_32bb_normal_h100_eager_k2_p128_q64_s104830_8236a61/report.json`.


Continuation interruption: the first RB small-pack attempt completed its warmup
but no timed sample before its exec process group was terminated. GPU process
absence and the missing exec handle were both checked. Its partial GPM/progress
records are retained under `...rb_h100_eager_k1_p32_q8_s104830_8236a61/`
`interrupted_attempt_1/`; no timing is accepted. The orphaned reset checkpoint
and failed config scratch were removed. Completed Normal reports are reused
only after validating timing isolation, source revision and harness SHA. A
detached controller waits for the verified live earlier-thread capture parent
before resuming, then guards GPU idle/process state before each arm. This is
continuation/recovery handling, not a source-code or queue-design change.


The stopping agent's complete handoff was read and reconciled:
[h100_validation_agent_handoff_20261006.md](h100_validation_agent_handoff_20261006.md).
It confirms no production patches remain to merge; `a7a20a1` and `fdb58ce`
have identical complete Git tree `5ec299353850cf0e1f5feafd41235c68f7764678`.
The later `8236a61` adds GPM/documentation only. Its RB capture smoke was stopped
at the user's request before completion; it supplies no correctness/performance
evidence. Native captured Normal K=2 and RB K=1/2, RB larger packs/cache/seeds,
whole-pipeline larger K and warmed compilation measurements remain this
thread's responsibility. Completed cache and full-suite evidence is retained
at the handoff's cited authoritative paths. No further GPU work from the
stopping thread is expected.


## Completed RB small-pack baseline

RB K=1, pre=32/post=8, cache off, batch=256, steps=116, roots=64,
completion samples=256 per root (16,384 total), eager IEEE/BF16 at `8236a61`:
152.107, 151.891, 152.758 seconds after one warmup; median **152.107 seconds**.
Every sample retains 169 fresh nodes (118 pre / 51 post), 29,696 learner
presentations and 116 optimizer steps. Physical learner calls are 699 pre +
1,164 post = 1,863, exactly the baseline packing workload. Source and ownership
checks pass. This completes the previously interrupted arm; its interrupted
first attempt remains excluded. Raw report:
`runs/experiments/nlhe_hu_32bb_rb_h100_eager_k1_p32_q8_s104830_8236a61/report.json`.

At this safe stage boundary, the scheduler was replaced while leaving the GPU
child's repetitions untouched. The continued matrix prioritizes a same-source
RB large-pack/cache comparison and RB phase attribution, then the missing
captured Normal K=2 and RB K=1/2 gates. This is scheduling of experiments only;
production FIFO/queue layouts are unchanged.


## Matched RB pack/cache gains at `8236a61`

HU 32bb R=13, BF16/IEEE, six layers d=128, K=1, roots=64, batch=256,
116 steps, completion_samples=256 (16,384 completions per iteration),
rollout/completion packs 8/64, seed104830. Each arm uses one warmup and three
isolated checkpoint-reset repetitions against the same committed source.

| Physical pre/post caps | Reference cache | Repeated seconds | Median | Presentations/s |
| --- | --- | --- | --- | --- |
| 32 / 8 | Off | 152.107, 151.891, 152.758 | 152.107 | 195.230 |
| 256 / 128 | Off | 42.960, 43.194, 43.132 | 43.132 | 688.496 |
| 256 / 128 | 1 GiB budget | 31.950, 32.086, 32.139 | 32.086 | 925.520 |

All three preserve 169 fresh nodes (118 pre, 51 post), 29,696 presentations,
116 optimizer steps and all 16,384 completion samples. Source consistency
and process-ownership checks pass. Pack tuning yields **3.53x**, caching adds
**1.34x**, and the combined same-source RB gain is **4.74x**. This combined
claim is legitimate here because all three arms are same-source matched work;
the separate cross-revision Normal comparisons must not be multiplied this way.
Per-sample pre/post calls are 699/1,164 versus 116/116 for both large-pack arms.
Cached large-pack RB has zero repeated reference forward presentations and
29,696 cached reference presentations; fresh cache construction is included
in wall time and excluded from these learner presentation counters.

Large-pack cache-off allocated/reserved peaks: 22,319,663,616 / 45,906,657,280
bytes. Cache-on: 22,335,695,872 / 45,057,310,720 bytes. These are allocator
measurements and do not quantify payload residency. Reports/configs under
`runs/experiments/nlhe_hu_32bb_rb_h100_eager_k1_p32_q8_s104830_8236a61/`,
`...p256_q128_s104830_8236a61/` and `...p256_q128_s104830_8236a61_cache/`.
Additional seeds, repeated reference refresh/resume, warmed compilation and
larger K remain separate gates before choosing a production recommendation.

## Bounded final run and offline continuation

The user's remaining Runpod budget is at most two hours. A detached tmux series
`h100-final` is now capped at **2026-10-07 00:01:48 UTC**, with GPU benchmark
work stopping two minutes earlier. The unbounded launcher was stopped while
its active Normal K=2 capture continues; the bounded controller adopts that
child and then runs only the priority arms described in
[h100_bounded_final_handoff_20261006.md](h100_bounded_final_handoff_20261006.md).
The earlier broad remaining-work list is deferred where it exceeds this budget.
Final status and the offline archive are documented there. Do not launch a
new open-ended matrix or claim deferred capture/warmup gates are complete.

RB phase telemetry completed: learner 42.033 / 43.282 seconds (97.1%), rollout
1.122 seconds. Learner weighted GPM SM activity 82.75%, occupancy 45.17%,
Tensor/HMMA activity 9.40%, DRAM bandwidth 38.61%. Rollout SM activity is 6.21%
but occupies only 2.6% of wall time. These support prioritizing learner packing
and execution efficiency. They do not establish per-core idle cycles or MFU.
Evidence is the RB `...p256_q128_s104830_8236a61_phase/phase_gpm_summary.json`.

The subsequent Tensor Core investigation and alignment experiments are detailed
in the bounded handoff's latest-evidence section. Dense BF16 GEMM calibration
reached 92.90% Tensor activity / ~680.67 TFLOP/s. Actual large-pack learner
profiles spend 28.41% (pre) / 46.50% (post) of summed GPU kernel time in
matrix multiplication plus attention, with substantial unfused non-matrix work.
Exact-zero working padding 343->344 switches post static FFN GEMMs to Hopper
kernels and improves model-only probes ~4%; three CUDA gradient/address oracle
checks pass. No production padding/normalization change has been adopted.

**New unresolved correctness finding:** ordinary asynchronous native captured
Normal K=2 failed with CUDA illegal memory access. Its synchronous diagnostic
with CUDA_LAUNCH_BLOCKING=1 passed 5 graphs/7 replays after cold compilation,
but this is not a fix or evidence the asynchronous path is safe. The original
failure log remains authoritative. RB captures were deferred under the bounded
budget. The earlier 572-test suite does not cover/clear this newly found full-size
capture failure; the goal remains unfinished pending final/offline analysis.


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
