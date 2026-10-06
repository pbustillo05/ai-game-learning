# H100 replica validation and performance evidence

Status: CUDA correctness gates passed; full-pipeline tuning and native capture
validation are in progress. Production Normal eager timing is available.

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

Nsight Compute 2025.1.1 is installed, but hardware counter access fails with
`ERR_NVGPUCTRPERM`. Shared driver policy is unchanged. CUDA timelines and NVML
clocks/power/memory/utilization can be recorded; Tensor Core activity, achieved
SM occupancy and idle-cycle percentages cannot be measured directly here.

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
for those measurements. No driver policy or container permissions were changed.

## Measured optimization priorities

1. Tune physical packs before increasing K: the default K=1 physical caps leave
   much of this H100 unused, and larger caps already yielded a whole-pipeline gain.
2. Measure existing pointwise fusion and graph replay. Thousands of eager
   launches and FP32 elementwise traffic remain visible; large captured packs
   currently lose graph admission under the 8 GiB limit. Compare retaining
   compiled execution after a graph denial if compiled timing supports it.
3. Evaluate device reference-output caching after matched cache-off baselines.
   The same frozen reference is currently recomputed for reused learner rows.
4. Test a static postflop compact bound based on at least three board cards.
   All compatible private hands fit in $\binom{4R-3}{2}$ rows; the current default
   map allocates $\binom{4R}{2}$. Existing gather/scatter masks can preserve full
   output IDs while reducing private token work. This needs CUDA parity and
   performance measurements before adoption.
5. Inspect odd static feature widths and alignment-sensitive GEMMs, and compare
   fused attention backends while preserving the learned additive key prior.
   These remain hypotheses; Tensor Core counters would improve attribution.
