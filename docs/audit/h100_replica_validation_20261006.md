# H100 replica validation and performance evidence

Status: validation in progress; production throughput has not been measured.

The active validation workstream is Codex conversation
`01a112e3-cf04-72c3-aa47-8a7c4f333c52`. The earlier conversation detected these
shared edits and stopped its pilot. The pilot prepared at `629b2a8` must not
resume against changed production code. Its correctness log remains evidence
for that revision; its interrupted autotuning does not establish throughput.
A new committed revision and plan will bind the next benchmark.

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

Nsight Compute 2025.1.1 is installed, but hardware counter access fails with
`ERR_NVGPUCTRPERM`. Shared driver policy is unchanged. CUDA timelines and NVML
clocks/power/memory/utilization can be recorded; Tensor Core activity, achieved
SM occupancy and idle-cycle percentages cannot be measured directly here.

## Remaining work

1. Commit the CUDA-tested reproducibility repair.
2. Prepare a fresh hash-bound pilot, complete Normal/RB K=1/2 smoke, and measure
   matched batch=256/steps=116 K=1/16 arms with warmed repeated samples.
3. Test smaller-pack K=32 or admitted K=8 where useful; validate selected settings
   across additional seeds and reference refreshes.
4. Profile phase/model costs separately from isolated unprofiled measurements;
   report aggregate throughput and per-run latency with explicit denominators.
5. Tune measured launch, layout, capture, or memory bottlenecks and rerun affected
   CUDA gates. One H100 cannot validate multi-rank NCCL; MPS/SM75 validation also
   requires the corresponding hardware.
