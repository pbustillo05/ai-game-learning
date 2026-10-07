# H100 validation agent handoff — 2026-10-06

This validation agent stopped at the user's explicit request so the other
optimization agent can take full control. No further tests, benchmarks, or
production edits are planned by this agent. The original goal is paused,
not complete: RB performance, wider full-pipeline scaling, and several capture
checks remain unfinished.

## Ownership and current repository state

- Shared checkout: `/workspace/ai-game-learning`, observed HEAD `8236a61`.
- Continuing optimization conversation: `01a112e3-cf04-72c3-aa47-8a7c4f333c52`.
- Stopping validation conversation: `01a112c4-94f3-7cf1-b9bf-f61053f42f72`.
- Isolated review worktree: `/tmp/h100-scheduler-review`, branch
  `h100-cohort-tail-review`, HEAD `fdb58ce6deb8ee36419a498064c04c5261665dd1`.
  All its production fixes are already integrated into the shared checkout.
  No additional production patch from this agent needs merging.
- That isolated revision and shared commit `a7a20a1` have the identical complete
  Git tree `5ec299353850cf0e1f5feafd41235c68f7764678`. Commit `8236a61` adds
  GPM monitoring/documentation without changing the tested production code/tests.
- The existing dirty report
  [h100_replica_validation_20261006.md](h100_replica_validation_20261006.md)
  belongs to the continuing agent. This handoff does not overwrite it.
  Successful experimental configs remain available under `configs/experiments/`.

The only live GPU task owned by this agent was the RB K=1 captured smoke, PID
130377. It and its verified descendant monitor/compiler processes were stopped
with SIGTERM; no surviving owned worker required SIGKILL. Its wrapper exited
with child status -15. Immediately afterward the continuing agent's RB eager
benchmark, PID 134101, acquired the GPU; it was left untouched. Thus our GPU
allocation was released, and the device is now under the other agent's control.

The interrupted smoke has **no completed correctness or throughput result**.
Its state is explicitly `stopped_by_user`; partial logs/NVML/progress evidence
were retained, and its interrupted runtime/config directories were removed.
Cleanup records are in
`runs/experiments/nlhe_hu_32bb_normal_h100_cache_review/owned_smoke_cleanup.json`
and `captured_smoke_state.json` in that same directory.

## Recovered environment

The earlier evaluator access/pin problem is resolved. The user's GitHub
authentication and push made the required revision available; no alternative
revision was substituted.

| Item | Validated value |
| --- | --- |
| Accelerator | H100 80GB HBM3, MIG disabled |
| GPU UUID | `GPU-c0a3cd37-65a1-2be4-42ea-79c44f710069` |
| Driver | 580.159.04 |
| Torch-reported memory | 85,017,493,504 bytes |
| Usable admission budget after 10 GiB reserve | 74,280,075,264 bytes |
| Python / venv | 3.10.21, `/home/pablob/.venvs/poker` |
| Torch / Triton | 2.12.1+cu130 / 3.7.1 |
| CUDA compiler | 13.0.88 |
| Evaluator | 0.7.0, contract v2 |
| Evaluator checkout | `/home/pablob/poker-gpu-evaluator` |
| Evaluator revision / imported BUILD_REVISION | `caeeb976045a17454d12293811469863e5e1ce61` |

Pinned requirements installed successfully and `pip check` passed. Native BF16
fused attention forward/backward passed on this H100. The evaluator API file
was read without modification; evaluator-dependent production work stays on
CUDA. No CPU fallback was introduced.

The prepared shell environment is `/tmp/h100_env.sh`:

```bash
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=GPU-c0a3cd37-65a1-2be4-42ea-79c44f710069
export MAX_JOBS=4
```

The CUDA wheel initially lacked the unversioned `libcudart.so` linker alias;
[setup-h100-toolkit.sh](../../tools/setup-h100-toolkit.sh) handles that case.
Neither `nvcc` nor `ninja` is missing from the configured venv.

## Bugs fixed and integrated

| Commit in shared checkout | Finding and resulting behavior |
| --- | --- |
| `5ba5653` | Fused CUDA attention rejected head widths 4/6. Q/K/V feature dimensions are padded to a multiple of eight, the original attention scale is retained, and output is cropped. Output/input/parameter-gradient parity passes at d=16/24/128 in FP32/FP16/BF16. Also repairs CUDA wheel linking. |
| `629b2a8` | Repairs native compile/capture replay tests, current shared-model/static-gate fixtures, benchmark parameter-layout metadata, and obsolete wrapper remnants. Live controls/weights and pointer stability are tested. |
| `5024269` | Uneven lane family tails destroyed compatible cohorts. Scheduling now drains earlier FIFO families and selects the largest exact contiguous compatible cohort, preserving each lane's samples, losses, Adam transitions, and publication. |
| `b14850e` | Graph residency rejection discarded already-warmed compilation. A rejected graph pool is released while the compiled function remains in the bounded cache. Pre-capture admission failures retain their existing eager behavior. |
| `a7a20a1` | Reference-cache admission estimated reuse from fresh store chunks, ignoring the scheduled learner presentations. It now uses sampler metadata without advancing RNG, including pooled/stratified family allocations. Known unprofitable caches are rejected before a reference forward. |

The last three changes originated in the isolated review worktree and were
cherry-picked by the continuing agent. Source patches and test records are
retained under
`runs/experiments/nlhe_hu_32bb_normal_h100_pilot_629b2a8/reports/`.

The continuing agent separately repaired and integrated:

- `3a2d9ac`: strict CUDA reproducibility now actually configures deterministic
  dispatch and cuBLAS workspace policy. All default profiles retain IEEE FP32;
  enabling TF32 broke six existing precision gates. No tolerance was widened.
  Strict K=2/K=16 checkpoint/reference-refresh replay became bitwise identical.
- `78aaaf1`: branched packet capacities now cover successive action and
  flop/turn/river expansion; topology admission checks each replica's forest
  against its own limits while retaining combined upload-byte limits.
- `83e9f55`: updates a stale rollback-test launch-count assumption after cohort
  scheduling changed. Actual rollback checks for parameters, buffers, Adam,
  scaler, RNG, versions and counters remain intact.

No changes were made to protected mathematics, queue partitioning, or the
private-hand compact bound.

## Correctness evidence

The final completed full CUDA suite passed **572 tests, 2 skipped, 15 warnings
in 273.60 seconds** at frozen `fdb58ce`, whose tree exactly matches `a7a20a1`.
The durable log is
`runs/experiments/nlhe_hu_32bb_normal_h100_pilot_629b2a8/reports/cache_fix_full_cuda.log`.
The 20 directly affected cache tests also passed in 50.78 seconds. Static
production audit reported zero violations.

The retained tests cover cache gathers versus live reference output, fixed-step
and epoch reuse, explicit family quotas, no RNG advancement, zero wasted
reference calls on rejected cache admission, and bounded payload admission.
Scheduler regressions cover uneven K=2/4/16 tails with independent parameter
and moment oracles. Graph-denial regressions exercise compiled reuse with
mutable inputs after capture rejection.

Shared evidence additionally includes 18 full-width native-model regressions
and all 12 public six-layer BF16 consumer variants: HU/6max, sequential/fixed/KKT
branching, sampled/RB, three iterations and reference refresh for each variant.
Those consumer checks validate finite losses/gradients/masters, applied Adam
updates and zero-copy ownership; they are untimed compatibility checks.
See `runs/experiments/nlhe_hu_32bb_normal_h100_validation/reports/` and the
continuing agent's audit for detailed logs.

A Normal K=1 full-width captured smoke completed at `3a2d9ac`: four retained
graphs, six replays, zero denied shapes. Its cold 697.6-second iteration used
ATEN-only compile search and overlapped untimed checks. It proves capture
compatibility, not steady-state speed. The interrupted RB smoke from this
agent adds no completed result.

Earlier full runs with a stale rollback fixture are diagnostic history, not
final passing suites. The repaired final 572-test run supersedes them. A
duplicate full run stopped by the other agent while waiting on shared extension
compilation is also incomplete. Multi-rank NCCL, MPS and SM75 behavior cannot be
validated with the hardware available in this pod.

## Matched full-pipeline performance

Both comparisons below use native BF16, six layers, d=128, HU 32bb, R=13,
Normal mode, K=1, IEEE FP32, eager execution, seed 104830, 64 rollout roots,
statistical batch 256 and 116 optimizer steps. Every timed sample presents
29,696 learner rows and generates 169 fresh player nodes. Each arm has two
warmups and three checkpoint-reset samples without competing CUDA jobs.

| Comparison | Baseline median | Candidate median | Whole-pipeline gain |
| --- | --- | --- | --- |
| Physical pre/post caps 32/8 → 256/128, cache off, `3a2d9ac` | 164.894 s | 41.360 s | 3.99× |
| Reference cache off → 1 GiB budget, caps 256/128, frozen `fdb58ce` | 41.952 s | 30.681 s | 1.37× |

The physical-pack measurement is the continuing agent's isolated comparison,
independently reviewed by this agent. Learner cohort calls fall from 1,863 to
232; statistical batch, row presentations and optimizer work are unchanged.
Peak allocated memory rises from approximately 1.90 GiB to 19.00 GiB.
Authoritative reports:

- `runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_3a2d9ac/report.json`
- `runs/experiments/nlhe_hu_32bb_normal_h100_eager_k1_large_isolated_3a2d9ac/report.json`

For this agent's cache pair, off samples are 41.952039, 41.951975 and 41.949608
seconds; on samples are 30.680709, 30.694026 and 30.581756 seconds. Throughput
rises from **707.857 to 967.905 presentations/s**. All samples retain 232 cohort
calls and exactly the same pre/post statistical presentation counts.

With caching, the learner consumes 29,696 cached reference presentations and
performs zero repeated reference-forward presentations. The initial reference
cache construction still forwards the 169 fresh rows and is included in whole
iteration timing; those construction forwards are outside learner counters.
Peak allocated bytes increase from 20,403,443,200 to 20,408,236,544, an extra
4,793,344 bytes (4.57 MiB). That allocator difference is not a measurement of
cache payload, and a 1 GiB budget does not imply 1 GiB occupied memory.

Authoritative cache evidence:

- `runs/experiments/nlhe_hu_32bb_normal_h100_cache_review/matched_cache_results.json`
- `runs/experiments/nlhe_hu_32bb_normal_h100_cache_off_fdb58ce/report.json`
- `runs/experiments/nlhe_hu_32bb_normal_h100_cache_on_fdb58ce/report.json`
- Exact experiment configs under the corresponding `configs/experiments/`
  directories; harness at
  `runs/experiments/nlhe_hu_32bb_normal_h100_cache_review/scripts/benchmark_cache.py`.

These are separate comparisons at different source revisions. Do not multiply
their gains into a same-source aggregate claim. Neither establishes long-run
learning quality, RB performance, captured throughput, or wider replica scaling.
Reference caching remains opt-in; this experiment did not change the default.

An earlier large-pack timing with overlapping CUDA work was rejected and is
excluded. The initial `629b2a8` pilot was interrupted and must not be resumed
against changed source; it has correctness evidence but no completed benchmark.

## Other useful findings and next experiments

The continuing agent's phase instrumentation at `8236a61` attributes 40.290 of
41.442 seconds (97.2%) to the learner with cache off, and approximately 1.079
seconds to rollout. With caching, learner time is 29.380 of 30.544 seconds;
rollout remains about 1.090 seconds. Prioritize remaining learner costs.
Instrumented samples provide attribution and should remain separate from the
unprofiled three-repeat medians above.

Read-only NVML GPM counters work on this H100. The retained tool is
[h100-gpm-monitor.py](../../tools/h100-gpm-monitor.py). Cache-off learner intervals
show approximately 80.29% SM activity, 42.79% warp occupancy, 9.52% Tensor/HMMA
activity and 36.65% DRAM bandwidth activity. Cache-on wall time improves even
though SM activity falls slightly; optimize elapsed work, not utilization alone.
Per-kernel Nsight Compute counters still fail with `ERR_NVGPUCTRPERM`; no shared
driver permissions or clocks were changed. Short completion/target phases have
no full 250 ms sample interval, so their GPM activity is unmeasured.

Increasing only post cap 128 → 192 did not improve the measured HU workload:
both caps already execute 116 post calls. A K=2 pre128/post64 cache-off run
measured 89.187 seconds and 665.927 aggregate presentations/s, below the large
K=1 baseline. Unequal lane/family sizes and extra physical calls matter;
replica count alone does not establish a speedup.

Production storage admission accepts default K=1/8/16 but rejects default K=32
before allocation: Normal learner storage needs 80,588,599,296 bytes and RB
84,999,761,920, exceeding the usable budget. Model-only K=32 with smaller packs
was finite, but excludes queues, targets, references, Adam and graph pools and
therefore does not establish production admission.

Suggested continuation, all still uncompleted by this agent:

1. Finish RB eager baseline and matched larger-pack/cache arms; the continuing
   agent's RB baseline was running at handoff. Validate gains across additional
   seeds and reference refreshes before selecting defaults.
2. Finish full-width captured Normal/RB K=1/2 checks and compare warmed compiled
   and captured execution with matched eager statistical work. Measure graph
   residency and verify compiled reuse after graph denial.
3. Investigate existing pointwise fusion/launch overhead and attention backends
   while preserving the learned additive key prior, numerical contracts, and
   current queue layouts. These are hypotheses, not measured improvements.
4. Measure admitted K=8/16 whole-pipeline scaling and physical pack choices.
   Only consider K=32 after the actual storage plan admits smaller packs.
   Preserve per-lane FIFO, independent optimizer semantics and exact row counts.
5. Keep profiler/instrumented runs separate from isolated timing, record source
   identity and runtime precision policy, and retain meaningful CUDA parity and
   rollback gates for each implemented change.

This handoff is intentionally a separate report so the continuing agent can
update its own audit, configurations and benchmarks without conflicting edits.
