# HU 32 bb sequential-RB optimization regression — 2026-07-17

> Historical performance evidence: this run used the pre-restoration rollout
> data plane. Preserve its learner-performance lessons, but do not compare its
> phase ownership or throughput directly with tiny 2026-07-21 acceptance runs.

## Purpose and immediate status

This is the self-contained handoff for optimizing the full-deck heads-up
32 bb sequential Rao-Blackwell training path after the July 17 storage/learner
refactor. Read this note before changing the learner.

The production attempt was stopped at approximately 22:13 CDT on 2026-07-17.
The tmux session
`pbg-nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_tuned-` no longer
exists. `nvidia-smi` after stopping showed GPUs 0 and 1 at 0% utilization and
6 MiB each. GPU 2 had another user's approximately 310 MiB process and was not
touched.

The stopped attempt is under
`runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_tuned/`. It was stopped
before iteration 0 completed, so it has no `metrics.jsonl`, checkpoint, or
useful completed-iteration state. Its `status.json` incorrectly still says
`running`: killing the owning tmux session sends a signal that the launcher
does not turn into a final status update. Trust the absent tmux session and the
clean NVML state, not that stale field.

Do not resume this attempt. Fix and validate optimization first, then launch a
fresh output directory initialized from
`runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0000020.pt`.

## Repository and environment snapshot

- Base commit: `8d947242a2c6bcce1060d153428a750ae6427578`
  (`refactor(core): complete memory-safe training pipeline`).
- Important ancestry:
  - `86756b8`: bounded distributed learner.
  - `04bd089`: sealed storage and memory-width control.
  - `165cfb1`: pre-refactor implementation used for conceptual comparison.
- The worktree was already dirty and contains concurrent/user changes. Do not
  reset or overwrite unrelated changes. Inspect `git status` and diffs before
  editing.
- Python environment: `/home/pablob/.venvs/poker`.
- Use `./tools/poker-python` for checked Python/evaluator execution.
- GPUs: three RTX 2080 Ti cards with 11,264 MiB reported each. Use only idle
  cards after checking `nvidia-smi`; GPU 2 commonly has another user's job.
- The sibling evaluator at `/home/pablob/poker-gpu-evaluator` is read-only.
  Its binding contract is
  `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`.
- Relevant prior audits:
  - `docs/audit/distributed_learner_reference_cache_20260717.md`
  - `docs/audit/migration_cleanup_gpu_acceptance_20260717.md`
  - `docs/audit/sealed_storage_memory_control_20260717.md`

The migration audit already marked throughput as the remaining release
blocker on a small benchmark. The full production geometry investigated here
amplifies that blocker from roughly 20% to several-fold learner slowdown.

## Production configuration and intended workload

The active scientific config is `configs/train/nlhe_hu_32bb_rb.yaml` and the
hardware profile is `configs/hardware/local_2x2080ti.yaml`.

Key settings at the time of diagnosis:

```yaml
rollout:
  sequential_roots_per_gpu_initial: 2048
  sequential_roots_per_gpu_maximum: 2560
  family_batch_rows: 64
  max_active_trajectories: 64
  execution_memory_bytes: 1610612736
  memory_growth_fraction: 0.10
  memory_growth_delay: 8

training:
  mode: rao_blackwell
  iters: 90
  gd_steps_per_iter: 232
  minibatch_size: 64
  amp: true

loss:
  name: rb_ppo
  c_kl: 0.2
```

The run is initialized from iteration-20 model weights with
`--init-checkpoint`, not resumed with optimizer/controller/RNG state. It is 90
new iterations numbered 0 through 89.

Each optimizer step currently draws 64 statistical rows per rank, divided
rigidly into 32 preflop and 32 postflop rows. Therefore each iteration requests
14,848 row uses per GPU and 29,696 globally:

$$
232 \times 64 = 14{,}848, \qquad
2 \times 14{,}848 = 29{,}696.
$$

Three fixed-2,048-root probes produced 17,674–18,522 global public-tree nodes,
11,496–12,145 global decisions, and exactly 4,096 global leaves per iteration.
The means were 18,207 nodes and 11,899 decisions. Linear estimates under clean
10% controller growth are:

| Iterations | roots/GPU | expected global nodes | approximate mean retained-row reuse |
|---|---:|---:|---:|
| 0–7 | 2,048 | 18,200 | 2.52x |
| 8–15 | 2,252 | 20,000 | 2.29x |
| 16–23 | 2,477 | 22,000 | 2.08x |
| 24–89 | 2,560 | 22,800 | 2.02x |

Growth is conditional on the measured phase envelopes remaining covered. Only
retained training decision rows are sampled. Chance nodes, terminal nodes, and
filtered decision rows have zero minibatch reuse.

At 2,048 roots/GPU, rank-0 retained partition means across the probes were
3,108.7 preflop rows and 2,780.7 postflop rows. With 7,424 draws assigned to
each partition per iteration, expected reuse was approximately 2.39x per
retained preflop row and 2.67x per retained postflop row.

## Central finding

The current optimization step is materially non-optimal. The long iteration
is not caused by rollout volume.

- Rollout: approximately 34 seconds per iteration.
- Completion/backward target construction: approximately 4–5 seconds.
- Current hot optimizer step: approximately 6.2 seconds.
- Current optimization projection: $232 \times 6.2 \approx 1{,}438$ seconds.
- Projected total iteration: approximately 24–25 minutes.
- Projected 90-iteration run: approximately 36–38 hours.

The config comment projecting approximately 10.5 hours is stale for the new
learner. To finish in about 10.5 hours, sustained optimizer-step time must be
near 1.6 seconds, close to the pre-refactor behavior.

## Baseline comparison

The earlier run `runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl` contains
11 completed, timed iterations before iteration 20 with 3,969 optimizer steps.
Their weighted optimization time was:

| implementation/evidence | optimizer seconds/step |
|---|---:|
| earlier timed iterations, weighted mean | 1.362 |
| earlier minimum iteration mean | 1.007 |
| earlier maximum before iteration 20 | 1.570 |
| current row-3 probes, three-step mean | 5.265 |
| current heat-soaked row-3 step | 6.232 |

Iteration 20 of the earlier run is not a clean baseline: it reports 1.866
seconds/step and many more physical model calls, consistent with code changing
around the transition. Exclude it when estimating the old steady learner.

The current mean is approximately 3.9x the clean earlier weighted mean, and
the heat-soaked step is approximately 4.6x. The user's observation that the
old version was “a lot faster” is correct.

## Probe evidence

### Before allocator-cache boundary fix: singleton postflop

Artifact:
`runs/experiments/nlhe_hu_32bb_ckpt20_memory_probe_fixed2048/metrics.jsonl`.

| iteration | postflop physical rows | physical subspans/step | optimization seconds | training allocated peak |
|---:|---:|---:|---:|---:|
| 0 | 1 | 35 | 6.223 | 1.664 GB |
| 1 | 1 | 35 | 6.232 | 1.765 GB |
| 2 | 1 | 35 | 6.373 | 1.765 GB |

Rollout left about 5.3 GiB of inactive blocks in the PyTorch allocator cache.
`cudaMemGetInfo` reported those blocks as driver-used, so the learner planner
mistook reusable cache slack for live storage and admitted only one postflop
row.

An uncommitted worktree change now calls `torch.cuda.empty_cache()` once at the
rollout/sealed-storage-to-learner phase boundary, immediately before
`_value_workspace_budget` in `pbg/core/training.py`. It returns inactive cache
segments without releasing live sealed storage. This is narrow and passed:

```text
./tools/poker-python -m pytest -q \
  tests/test_distributed_learner.py tests/test_adaptive_work_control.py
21 passed
```

Do not assume this patch alone solves throughput. It fixes incorrect admission
and improves cold batching, but thermal steady-state speed remains poor.

### After boundary fix: three-row postflop

Config:
`configs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe.yaml`.

Artifact:
`runs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe/metrics.jsonl`.

| iteration | pre/post physical rows | subspans | optimization seconds | training allocated peak | iteration reserved peak |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 / 3 | 14 | 4.223 | 2.160 GB | 7.055 GB |
| 1 | 15 / 3 | 14 | 5.339 | 2.255 GB | 7.439 GB |
| 2 | 15 / 3 | 14 | 6.232 | 2.247 GB | 7.529 GB |

GPU utilization rose from roughly 73–76% to roughly 88–96%. Step time degraded
as the cards heated. GPU 1 reached 83–84 C on the automatic fan curve.

The declared transient predictions were 1,169,497,620 bytes for 15 preflop
rows and 6,130,138,104 bytes for three postflop rows. The measured complete
training-phase allocated peak was only 2.16–2.25 GB and its reserved peak was
2.25–2.36 GB after releasing rollout cache. The postflop transient contract is
therefore highly conservative relative to observed end-to-end learner memory.
Do not simply delete the guard; calibrate it with staged multirow probes and
retain the hardware reserve.

### Two-row experiment

Artifact:
`runs/experiments/nlhe_hu_32bb_ckpt20_learner_rows2_probe_corrected/metrics.jsonl`.

The corrected temporary hardware envelope admitted 11 preflop and two
postflop rows, producing 19 subspans. Optimization times were 4.575, 6.848,
and 7.319 seconds. This was worse than row 3. The temporary row-2 config/profile
was removed; the preserved run artifacts are diagnostic only.

### GPU placement experiment

Artifact:
`runs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe_gpu02/metrics.jsonl`.

Using GPUs 0 and 2 instead of 0 and 1 with row-3 batching produced 4.296,
5.417, and 6.430 seconds. GPU 2 also heated to approximately 83 C. Placement
did not materially improve sustained speed. GPU 2 later again acquired another
user's process and must not be selected without a fresh idle check.

### Fan-control experiment

Read-only `nvidia-settings` queries showed automatic fan control. Attempts to
set manual 75% fans failed with `Operation not permitted for the current user`.
No manual fan setting was successfully applied or left behind. Do not build
the optimization plan around unavailable fan privileges.

Thermals reveal wasted kernel/runtime work but are not the underlying software
regression: the old learner sustained approximately 1.0–1.6 seconds/step on
the same cards.

## Root-cause analysis

### 1. Fixed 32/32 statistical allocation overuses expensive postflop rows

Current code in `_one_iteration` divides `training.minibatch_size` equally
over nonempty partitions:

```python
base, extra = divmod(cfg.training.minibatch_size, len(active_ids))
statistical_counts = {
    partition_id: base + int(position < extra)
    for position, partition_id in enumerate(active_ids)
}
```

For HU this forces 32 preflop and 32 postflop samples every optimizer step.
Postflop is much more expensive because the natural exact domain has 1,326
hands and the padded policy sequence is 2,554 tokens.

Earlier production metrics show one preflop and one postflop model call per
step for iterations 0–19. In iteration 0, 302 steps processed 16,819 preflop
rows and 2,509 postflop rows: about 55.7 preflop and 8.3 postflop rows/step.
At iteration 19, the averages were about 50.4 and 13.6. The previous
`_stratified_decision_rows` path allocated samples using stratum mass and a
gradient-variance proxy rather than forcing equal counts.

The new equal split raises costly postflop statistical work to 32 rows/step,
roughly 2.4–3.9x the earlier observed postflop count. Correct exact stratum
coefficients can keep a non-equal sampler unbiased; any replacement must prove
that property and preserve distributed SUM semantics.

### 2. One logical step is fragmented into 14 forward/backward subspans

With the current profile:

- Preflop padded width is about 535 tokens. `training_max_tokens: 8192`
  admits 15 rows.
- Postflop padded width is 2,554 tokens. The same cap admits only three rows.
- The token-pair cap is 67,108,864, admitting roughly ten full-width postflop
  rows; the token cap binds first.

One step therefore becomes:

- preflop: 15 + 15 + 2 rows, three subspans;
- postflop: ten 3-row subspans plus one 2-row tail, eleven subspans;
- total: 14 gradient-bearing subspans.

For every subspan, `DistributedFixedRolloutUpdate.__call__` gathers inputs,
forwards the reference model when uncached, forwards the trainable model,
computes loss, and calls backward before releasing it. Earlier production
usually issued one preflop and one postflop training model call per step.

The 14-way fragmentation increases Python dispatch, gathers, model launches,
loss launches, reference launches, and backward launches. It is the clearest
direct explanation for low efficiency.

### 3. Production reference-cache selection is not wired

`pbg/core/learner.py` implements `ReferenceCacheCandidate`, exact subset
selection, transactional cache construction, and fallback. The acceptance
tests exercise manually supplied candidates. Production does not supply them.

`pbg/core/training.py` currently creates `DistributedLearnerConfig` with
`statistical_counts`, `physical_shapes`, `available_transient_bytes`, and
`resident_base_bytes`, but not `reference_candidates` or `hard_bytes`.
Consequently the default candidate tuple is empty.

Observed production-geometry metrics:

```text
reference_cache_partitions = 0
reference_rows = 64 per optimizer step
model_rows = 64 per optimizer step
reference_cache_predicted_seconds = 0
```

The zero predicted seconds is not evidence that recomputation is free; it is
the empty-candidate selector result. With `c_kl: 0.2`, every sampled row is
forwarded through the frozen reference model again on every one of the 232
steps. The reference model and sealed inputs do not change within the
iteration, so a correctly bounded per-partition output cache is scientifically
exact and potentially high leverage.

The earlier audit explicitly left “route production configuration/calibration
into `DistributedLearnerConfig`” as a phase-3 item. It remains incomplete.

### 4. Physical memory prediction is far above measured learner peaks

`PokerRolloutBackend.training_transient_bytes` uses the family maximum padded
width, layers, fp32 attention matrices, feature activations, and a 1.5 guard.
It is a conservative analytical upper bound, not a calibrated prediction.

For the row-3 postflop shape it reports 6.13 GB, while the entire learner phase
peaks around 2.25 GB allocated. This forces small physical shapes even with
several GiB of genuine headroom. The profile's static token cap independently
limits postflop to three rows.

Calibrate actual training shapes with CUDA peaks and preserve a hard reserve.
Do not infer safe row 8/16/32 operation merely by linear interpolation: the
attention workspace and allocator bins must be measured.

### 5. Activation-checkpoint metadata is not an active optimization path

`PhysicalShape` and `PhysicalSubspan` carry an `activation_checkpointing`
field and metrics count it. The production-created shapes leave it false, and
the poker forward path does not switch implementation based on the subspan
flag. Treat activation checkpointing as unimplemented until code inspection
and a real GPU test prove otherwise.

### 6. Gradient synchronization is probably secondary but must be timed

The new learner uses a 44,536,952-byte fp32 flat-gradient buffer and performs
one SUM collective per optimizer step, followed by one common finite decision,
unscale/clip, and optimizer transition. This is architecturally preferable to
many DDP reductions and should remain unless measurement disproves it.

Current full-production telemetry does not break optimization into sampling,
reference forward, model forward, loss, backward, all-reduce, finite check,
and optimizer update. The old path accumulated host subphase timings; the new
path exposes only total optimization time. Restore low-distortion timing before
guessing at the collective contribution.

## Recommended optimization sequence

### Step 0: build a trustworthy sustained benchmark

Do not use GPU utilization or a single cold step as the primary metric. Create
a production-geometry experiment that:

- initializes from `ckpt_0000020.pt`;
- uses 2,048 roots/GPU fixed at a 2,048 maximum so rollout identity is stable;
- performs enough optimizer steps in one iteration to heat-soak the cards,
  initially 20–40 rather than 232;
- records tail/median seconds per optimizer step;
- records physical shapes, statistical counts, model/reference rows and calls,
  CUDA allocated/reserved peaks, temperatures, clocks, power, and collectives;
- runs on GPUs 0 and 1 only after confirming both are idle;
- leaves at least the hardware-profile VRAM reserve and stages shape increases
  conservatively to avoid OOM.

Three iterations with one optimizer step each were useful for memory and
thermal direction, but a continuous 20–40-step optimization phase is the
correct steady learner benchmark.

Add aggregate CUDA-event or phase-boundary timing without synchronizing every
subspan. Required buckets are:

1. index draw and gather;
2. reference forward/cache lookup;
3. current forward;
4. loss;
5. backward;
6. gradient all-reduce and finite agreement;
7. unscale/clip/AdamW.

### Step 1: wire and measure exact reference caching

Construct real `ReferenceCacheCandidate` entries for each sealed partition and
pass them, plus the correct hard ceiling, into `DistributedLearnerConfig`.
Candidate measurements must include:

- final cache destination bytes from actual output schema;
- bounded build workspace for each calibrated physical shape;
- cached versus uncached learner workspace;
- measured build, cached-step, and uncached-step times.

The selection transaction already reserves destinations before build and
rolls back to exact uncached recomputation on failure. Preserve those semantics,
no-reroll behavior, and learner RNG invariance. Add a production integration
test proving candidates are nonempty and at least one cache is selected when
it fits. Also verify cached and uncached objectives/gradients remain equal.

This should remove repeated frozen-model forwards across 232 steps. It is the
first high-leverage optimization that need not change sampling or the
scientific objective.

### Step 2: calibrate larger physical shapes safely

After caching, probe postflop physical widths 4, 6, 8, and possibly higher.
Change one execution knob at a time. For every width:

- begin with one optimizer step;
- record current allocation, reserved allocation, NVML use, and temperature;
- require substantial headroom before increasing;
- then run the sustained 20–40-step benchmark;
- compare tail seconds/step, not just utilization;
- verify identical fixed-seed sampled statistical indices and objective within
  throughput-mode tolerances.

Reconcile the analytical transient contract against measurements rather than
blindly raising `training_max_tokens`. A hardware profile should express the
validated safe tier after the code path is efficient, not compensate for a
miscalibrated predictor.

### Step 3: restore efficient unbiased statistical allocation

Replace the rigid 32/32 split with a measured allocation that gives each
nonempty stratum positive support and minimizes variance/time. The earlier
mass/gradient-variance method is a reference, not something to copy blindly.

Requirements:

- preserve exact global stratum coefficients and flat-gradient SUM semantics;
- prove the estimator is unbiased under unequal per-partition counts;
- avoid quantizing away small strata;
- preserve deterministic logical RNG/checkpoint behavior;
- expose selected counts and predicted variance/time in metrics;
- compare learning quality as well as wall time.

A likely initial production allocation is closer to the earlier observed
roughly 50–56 preflop and 8–14 postflop rows than 32/32, but the final choice
must be derived from current gradient variance and time measurements.

### Step 4: remove remaining per-subspan overhead

Only after Steps 1–3, profile:

- repeated construction of full postflop weights/masks in
  `policy_forward_partition`;
- redundant public encodings or immutable compatibility masks that can be
  sealed/cached safely;
- loss-side duplicate indexing/log-probability work;
- optimizer/finite-check small-kernel launch overhead;
- safe graph capture or compilation for warmed fixed physical shapes.

Do not introduce a cache whose key depends on mutable model outputs or current
parameters. Public encodings, immutable sealed inputs, and frozen reference
outputs are the safe candidates.

## Correctness constraints that optimization must preserve

- NLHE cash is the only active game; do not revive archived games.
- Sequential mode has branching only at the distinguished root; this learner
  work must not change rollout topology or root multiplicity.
- Preserve natural preflop width 169 and exact/full-deck postflop width 1,326.
- The sibling evaluator remains the sole owner of posterior/equity simulation
  and collision conditioning.
- Preserve immutable `TrainingStorage`, partition routing, exact quotient map
  $\phi$, old-policy identity, and selected-action/legal-mask semantics.
- Preserve clipped RB-PPO objective, player-clock GAE targets, reference
  `KL(current || reference)`, loss scaling, and exact stratum normalization.
- Preserve one flat fp32 gradient SUM across ranks with no world-size division,
  common nonfinite decision, common clipping, and identical optimizer state.
- Preserve repeated sampled indices and their order through physical splitting.
- Physical shape changes must not change statistical sampling or scientific
  configuration identity.
- No sampled-iteration reroll after memory failure.
- Checkpoint/resume must retain model, optimizer, scaler, reference schedule,
  per-rank RNG, adaptive memory controller, and gap-free metrics invariants.
- Throughput mode permits low-order floating differences but not different
  logical draws, counts, targets, or optimizer-step decisions.

## Suggested acceptance targets

These are engineering targets, not mathematical requirements:

- Primary: sustained heat-soaked optimizer time no worse than 1.6 seconds/step
  on two RTX 2080 Ti cards at this production geometry.
- Interim: below 2.0 seconds/step, which projects an iteration near 8.5 minutes
  and 90 iterations near 13 hours before checkpoint overhead.
- Reference cache selected in production when admitted, with zero cached versus
  uncached objective/gradient discrepancy within tested precision.
- No more than a small number of postflop physical subspans per step; justify
  the exact number from memory and throughput measurements.
- Peak allocated/reserved memory remains below the profile ceiling with the
  configured reserve; no OOM is acceptable.
- GPU utilization is diagnostic only. Optimize statistical rows/second and
  optimizer seconds/step.
- Full CPU suite and focused one-/two-GPU learner acceptance pass.

## Relevant code map

- `pbg/core/training.py`
  - `_value_workspace_budget`: driver-free learner workspace estimate.
  - `_one_iteration`: sealing, equal statistical counts, physical shape
    admission, learner construction, phase metrics.
  - `run_training`: reference-model lifecycle and adaptive root controller.
- `pbg/core/learner.py`
  - `PhysicalPlanner`: physical splits and tails.
  - `ReferenceCacheCandidate`, `select_reference_cache_set`: transactional
    subset selection.
  - `DistributedLearnerConfig`: production currently leaves candidates empty.
  - `DistributedFixedRolloutUpdate.__call__`: reference/current forward,
    per-subspan backward, flat synchronization, optimizer step.
- `pbg/core/optimization.py`
  - flat-gradient owner, SUM, finite gate, clipping, optimizer transition.
- `pbg/games/poker/rollout_backend.py`
  - `_family_maximum_width`: pre/post padded sequence widths.
  - `policy_forward_partition`: sealed natural-width learner forward.
  - `training_transient_bytes`: conservative workspace contract.
- `pbg/games/poker/policy_planning.py`
  - older/dynamic physical packing behavior useful for comparison.
- `configs/hardware/local_2x2080ti.yaml`
  - current 8,192 training-token and 67,108,864 token-pair caps.
- `configs/train/nlhe_hu_32bb_rb.yaml`
  - production scientific schedule. Its current output path points to the
    stopped `_tuned` attempt and must be changed for a fresh launch.

## Artifact inventory and useful commands

Primary artifacts:

```text
runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl
runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0000020.pt
runs/experiments/nlhe_hu_32bb_ckpt20_memory_probe_fixed2048/metrics.jsonl
runs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe/metrics.jsonl
runs/experiments/nlhe_hu_32bb_ckpt20_learner_rows2_probe_corrected/metrics.jsonl
runs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe_gpu02/metrics.jsonl
runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_tuned/
```

Recompute the clean earlier step baseline:

```bash
jq -s '
  map(select(.phase_optimization_seconds != null and .iter < 20)) |
  {
    iterations: length,
    steps: (map(.optimizer_steps) | add),
    seconds_per_step:
      ((map(.phase_optimization_seconds) | add) /
       (map(.optimizer_steps) | add))
  }' runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl
```

Inspect the current row-3 probe:

```bash
jq -c '{
  iter,
  phase_optimization_seconds,
  partition_0_physical_rows,
  partition_1_physical_rows,
  physical_subspans,
  reference_cache_partitions,
  reference_rows,
  model_rows,
  training_peak_allocated_bytes,
  training_peak_reserved_bytes
}' runs/experiments/nlhe_hu_32bb_ckpt20_learner_batch_probe/metrics.jsonl
```

Before every CUDA run:

```bash
nvidia-smi
```

Use the repository launcher with explicit GPUs and a fresh output directory.
Torchrun needs local rendezvous access; if the execution sandbox blocks its
socket, request the ordinary escalation rather than changing networking code.

After every probe, confirm selected GPUs have returned to their display-only
idle footprint and that no training worker remains. Never stop or modify the
GPU 2 process unless it belongs to the probe just launched.

## Bottom line

The memory-safe refactor preserved important correctness and ownership
contracts but did not complete production performance integration. Three
issues compound at full-deck HU geometry: equal 32/32 sampling spends too much
statistical work on expensive postflop rows; conservative caps fragment one
step into 14 subspans; and the implemented reference-cache selector receives
no production candidates, so frozen-reference work is recomputed 232 times.

Fix reference caching first, measure continuous per-subphase time, calibrate
larger physical batches safely, then restore an unbiased time/variance-aware
statistical allocation. Do not spend another 36-hour production run until a
heat-soaked production-geometry benchmark is back near the earlier 1.0–1.6
seconds per optimizer step.
