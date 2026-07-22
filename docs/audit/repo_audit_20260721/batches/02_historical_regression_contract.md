# Historical regression and performance contract

Audit point: `b55df976e80b82c474c86202c90f3b32072af03c` (`HEAD` when this
batch was inspected). Historical comparison begins at
`537700f5b94873c3daa6e6a03ad00cfef4b8e88b`. This pass used Git and
artifact inspection only; it did not run tests or GPU work. The existing
worktree modification to
`docs/todo/restore_authoritative_storage_lifecycle_20260721.md` was not read as
historical evidence and was not modified.

## Executive result

The July 18 learner optimization, adaptive KKT implementation, and protected
root refill are still identifiable in the current production path. Their main
symbols and focused tests survived the storage refactor. The refactor also
replaced global node capacity with page-level transactional admission.

There are nevertheless two material regression risks that cannot be cleared
from the checked-in evidence:

1. The current normal and sequential-RB target builders perform Python scalar
   reads from device topology and decision tensors inside per-trajectory and
   per-decision loops (`pbg/core/targets.py:169-182` and `251-312`). This
   reintroduces the exact host-synchronization pattern that commit `b94941c`
   removed from tree validation. The previous vectorized value reducer was
   deleted in `dce75ff`; no equal-work post-refactor target benchmark exists.
2. Fixed-$\beta$ production training has page-allocation transactions but no
   downstream-envelope-aware closure admission. The former upper-node cutoff
   is explicitly evaluation-only (`pbg/core/branching.py:52-64`), while
   `docs/todo/optimization.md:454-477` still describes production fixed-$\beta$
   OOM admission as future work. Thus the large-tree safeguard is only partly
   preserved for the fixed-$\beta$ core mode.

The repository has no same-hardware, same-workload run of current `HEAD` that
can establish absence of an end-to-end efficiency regression. Historical
numbers below are therefore either artifact-backed pre-restoration baselines,
document-only claims, or non-comparable smoke evidence, as labeled.

## Commit timeline

| Commit | Historical responsibility | Current disposition |
|---|---|---|
| `537700f` | Cost-aware fixed statistical allocation, sampled-row reference caching, fused policy terms, native biased SDPA, separately planned cache-build shapes, CUDA learner phase telemetry, and tuned learner/hardware profiles | Core symbols and tests retained |
| `818aa17` | Geometry-only upper/stop cutoffs for branched experiments | Retained as `BranchingDiagnosticLimits`, explicitly excluded from training configuration |
| `b94941c` | Vectorized `TreeBatch.validate()` to remove thousands of per-node/per-edge CUDA-to-host scalar synchronizations | Vectorized validator retained; analogous scalar loops were introduced in new target builders |
| `79a354b` | Explicit fixed/dynamic branching mode, KKT intensity/controller/path recurrences, T/C heads and losses, checkpoint state, and tests | Mathematical primitive file is unchanged from this commit; wiring migrated |
| `1cde6de` | Protected-root minimum-tree refill, unbiased protected target prefix, dynamic counterfactual correction; normalized-value fixes and extra diagnostics | Refill semantics retained through paged rollout, compact topology, and target builders |
| `a399e30`, `9af6a29`, `759cde6`, `dce75ff` | Byte-admitted paged storage, migration of sequential and branched paths, removal of legacy global node capacity | Current sole path; page-level admission retained |
| `d0d246c` | Sole-path acceptance and current GPU lifecycle tests | Current acceptance suite; tiny rather than historical production geometry |

## Binding historical contracts

### HIST-PERF-01 — Preserve the accelerated distributed learner

The behavior introduced at `537700f` is not one optimization but a connected
set. A replacement learner must preserve all of the following:

- positive arbitrary per-stratum statistical counts using the cost proxy and
  exact $\alpha_m/B_m$ weighting;
- one reusable sampled-index buffer and the stable flat fp32 gradient buffer;
- bounded physical subspans and separately bounded no-grad reference-cache
  build shapes;
- sampled-row cache construction that preserves both the update and generator
  state when compared with uncached recomputation;
- one fused natural policy composition for selected log probability, entropy,
  and KL;
- native biased SDPA on the validated NVIDIA profile;
- SUM gradient synchronization followed by the finite/clip/update gate;
- telemetry sufficient to attribute gather, reference, forward, loss,
  backward, synchronization, and optimizer time.

Current evidence:

- `pbg/core/learner.py:212-264`, `allocate_fixed_statistical_counts`;
- `pbg/core/learner.py:610-665`, `_policy_terms`;
- `pbg/core/learner.py:881-959`, `DistributedLearnerConfig` and the stable
  `FlatGradientBuffer`/planner owners;
- `pbg/core/learner.py:1144-1210`, sampled unique-row cache planning;
- `pbg/core/learner.py:1264-1447`, direct accumulated backward, flat SUM, and
  CUDA bucket telemetry;
- `pbg/core/training.py:2343-2420`, cost allocation and separate physical/cache
  build plans;
- `pbg/games/poker/adapter.py:237`, fused `policy_terms`;
- `pbg/games/poker/model/primitives.py:181-211` and `319-349`, biased SDPA
  implementation and dispatch;
- `configs/hardware/local_2x2080ti.yaml:3`, `attention_backend: sdpa`;
- `tests/test_distributed_learner.py:86-103`, cost allocation;
  `372-409`, flat-gradient aliasing; `411-461`, sampled-cache update/RNG
  equality; and `tests/test_p6_primitives.py:72-118`, SDPA forward/backward and
  CUDA throughput-precision comparisons.

Disposition: **strong static preservation**, but not performance-verified on
current `HEAD`. In particular, the current path still calls
`torch.cuda.empty_cache()` at `pbg/core/training.py:2360` and `2808`, so the
optimization TODO's allocator-cycle concern is still live.

### HIST-PERF-02 — Preserve the target/value synchronization fix

Commit `b94941c` replaced scalar node/edge validation with bounded tensor
operations. Its surviving implementation is visible at
`pbg/core/tree.py:110-175`: vectorized child counts, `repeat_interleave`,
`index_add_`, vector parent/depth/weight comparisons, and `bincount`.

The historical vectorized benchmark artifacts are:

- `runs/experiments/nlhe_hu_32bb_sequential_value_vectorized_bench/metrics.jsonl`:
  one GPU, 2,048 roots, 5,051 decisions, one optimizer step, target phase
  1.055 s, profiled iteration 18.866 s;
- `runs/experiments/nlhe_hu_32bb_sequential_value_vectorized_bench_2560/metrics.jsonl`:
  one GPU, 2,560 roots, 6,343 decisions, one optimizer step, target phase
  1.313 s, profiled iteration 23.718 s.

Those runs occurred before `b94941c` was committed and are consistent with the
worktree version that became that commit, but neither artifact stores a Git
commit hash. They establish an absolute historical observation, not a
reproducible before/after ratio.

Current risk: `build_normal_targets()` converts CUDA `leaf_row`, CSR pointers,
partition ids, rows, owners, and old values to Python scalars at
`pbg/core/targets.py:169-182`. `build_sequential_rb_targets()` does the same in
both its extent pass and completion/reverse passes at lines `251-312`, including
`bool(pool_valid[pool_row])`. For CUDA inputs these are repeated implicit host
synchronizations. Commit `dce75ff` deleted `pbg/core/values.py`, including its
vectorized streamed reducer, and retained these loops as the sole path.

Disposition: **high-confidence algorithmic performance regression risk**.
Correctness tests (`tests/test_refactor_storage_targets.py:274-379`) cover
returns, posterior deduplication, and exact/model widths, but no performance or
no-host-sync gate covers these target loops. A current equal-work GPU benchmark
is required before this contract can be marked preserved.

### HIST-KKT-01 — Preserve adaptive KKT exactly

Commit `79a354b` introduced the dynamic controller. The complete file
`pbg/core/adaptive_branching.py` has no diff between `79a354b` and `HEAD`.
The binding primitive set is:

- `gamma_deviance`, lines `20-49`;
- ordinary randomized rounding, lines `52-62`;
- clipped log-domain KKT intensity, lines `65-90`;
- player-ancestor and ancestor-product recurrence, lines `93-108`;
- damped/checkpointed $\log\kappa$ controller, lines `111-209`.

Current wiring computes fixed or dynamic intensity and records nominal versus
executed offspring at `pbg/core/branching.py:438-499`. The controller update
uses the refill-corrected counterfactual tree size at
`pbg/core/training.py:1027-1064`. Focused evidence remains in
`tests/test_adaptive_kkt_branching.py:36-216` (loss, intensity, rounding,
controller, checkpoint), `276-359` (warmup, active KKT, and safeguard), and the
real T/C target/loss tests later in that file. Tiny one- and two-GPU dynamic
acceptance wiring is in `tests/test_refactor_gpu_acceptance.py:238-269` and
`351-390`.

Disposition: **strong static and focused-test preservation**. This pass did not
re-run those tests. The batch-local `safe_tree_size` suppression changes
executed multiplicity relative to nominal and relies on the counterfactual
correction; its mathematical fidelity belongs in the math audit, not in this
historical contract.

### HIST-SMALL-01 — Preserve the unbiased small-tree refill

Commit `1cde6de` added `branching_min_nodes` by appending root samples after an
ordinary root cohort, while excluding appended samples from root target
averages and controller observations. Current preservation points are:

- input validation and refill execution at `pbg/core/branching.py:165-173` and
  `837-900`;
- compact protected-prefix metadata and validation in
  `pbg/core/topology.py:162-185` and `252-268`;
- protected root target degree in `pbg/core/targets.py:556-625`;
- refill-corrected dynamic controller observation in
  `pbg/core/training.py:1037-1061`;
- behavior tests in `tests/test_c09_labeled_branching.py:343-389`, compact
  topology/target tests in `tests/test_refactor_storage_targets.py:209-241` and
  `538-582`, and GPU refill coverage in
  `tests/test_refactor_gpu_acceptance.py:238-269`.

Disposition: **strong static preservation**. The storage representation
changed, but the protected ordinary prefix, final uniform root weights,
nominal count, and controller exclusion all remain explicit.

### HIST-OOM-01 — Fail before allocation without rerolling statistical work

There were three historically distinct safeguards, which must not be conflated:

1. `818aa17` added `branching_upper_cutoff_nodes` and stop cutoffs for geometry
   experiments. Suppressing future multiplicity to one changes the sampled
   law; it was never sufficient as a transparent training OOM fix.
2. `79a354b` added dynamic KKT's batch-local `safe_tree_size` multiplicity
   suppression with nominal-count correction.
3. The storage refactor (`a399e30` through `dce75ff`) introduced exact
   byte-accounted pages and grouped child-wave transactions.

The current grouped child transaction is
`pbg/core/branching.py:351-380`: tree nodes/edges, occurrence arena,
controller rows, reach pool, poker state, and history pages are all required
before draws or state mutation; every owner rolls back page growth on error.
`tests/test_p8_tree.py:138-177` checks grouped rollback, and lifecycle acceptance
documents the sole path in
`docs/audit/restore_storage_lifecycle/phase_5_sole_path_and_memory_control.md:32-45`.

However, `BranchingDiagnosticLimits` is explicitly "evaluation-only" at
`pbg/core/branching.py:52-64`; training config has no fixed-$\beta$ upper cutoff.
The transaction accounts for currently requested execution owners but not an
analytical unit-multiplicity closure of every live frontier plus retained
completion, target, reference, gradient, optimizer, driver-reserve, and
fragmentation envelopes. `docs/todo/optimization.md:454-477` states that this
production safeguard remains to be built.

Disposition: **partly preserved / open high-severity gap for fixed $\beta$**.
Page over-allocation now fails transactionally, but a large fixed tree lacks a
first-class predictive closure contract. No current realistic fixed-$\beta$
soak demonstrates a clean pre-OOM stop under the sole production path.

## Performance evidence inventory

### Artifact-backed measurements

| Artifact | What it establishes | Comparability limits |
|---|---|---|
| `runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl` | 1,110 completed two-rank iterations. Direct aggregation gives mean profiled iteration 143.340 s, rollout 44.255 s, targets 7.642 s, optimization 91.386 s, learner forward 23.371 s, backward 42.750 s, reference-cache build 14.095 s, and gradient synchronization 6.953 s. Mean decisions were 11,188.8. These reproduce the headline budget in `docs/todo/optimization.md:3-14`. | Pre-restoration data plane; policy and root width evolve over 1,110 iterations. Valid as a historical production budget, not a current-HEAD acceptance run. |
| The two `sequential_value_vectorized_bench*` metrics files described above | Absolute one-GPU target/iteration timing at 2,048 and 2,560 roots after the validator optimization | One optimizer step, no embedded commit hash, and pre-restoration storage |
| `runs/experiments/nlhe_hu_32bb_normalized_loss_smoke_20260721/metrics.jsonl` and `...cv016...` | Three-iteration and one-iteration normalized-loss smokes; target phases about 0.24-0.27 s for roughly 2,100 decisions | Only 512 roots/GPU and 16 optimizer steps; changed loss units; run before `1cde6de`, so not a refactor comparison |
| `runs/experiments/nlhe_hu_32bb_normalized_loss_pilot40_20260721/metrics.jsonl` | 32 completed pilot iterations with finite timing records | Same non-comparability as the normalized smokes; incomplete relative to the intended 40 |
| `runs/experiments/branching_hu32_ckpt814/*.json` and sample JSONL files | Fixed-$\beta$ geometry/censoring experiments underlying the branching section of the TODO | Cutoff-modified trees, checkpoint 814, benchmark/evaluation path, and not current production training throughput |
| `runs/nlhe_6max_10bb_full_branched/metrics.jsonl` | Historical long branched run | Predates explicit five-mode/KKT/refactor contracts; unsuitable as current fixed/dynamic regression baseline |

### Documentary claims whose raw artifacts are absent

`docs/audit/nlhe_hu_32bb_optimizer_speed_record_20260717.md` reports a final
two-RTX-2080-Ti 232-step update of 69.427 s (0.2993 s/step), with 18.828 s
forward, 33.873 s backward, 3.196 s gradient synchronization, 2.328 s loss,
1.989 s optimizer update, 8.794 s reference build, and 3.64/3.88 GB learner
allocated/reserved peaks. It identifies
`runs/experiments/nlhe_hu_32bb_optimizer_final_widecache_232step/metrics.jsonl`,
but that directory is not present in the current workspace. The numbers are a
precise historical acceptance claim, not independently re-aggregatable raw
evidence here.

The same limitation applies to most intermediate optimizer probes named in
`docs/audit/nlhe_hu_32bb_optimization_regression_20260717.md` and
`docs/audit/nlhe_hu_32bb_training_parameter_tuning_20260717.md`. Their configs
remain, but their run directories are absent. Do not use their stated ratios as
current regression gates without restoring or reproducing the artifacts.

### Comparison that is currently impossible

There is no realistic post-`dce75ff` run matching the historical full-deck HU
32 bb workload, two RTX 2080 Ti GPUs, SDPA/AMP, roots per GPU, completion
samples, statistical presentations, loss, reference-cache cadence, and warmup.
The storage lifecycle acceptance report explicitly uses tiny tests. Therefore:

- a current learner backward regression cannot be ruled out by static symbol
  survival;
- the new target-loop regression risk cannot be quantified;
- end-to-end sampled decisions/s cannot be compared fairly;
- fixed-$\beta$ and KKT production memory/throughput regressions cannot be
  measured from `runs/`.

## Optimization TODO triage

The current `docs/todo/optimization.md` mixes still-valid ideas, items partly
addressed by the refactor, and stale implementation descriptions.

### Still live and evidence-backed

- Fuse rollout action/log-prob/reach/retention work. Current sequential rollout
  still calls `backend.reach_update` and then `_selected_policy_logp`
  separately (`pbg/core/rollout.py:819-845`); branched does likewise at
  `pbg/core/branching.py:608-672`.
- Compile/capture stable rollout and learner shapes. There is a small compiled
  attention helper in `pbg/games/poker/model/primitives.py:58`, but no
  end-to-end rollout or learner `CUDAGraph` path.
- Streamline reference-cache construction and cache only consumed KL outputs.
  The historical 14.095 s/run average is artifact-backed.
- Improve postflop physical packing/street bucketing. Large sections still use
  `comb(n_cards - 3, 2)` physical width in
  `pbg/games/poker/rollout_backend.py:1870`, `1981`, and `2072`.
- Remove/consolidate repeated hot-path device synchronizations. The new target
  loops make this more urgent than the TODO states.
- Overlap gradient communication with backward. The flat gradient SUM remains
  synchronous at `pbg/core/optimization.py:133`.
- Compact preflop live reach. Branched root reach is still allocated with the
  maximum exact width in `pbg/core/branching.py:205-218`.
- Eliminate the rollout-to-learner allocator flush. Two current production
  calls remain at `pbg/core/training.py:2360` and `2808`.
- Fixed-$\beta$ closure/OOM admission remains safety-critical, as described in
  HIST-OOM-01.

### Partly implemented or requiring rewrite against the new owners

- B3/B4 (direct branched caches and chunk capacity owners): payload-proportional
  Python tensor lists and global node arrays were replaced by
  `PagedRowAccumulator`, `PagedTreeSource`, paged reach, and controller arenas.
  The final node cache still seals with an order/gather at
  `pbg/core/branching.py:962-986`; optimization should target that exact owner,
  not the deleted list/`cat` implementation.
- "Specialized sequential backward" is stale wording. The current problem is
  the Python-scalar target implementation in `pbg/core/targets.py`, so any
  optimization must implement the authoritative bounded segmented algorithms
  directly rather than resurrecting deleted `values.py` storage.
- Graph-blocking validation work was partly solved by `b94941c`, but the
  repository needs a broader no-host-sync target/learner gate.

### Stale or unsafe as written

- File/line references in the TODO describe the pre-restoration data plane and
  should not be used as architectural authority.
- The fixed-$\beta$ benchmark's "upper cutoff" produces censored, altered-law
  trees. It is valid for geometry measurements only and must not be promoted as
  a transparent production memory fix. Production admission must fail or close
  under an explicitly authorized mathematical rule.
- The B4 claim that tree/reach/queues all allocate the complete node budget is
  obsolete after the paged-storage refactor.
- The TODO says specialized sequential propagation has only about a 0.4 s
  ceiling. That estimate predates the new scalar target loops and cannot be
  carried forward without measurement.

## Required regression gates for synthesis

1. Add a static/no-host-sync test that rejects Python scalar reads of CUDA
   topology and decision columns inside normal and sequential-RB target loops.
2. Run a one-GPU equal-seed 2,048-root sequential-RB target benchmark and
   compare against the artifact-backed 1.055 s target phase, then repeat at
   2,560 roots against 1.313 s. Match completion samples and separate completion
   from recurrence time.
3. Run the historical learner geometry (SDPA/AMP, 53/11 or the current
   mathematically equivalent cost allocation, 232 steps) before claiming the
   0.2993 s/step optimization survived. Record cache-build, forward, backward,
   SUM, loss, update, physical spans, and memory peaks separately.
4. Add a fixed-$\beta$ synthetic-ceiling test that proves child/frontier closure
   admission fails before allocation and without consuming/rekeying logical
   draws; then stage a realistic GPU tree only after its complete downstream
   byte envelope is known.
5. Re-run KKT checkpoint/resume, protected refill, and tiny one-/two-GPU dynamic
   acceptance tests, preserving nominal/executed counts and controller state.
6. Only after those gates, compare end-to-end current throughput with the
   artifact-backed 1,110-iteration historical budget. Report measured values,
   documentary values, and inferred values in separate columns.
