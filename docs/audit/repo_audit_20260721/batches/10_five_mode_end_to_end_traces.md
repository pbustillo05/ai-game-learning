# Five-mode end-to-end production traces

## Scope and conclusion

This pass read batch reports `00` through `07` first and then independently
traced the installed configuration, training, evaluation, rollout, storage,
target, learner, distributed-update, and checkpoint paths. It made no code or
test changes and ran no GPU workload.

The implementation has **three internal training/storage labels** and exactly
five intended external semantic combinations:

1. `training.mode == "normal"`;
2. `training.mode == "rao_blackwell"` with no `probe`;
3. `training.mode == "branched_rao_blackwell"` and
   `rollout.branching_mode == "fixed"`;
4. `training.mode == "branched_rao_blackwell"` and
   `rollout.branching_mode == "dynamic"`;
5. `training.mode == "rao_blackwell"` with a non-null `probe`.

That is the correct conceptual mapping of the requested five modes onto the
three storage families described by `AMB-002`. The fifth combination is an
approximate BR probe, not an authoritative exact best-response algorithm.
Fixed and KKT branched RB intentionally share `TrainingStorage.mode ==
"branched_rb"`; their semantic distinction exists in rollout geometry and
checkpointed controller state, not in targets or learner loss.

All five combinations have a CUDA-capable code path, but none is
NVIDIA-exclusive: `completion.backend` still permits `cpu` and `auto`, and the
launcher falls back to CPU. BR has additional qualifications: it has no
authoritative algorithm, no committed launch preset or CUDA acceptance test,
and its multi-player early-fold path is broken. Therefore only the four
training estimators are statically NVIDIA-production reachable over the full
declared poker game domain; BR is only partially so.

The cross-cutting trace confirms the major storage/queue findings in batch 04,
the learner and failure findings in batch 05, and the poker/model findings in
batches 06-07. It refines `AUD-RSQ-011`: the installed validated poker training
path **does** hard-require and train T/C in fixed mode, despite the rollout
helper's locally permissive schema inference.

## Compact semantic and reachability matrix

| External mode | Strict selector | Rollout entry and realized object | Target/storage label | Learner branch | CUDA production status |
|---|---|---|---|---|---|
| `normal` | `training.mode=normal`, `probe=null`, no branching fields, `completion.n_samples=null` | `sample_trees_sequential(..., return_profiles=True)`; $G_k$ independent terminal trajectories | `build_normal_targets`; `TrainingStorage("normal")` | `mode_loss("normal")` | Reachable on CUDA, but CPU/auto also accepted; no committed normal preset |
| `sequential_rb` | `training.mode=rao_blackwell`, `probe=null`, no branching fields, positive completion samples | `sample_trees` -> `sample_trees_sequential`; $G_k$ independent public pre-terminal paths | `build_sequential_rb_targets`; `TrainingStorage("sequential_rb")` | `mode_loss("sequential_rb")` | Reachable and has a sequential-RB preset/CUDA smoke; CPU/auto still accepted |
| branched RB, fixed beta | `training.mode=branched_rao_blackwell`, `branching_mode=fixed`, `adaptive_kkt=null`, one root, T/C enabled | `sample_trees` -> `sample_branched_tree`; one labeled tree, fixed beta table, randomized rounding | `build_branched_rb_targets`; `TrainingStorage("branched_rb")` with T/C targets | `mode_loss("branched_rb")` including Gamma-deviance T/C | Reachable and preset/CUDA smoke exists; fixed queue safeguard is missing |
| branched RB, KKT | same branched mode, `branching_mode=dynamic`, required `adaptive_kkt`; fixed warmup then KKT | same one-tree path; `dynamic_active` gates learned player beta after warmup; chance beta remains fixed | Same branched target/storage path, including nominal/executed corrections and T/C | Same branched loss; controller changes only future geometry | Reachable and tiny CUDA smoke exists; no committed KKT preset |
| `BR` probe | `training.mode=rao_blackwell` plus `probe.target_checkpoint`; fixed opponent, sampled probe seat | same sequential rollout with mixed policy and `ProbeRollout`; retains only probe-seat decisions | same sequential-RB target/storage path | same sequential-RB PPO loss, updating only probe model | Approximate BR only; no authority/CUDA gate/preset, CPU fallback active, multi-player early-fold path broken |

## Shared entry, bundle, optimization, and checkpoint trace

### Entry and semantic selection

- The installed training command is `pbg-train` (`pyproject.toml:14`) ->
  `pbg.cli.train.main` (`pbg/cli/train.py:11-40`) -> `load_config` ->
  `pbg.runtime.train` -> `run_training` (`pbg/config.py:11-21`,
  `pbg/runtime.py:24-38`, `pbg/core/training.py:2529-2538`).
- The strict parser exposes only three base training strings at
  `pbg/core/config.py:282-287`. `ProbeConfig` is a nullable overlay
  (`:123-129`, `:155-166`), and a probe is allowed only on
  `rao_blackwell` (`:288-305`). Branched fixed versus dynamic is an explicit
  secondary selector (`:306-357`). This yields the five combinations above;
  no sixth estimator label passes the strict loader.
- Normal requires no completion samples; both RB families require a positive
  sample count (`pbg/core/config.py:504-516`). Branched validation requires one
  root on every rank, explicit beta, T/C model heads, and positive T/C loss
  coefficients (`:306-325`, `:380-405`).
- The selection is not canonicalized into a single semantic enum. This is why
  benchmark conversion can collapse KKT into fixed mode and why branching
  heads can remain enabled on non-branched modes, confirming `MODE-003` and
  `MODE-005`.

### Device and bundle construction

- `run_training` selects CPU for `completion.backend=cpu`, CUDA for `cuda`, and
  CUDA-if-present otherwise (`pbg/core/training.py:2543-2558`). Multi-rank uses
  NCCL on CUDA and Gloo on CPU (`:2708-2717`). This confirms `MODE-001` and
  `AUD-TL-005`: CUDA reachability is real, but NVIDIA-only execution is not
  enforced.
- `build_bundle` constructs `PokerModelBundle`, the policy adapter, posterior
  sampler, partition adapter, paged poker rollout backend, and (except normal)
  the sealed completion service on the selected device
  (`pbg/games/poker/bundle.py:639-717`). The `sequential` memory-plan bit is
  derived solely from whether the mode is branched (`:669-680`).
- A probe target is hash-checked and game-checked on CPU, then a frozen opponent
  model is created on the requested device and registered beside the trainable
  model (`pbg/core/training.py:2600-2635`, `:2686-2707`). Only the probe model's
  parameters enter the optimizer (`:2718-2725`).

### Common frozen rollout-to-learner lifetime

The live model is put in evaluation mode and remains immutable through rollout,
sealing, completion, and targets; no second rollout-model clone is made
(`pbg/core/training.py:1070-1108`). Each mode then follows:

```text
model/backend owners
  -> paged rollout sources + queues + mutable poker states/reach
  -> SampledTree + policy/leaf/input stores
  -> post-hoc RolloutStorage + compact target topology
  -> exact immutable TrainingStorage
  -> bounded sampled learner subspans + direct flat gradients
  -> optimizer state / checkpoint
```

The wrapper releases backend rollout owners in `finally` blocks
(`pbg/core/rollout.py:964-987`, `pbg/core/branching.py:67-92`). The iteration
builder deletes sampled tree/policy owners after sealing and deletes rollout and
completion owners after target construction (`pbg/core/training.py:1213-1226`,
`:1301-1324`, `:1390-1409`). The learner deletes every gathered subspan and
graph immediately after backward (`pbg/core/learner.py:1294-1383`), and the
iteration deletes learner and final storage after the update
(`pbg/core/training.py:2524`).

This apparent last-consumer discipline does not cure the architectural overlap:
all four training estimators first finalize the larger `SampledTree`, then
allocate the normative topology while that tree remains live. The conversion at
`pbg/core/training.py:1725-1739` and `:1765-1826` confirms `AUD-RSQ-001` and
`AUD-RSQ-002` across mode boundaries.

### Common learner/distributed/checkpoint path

- Every target builder emits immutable partitioned `TrainingStorage`.
  `validate_learner_input` finalizes it, rejects forbidden fields, and requires
  uniform sampling for sequential modes versus aliases for branched mode
  (`pbg/core/learner.py:55-87`).
- `_one_iteration` allocates fixed statistical counts by local mass and model
  width, builds separately bounded learner/reference shapes, and creates
  `DistributedFixedRolloutUpdate` (`pbg/core/training.py:2323-2449`).
- The learner all-gathers stratum log masses, samples uniform indices for
  `normal`/`sequential_rb` and alias indices for `branched_rb`, forwards one
  physical subspan, applies `mode_loss`, scales once by
  `alpha/count`, and backpropagates directly into the flat gradient buffer
  (`pbg/core/learner.py:109-150`, `:1225-1383`). It performs one scaled-gradient
  SUM and one finite-gated optimizer step (`:1384-1401`). This confirms the
  preserved direct-backward path in `HIST-PERF-01`.
- Rank zero atomically checkpoints model, optimizer, scaler, scientific config,
  per-rank RNG, sequential memory controller, optional KKT controller, and
  probe target identity (`pbg/core/checkpointing.py:148-210`;
  `pbg/core/training.py:3163-3202`). Resume explicitly rejects KKT controller
  state in fixed mode and requires it in dynamic mode
  (`pbg/core/training.py:2788-2800`). Thus fixed and KKT do not silently change
  semantics on checkpoint resume.
- Mandatory rollout/target/learner exceptions still occur before a common
  failure flag and can strand peers already entering collectives. Nothing in
  the per-mode branches repairs this, confirming blocker `AUD-TL-001` for all
  five combinations.

## Mode 1: normal

### Exact wiring

1. Strict selection is `training.mode == "normal"`, with null completion
   samples and no branched fields (`pbg/core/config.py:282-287`, `:336-348`,
   `:504-508`). A probe is rejected (`:288-292`).
2. `_build_iteration_data` calls `sample_trees_sequential` with
   `return_profiles=True` (`pbg/core/training.py:1109`, `:1143-1173`).
3. `_sample_trees_sequential` fixes $G_k=roots_per_gpu$, allocates independent
   root addresses and profiles, maintains one live occurrence/reach row per
   trajectory, and samples player/chance events with logical RNG keys
   (`pbg/core/rollout.py:284-436`, `:671-877`).
4. At each stopped leaf, `capture_normal_leaf` settles one sampled private
   profile and runout (`pbg/core/rollout.py:565-610`;
   `pbg/games/poker/rollout_backend.py:2579-2588`). Sealing projects each
   decision to its realized private type and retains scalar old log probability
   and value plus terminal reward (`pbg/core/training.py:1544-1565`,
   `:1635-1646`).
5. `SequentialDecisionCSR.build` is constructed post hoc and
   `build_normal_targets` performs the reverse player-clock lambda recurrence
   (`pbg/core/training.py:1725-1739`; `pbg/core/targets.py:137-200`).
6. `mode_loss("normal")` gathers the realized type, computes frozen-policy PPO,
   scalar value loss, entropy, and optional reference KL
   (`pbg/core/learner.py:681-746`).

### Owners, conformance, and reachability

The rollout retains sampled root private profiles until terminal settlement and
sealing, then only realized-type scalar policy/value rows and terminal rewards
survive. Posterior pools, completion service, branched action tables, and T/C
targets are absent. This matches the intended mathematical consumer set.

Normal nevertheless carries full log-reach state through the shared sequential
rollout and first seals the full legacy tree; queue scheduling, schema inference,
preflop exact-width carriers, postflop packing, and memory-envelope findings
from batches 04, 06, and 07 all apply. Its target recurrence contains per-node
CUDA scalar reads (`pbg/core/targets.py:169-182`), confirming `AUD-TL-002` and
`HIST-PERF-02` risk.

The mode is CUDA reachable when `completion.backend=cuda`, despite normal not
using completion. It is not CUDA-only, and no committed normal production
preset exists (`MODE-006`).

## Mode 2: sequential Rao--Blackwell

### Exact wiring

1. The selector is `training.mode == "rao_blackwell"` with `probe is None`,
   no branching fields, and positive completion samples
   (`pbg/core/config.py:282-305`, `:336-357`, `:509-516`).
2. `_build_iteration_data` calls `sample_trees`, whose dispatch selects
   `sample_trees_sequential` (`pbg/core/training.py:1155-1163`,
   `pbg/core/rollout.py:990-1013`).
3. The sequential sampler maintains public reach and uses temporary posterior
   profiles to sample each event; at a stopped leaf it retains immutable RB
   completion inputs and stopped reach (`pbg/core/rollout.py:398-443`,
   `:565-610`; `pbg/games/poker/rollout_backend.py:2559-2569`).
4. `_seal_sequential_rollout_storage` creates a `CompletionSpec` around the
   sibling evaluator's `complete_sealed` API and logical completion keys
   (`pbg/core/training.py:1647-1723`).
5. `build_sequential_rb_targets` deduplicates posterior rows per
   trajectory/player and computes exact-hand lambda returns; the resulting
   storage owns one posterior pool and one return row per decision
   (`pbg/core/targets.py:203-341`).
6. `mode_loss("sequential_rb")` maps natural model rows to exact hands, applies
   posterior-weighted PPO/value/entropy/KL, and derives advantage at loss time
   (`pbg/core/learner.py:747-781`).

### Owners, conformance, and reachability

Mutable public state and live reach die after stopped-leaf/decision capture;
sealed leaves own exact completion inputs only until target microbatches are
projected. The final owner is the exact posterior pool plus natural-width frozen
model rows and exact-hand lambda returns. This is semantically aligned at the
consumer boundary.

The target implementation repeatedly converts CUDA topology/owner/valid values
to Python scalars (`pbg/core/targets.py:251-315`), and the rollout first expands
preflop outputs to exact-hand width then compresses them during seal
(`pbg/core/training.py:1597-1603`). Those independently confirm
`AUD-TL-002`, `AUD-RSQ-004`, and the historical target-performance regression
risk. The full-tree conversion, queue-policy, queue-accounting, evaluator
compatibility, folded-summary, and postflop-width findings also apply.

The mode is CUDA reachable and has a committed sequential-RB preset and tiny
CUDA acceptance coverage, but CPU/auto remains legal.

## Mode 3: branched Rao--Blackwell with fixed beta

### Exact wiring

1. Validation requires branched mode, `branching_mode == "fixed"`, explicit
   beta, `adaptive_kkt is None`, exactly one root per rank, T/C heads, and
   positive T/C coefficients (`pbg/core/config.py:306-357`, `:380-405`).
2. Dispatch is `sample_trees` -> `sample_branched_tree`
   (`pbg/core/rollout.py:1005-1012`). The sampler itself reasserts one root and
   fixed/dynamic validity (`pbg/core/branching.py:143-173`, `:293-307`).
3. `offspring_for` uses `backend.beta_fn`, ordinary randomized rounding, and
   independent logical keys for every node including root
   (`pbg/core/branching.py:438-508`). Every child independently samples a
   posterior profile and action/chance event (`:551-610`).
4. The completed tree optionally receives protected-root refill and corrected
   root/path masses (`pbg/core/branching.py:834-900`). Nominal and executed
   counts, logical paths, old policy/value, and frozen T/C are retained
   (`:901-1009`).
5. `_record_branched_controller_metrics` requires T/C **before** returning for
   fixed mode (`pbg/core/training.py:983-1004`). The branched seal creates the
   target-degree action prefix and `BranchedTreeSpan`, then completion and
   `build_branched_rb_targets` produce value, child posterior, per-edge
   leave-one-out advantage, trace target/mask, and cost target
   (`pbg/core/training.py:1744-2027`; `pbg/core/targets.py:393-716`).
6. `mode_loss("branched_rb")` uniformly reduces child action losses, uses child
   posterior for policy and mean child posterior for value/entropy/KL, and adds
   Gamma-deviance T/C losses (`pbg/core/learner.py:782-854`).

### Owners, conformance, and reachability

This confirms the one-root law, ordinary root rounding, sibling independence,
uniform edge mass, protected refill, nominal/executed retention, leave-one-out
baseline, and T/C training. Fixed mode is CUDA reachable and has committed
fixed presets plus a tiny CUDA smoke.

The installed path refutes the broad claim in `AUD-RSQ-011` that fixed
production can silently omit T/C: strict config requires the heads, the poker
models emit them, `_record_branched_controller_metrics` rejects omission, the
target builder emits T/C columns, and the learner trains them. The narrower
finding remains valid: `sample_branched_tree` and schema-on-first-write do not
locally reject both fields being absent, so direct helper callers bypassing the
validated training frame are permissive.

Fixed mode lacks the mathematical scientific queue safeguard because the only
`safe_tree_size` lives in `adaptive_kkt` and suppression is gated by
`dynamic_active` (`pbg/core/branching.py:474-481`). This confirms
`AUD-RSQ-009` and `HIST-OOM-01`. Its child materialization scales with logical
offspring (`:551-681`), and full-tree seal/conversion remains outside hard
admission. Thus reachability does not imply pseudocode-compliant memory safety.

## Mode 4: branched Rao--Blackwell with KKT

### Exact wiring

1. The selector is branched mode plus `branching_mode == "dynamic"` and a
   validated `AdaptiveKKTConfig`; the same one-root/T/C requirements apply
   (`pbg/core/config.py:306-405`).
2. `run_training` creates or restores `AdaptiveKKTController` only for dynamic
   mode (`pbg/core/training.py:2815-2824`).
3. `dynamic_active` is false until `iteration >= warmup_iters`; before that,
   geometry uses the fixed beta table and the controller does not advance
   (`pbg/core/branching.py:293-307`, `:438-460`;
   `pbg/core/training.py:1035-1036`).
4. After warmup, player-family nodes compute log-space KKT intensity from
   frozen output T/C, controller `log_kappa`, player-ancestor sum, and ancestor
   product. Chance nodes continue to use fixed beta because `family is None`
   (`pbg/core/branching.py:438-460`, `:654-661`).
5. The dynamic-only queue safeguard may replace executed counts by one while
   retaining nominal counts (`pbg/core/branching.py:472-507`). Controller
   observation uses nominal-executed cost correction and excludes refill-only
   root work before `controller.observe` (`pbg/core/training.py:1008-1067`).
6. Targets and learner are exactly the common branched path above. Dynamic
   geometry does not introduce a second target or optimization implementation.

### Owners, conformance, and reachability

The KKT primitive/controller wiring, warmup, player-only learned intensity,
counterfactual observation, refill exclusion, checkpoint restoration, and T/C
learning are statically preserved, confirming `HIST-KKT-01` and
`HIST-SMALL-01`.

The scientific safeguard is disabled during dynamic warmup because it is gated
on `dynamic_active`, even though the mode is already the KKT external mode.
This is the warmup half of `AUD-RSQ-009`. The same unbounded child wave,
post-hoc full-tree conversion, inferred schemas, queue ordering/accounting,
folded-summary, evaluator sync, and model-width findings apply.

KKT is CUDA reachable and has tiny CUDA acceptance coverage, but no committed
KKT production preset or five-mode benchmark case exists (`MODE-003`,
`MODE-006`).

## Mode 5: BR probe

### Exact wiring

There is no BR training enum. A probe is the sequential-RB mode plus a nullable
`ProbeConfig` whose target checkpoint is content-bound (`pbg/core/config.py:123-129`,
`:288-305`; `pbg/core/training.py:2600-2635`). Training is launched through the
ordinary `pbg-train` command. `pbg-br-eval` only evaluates already trained probe
checkpoints (`pbg/cli/br_eval.py:11-34`).

During rollout:

- one probe seat is selected per trajectory by configured seats or a logical
  RNG key (`pbg/core/rollout.py:357-389`);
- decisions owned by that seat use the trainable probe model; all other seats
  use the frozen target model (`:780-807`);
- only probe-seat policy/value/input rows are retained (`:833-857`);
- once the probe seat folds, tree ownership is overwritten with `PROBE_FOLD`
  so its continuation can be cut (`:860-868`);
- sealing and target construction then reuse the sequential-RB completion,
  posterior pool, lambda target, and PPO learner branches
  (`pbg/core/training.py:1510-1518`, `:1647-1723`, `:1265-1353`).

Evaluation hash-checks the paired target, reconstructs both models, and compares
mixed-policy versus target-only rollouts under logical addresses
(`pbg/eval/br_probe.py:87-170`). It reports an approximate lower-bound gain, not
an exact BR certificate (`:29-49`, `:179-200`).

### Authority, owners, and reachability

`core_math.tex` supplies no BR sampling law, objective, stopping condition, or
exploitability definition (`MATH-BR-001`). The implementation trains one shared
function approximator across sampled seats under sequential-RB PPO and calls
the result an approximate probe. Its fidelity cannot be adjudicated against
the designated authority.

The opponent model is a full additional frozen GPU owner for rollout and is
checkpoint-identified but not checkpoint-embedded. Probe decision rows alone
enter `TrainingStorage`; opponent decision rows die after successor/reach
materialization. The final checkpoint owns only the probe model plus target
hash/path metadata and logical-work counters.

BR is not fully certified as NVIDIA production: training accepts CPU, evaluation
defaults to CUDA-if-present but accepts explicit CPU
(`pbg/eval/br_probe.py:87-96`, `:129-140`), all current integration tests use
CPU, and no committed probe preset exists. More seriously, the declared
$P=2\ldots6$ domain is not end-to-end reachable for early probe folds; see
`TRACE-001` below.

## Cross-cutting finding reconciliation

| Prior finding | End-to-end disposition |
|---|---|
| `MODE-001`, `AUD-TL-005` CPU/Gloo path | Confirmed for all five combinations. CUDA capability exists but exclusivity does not. |
| `MODE-003` benchmark omits KKT and BR | Confirmed. This trace finds five semantic combinations, while benchmark conversion exposes only three and forces fixed branching. |
| `MODE-005` unused T/C outside branched | Confirmed. Non-branched validation has no converse gate; dead T/C parameters can accompany normal/sequential/BR. |
| `MODE-006`, `MODE-007` missing presets/BR authority and CUDA test | Confirmed. Only sequential RB and fixed branched have canonical presets; BR remains approximate and CPU-tested. |
| `AUD-RSQ-001`/`002` legacy full-tree post-hoc seal and unadmitted overlap | Confirmed in normal, sequential RB, fixed, KKT, and BR (which reuses sequential seal). |
| `AUD-RSQ-004` preflop H-to-K post-hoc compression | Confirmed in all RB training paths, including BR; normal selects a scalar realized type from the H-width carrier. |
| `AUD-RSQ-005`/`006` branched unbounded child wave and late address publication | Confirmed in both branched modes; learner subspan bounding does not repair rollout materialization. |
| `AUD-RSQ-007`/`008` queue scheduling/accounting | Confirmed across both sequential families (normal, sequential RB, BR) and both branched modes. |
| `AUD-RSQ-009` fixed/warmup safeguard absent | Confirmed. Fixed never applies it; KKT warmup does not either. |
| `AUD-RSQ-011` fixed T/C can silently disappear | **Refined/refuted for installed production:** strict configuration, poker model, and `_record_branched_controller_metrics` collectively require T/C, and fixed targets/loss train them. The local rollout helper remains insufficiently strict. |
| `AUD-TL-001` distributed failure agreement | Confirmed for every mode; all share the same unguarded failure-to-collective path. |
| `AUD-TL-002`, `HIST-PERF-02` target CUDA scalar loops | Confirmed for normal and sequential RB, therefore also BR. |
| `POKER-001`/`002`, `AUD-MODEL-002` folded-summary ownership | Confirmed as bundle/model dependencies shared by every mode. |
| `POKER-003` evaluator v1 compatibility and `POKER-005` syncs | Confirmed in shared posterior/completion rollout paths; normal also uses the profile sampler and terminal evaluator. |
| `AUD-MODEL-001`, `POKER-009` worst-width postflop execution | Confirmed in every mode through shared backend family forwards. |

## New standardized findings

### TRACE-001 — Multi-player BR probe early-fold leaves cannot be settled

- Severity: high
- Confidence: confirmed by static control/data flow
- Contract: requested $P\in\{2,\ldots,6\}$ production domain; `MATH-BR-001`
  authority gap; five-mode NVIDIA requirement
- Modes affected: BR probe for $P>2$ when the selected probe seat folds before
  the public hand becomes terminal
- Evidence:
  - `pbg/core/rollout.py:860-868`, `_sample_trees_sequential`, overwrites only
    the **tree owner** of the successor with `PROBE_FOLD`; it does not change the
    underlying poker state's phase.
  - `pbg/core/rollout.py:531-610`, `chance_close`, routes this marker as a leaf
    but passes only `stopped_states`, private profiles, and keys to
    `capture_probe_leaf`; it does not pass the tree owner or selected seat.
  - `pbg/games/poker/rollout_backend.py:2590-2609`, `capture_probe_leaf`, calls
    `self.classify(states)` again and tests that result for `PROBE_FOLD`.
    `PokerRolloutBackend.classify` can return only chance, terminal,
    pre-terminal, or the current actor (`:365-380`); it never derives
    `PROBE_FOLD` from poker state.
  - Consequently the artificial cutoff is treated as `ordinary` and passed to
    `sampled_terminal_reward_keyed`. That routine rejects every state whose
    reclassified owner is not `TERMINAL` or `PRETERM`
    (`pbg/games/poker/rollout_backend.py:772-788`).
  - `tests/test_br_probe.py:37-73` builds only heads-up configs. Its explicit
    fold test constructs a synthetic `SampledTree.owner=PROBE_FOLD` but calls
    the older completion API directly (`:172-213`); it never exercises
    `capture_probe_leaf` through the production rollout cutoff.
- Current behavior: in heads-up poker, a probe fold normally makes the poker
  state terminal and masks the defect. With three or more live players, a probe
  can fold while opponents continue; production creates a `PROBE_FOLD` tree
  leaf, loses that marker at capture, and raises during reward settlement.
- Required behavior: the immutable probe cutoff contract must carry the
  selected seat and cutoff identity to leaf capture/completion, and settle the
  selected player's public fold payoff without pretending the underlying hand
  is terminal. The authoritative BR document must specify whether and why this
  continuation truncation is valid.
- Consequence: the fifth mode is not production-reachable over the repository's
  advertised 2-6 player domain, on CUDA or CPU; paired evaluation shares the
  same sampler and is affected.
- Recommended disposition: pass an explicit cutoff mask/selected seat into the
  leaf materialization schema and test the real production sampler for 3-6
  players. Do not infer a synthetic tree owner by reclassifying state.
- Verification needed: deterministic three-player forced-fold train and paired
  evaluation tests on CUDA; verify selected posterior/payoff, absence of
  opponent target rows after cutoff, hash binding, and equality under queue
  rescheduling.

### TRACE-002 — The fifth mode is a nullable overlay rather than a closed semantic selector

- Severity: medium
- Confidence: confirmed
- Contract: exactly-five-mode requirement; production-only configuration
- Modes affected: all five, especially BR and branched fixed/KKT
- Evidence:
  - `pbg/core/config.py:105`, `TrainingConfig.mode`, is an unconstrained string
    later restricted to three labels at `:282-287`.
  - `ProbeConfig` is independent nullable state (`:123-129`, `:165`) and fixed
    versus KKT is another nullable field (`:55-66`).
  - Runtime branches on the base mode in `_build_iteration_data`
    (`pbg/core/training.py:1109-1436`) and discovers BR indirectly from
    `bundle.opponent_model` in rollout (`pbg/core/rollout.py:357`, `:780-792`).
  - The benchmark consequently recognizes only three labels and rewrites the
    branched case to fixed (`pbg/benchmark.py:28`, `:68-100`), as already found
    by `MODE-003`.
- Current behavior: the strict validator presently excludes most invalid
  combinations, so there are five intended combinations, but no single object
  names or exhaustively dispatches them. Consumers can accidentally collapse
  fixed/KKT or overlook the probe overlay.
- Required behavior: one canonical semantic selector/matrix should map the five
  externally supported modes to their three storage families and required
  config fields. It need not duplicate storage implementations.
- Consequence: cross-mode tooling, presets, benchmarks, acceptance tests, and
  audit reachability repeatedly disagree about how many modes exist.
- Recommended disposition: introduce a derived, closed semantic-mode enum or
  validated property used by runtime, benchmark, manifests, and tests; retain
  the existing three storage labels internally.
- Verification needed: exhaustive loader table proving exactly five valid
  semantic combinations and rejecting every extra combination; round-trip the
  semantic identity through run metadata and checkpoints.

### TRACE-003 — Non-branched branching heads create a shadow parameterized path

- Severity: medium
- Confidence: confirmed
- Contract: `MATH-NORMAL-004`, `MATH-SRB-005`, minimal production payload;
  exactly-five-mode requirement
- Modes affected: normal, sequential RB, BR
- Evidence:
  - Branched validation requires `use_branching_heads=true` and positive T/C
    coefficients, but there is no converse rejection at
    `pbg/core/config.py:380-388`.
  - `build_bundle` instantiates the model exactly from that flag and charges it
    in the rollout plan regardless of training mode
    (`pbg/games/poker/bundle.py:647-680`).
  - Sequential rollout stores only `output.v`, never T/C
    (`pbg/core/rollout.py:833-857`), and normal/sequential loss branches never
    consume them (`pbg/core/learner.py:711-781`).
  - The optimizer nevertheless owns every model parameter
    (`pbg/core/training.py:2718-2725`).
- Current behavior: a valid non-branched configuration may instantiate and
  forward T/C heads whose outputs have no target, loss, or gradient. This does
  not create a sixth estimator, but it is a production-reachable shadow model
  architecture and resource cost.
- Required behavior: non-branched modes must reject branching heads and require
  `c_t=c_c=0`; only the two branched semantics should own T/C.
- Consequence: unused objects remain reachable, memory/performance comparisons
  are biased, and checkpoints for the same named semantic mode can have
  different dead parameter sets.
- Recommended disposition: enforce symmetric config gates and canonicalize all
  five presets through the strict loader.
- Verification needed: parameter-count and allocation parity tests for normal,
  sequential RB, and BR; rejection of all branching-only fields outside the two
  branched modes.

## Extra semantics and missing gates

- No extra training estimator survives strict loading beyond the five
  combinations. The variance CLIs, diagnostic tree limits, CPU/MPS inference,
  and three-mode benchmark are additional active workflows/platforms, not sixth
  learner estimators. They remain production-surface violations identified by
  batches 03-07.
- `ppo_epochs` versus fixed `gd_steps_per_iter`, optional KL, value clipping,
  entropy schedules, utility normalization, optimizer choice, and fixed beta
  tables create parameterized loss/update variants inside a mode. The math
  authorizes optional KL/value clipping, but there is no closed canonical
  production profile proving these variants do not constitute unwanted mode
  drift.
- Missing gates are: CUDA-only launcher/config; converse absence of T/C outside
  branched modes; one canonical preset and CUDA acceptance test per semantic
  mode; a five-mode benchmark identity; BR authority; multi-player BR cutoff
  coverage; common distributed terminal-failure agreement; and direct compact
  storage/queue/memory structural assertions.

## Uncertainties and validation requirements

- BR cannot be labeled mathematically faithful until a separate authority
  defines exact versus approximate BR, seat weighting, fold cutoff, objective,
  stopping, and exploitability reporting. This report treats the current code
  only as an approximate probe.
- The multi-player BR cutoff defect is statically determined, but a focused
  three-player CUDA reproduction is still required to record the exact failure
  boundary and memory state.
- Fixed and KKT math wiring was traced statically; focused tests cited in batch
  02 were not rerun here. GPU validation should begin only after `nvidia-smi`
  confirms an idle device and should use tiny one-mode cases.
- No same-workload current run proves learner or end-to-end performance against
  historical artifacts. The target scalar loops, post-hoc tree seal, worst-width
  postflop execution, and fixed-tree closure gap require the regression gates
  specified in batch 02.
- The sibling evaluator authority path named by `AGENTS.md` was absent according
  to batch 06. This pass relied on the checked-in facade behavior and did not
  independently certify evaluator contract v2.

## Files and objects independently traced

Fully traced for cross-mode wiring:

- `pyproject.toml`; `pbg/cli/train.py`; `pbg/cli/br_eval.py`
- `pbg/config.py`; `pbg/runtime.py`; relevant validation in
  `pbg/core/config.py`
- `pbg/core/training.py`: mode dispatch, seal, target/learner handoff, device,
  controller, distributed loop, resume, checkpoint
- `pbg/core/rollout.py`; `pbg/core/branching.py`
- `pbg/core/targets.py`; mode-specific `pbg/core/learner.py`
- `pbg/core/checkpointing.py`; `pbg/core/distributed.py`
- `pbg/games/poker/bundle.py`; relevant leaf/model functions in
  `pbg/games/poker/rollout_backend.py` and model heads
- `pbg/eval/br_probe.py`; relevant BR and CUDA acceptance tests

Batch reports `00_math_and_modes_contract.md` through
`07_model_architecture_performance.md` were read before this independent trace.
