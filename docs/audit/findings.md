# Open numbered findings register

## Reading guide

Severities:

- **Critical:** a requested training mode is absent, a shipped path fails, or a production configuration is structurally infeasible.
- **Major:** material algorithm/model mismatch, correctness risk, or large unavoidable performance/memory loss.
- **Moderate:** incomplete operational support, dead configuration, documentation drift, or a narrower optimization/test gap.

This file contains only unresolved findings. After a recommendation has been
implemented and verified, move its entire entry to
[`resolved_findings.md`](resolved_findings.md). Preserve its original `AUD-*`
identifier, response, and implementation note; do not renumber remaining
findings or reuse identifiers.

`Response:` fields are for the user's decision or clarification and may be
filled before implementation.

Model recommendations are cost-conscious routing judgments, not claims about
published model pricing or guarantees. This register uses the following
working rubric: `5.5` for mechanical documentation/cleanup, `5.6 Luna` for
contained implementation, `5.6 Terra` for cross-module systems work, and
`5.6 Sol` for mathematically or architecturally high-risk work. Thinking
intensity is `low`, `medium`, or `high`.

## A. Training-mode and core-algorithm findings

### AUD-005 — H100 ceilings for exact fused MHA remain unmeasured

**Severity:** Major  
**Modes:** All poker modes  
**Evidence:** The former implementation routed nonzero reach priors through explicit fp32 `[B,H,N,N]` logits/probabilities and retained folded-seat query rows. It now uses mathematically exact compiled FlexAttention when supported, exact checkpointed query-chunk attention otherwise, live-seat postflop compaction, and hardware-profile token budgets. On an idle 2080 Ti, production-width four-layer full-flop forward/backward peaks measured 0.291/0.380/0.642 GiB for 2/3/6 live-player singleton states. Target H100 40/80 GiB batch ceilings have not yet been measured.

**Consequence:** Effective minibatch 128 is supported through accumulated physical batches on the local profile, but H100 throughput and safe per-rank ceilings cannot yet be advertised as production-validated.

**Recommendation:** Retain the exact MHA design. Quest access is available but intentionally deferred until the local readiness gate passes. On the first scheduled Quest validation job, validate one H100 before two/four-GPU jobs, measure 40/80 GiB tiers without trial-by-OOM, and write the resulting per-rank token ceilings into `configs/hardware/quest_slurm.yaml`.

**Model recommendation:** `5.6 Sol` with `high` thinking — selecting an exact scalable attention formulation requires architecture, kernel, autograd, and memory expertise.

**Response:** Implemented without changing the architecture or attention law. Postflop execution now scales with live players, while rare early-training six-way states remain supported as budgeted singletons. Flop/turn/river share postflop queues and may batch together; live-player count and actual padded token cost drive packing.

**Implementation:** Local profile complete and CUDA-validated on the active 2080 Ti server. Quest access is established, but the H100 profile remains deliberately conservative and opt-in; this finding remains open only for readiness-gated Quest measurement/tuning.

---

### AUD-009 — Partition summaries are recomputed and discarded on every postflop forward

**Severity:** Major  
**Modes:** All postflop poker modes  
**Evidence:** `PokerRolloutBackend.policy_forward` calls `_partition_summary` for every postflop queue before checking `evaluator_summary_dim`. The shipped poker model has the default `evaluator_summary_dim=0`, so the result is discarded. During gradient steps the call repeats for every sampled minibatch. There is no cache keyed by street/fold state, despite `MHA_model.md` and `folder_impact.md` requiring refresh only on a street or fold event.

**Consequence:** An evaluator call and large pair-summary allocation are paid even though the model cannot consume them. Enabling summaries repeats deterministic work many times per unchanged node.

**Recommendation:** Do not call the evaluator when the active model has no summary consumer. When enabled, compute and cache a compact model-ready summary at fold/street transitions, attach it to the public/model-input snapshot, and reuse it across rollout siblings/minibatches when its dependency key is unchanged.

**Model recommendation:** `5.6 Terra` with `medium` thinking — the caching rule is clear, but dependency keys and branch-safe lifetimes span several modules.

**Response:** Yes. Folded stats only need to be computed postflop and only when:
- flop/turn/river was just drawn,
- or a new player just folded.
Most spots don't require re-computing it (the coefficients would be identical as the previous node).


---

### AUD-010 — The evaluator/model boundary cannot currently supply the full MHA folder contract

**Severity:** Major  
**Modes:** All postflop poker modes  
**Evidence:** `pbg-v2 partition_many` returns only `base_log_z`, 52 unary log ratios, an optional `52x52` pair correction, and validity. `MHA_model.md` additionally requires per-folder `log F_k`, line/street embeddings, unnormalized card marginals/pressure, signed low-rank beta factors, and optional tight-folder contractions. The tensor backend records no real fold line metadata and uses the current board as every `board_at_fold`; `fold_line_features` is never supplied to the postflop net.

**Consequence:** Full MHA conformance cannot be achieved by changing only the current projection width. Some required data are neither produced by the sibling facade nor preserved locally.

**Recommendation:** Agree a versioned evaluator extension that returns the model-ready folder sufficient statistics while keeping private poker math in the sibling. Preserve public fold-time line metadata locally. Document which returned neutral fields may be decomposed/cached locally and which new fields must be evaluator-owned.

**Model recommendation:** `5.6 Sol` with `high` thinking — this changes a versioned cross-repository API and partitions ownership of mathematically sensitive data.

**Response:** Yes.


---

### AUD-011 — The engine does not expose the full fixed-denomination NLHE action set

**Severity:** Major  
**Modes:** All poker modes  
**Evidence:** The mathematical/poker scope says all legal actions in a fixed chip denomination. The engine uses float stacks and an affine `n_ticks` raise-to grid; the 100bb preset supplies only 32 raise ticks between min-raise and all-in. No chip denomination is configured, and most legal raise-to amounts are unavailable.

**Consequence:** The game being trained is a discretized no-limit abstraction, not full 100bb cash NLHE under the stated scope. Strategy quality and action probabilities depend on the arbitrary 32-point grid.

**Recommendation:** Make the scope decision explicit. Either add an integer chip unit and enumerate every legal raise-to amount, with a continuous/factored head evaluated on the dynamic legal ticks, or rename/document the product as a 32-size abstraction and test convergence as tick resolution changes.

**Model recommendation:** `5.6 Sol` with `high` thinking — continuous sizing touches game semantics, sampling distributions, policy heads, likelihoods, and training correctness.

**Response:** I want continuous bet sizing, meaning NO abstraction. The model should be able to pick any bet sizes, using a parametrized distribution with a few "modes" each having a center, width and height. And we need to sample from this distribution. Ideally we use 100bb with 0.5bb as the smallest units. But I'm also ok using as default 64bb with 0.5bb as the smallest unit so it becomes exactly 128 possible bet sizes (if you think a power of 2 would be better).

---

### AUD-012 — Fixed branching repeats identical model forwards per child

**Severity:** Major  
**Modes:** Branched RB  
**Evidence:** `_sample_branched_labeled` repeats parent states/reaches by offspring count, then calls `policy_forward` on those repeated rows. Every sibling from one player parent therefore recomputes the same policy and value output. The code later selects the first duplicate value for the parent, confirming the duplication is recognized after the expensive call.

**Consequence:** Model compute and activation/workspace scale with child count even though only posterior samples/events/reach updates need to be child-local. This is particularly harmful for the already expensive poker MHA.

**Recommendation:** Evaluate each unique active player parent once; sample child-local profiles; gather actor rows/events from the parent output; compute selected-action log probabilities for all sampled labels; expand only the lightweight reach/event operations. Add a call-count and numerical-parity test.

**Model recommendation:** `5.6 Terra` with `medium` thinking — this is a bounded vectorization refactor with a crisp numerical-parity oracle.

**Response:** Yes.


---

### AUD-013 — The rollout/completion/backward pipeline is not genuinely streamed

**Severity:** Critical  
**Modes:** Sequential RB, branched RB  
**Evidence:** Samplers preallocate the entire capacity. `_one_iteration` gathers every leaf, completes all leaves in one batch, constructs a complete `NodeTargets` table, and retains the frozen model/bundle and completion object while gradient steps run. `branched_value` chunks only its internal `P/M` workspace; all leaf results and final targets remain resident. Its default row cap can still permit roughly 20+ GiB of poker `P/M` workspace because the cap counts player rows, not bytes including 1326 types.

**Consequence:** Peak memory is the sum of multiple phases that need not overlap. Backward “streaming” does not prevent OOM at realistic tree/leaf counts.

**Recommendation:** Pipeline whole-tree chunks through rollout -> leaf completion -> backward -> compact target append, release transient state, then continue. Budget bytes, not abstract rows. Spill compact targets to host/pinned memory only if measured transfer cost is favorable. Release the frozen model and completion diagnostics before optimization.

**Model recommendation:** `5.6 Sol` with `high` thinking — pipeline ordering, target laws, object lifetimes, and GPU peak memory must all remain correct together.

**Response:** I think I agree? I want you to explain this more though. Also note that the "completion" data can be discarded after we do the backward pass: only the estimators (such as V^, A^, P^) are needed for minibatches.


---

### AUD-014 — Completion retains unused large diagnostics and ignores quality signals

**Severity:** Major  
**Modes:** Sequential RB, branched RB  
**Evidence:** Poker completion retains `equity_share[B,P,1326,K]`, `output_valid`, `estimator_stddev`, `sample_count`, `vvec_hat`, `u_hat`, and `n_eff`. The active `branched_value/rb_ppo` path consumes only `p_hat` and `v_hat`. For six players, `equity_share` alone is about 191 kB (186.5 KiB) per leaf. Conversely, `n_eff`, validity, and estimator error are not used to reject or adapt noisy targets.

**Consequence:** Memory is spent on dead fields while target quality is not governed. Many leaves can turn this into gigabytes.

**Recommendation:** Add a minimal completion-return mode for active consumers, or immediately detach only required learning fields and release the sibling result. If quality fields are retained, define thresholds, logging, invalid-row policy, and optional adaptive sampling rather than silently ignoring them.

**Model recommendation:** `5.6 Terra` with `medium` thinking — DTO minimization is contained, while quality-policy choices require careful evaluator integration.

**Response:** Yes.


---

### AUD-015 — Folded and live reach lifetimes are not mode-specific

**Severity:** Major  
**Modes:** All, especially normal  
**Evidence:** Both samplers maintain a dense `[node,P,H]` reach arena and `ReachCache` exposes whole rows. Folded seats stay in the same full-width representation. There is no transition from a full fold-time range to the compact statistics required by the model, nor any release after normal terminal rollout or RB leaf completion.

**Consequence:** The optimization explicitly requested by the user—discarding folded ranges after their last consumer—is absent. Normal mode would inherit RB memory unless a specialized lifecycle is designed.

**Recommendation:** Implement the lifetimes in `mode_storage_compute.md`: retain folded full ranges only through future summary refreshes and, for RB, leaf completion; persist model-ready folded summaries per decision; then free full folded rows. Treat normal, one-path RB, and tree RB as different compact target layouts.

**Model recommendation:** `5.6 Sol` with `high` thinking — mode-specific reach ownership and release points interact with completion, replay, summaries, and branching independence.

**Response:** We do need full folded ranges to be fed to the "completion" equity/posterior API (which isn't used in normal mode). But otherwise, they are only used through the folded stats. Make sure this is efficiently handled.


---

## B. Game-family, theory, operations, and validation findings

### AUD-017 — The synthetic dial is not an interpolation to the target poker game

**Severity:** Major  
**Modes:** All synthetic modes  
**Evidence:** `dial(public_scale, private_scale, n_players)` creates a capped limit game with only fold/check-call/one fixed raise and sets `board_per_round=(0,...,0)`. Increasing public scale adds betting rounds and raise caps but no public chance. Increasing private scale changes hand/deck size, while showdown strength remains the simple “paired with board, then max private rank” rule. It never approaches dynamic no-limit sizing, four Hold'em streets/board structure, side pots, or Hold'em hand ranking.

**Consequence:** The dial can stress generic dimensions but cannot support scientific claims about a spectrum between Kuhn, Leduc, and 6-max NLHE. Results across the endpoints are not controlled changes of one game family.

**Recommendation:** Define the intended interpolation axes explicitly: players, private cards/types, public board schedule, betting rounds, raise-size resolution, stack depth/all-ins/side pots, and showdown evaluator. Provide named points that exactly reproduce Kuhn and Leduc and a sequence that approaches the chosen NLHE abstraction. Add invariant/property tests across the grid.

**Model recommendation:** `5.6 Sol` with `high` thinking — designing a scientifically meaningful two-axis family requires strong game abstraction and experimental-design judgment.

**Response:** Ok yes this is problematic. But I also don't want 20 different dial axis. I think we should only have two axis:
- private space and board: increase deck size
- action space: increase bet options (tree width) and number of streets (tree depth).
Probably we can keep the same poker rules throughout. We can brainstorm this more. The design should give only 2 axis (designed to showcase the strenght of rao-blackwellization and of branching) and should allow us to maximally reuse the same implementation.

---

### AUD-018 — The shipped dial example aliases distinct public histories in the model

**Severity:** Major  
**Modes:** Synthetic  
**Evidence:** `SyntheticNet` embeds `public_key % n_buckets`. The shipped dial has 10,826 enumerated public states but only 1,024 buckets, so distinct perfect-recall public histories necessarily share embeddings. There is no collision check or collision-free encoding. Kuhn (9/128) and Leduc (465/512) happen to fit.

**Consequence:** The dial policy cannot represent an arbitrary behavioral strategy on its own public tree and may conflate different legal/history contexts. It is not a faithful PBG policy parameterization at the shipped setting.

**Recommendation:** Use collision-free state embeddings for enumerated synthetic games, dynamically size the table after tree enumeration, or use a lossless structured/history encoder. Reject configurations whose hash space is smaller than the reachable public-state set if exact information-state representation is intended.

**Model recommendation:** `5.6 Luna` with `medium` thinking — this is a localized representation/configuration fix with straightforward collision tests.

**Response:**


---

### AUD-019 — Kuhn/Leduc/dial do not use the required MHA poker design

**Severity:** Major if “each version of poker” includes these games; otherwise Moderate clarification  
**Modes:** Synthetic poker variants  
**Evidence:** All synthetic games use `SyntheticNet`, a small public/type embedding MLP. None uses the MHA hand/class rows, reach priors, action anchors, incidence residual, factored continuous raise head, or folded-summary design.

**Consequence:** Model comparisons across small poker and full poker confound game complexity with a completely different architecture. The user's requirement that `docs/MHA_model.md` be used “for each versions of poker” is not met under the natural reading.

**Recommendation:** Decide scope. If binding, create scaled MHA adapters for Kuhn/Leduc/dial with the same architectural concepts and game-appropriate row/action heads. Retain `SyntheticNet` only as an explicit baseline. If the requirement applies only to Texas Hold'em variants, state that and add matching Hold'em spectrum points.

**Model recommendation:** `5.6 Terra` with `high` thinking — scope clarification is simple, but the binding branch becomes a cross-game architecture and comparability project.

**Response:**


---

### AUD-020 — Synthetic `exact` completion is actually sampled MC

**Severity:** Moderate  
**Modes:** Sequential RB, branched RB synthetic  
**Evidence:** `SyntheticCompletion.complete` always samples `n_samples` posterior profiles, bins occurrences, and averages rewards in Python loops. `completion.backend: exact` selects this same implementation; the exact enumerators in `games/synthetic/exact.py` are not used in production target construction.

**Consequence:** Preset names and documentation overstate exactness, and small-game estimator tests cannot cleanly distinguish completion MC variance from rollout/branching variance.

**Recommendation:** Either implement an exact completion backend under feasibility guards and retain `mc` separately, or rename the current backend. Use exact Kuhn/Leduc completion as an oracle for all three training-mode target implementations.

**Model recommendation:** `5.6 Terra` with `medium` thinking — finite-game enumeration is bounded and testable but must integrate correctly with all three training modes.

**Response:**


---

### AUD-021 — `core_math.tex` lambda-return displays are off by one relative to code

**Severity:** Major specification defect, implementation currently follows the corrected companion  
**Modes:** Normal, sequential RB  
**Evidence:** Equations `sequential-lambda-return` and `rao-blackwell-lambda-return` use terminal exponent `|Tau ∖ sigma|_p`, which includes the current decision under the document's depth convention. The correct player-clock exponent is the number of downstream decisions by the same player, `|Tau|_p - |sigma|_p - 1`. `NOTATION.md` explicitly normalizes this typo, and the backward implementation applies one lambda mix at each downstream owner decision, which is the corrected behavior.

**Consequence:** A future direct transcription of the authoritative TeX would disagree with the current code and standard player-clock GAE, especially visible at the last decision where the terminal reward should have exponent zero.

**Recommendation:** Correct the equations in `core_math.tex` rather than relying on a companion normalization. Add closed-form trajectory tests for zero, one, and multiple downstream same-player decisions at lambda 0, intermediate lambda, and 1.

**Model recommendation:** `5.6 Sol` with `high` thinking — an authoritative mathematical correction should be derived against code and closed-form edge cases, not patched mechanically.

**Response:**


---

### AUD-022 — Global policy-support assumptions remain unresolved

**Severity:** Major theoretical/robustness issue  
**Modes:** All  
**Evidence:** The existing `core_math_audit.md` already records missing full-support/absolute-continuity hypotheses. Code partly enforces numerical support by clamping probabilities before logs (`eps=1e-12` poker, finite softmax synthetic), but the mathematical strategy space permits zeros and PPO divides current by cached rollout probability. Chance/posterior conditionals also depend on reachable positive-mass events.

**Consequence:** Theorems and loss formulas are not globally well-defined as written, while implementation clamps silently alter exact zero-support semantics.

**Recommendation:** Add one binding support convention to the TeX/contracts: full-support parameterized legal-action policies with a documented probability floor, plus positive-reach restrictions for conditionals. Test that stored actions always have finite old log probability and document the bias introduced by any clamp.

**Model recommendation:** `5.6 Sol` with `high` thinking — this establishes global mathematical assumptions and must reconcile exact theory with numerical implementation bias.

**Response:**


---

### AUD-023 — Configuration exposes dead or misleading controls

**Severity:** Moderate  
**Modes:** All  
**Evidence:** `use_suit_consistency` is parsed but never used. `debug_asserts` is parsed but has no production consumers. `clip_eps_v` exists and `l_v_clip` is implemented, but `rb_ppo` always uses unclipped `l_v`. Synthetic `model.e_max` in YAML is silently ignored. Game/model mappings are not strict at top-level load despite documentation describing strict configuration.

**Consequence:** Users can believe an optimization/regularizer is enabled when it is not, and misspelled/obsolete game-model keys can survive until late or be ignored.

**Recommendation:** Remove deferred fields until wired, or make them functional and report them in metrics/checkpoints. Give each game factory a strict typed schema. Add a test that toggling every accepted boolean/enum either changes a traceable behavior or is rejected.

**Model recommendation:** `5.6 Luna` with `medium` thinking — mostly mechanical schema cleanup and toggle tests, with limited architectural judgment.

**Response:**


---

### AUD-024 — Branch and memory budgets are not preflighted

**Severity:** Major  
**Modes:** Branched RB  
**Evidence:** `node_budget` is a hard per-tree capacity multiplier, and overflow is detected only when appending children. There is no expected-node calculation from beta/horizon, high-quantile safety estimate, free-memory query, completion-leaf estimate, or staged dry run. The implementation correctly refuses to truncate, but that only turns budget mismatch into a late failed iteration.

**Consequence:** Fixed beta can explode the tree or allocate massive unused headroom. On a shared GPU server, late capacity failure wastes work and increases OOM risk.

**Recommendation:** Add an analytical and empirical planner: estimate offspring-by-stage tree size, leaf count, target bytes, and model workspace; choose chunk/tree counts under a configured memory fraction; run a tiny calibration before scale-up; log realized branching and peak memory. Keep loud overflow as a final invariant.

**Model recommendation:** `5.6 Sol` with `high` thinking — safe preflight planning combines stochastic tree growth, byte accounting, calibration, and shared-GPU safeguards.

**Response:**


---

### AUD-025 — Frozen rollout/model and transient objects live through optimization

**Severity:** Moderate  
**Modes:** All current modes  
**Evidence:** `_one_iteration` deep-copies the full model, places it in `frozen_bundle`, builds completion/targets, and then enters gradient steps without deleting the frozen model/bundle or completion object. Python references remain live until the function returns.

**Consequence:** Model parameters/buffers and large transient evaluator fields overlap with train-time activations and optimizer state. This materially raises peak memory for poker.

**Recommendation:** Snapshot only the rollout quantities required by PPO, explicitly release frozen model/bundle/completion/leaf caches before training, and synchronize/empty allocator caches only when measurement shows value. Consider maintaining two reusable model buffers rather than deep-copy allocation every iteration.

**Model recommendation:** `5.6 Terra` with `medium` thinking — the lifetime refactor is bounded but requires careful reference and allocator verification.

**Response:**


---

### AUD-026 — Device-resident hot paths contain avoidable host synchronization

**Severity:** Moderate  
**Modes:** All GPU modes  
**Evidence:** `AliasTable.build` copies weights to CPU and builds via Python lists in float64. `build_compact_map` uses `.item()` and a Python loop per batch row on every postflop forward. `MHA.forward` executes `count_nonzero(...).item()` every layer to choose a path even though reach bias is known to be present. Poker adapters and routing use multiple `bool(tensor.any())`/scalar conversions.

**Consequence:** GPU pipelines repeatedly stall, and the “device residency” tests miss synchronization that does not spell `.cpu()` in the few scanned functions.

**Recommendation:** Use device-native categorical sampling/alias construction, vectorized fixed-per-street compact maps, an explicit biased-attention call path, and tensorized validation outside the steady-state hot path. Profile synchronization counts with CUDA tooling and add a no-host-sync benchmark gate for model/minibatch execution.

**Model recommendation:** `5.6 Terra` with `high` thinking — several CUDA hot paths need profiling-led rewrites without changing sampling or model semantics.

**Response:**


---

### AUD-027 — Global deterministic mode conflicts with maximum throughput

**Severity:** Moderate  
**Modes:** All GPU modes  
**Evidence:** `train` unconditionally enables `torch.use_deterministic_algorithms(True)` and deterministic cuDNN behavior. There is no reproducibility/performance configuration. Memory-efficient attention and some optimized kernels may be unavailable or slower under this global setting.

**Consequence:** The repository cannot claim “no optimization left” while forcing the strictest deterministic mode for every serious run, especially when the model already needs memory-efficient kernels.

**Recommendation:** Add explicit `reproducibility: strict | seeded | throughput` profiles. Keep exact seeds/checkpoint RNG in every profile, document backend-level reproducibility, and benchmark/validate statistical equivalence for the throughput profile.

**Model recommendation:** `5.6 Luna` with `medium` thinking — the configuration and backend toggles are contained and can be guarded by reproducibility tests.

**Response:**


---

### AUD-028 — Training cannot resume as an end-to-end workflow

**Severity:** Moderate  
**Modes:** All  
**Evidence:** Checkpoints contain model, optimizer, scaler, iteration, config, and RNG, and `load_checkpoint` can restore them. `train` has no resume argument/path, always creates a new optimizer, starts the iteration loop at zero, and appends to the target metrics file without run identity checks.

**Consequence:** Long full-poker training cannot reliably restart after interruption using the public training API. Manual reconstruction risks iteration/schedule/RNG mistakes.

**Recommendation:** Add a resume entry point that validates config compatibility, restores all state, starts from the stored iteration, preserves schedules, and safely handles metrics/checkpoint naming. Test bitwise continuation in strict mode.

**Model recommendation:** `5.6 Luna` with `medium` thinking — checkpoint primitives already exist, making this a contained workflow and validation task.

**Response:**


---

### AUD-029 — Existing tests do not gate the requested capabilities

**Severity:** Major  
**Modes:** All  
**Evidence:** The suite passes 124 tests with 4 skips, but no test invokes normal training because it does not exist. The 3-iteration real poker smoke uses 2 players, 2bb, 2 ticks, tiny one-layer models, sequential RB, `reach_storage: full`, and batch size 2; it does not exercise the shipped recompute path or realistic postflop attention. The H100 preset test only loads fields. MHA tests are shape/symmetry primitives, not architectural conformance. No mode-to-exact-gradient comparison exists.

**Consequence:** Green CI is compatible with missing modes, a crashing shipped storage mode, an infeasible preset, and a nonconforming model.

**Recommendation:** Implement the gates in `validation_matrix.md`, especially: all three modes on exact Kuhn/Leduc; beta-one equivalence; poker chance replay; full public-history distinguishability; folder no-op/known-effect tests; engine differential fuzzing; and measured memory/throughput ceilings.

**Model recommendation:** `5.6 Terra` with `high` thinking — the work is test-focused but spans every algorithmic subsystem and must distinguish real gates from superficial coverage.

**Response:**


---

 

---

### AUD-032 — There are no comparative performance gates for the three modes

**Severity:** Major relative to the “all modes optimally” requirement  
**Modes:** All  
**Evidence:** No benchmark records decisions/second, terminal or completion samples/second, peak allocated/reserved bytes, model/evaluator time, target bytes per decision, or effective samples per unit cost by mode. Batch histograms are retained but no throughput thresholds exist. Normal mode is unavailable, preventing comparison entirely.

**Consequence:** Regressions that make normal mode pay RB costs, or sequential RB pay branching costs, would not be detected. “Modularity” has no enforceable efficiency contract.

**Recommendation:** Add stable microbenchmarks and end-to-end profiles on Kuhn, Leduc, a defined dial grid, and tiny/medium NLHE. Report phase timings and peak bytes, normalize by decisions and effective completion samples, and set regression tolerances per mode/device. Require clean GPU state before/after benchmark jobs.

**Model recommendation:** `5.6 Terra` with `high` thinking — useful gates require careful metric design, workload stabilization, GPU profiling, and non-flaky tolerances.

**Response:** Well, we're going to have to use our judgement on what "optimality" means. Only "normal" mode exists in the literature and could be benchmarked, but also it should be easy for you to look at it and figure out if the code has big bottlenecks or not. I say we will just try to improve against our own times (do round of optimization where we benchmark a rollout, then try to tweak the code to make it faster). 

---
