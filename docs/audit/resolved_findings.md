# Resolved findings register

This file preserves audit findings whose recommended changes have been
implemented and verified. Entries retain their original `AUD-*` identifiers,
evidence, user responses, and implementation notes; identifiers are never
renumbered or reused.

## A. Training-mode and core-algorithm findings

### AUD-003 — Poker `reach_storage: recompute` fails after public chance

**Severity:** Critical  
**Modes:** Sequential RB, branched RB  
**Evidence:** `ReachCache.rows_for` replays chance edges through `_replay_reach_update`, passing a plain event tensor and a dummy policy tensor. `PokerRolloutBackend.reach_update` requires an `EventBatch` and interprets non-`None` policy output as a player action. A targeted 2-player CPU rollout reached six postflop decisions and failed with `AttributeError: 'Tensor' object has no attribute 'public_id'`. The shipped `nlhe6_h100.yaml` selects `recompute`.

The problem is deeper than the immediate type error: `TreeBatch` stores only the integer public chance label, not the `cards` payload required to reproduce board compatibility. `_replay_reach_update` also sizes a dummy event dimension from the large chance label (offsets start at 100,000), which could itself allocate an enormous tensor.

**Consequence:** The advertised poker preset cannot train postflop. A superficial wrapper fix would still reconstruct the wrong reach law unless chance payloads or an exact decoder are retained.

**Recommendation:** Redesign replay around game-owned edge payload storage or a game-owned `replay_reach(parent_state, child_state, public_id, cached_logp)` API. Add exact parity tests for roots, action edges, flop, turn, river, folds, and duplicate branched children. Until fixed, reject poker `recompute` at config load and use a memory-feasible checkpoint/snapshot mode.

**Model recommendation:** `5.6 Terra` with `high` thinking — exact replay semantics cross tree storage, poker chance payloads, reach updates, and parity testing.

**Response:** Ok yea if you can keep the changes minimal.

**Implementation:** Completed with no poker-specific edge payload added to
`TreeBatch`. Generic chance replay now calls the typed `EventBatch` backend
boundary without synthesizing an event-sized policy tensor. Poker supplies
`replay_chance_reach`, which uses the child state’s persisted board to apply
the same hole-card compatibility mask as rollout; reapplying already-known
board cards is idempotent. This removes both the type mismatch and the
large-label allocation hazard.

**Verification:** CPU parity tests compare `full` and `recompute` reaches at
every node through action edges, flop, turn, river, and a fold. They pass for
both sequential RB and fixed branching, including duplicate branched children:
`3 passed in 1.69s` for the targeted tests. This closes recompute correctness;
the later `AUD-004` work removes the avoidable all-node arena allocation.

---

### AUD-004 — The shipped H100 preset is structurally out of memory

**Severity:** Critical  
**Modes:** Sequential RB, branched RB; poker  
**Evidence:** The preset sets `n_trees=256`, `node_budget=8192`, so capacity is 2,097,152 states. Both samplers allocate `reach_rows[capacity,6,1326]` even for `recompute`: 66,739,765,248 bytes (62.16 GiB). `PokerTensorStateStore` computes `max_rollout_steps=2451` and allocates `action_history[capacity,2451]`: 20,560,478,208 bytes (19.15 GiB). These two tensors alone total 81.30 GiB, before tree buffers, the remaining state tensors, two model copies, completion, targets, optimizer state, or activations.

**Consequence:** It cannot fit an “80 GB” H100 and will generally fail during bundle construction or rollout allocation. The configuration comment describing it as conservative is incorrect.

**Recommendation:** Replace per-iteration worst-case preallocation with pooled/chunked whole-tree scheduling and allocate reach storage according to the selected storage mode. Store action history parent-linked or in a packed event arena rather than copying the maximum history into every state. Add byte-level peak-memory estimation against free memory before allocation.

**Model recommendation:** `5.6 Sol` with `high` thinking — this is a foundational GPU-memory and scheduling redesign with severe OOM consequences.

**Response:** Sounds good: we need to be very careful to avoid OOM.

**Implementation:** Completed for capacity-sized rollout allocations.
`recompute` no longer allocates a `[capacity,P,H]` reach arena in either
sampler; it retains roots and uses wave-local reach tensors. Branched rollout
also accumulates only realized policy-edge and policy-node cache rows instead
of reserving capacity-sized likelihood/value tables. `PokerTensorStateStore`
now represents public player history as one parent handle plus one event per
successor, eliminating the copied `action_history[capacity,max_steps]` tensor.
The shipped preset is one sampled branching tree per rank with a 4,096-node
hard ceiling, preserving the former total per-rank capacity without root
replicas.

`estimate_rollout_memory` provides a conservative byte plan (two full reach
frontiers plus state/tree/cache bounds), and CUDA bundle construction rejects a
plan exceeding half of currently free device memory before allocating the
backend. The revised preset's rollout plan is 306,149,260 bytes (about
291 MiB) on CPU construction.

**Verification:** Static tests bound the legacy 2,097,152-node recompute
frontier above 100 GiB and legacy full storage above 60 GiB, while the new
preset plan is below 1 GiB. Parent/event links are checked against random
legal engine trajectories. CPU construction of the preset succeeds, the full
suite passes (`133 passed, 4 skipped`), and a tiny explicitly targeted CUDA
smoke passed on idle GPU 1; GPU 1 returned to 9 MiB and 0% utilization.
`AUD-005` and later model allocations remain separate limits; `AUD-013` now
owns the completion/backward byte plan. This does not claim that the preset is
a production-scale H100 run.

---

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

### AUD-006 — The poker policy does not receive the full public history

**Severity:** Major  
**Modes:** All poker modes  
**Evidence:** `PokerTensorStateStore` retains parent-linked `history_parent`, `history_event`, and `history_length`, but `encode_public` does not expose the reconstructed history. `PokerPublicBatch` contains street, board, pot, stacks, total commitments, actor, a coarse legality summary, tick count/bounds, status, seats, and button. It omits exact action sequence, bet sizes/actions by street, current street commitments, call amount, and explicit min-raise/current-bet state. The preflop/postflop networks therefore cannot tokenize the public trajectory required by `docs/MHA_model.md`.

**Discrepancy:** The PBG information state is `(x_p, sigma)` with full public trajectory. The MHA input contract explicitly includes public history and exact public arithmetic.

**Consequence:** Histories with strategically different signaling can alias at the model input. Belief ranges indirectly encode old-policy actions, but they are not a substitute for the public information state and create undesirable policy-dependent input semantics.

**Recommendation:** Define and test a packed, lossless public action-history representation with street separators, action type, exact raise-to/chip amount, acting seat, and legality arithmetic. Feed it as public tokens. Remove the current per-state full-history copy when the packed parent-linked representation is in place.

**Model recommendation:** `5.6 Sol` with `high` thinking — the representation becomes a binding information-state and storage contract across engine, rollout, and model code.

**Response:** That is indeed a big issue. We should fix it. I just don't understand what you mean by "Remove the current per-state full-history copy when the packed parent-linked representation is in place."

**Implementation:** Completed. The previously introduced parent/event chain is
now lossless: each successor stores the acting seat, street, action kind, exact
call or raise-to amount, and the pre-action pot/call/current-bet/min/max-raise
arithmetic once. Chance successors add explicit street separators. Both tensor
and reference-engine adapters reconstruct a chronological `PackedPublicHistory`
only for the requested model batch. `PokerPublicBatch` also carries current
street commitments, exact call/current/min/max/last-raise values, the complete
legality mask, and every raise tick. Separate preflop and postflop encoders turn
these fields and history positions into public sequence tokens.

**Verification:** A scripted raise/call/flop trajectory checks every reconstructed
history field and separator. A model test holds all other inputs fixed and
confirms that changing the historical action/amount changes output. The full CPU
suite passes (`139 passed, 6 skipped`), and the targeted CUDA model/adapter smoke
passes on explicitly selected idle GPU 1 (`4 passed`).

---

### AUD-007 — Most binding MHA architecture components are missing

**Severity:** Major  
**Modes:** All poker modes  
**Evidence:** The networks implement class/combo embeddings, dense pre-LN attention, action anchors, reach key bias, factored action heads, actor values, and optional public moments. They do not implement the specified incidence residual (`O_card`, identical-combo `O_=`, own-range channel), preflop orbit-averaged card geometry/overlap scalars, postflop two-hot and signed-beta geometry, bilinear side units, per-seat nonactor/folder readout prescribed in the document, folder pressure channel, or tight-folder terms. `model/incidence.py` contains only compact gather/scatter/permutation helpers and is not used by either network as an incidence residual.

**Consequence:** Passing shape tests does not establish conformance to `docs/MHA_model.md`; the learned function class lacks exact paths the design relies on for blockers and identical-combo effects.

**Recommendation:** Turn the MHA document into an executable conformance checklist with tensor-level tests for each path. Implement in stages: public history; incidence residual; preflop geometry; folder-corrected reach/signed beta coordinates; seat/folder summaries; bilinear units; then optional order-3/public moments.

**Model recommendation:** `5.6 Sol` with `high` thinking — this is a large theory-to-architecture implementation whose stages must remain mutually consistent.

**Response:** Yes.

**Implementation:** Completed for the binding trunk/readout paths. Every layer
now runs the exact unnormalized card accumulator with separate card-overlap,
identical-row, and own-range channels. Preflop rows receive exact orbit-averaged
52-card geometry, class sizes, and within/cross-class overlap scalars. Postflop
rows receive two-hot card geometry, evaluator-derived folder-corrected log reach,
and signed positive/negative beta coordinates. Layer FFNs consume the incidence
residual, card/equality products, beta-coordinate squares, folded pressure, and
learned tight-folder delta products. Readout uses action-anchor actor summaries,
reach-weighted live-seat summaries, folder summaries, button-relative ordering,
and the prescribed global feature. Per-folder finite-difference calls stay
behind the evaluator boundary and supply log mass, card marginals, line context,
pressure, and folder-seat metadata without local private-assignment math.

**Verification:** Tensor tests compare every incidence channel against a brute-
force small-row calculation, verify orbit mass/overlap identities, reconstruct
the evaluator beta matrix from signed coordinates, and show that folder pressure
changes output. Existing compact scatter, class-permutation, autocast, and
all-parameter gradient tests remain green. The full CPU and targeted CUDA results
are the same as recorded under `AUD-006`.

---

### AUD-008 — Folded-summary semantics do not match `folder_impact.md`

**Severity:** Major
**Modes:** All postflop poker modes
**Evidence:** `_partition_summary` sorts live and folded ranges and `PokerGPUPartitionAdapter.summarize` concatenates both before calling neutral `partition_many`. `folder_impact.md` defines `Phi` as the folded subsystem integrated out; with no folders, its correction should be zero. The current call with no folders still partitions all live-player ranges and returns nonzero collision finite differences. When model summaries are enabled, those fields are projected generically rather than converted into per-hand `Lambda_B`, signed beta factors, or folder pressure.

**Consequence:** The enabled feature would represent a different partition law from the claimed folded-player effective potential. It may double-count live-player collision geometry already intended for attention/incidence.

**Recommendation:** Specify exactly which evaluator rows define the folder subsystem. For the current theory, call the neutral partition API on folded rows only, derive allowed transformations of returned unary/pair fields, and separately handle live-range incidence. Add the no-folder zero-correction invariant and small exact folder-oracle tests.

**Model recommendation:** `5.6 Sol` with `high` thinking — subtle probabilistic semantics and double-counting risks make this unsuitable for a cheap mechanical pass.

**Response:** Yes.

**Implementation:** Completed. The adapter sends only frozen folded range rows
to the neutral partition facade; live ranges are no longer part of that law.
With no folders it skips the facade and returns an exactly neutral valid
summary. Combined folded finite differences produce per-hand `Lambda_B` and
signed beta geometry, while separate one-folder facade calls produce model-ready
per-folder mass/marginal fields. Live-player interaction remains solely in the
incidence residual, preventing double counting. Real fold-time board and
eight-field public line metadata are persisted on the folding successor and
carried with the frozen range.

**Verification:** Adapter mapping tests assert folded-only row shapes and masks,
the no-folder invariant asserts zero unary/pair corrections and zero evaluator
calls, and model tests cover `Lambda_B`, signed beta reconstruction, pressure,
and folder readout. The real sibling CPU smoke and the full suite pass; the
targeted CUDA DTO/model smoke also passes with clean GPU teardown.

---

### AUD-001 — Ordinary terminal-trajectory training does not exist

**Severity:** Critical  
**Modes:** Normal  
**Evidence:** `RootConfig` has no training-estimator mode. `sample_trees` only dispatches on `rollout.beta_mode` to sequential pre-terminal rollout or fixed branching. `_one_iteration` always calls the completion backend, always calls `branched_value`, and the only registered composite is `rb_ppo`. Poker rollout classifies showdowns as `PRETERM`, not terminal, and has no sampled-profile terminal showdown continuation in the training path.

**Discrepancy:** `SampleTerminal` and `SequentialPG` in `core_math.tex` require a full sampled trajectory, a sampled private profile in the loss, and direct terminal reward. The code's `beta_mode: sequential` implements `SamplePreTerminal` plus private-state Rao--Blackwellization instead.

**Consequence:** The repository cannot run the first of the requested three modes. It cannot measure the variance/throughput trade-off against RB, and it necessarily pays completion and all-type target costs even when the user requests sampled terminal rewards.

**Recommendation:** Add an explicit `training.mode: normal | rao_blackwell | branched_rao_blackwell`. Implement a specialized terminal sampler and flat scalar decision dataset. For poker, continue the sampled profile through all remaining board cards and settle the sampled showdown through `evaluate_showdowns`; do not invoke `complete` or allocate posterior tensors.

**Response:** Yes, we should do this: I want to be able to compare all three modes to see which is best in experiments. Ideally, we keep the code modular and don't make it explode in size just because we added this normal mode. Try to implement this with minimal modifications to the codebase, while still being very efficient.

**Implementation:** Completed. `training.mode` now selects `normal`,
`rao_blackwell`, or `branched_rao_blackwell`. Normal mode reuses the sequential
frontier sampler but emits a flat scalar decision dataset, bypasses completion,
samples missing poker runout cards, and settles showdowns through
`evaluate_showdowns`.

---

### AUD-002 — Branching choice is being used as a proxy for estimator choice

**Severity:** Major  
**Modes:** All  
**Evidence:** `beta_mode` accepts only `sequential` and `fixed`; loss accepts only `rb_ppo`. Completion budget is mandatory in every YAML. Data structures are always `TreeBatch`, `BatchedCompletionStats`, and posterior-valued `NodeTargets`.

**Discrepancy:** “Sequential versus fixed branching” and “sampled-private versus Rao--Blackwell target” are orthogonal choices in the mathematics. The requested product is three deliberately supported modes, with a future possibility of other losses and variable beta.

**Consequence:** Adding normal mode as another `beta_mode` would entrench the wrong abstraction and continue forcing maximal storage on simpler algorithms.

**Recommendation:** Separate configuration axes: estimator/training mode, public branching mode/intensity, completion policy, and objective. Validate legal combinations. Dispatch early to specialized rollout/target/minibatch implementations behind a small shared training interface.

**Response:** I think I agree? I'm just not sure I understand what these words mean: "estimator/training mode, public branching mode/intensity, completion policy, and objective". Explain them one by one, then we can decide to implement. Seems to me like we want as options:

- training.mode: normal | rao_blackwell | branched_rao_blackwell
- choice of loss function
- beta (only in branched)

then obviously picking poker variants.

**Implementation:** Completed. `rollout.beta` is legal only for branched RB;
normal requires a null completion sample budget; and registered losses declare
the training modes they support. Training dispatches before completion/target
allocation while sharing game navigation, policy evaluation, and optimizer
code.

---

### AUD-009 — Partition summaries are recomputed and discarded on every postflop forward

**Severity:** Major
**Modes:** All postflop poker modes
**Evidence:** `PokerRolloutBackend.policy_forward` called `_partition_summary` for every postflop queue before checking whether the model consumed it, and there was no cache keyed by street/fold state.

**Consequence:** An evaluator call and large pair-summary allocation were paid even when unusable, and deterministic work repeated across unchanged siblings/minibatches.

**Recommendation:** Skip the evaluator for models with no summary consumer. Otherwise compute and cache a compact model-ready summary at fold/street transitions and reuse it while its dependency key is unchanged.

**Model recommendation:** `5.6 Terra` with `medium` thinking — the caching rule is clear, but dependency keys and branch-safe lifetimes span several modules.

**Response:** Yes. Folded stats only need to be computed postflop and only when:

- flop/turn/river was just drawn,
- or a new player just folded.

Most spots don't require re-computing it (the coefficients would be identical as the previous node).

**Implementation:** Completed. State rows now carry a four-byte partition-cache
anchor. Ordinary actions inherit it; chance deals and folds replace it with the
successor row. The backend batches dependency-distinct misses, stores a compact
model-ready `PartitionSummaryBatch`, and reuses it across siblings and later
forwards. `PostflopNet` explicitly declares its consumer contract; models
without a consumer bypass the evaluator. Regression tests cover reuse, fold
invalidation, and the no-consumer fast path.

---

### AUD-010 — The evaluator/model boundary cannot currently supply the full MHA folder contract

**Severity:** Major
**Modes:** All postflop poker modes
**Evidence:** `pbg-v2 partition_many` returned only combined base/unary/pair fields, while the MHA contract also required per-folder mass/pressure, signed beta factors, tight contractions, and real fold-line metadata.

**Consequence:** Full MHA conformance could not be achieved by changing only the local projection width; some required data were absent or locally fabricated.

**Recommendation:** Add a versioned evaluator extension returning model-ready folder sufficient statistics while keeping private poker math in the sibling, and preserve public fold-time metadata locally.

**Model recommendation:** `5.6 Sol` with `high` thinking — this changes a versioned cross-repository API and partitions ownership of mathematically sensitive data.

**Response:** Yes.

**Implementation:** Completed across both repositories. The sibling evaluator
now exposes `partition_folders` with capability version `pbg-folder-v1`,
returning combined and per-folder finite differences, unnormalized card
marginals, deltas/covariances, tight-cluster contraction, and signed beta
factors in one public call. Local code validates the rich DTO, retains only the
compact model-facing fields, and attaches the locally preserved fold board,
line, and seat metadata. The postflop model consumes evaluator-owned beta,
pressure, delta, and tight-contraction fields without recomputing poker math.

---

### AUD-011 — The engine does not expose the full fixed-denomination NLHE action set

**Severity:** Major
**Modes:** All poker modes
**Evidence:** The engine used an affine `n_ticks` grid; the 100bb preset exposed only 32 raise sizes and configured no chip denomination.

**Consequence:** Training used a discretized sizing abstraction rather than full fixed-denomination NLHE.

**Recommendation:** Add an integer chip unit and enumerate every legal raise-to amount, using the factored continuous head on the dynamic legal ticks.

**Model recommendation:** `5.6 Sol` with `high` thinking — continuous sizing touches game semantics, sampling distributions, policy heads, likelihoods, and training correctness.

**Response:** I want continuous bet sizing, meaning NO abstraction. The model should be able to pick any bet sizes, using a parametrized distribution with a few "modes" each having a center, width and height. And we need to sample from this distribution. Ideally we use 100bb with 0.5bb as the smallest units. But I'm also ok using as default 64bb with 0.5bb as the smallest unit so it becomes exactly 128 possible bet sizes (if you think a power of 2 would be better).

**Implementation:** Completed with the preferred 100bb/0.5bb default. Both
public engines enumerate all 200 global raise-to denominations and mask the
exact state-legal interval, including short all-ins; configuration rejects
stacks or blinds outside the denomination. The logistic-mixture head is
renormalized over legal denominations, the all-in atom targets the current
maximum, and rollout samples from that exact law. Production configs no longer
contain `n_ticks: 32`. Tests cover complete enumeration, engine parity, dynamic
all-in placement, sampling probabilities, and invalid denominations.

---

### AUD-012 — Fixed branching repeats identical model forwards per child

**Severity:** Major
**Modes:** Branched RB
**Evidence:** `_sample_branched_labeled` repeats parent states/reaches by offspring count, then calls `policy_forward` on those repeated rows. Every sibling from one player parent therefore recomputes the same policy and value output. The code later selects the first duplicate value for the parent, confirming the duplication is recognized after the expensive call.

**Consequence:** Model compute and activation/workspace scale with child count even though only posterior samples/events/reach updates need to be child-local. This is particularly harmful for the already expensive poker MHA.

**Recommendation:** Evaluate each unique active player parent once; sample child-local profiles; gather actor rows/events from the parent output; compute selected-action log probabilities for all sampled labels; expand only the lightweight reach/event operations. Add a call-count and numerical-parity test.

**Model recommendation:** `5.6 Terra` with `medium` thinking — this is a bounded vectorization refactor with a crisp numerical-parity oracle.

**Response:** Yes.

**Implementation:** Completed. Each rollout wave now calls `policy_forward`
once per unique active player parent. A tensor-structured row gather expands
the parent outputs for child-local profile sampling, event selection, reach
updates, and selected-action likelihoods; value rows remain parent-local.
Tests verify the exact unique-parent call sizes across two branching levels and
numerical parity between gathered and explicitly repeated policy outputs.

---

### AUD-014 — Completion retains unused large diagnostics and ignores quality signals

**Severity:** Major
**Modes:** Sequential RB, branched RB
**Evidence:** Poker completion retains `equity_share[B,P,1326,K]`, `output_valid`, `estimator_stddev`, `sample_count`, `vvec_hat`, `u_hat`, and `n_eff`. The active `branched_value/rb_ppo` path consumes only `p_hat` and `v_hat`. For six players, `equity_share` alone is about 191 kB (186.5 KiB) per leaf. Conversely, `n_eff`, validity, and estimator error are not used to reject or adapt noisy targets.

**Consequence:** Memory is spent on dead fields while target quality is not governed. Many leaves can turn this into gigabytes.

**Recommendation:** Add a minimal completion-return mode for active consumers, or immediately detach only required learning fields and release the sibling result. If quality fields are retained, define thresholds, logging, invalid-row policy, and optional adaptive sampling rather than silently ignoring them.

**Model recommendation:** `5.6 Terra` with `medium` thinking — DTO minimization is contained, while quality-policy choices require careful evaluator integration.

**Response:** Yes.

**Implementation:** Completed. Completion results are converted immediately
to a learning-only `p_hat`/`v_hat` view. The conversion rejects non-finite or
negative posterior mass, non-finite values, invalid outputs carrying posterior
mass, nonpositive effective/sample counts, and inconsistent effective counts.
It logs scalar effective-sample, invalid-mass, estimator-error, and error-
coverage summaries, then releases `equity_share`, elementwise validity/error
tensors, auxiliary moments, leaf reaches, and the original completion DTO
before target construction/optimization.

---

### AUD-015 — Folded and live reach lifetimes are not mode-specific

**Severity:** Major
**Modes:** All, especially normal
**Evidence:** Both samplers maintain a dense `[node,P,H]` reach arena and `ReachCache` exposes whole rows. Folded seats stay in the same full-width representation. There is no transition from a full fold-time range to the compact statistics required by the model, nor any release after normal terminal rollout or RB leaf completion.

**Consequence:** The optimization explicitly requested by the user—discarding folded ranges after their last consumer—is absent. Normal mode would inherit RB memory unless a specialized lifecycle is designed.

**Recommendation:** Implement the lifetimes in `mode_storage_compute.md`: retain folded full ranges only through future summary refreshes and, for RB, leaf completion; persist model-ready folded summaries per decision; then free full folded rows. Treat normal, one-path RB, and tree RB as different compact target layouts.

**Model recommendation:** `5.6 Sol` with `high` thinking — mode-specific reach ownership and release points interact with completion, replay, summaries, and branching independence.

**Response:** We do need full folded ranges to be fed to the "completion" equity/posterior API (which isn't used in normal mode). But otherwise, they are only used through the folded stats. Make sure this is efficiently handled.

**Implementation:** Completed. Normal mode converts decision reaches to a
game-owned cache as soon as sampled terminal targets are built. RB modes retain
full folded ranges through leaf completion and target construction, then
replace the tree `ReachCache` with the same decision cache. Poker implements
that cache as ragged live-seat `[T,1326]` rows plus seat ids and row pointers;
dense `[B,P,1326]` rows exist only for sampled physical minibatches. Folded
model-ready summaries remain in the street/fold-keyed partition cache, and a
regression test verifies that full folded rows are absent while cached folder
summaries still serve later training forwards.

---

### AUD-016 — Player-node branching samples more hidden state than the event law needs

**Severity:** Moderate  
**Modes:** Branched RB poker  
**Evidence:** Each abstract player child calls the exact joint-profile rejection sampler and then uses only the acting player's combo to draw the action. Full joint profiles are required for chance events, but the player-event mixture requires only the collision-conditioned actor marginal.

**Consequence:** Multiway rejection sampling is paid at every branched player edge and may dominate rollout as ranges tighten.

**Recommendation:** Extend the sibling evaluator with an exact actor-marginal sampler (or direct public action-mixture sampler) for player nodes while retaining full joint sampling at chance. Preserve independent child-local draws. Benchmark acceptance and end-to-end cost before and after.

**Model recommendation:** `5.5` with `low` thinking — the user rejected this implementation direction, so the remaining work is to document and close the finding without changing the sampler.

**Response:** No. Mathematically it's the same difficulty to do an "actor-marginal sampler" as it is to do a full joint profile sampler. The reaches are NOT posteriors. Unless you have an idea on how to sample exactly on the true acting player posteriors, we should keep the current common rejection sampler. Sure it's outputing too much data usually, but that seems like a small memory cost to make it into a unified kernel that is very parallelizable (and we can discard this data right after it's used).

**Implementation:** Closed by inspection, with no sampler change. In branched
RB rollout, `_sample_branched_labeled` keeps `profiles =
backend.sample_profiles(...)` as a wave-local tensor, uses it immediately in
`backend.sample_events(...)` for player actions and chance board draws, and
stores only public event ids, reach rows, selected-action log probabilities,
and value rows. The returned `TreeBatch`, `ReachCache`, and `PolicyCache`
schemas contain no private-profile field, so the sampled joint profile data is
discarded after the wave has produced its public event/reach update.

---

## B. Game-family, theory, operations, and validation findings

### AUD-019 — Kuhn/Leduc/dial do not use the required MHA poker design

**Severity:** Major if “each version of poker” includes these games; otherwise Moderate clarification  
**Modes:** Synthetic poker variants  
**Evidence:** All synthetic games used `SyntheticNet`, a small public/type embedding MLP. None used the MHA hand/class rows, reach priors, action anchors, incidence residual, factored continuous raise head, or folded-summary design.

**Consequence:** Model comparisons across small poker and full poker confounded game complexity with a completely different architecture. The user's requirement that `docs/MHA_model.md` be used “for each versions of poker” was not met under the natural reading.

**Recommendation:** Decide scope. If binding, create scaled MHA adapters for Kuhn/Leduc/dial with the same architectural concepts and game-appropriate row/action heads. Retain `SyntheticNet` only as an explicit baseline.

**Response:** Yes

**Implementation:** Synthetic poker now defaults to `SyntheticMHANet`. It runs
player/type rows with public-history and discrete action-anchor tokens, adds
normalized log reach as an untempered key prior, and applies exact card,
identical-type, and own-range incidence channels in every layer. Folded rows are
excluded from attention and integrated through finite-game card pressure. The
discrete raise menu uses event anchors, the appropriate counterpart of NLHE's
factored tick head. `SyntheticNet` remains available only through explicit
`model.architecture: mlp_baseline`. Exact policy export propagates the model's
own reach factors down the public tree.

**Verification:** Tests assert MHA is the default, validate card incidence,
show reach-prior and folded-summary effects, verify exact duplicate-state
expansion, and cover explicit MLP selection. Existing adapter, rollout, exact
export, and all-mode training tests pass; the full CPU suite reports `196
passed, 7 skipped`.

---

### AUD-032 — There are no comparative performance gates for the three modes

**Severity:** Major relative to the “all modes optimally” requirement  
**Modes:** All  
**Evidence:** No benchmark recorded normalized throughput, phase time, retained
target bytes, or device peak memory by mode.

**Consequence:** Mode-specific performance regressions had no enforceable
efficiency contract.

**Recommendation:** Add stable microbenchmarks and end-to-end profiles with
per-mode/device tolerances and clean GPU setup/teardown.

**Response:** Use judgment about optimality and improve against our own measured
times with a benchmark-then-optimize pass.

**Implementation:** `pbg-benchmark` now runs any scientific config through all
three modes, records median phase times, topology/offspring, decisions and
terminal/completion samples per second, effective completion samples, compact
target bytes per decision, and CUDA allocated/reserved peaks. Baselines are
matched by device, compute capability, thread count, and workload; throughput,
time, storage, and CUDA-memory regressions have configurable tolerances. The
runner requires clean CUDA allocator state before and after a job. The committed
matrix covers Kuhn, Leduc, a third dial point, tiny/medium local NLHE, and the
readiness-gated production chunk.

The optimization pass deduplicates identical synthetic public states before
model evaluation and expands the output exactly. On the local 14-thread CPU,
the controlled 512-tree Kuhn normal profile improved median rollout from
0.2233 s to 0.1968 s (11.8%), optimization from 0.0287 s to 0.0163 s (43.2%),
and total throughput from 4,355 to 5,136 decisions/s (17.9%).

**Verification:** Gate-unit tests cover within-tolerance acceptance,
multi-metric regression failures, and cross-environment refusal. Final
three-mode Kuhn and Leduc profiles emitted `pbg-performance-v1` records. A tiny
heads-up 5bb normal-mode CUDA smoke completed on idle GPU 1 with a 46 MiB peak
reserved footprint; after process exit GPU 1 returned to 9 MiB and 0%
utilization. Larger CUDA/NLHE baselines remain machine-local operations and
follow the shared-server workload matrix rather than being encoded as portable
absolute timings.

---

### AUD-030 — CUDA verification is presently blocked in the prescribed venv

**Severity:** Moderate validation blocker, not by itself a repository-code defect  
**Modes:** GPU  
**Evidence:** The shared venv at `/home/pablob/.venvs/poker` uses Python 3.10.12 and PyTorch `2.12.1+cu130`. The project requirement was aligned to Python 3.10+, preserving the common venv for this repository and the sibling evaluator. With elevated GPU access, `nvidia-smi` confirmed the NVIDIA 580.159.04 driver and GPU 1 was idle. A tiny CUDA tensor smoke passed, and the targeted CUDA parity and rollout tests passed (`2 passed, 17 deselected`) with `CUDA_VISIBLE_DEVICES=1`. A final GPU check showed no audit workload remained.

**Consequence:** CUDA residency, parity, and evaluator CUDA behavior are now revalidated in the prescribed shared venv. This does not establish H100 feasibility; the independent static memory findings remain open.

**Recommendation:** Repair or recreate the shared venv with a supported Python and working driver/CUDA compatibility under server policy, then run tiny CUDA smoke tests on an explicitly selected idle GPU before any benchmark. Do not attempt a production-scale H100 run until `AUD-005` is resolved.

**Model recommendation:** `5.6 Luna` with `medium` thinking — environment repair is operationally contained but needs cautious shared-server and GPU hygiene.

**Response:** Dude that's up to you to figure out! Feel free to modify "/home/pablob/.venvs/poker". The idea is just to use a common venv for this repo and the sibling repo. It doesn't really matter to me which version of python, pytorch, cuda.... is installed there. We just need to remember this is a shared server so everything must be done locally in my folder "/home/pablob/" without impacting global python or torch or whatever. You should figure out a way to use the GPUs for testing, this is absolutely crucial and non-negotiable.

**Implementation:** Completed. The shared venv already had a compatible CUDA-enabled PyTorch installation, but sandboxed GPU access had hidden the driver. The project metadata and documentation now support the shared Python 3.10 environment. GPU visibility was verified outside the sandbox, and tiny CUDA validation ran only on explicitly selected idle GPU 1 with clean teardown.

### AUD-031 — Authoritative paths and implementation notes have drifted

**Severity:** Moderate  
**Modes:** Documentation/maintenance  
**Evidence:** Root `POKER_GPU_EVALUATOR_IO.md`, `PREAMBLE.md`, `Prompts/`, and `docs/latex/` are deleted/moved in the current worktree, while `CONTRACTS.md`, `NOTATION.md`, and MHA docs still refer to old paths/casing (`Docs/MHA_model.md`, root evaluator IO, `docs/latex/core_math.tex`). `IMPLEMENTATION_NOTES.md` still describes superseded Poisson/deduplicated/clamped branching before later corrections, and calls the H100 integration/preset ready despite the failures above. The existing `core_math_audit.md` explicitly excludes implementation and therefore must not be read as a repository-readiness audit.

**Consequence:** A future implementer can follow stale behavior or broken links and incorrectly treat mathematical coherence as end-to-end readiness.

**Recommendation:** After current file moves settle, update all authoritative pointers, mark historical sections as such at their source, and link this audit from the repository docs index. Keep one current execution/status matrix rather than contradictory chronological claims.

**Model recommendation:** `5.5` with `low` thinking — this is primarily link repair, documentation reconciliation, and consistency checking.

**Response:** Yes, feel free to fix the paths to the new locations. It's my bad: I moved some of these files.

**Implementation:** Completed. Live documentation now points to the moved
authorities under `docs/theory/`, `archive/`, and `archive/prompts/`; casing
and folder-impact references were corrected; `docs/audit/README.md` is linked
from the documentation index; `IMPLEMENTATION_NOTES.md` is explicitly marked as
a historical ledger; and H100/preset language points readers to the active
audit findings rather than claiming readiness.

---

### AUD-018 — The shipped dial example aliases distinct public histories in the model

**Severity:** Major  
**Modes:** Synthetic  
**Evidence:** `SyntheticNet` embedded `public_key % n_buckets`. The shipped dial
had 10,826 enumerated public states but only 1,024 buckets, so distinct
perfect-recall public histories necessarily shared embeddings.

**Consequence:** The dial policy could not represent an arbitrary behavioral
strategy on its own public tree and could conflate different legal/history
contexts.

**Recommendation:** Use collision-free state embeddings for enumerated
synthetic games, dynamically size the table after tree enumeration, or reject
insufficient capacity.

**Model recommendation:** `5.6 Luna` with `medium` thinking.

**Response:**

**Implementation:** `SyntheticNet` now indexes public keys directly and rejects
negative/out-of-capacity keys. The synthetic bundle enumerates the complete
public tree before model construction, verifies key uniqueness, derives the
embedding capacity from the maximum reachable key, and rejects a smaller
user-supplied capacity. Shipped synthetic presets no longer guess a bucket
count.

**Verification:** A regression builds the shipped dial and proves that every
enumerated history has a unique key and an in-capacity embedding row; a model
unit test proves that an out-of-capacity key is rejected instead of wrapped.

---

### AUD-023 — Configuration exposes dead or misleading controls

**Severity:** Moderate  
**Modes:** All  
**Evidence:** `use_suit_consistency` and `debug_asserts` had no production
consumers, `clip_eps_v` was not selected by the composites, synthetic
`model.e_max` was ignored, and game-owned mappings admitted obsolete keys.

**Consequence:** Users could believe a behavior was enabled when it was not,
and misspelled settings could survive configuration loading.

**Recommendation:** Remove deferred fields until wired or make them functional;
give each game factory a strict schema and test accepted toggles.

**Model recommendation:** `5.6 Luna` with `medium` thinking.

**Response:**

**Implementation:** The two no-op flags and synthetic `e_max` were removed from
the accepted schema and presets. Synthetic game/model and poker game/model
mappings are validated during top-level load and reject unknown fields.
`loss.value_loss: mse | clipped` now explicitly selects `l_v` or `l_v_clip` in
both normal and RB composites, so `clip_eps_v` has a real consumer.

**Verification:** Configuration regressions reject both removed flags,
synthetic `e_max`, and misspelled game keys. Loss tests demonstrate that the
two accepted value-loss settings select observably different objectives.

---

### AUD-027 — Global deterministic mode conflicts with maximum throughput

**Severity:** Moderate  
**Modes:** All GPU modes  
**Evidence:** Training unconditionally enabled deterministic algorithms and
deterministic cuDNN behavior, with no reproducibility/performance choice.

**Consequence:** Optimized kernels and autotuning were unavailable even for
performance runs.

**Recommendation:** Add `strict | seeded | throughput` profiles while keeping
seeds and checkpoint RNG state in every profile.

**Model recommendation:** `5.6 Luna` with `medium` thinking.

**Response:**

**Implementation:** `training.reproducibility` is a strict enum. `strict`
enables deterministic algorithms and disables benchmarking/TF32; `seeded`
keeps explicit seeds while allowing nondeterministic kernels without cuDNN
autotuning; `throughput` enables cuDNN benchmarking and TF32. Hardware metadata
records the selected profile and effective backend flags. The production NLHE
presets select `throughput`; small correctness presets remain strict or seeded.

**Verification:** Profile tests assert each backend-control transition, and all
profiles retain explicit rank-local generators and checkpointed process RNG.

---

### AUD-028 — Training cannot resume as an end-to-end workflow

**Severity:** Moderate  
**Modes:** All  
**Evidence:** Checkpoint primitives existed, but `train` always created a new
optimizer, began at zero, and appended metrics without run identity checks.

**Consequence:** Long training could not reliably restart through the public API.

**Recommendation:** Add a validated resume entry point restoring all state and
preserving iteration schedules; test bitwise strict continuation.

**Model recommendation:** `5.6 Luna` with `medium` thinking.

**Response:**

**Implementation:** `train(cfg, resume_from=...)` and CLI `--resume` restore the
model, optimizer, scaler, completed iteration, explicit generators, and process
RNG. Checkpoints store per-rank RNG for DDP. Resume requires the original output
directory, scientific configuration, run identity, DDP world size, schedule,
and exact gap-free metrics prefix. New runs refuse accidental append over
existing artifacts, and every metric/checkpoint carries a stable run identity.

**Verification:** The end-to-end test simulates an interruption after a real
checkpoint, resumes through the public API, and compares its final model tensors
and metrics with an uninterrupted strict-mode run; they are bitwise identical.

---

### AUD-020 — Synthetic `exact` completion is actually sampled MC

**Severity:** Moderate
**Modes:** Sequential RB, branched RB synthetic
**Evidence:** `SyntheticCompletion.complete` always sampled `n_samples`
posterior profiles. `completion.backend: exact` selected the same implementation.

**Consequence:** Preset names overstated exactness and small-game estimator tests
mixed completion noise with rollout/branching variance.

**Recommendation:** Implement feasibility-guarded exact completion and retain MC
as a separate backend; use exact Kuhn/Leduc completion as a training-mode oracle.

**Model recommendation:** `5.6 Terra` with `medium` thinking.

**Response:**

**Implementation:** `SyntheticExactCompletion` now enumerates every
collision-free private profile compatible with the public board, weights it by
the product of the stored per-player reaches, and returns exact posterior mass
and conditional values. It calls the production enumerator in
`games/synthetic/exact.py`, fails before exceeding the exact-profile guard, and
is independent of the configured sample budget and RNG. The former sampler is
retained explicitly as `SyntheticMCCompletion`; bundle dispatch distinguishes
`exact` from `mc`.

**Verification:** Exhaustive posterior/value calculations match an independent
Kuhn enumeration, exact results are bitwise seed/sample-budget invariant, and a
guard regression fails before infeasible enumeration. All three training modes
complete production-path iterations on both exact Kuhn and exact Leduc.

---

### AUD-025 — Frozen rollout/model and transient objects live through optimization

**Severity:** Moderate
**Modes:** All current modes
**Evidence:** `_one_iteration` formerly owned the deep-copied rollout model and
completion/leaf objects in the same Python frame as optimization.

**Consequence:** Frozen parameters and evaluator transients could overlap with
train-time activations and optimizer state.

**Recommendation:** Explicitly end frozen-model, completion, leaf-reach, and
full-reach-cache lifetimes before training; avoid unconditional allocator flushes.

**Model recommendation:** `5.6 Terra` with `medium` thinking.

**Response:**

**Implementation:** Rollout, completion, and target construction now run in the
separate `_build_iteration_data` frame. Only compact normal data or the tree,
targets, policy cache, alias table, and compact decision reaches cross its return
boundary. The frozen model/bundle, diagnostic completion result, leaf reaches,
learning completion view, and full reach cache become unreachable before the
first optimization forward. No synchronization or allocator flush is forced.

**Verification:** A weak-reference lifetime regression observes both the copied
model and the full completion result being collected at the phase boundary while
the returned targets remain usable. Focused normal/RB training and strict-resume
tests pass.

---

### AUD-029 — Existing tests do not gate the requested capabilities

**Severity:** Major
**Modes:** All
**Evidence:** The audit-time suite did not invoke normal training, exact
cross-mode gradients, shipped reach replay, or architectural conformance paths.

**Consequence:** A green suite was compatible with missing or broken requested
capabilities.

**Recommendation:** Implement the correctness gates in `validation_matrix.md`,
including all modes on exact small games, beta-one equivalence, poker replay,
history/folder effects, and engine differential fuzzing. Target H100 ceilings
remain separately tracked by `AUD-005`.

**Model recommendation:** `5.6 Terra` with `high` thinking.

**Response:**

**Implementation:** The suite now runs normal, sequential RB, and beta-one
branched RB through exact Kuhn and Leduc production paths. A detached-target
oracle exhaustively enumerates Kuhn continuation values and gates each mode's
empirical root policy-gradient vector. Existing beta-one public-law,
duplicate-label, chance replay through every street, board-conflict, lossless
history/input-effect, folder no-op/effect, MHA incidence/geometry, and normal
poker settlement tests are retained. Engine differential fuzzing now spans
random legal histories for every supported table size from two through nine
players, alongside deterministic all-in/reopening/side-pot cases.

**Verification:** Focused exact/cross-mode/lifetime/branching/replay/history/
folder/engine gates pass on CPU. CUDA-targeted model and rollout checks are run
under the shared-server hygiene procedure. Device-local performance baselines
are now implemented by `AUD-032`; Quest H100 ceilings remain owned by
`AUD-005`, rather than being silently inferred from correctness CI.

---

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

**Implementation:** The public helper is now exactly
`dial(card_scale, action_scale)` and fixes player count at two. `card_scale`
alone grows deck size, private types, and board-chance support. `action_scale`
grows the number of streets, raise cap, and distinct per-round raise increments,
so it controls both tree depth and width. The single synthetic state machine now
supports multiple raise events without named-rule branches or event-id overlap.
The two anchor points exactly reproduce the existing presets:
`dial(1, 1) == kuhn()` and `dial(2, 2) == leduc()`.

The intended scientific claim is explicit: this is a controlled two-axis
algorithmic stress family for Rao--Blackwellization and branching, not a claim
that a scalar path converges rule-for-rule to 6-max NLHE. Full NLHE remains the
separate production endpoint. This removes the former hidden player-count axis
and avoids presenting unrelated rule changes as independent experimental dials.

**Verification:** Grid properties cover every action scale through four streets
across multiple card scales, monotonic private/chance growth, fixed rule fields,
raise-width/depth growth, exact Kuhn/Leduc anchors, distinct raise transitions,
and disjoint player/chance event namespaces. The shipped example is one
documented point on this two-axis helper.

---

### AUD-026 — Device-resident hot paths contain avoidable host synchronization

**Severity:** Moderate
**Modes:** All GPU modes
**Evidence:** `AliasTable.build` copies weights to CPU and builds via Python lists in float64. `build_compact_map` uses `.item()` and a Python loop per batch row on every postflop forward. `MHA.forward` executes `count_nonzero(...).item()` every layer to choose a path even though reach bias is known to be present. Poker adapters and routing use multiple `bool(tensor.any())`/scalar conversions.

**Consequence:** GPU pipelines repeatedly stall, and the “device residency” tests miss synchronization that does not spell `.cpu()` in the few scanned functions.

**Recommendation:** Use device-native categorical sampling/alias construction, vectorized fixed-per-street compact maps, an explicit biased-attention call path, and tensorized validation outside the steady-state hot path. Profile synchronization counts with CUDA tooling and add a no-host-sync benchmark gate for model/minibatch execution.

**Model recommendation:** `5.6 Terra` with `high` thinking — several CUDA hot paths need profiling-led rewrites without changing sampling or model semantics.

**Response:** Yes

**Implementation:** Decision-law construction now normalizes weights and samples
categorically with `torch.multinomial` on the source device; the historical
`AliasTable` name remains API-compatible, but no tensor is copied through CPU
lists. Compact combo maps use a vectorized prefix-rank/scatter construction and
vectorized inverse gather. Production poker rollout queues are split by street
and pass the exact static compatible-combo width (1176/1128/1081), so no device
reduction is read by the host. Reach-biased MHA has an explicit call path, and
CUDA value validation in the policy heads, pre/post models, policy adapter,
decision sampler, and compact-reach expansion uses asynchronous device asserts.

**Verification:** Distribution tests retain the weighted decision law and
compact gather/scatter gradient semantics. A CUDA sync-debug regression runs
device categorical construction/sampling, fixed-street compaction, biased
attention, and backward with host synchronization configured as an error.
Static guards also reject reintroducing CPU/list alias construction or a scalar
attention-path probe.

---

### AUD-013 — Completion and backward propagation were not genuinely streamed

**Severity:** Critical  
**Modes:** Sequential RB, branched RB  
**Evidence:** Training formerly completed every leaf in one batch, retained the
full completion result, and allocated `P/M [N,P,X]` for a whole root replica.
The abstract default of 2,000,000 player rows permitted 21,216,000,000 bytes
(19.76 GiB) for poker `P/M` alone, before targets and other live allocations.
The implementation also treated branched rollout as a forest of replicas of
the identical initial state.

**Consequence:** Completion, whole-tree moments, targets, reaches, topology, and
the frozen rollout model overlapped unnecessarily. The advertised chunking did
not bound the value-phase peak in bytes.

**Recommendation:** Use trajectories as the sequential unit and exactly one
sampled branching tree per rank/GPU. Feed byte-limited completion microbatches
into an upward `P/M` reduction frontier, replace completed siblings by a parent
boundary summary, and retain only compact decision data for optimization.

**Model recommendation:** `5.6 Sol` with `high` thinking.

**Response:** Approved after clarifying that sequential samples are
trajectories, branched RB has exactly one tree per GPU, and the retained
boundary consists of `P-hat/M-hat` summaries at the leaves of the progressively
contracted tree.

**Implementation:** Branched configuration and sampling now require
`n_trees=1` per rank; presets preserve their former total node ceiling by
moving it into that tree's `node_budget`. Sequential modes continue to use the
legacy serialized `n_trees` field as a trajectory count and documentation does
not call those samples trees.

Training selects completion microbatches from an explicit byte budget (at most
256 leaves and at most one eighth of the phase budget). Each learning-only
`p_hat/v_hat` batch is immediately inserted into the backward frontier. When
all children of a parent are present, their `P/M` rows are averaged, decision
targets are emitted, owner `M` is lambda-bootstrapped, and the children are
replaced by one parent row. The persistent moment boundary never exceeds the
initial leaf cut and no `[N,P,X]` moment arena exists. A conservative preflight
counts completion fields, retained targets, `P/M`, overlapping gathers/copies,
private-type width, dtype bytes, and structural scratch; over-budget plans fail
before target allocation. The former abstract row cap is removed.

The previous completion DTO and leaf reaches are released before the next
completion call. The frozen rollout snapshot is released before completion.
Full reaches and topology do not cross into optimization: decision state refs,
owners, old outputs, alias weights, targets, and game-compacted reaches do.
Sequential one-child parent/child posteriors share one storage allocation.
Pinned-host target spilling was not assumed; the byte preflight rejects a
resident target set that does not fit.

**Verification:** Full and one-leaf-at-a-time target construction are bitwise
equal across parent/child posteriors, advantages, values, and optional Q.
Regressions cover duplicate labels, lambda boundaries, auxiliary bootstrap
ordering, byte-budget rejection, one-tree enforcement, sequential posterior
storage sharing, completion-result collection between microbatches, exact
three-mode small-game gradients, and compact optimization without retained
topology. The complete CPU suite and a conservative CUDA smoke are recorded in
the implementation handoff: 196 CPU tests passed with 7 CUDA-only skips, and
the targeted frontier CPU/CUDA parity test passed on an otherwise idle GPU.

---
