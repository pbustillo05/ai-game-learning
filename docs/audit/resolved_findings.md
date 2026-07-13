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
The shipped preset is a conservative 8-tree, 512-node bootstrap configuration.

`estimate_rollout_memory` provides a conservative byte plan (two full reach
frontiers plus state/tree/cache bounds), and CUDA bundle construction rejects a
plan exceeding half of currently free device memory before allocating the
backend. The revised preset's rollout plan is 306,318,404 bytes (about
291 MiB) on CPU construction.

**Verification:** Static tests bound the legacy 2,097,152-node recompute
frontier above 100 GiB and legacy full storage above 60 GiB, while the new
preset plan is below 1 GiB. Parent/event links are checked against random
legal engine trajectories. CPU construction of the preset succeeds, the full
suite passes (`133 passed, 4 skipped`), and a tiny explicitly targeted CUDA
smoke passed on idle GPU 1; GPU 1 returned to 9 MiB and 0% utilization.
`AUD-005`, `AUD-013`, and later model/completion allocations remain separate
limits; this does not claim the preset is a production-scale H100 run.

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
