# Poker backend and sibling-evaluator audit

## Scope, authorities, and result

Reviewed the NLHE backend, immutable poker inputs, completion and folder-summary
facades, compact deck geometry, tensor state/history owners, and the directly
relevant tests. This was a read-only implementation audit: no production code,
tests, configuration, or sibling-repository file was changed.

The path named as binding by `AGENTS.md`,
`/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`, does not exist in the
workspace. The sibling repository instead contains `docs/api.md` and
`docs/stale_api_pbg.md`. Per coordinator direction, this audit read
`/home/pablob/poker-gpu-evaluator/docs/api.md` completely (380 lines) as the
current contract-v2 evidence and did **not** use `stale_api_pbg.md` as an
authority. Finding POKER-008 records this authority-path drift.

Overall result: the tensor-native backend preserves the active NLHE dials,
compact-deck evaluator identity, exact-hand completion width, keyed sibling
posterior sampling, one branched root, immutable leaf inputs, natural-width
preflop/postflop partitions, and recyclable state/history ownership. Four
material inconsistencies remain:

1. `post_P` materializes a neutral folded summary even though the storage
   authority explicitly prohibits it.
2. Folded summaries are recomputed and retained per scheduler pack and only
   deduplicated during seal, rather than being stored once at their dependency
   anchor.
3. The facades and tests preserve evaluator contract v1 compatibility despite
   the current binding declaring contract v2.
4. A large list-based/CPU implementation remains reachable from evaluation and
   inference, while training itself still explicitly selects CPU.

No GPU work was run for this static batch. Findings POKER-005 and POKER-009
specify focused CUDA gates needed before performance fidelity can be claimed.

## Findings

### POKER-001 — `post_P` materializes the forbidden neutral folded summary

- Severity: high
- Confidence: confirmed
- Contract: STORE-005, STORE-006, STORE-014, STORE-031, TEST-STORE-006
- Modes affected: all five modes when an all-live postflop model forward
  consumes evaluator summary features
- Evidence:
  - `pbg/games/poker/rollout_backend.py:1840-1872`,
    `PokerRolloutBackend.policy_forward_family`
  - `pbg/games/poker/rollout_backend.py:1960-1983`,
    `PokerRolloutBackend.policy_forward_partition`
  - `pbg/games/poker/rollout_backend.py:2049-2075`,
    `PokerRolloutBackend.functional_policy_forward_partition`
  - `pbg/games/poker/rollout_backend.py:2466-2491`,
    `PokerRolloutBackend._partition_summary_all_live`
  - `pbg/games/poker/folded.py:264-289`,
    `PokerGPUPartitionAdapter.summarize`
  - `tests/test_storage_lifecycle_poker_inputs.py:76-100`
- Current behavior: each of the rollout, sealed learner, and functional/JVP
  forward paths calls `_partition_summary_all_live` for `post_P` whenever the
  postflop model consumes summaries. That helper constructs a zero-folder
  request; `PokerGPUPartitionAdapter.summarize` then allocates neutral
  `hand_log_ratio[B,H]`, `beta_eigenvalue[B,Q]`,
  `beta_eigenvector[B,D,Q]`, and other zero tensors. The current test proves
  only that no *persistent cache* is allocated; it does not reject these
  transient neutral summary allocations.
- Required behavior: `post_P` has no summary reference and must never
  materialize a neutral summary. The model/input contract must represent the
  no-folder case without allocating these tensors.
- Consequence: direct contradiction of the authoritative poker storage layout,
  avoidable per-forward allocation and bandwidth, and an uncharged transient
  whose size grows with batch, hand width, deck width, and beta rank.
- Recommended disposition: fix. Make absence of a summary an explicit
  `post_P` model condition; remove `_partition_summary_all_live` from all
  production paths if it then has no consumer.
- Verification needed: extend `TEST-STORE-006` to monkeypatch
  `PokerGPUPartitionAdapter.summarize` to fail for `post_P`, and cover rollout,
  sealed learner, and functional forward on CUDA.

### POKER-002 — Folded summaries are duplicated across packs before seal

- Severity: high
- Confidence: confirmed
- Contract: STORE-006, STORE-008, STORE-010, STORE-014, STORE-021, STORE-030
- Modes affected: normal, sequential RB, fixed-beta branched RB, KKT branched
  RB, and BR probe whenever postflop descendants share a fold/street anchor
- Evidence:
  - `pbg/games/poker/rollout_backend.py:284-292`, `_dependency_anchors`
  - `pbg/games/poker/rollout_backend.py:584-606`, fold-anchor creation
  - `pbg/games/poker/rollout_backend.py:647-704`, street-anchor creation
  - `pbg/games/poker/rollout_backend.py:1146-1182`,
    `capture_decision_inputs`
  - `pbg/games/poker/rollout_backend.py:1191-1244`,
    `append_decision_input_capture`
  - `pbg/games/poker/rollout_backend.py:1329-1370`, seal-time deduplication
- Current behavior: anchor identity is correctly refreshed exactly on a fold or
  street transition. However, every scheduler pack independently uniques only
  the anchors present in that pack, calls the sibling folder summarizer again,
  and appends the resulting payload to the rollout-local summary arena. If the
  same dependency anchor is used by decisions in later packs, its large summary
  is recomputed and retained again. Only `seal_decision_input_captures` uniques
  all appended anchors and discards duplicate payloads.
- Required behavior: one immutable summary row is generated and owned at each
  dependency anchor and referenced by descendant decisions until the next
  dependency change. It is not a per-decision or per-pack payload.
- Consequence: evaluator work and rollout memory scale with packs/decisions
  rather than realized anchors. Seal peak includes all duplicates plus the
  exact destination, contradicting the intended memory saving and threatening
  large-tree admission.
- Recommended disposition: fix. Give the rollout-local summary owner an
  anchor-indexed lookup/append transaction and compute only missing anchors;
  append only the integer summary reference to each decision row. Seal each
  anchor row once without post-hoc payload deduplication.
- Verification needed: force one fold anchor to occur in multiple queue packs
  and assert exactly one evaluator call, one summary source row, stable decision
  references, and a seal peak based on distinct anchors rather than decisions.

### POKER-003 — Production facades intentionally accept obsolete evaluator contract v1

- Severity: high
- Confidence: confirmed
- Contract: STORE-001, STORE-007; sibling `docs/api.md` “Production” and
  “Identity and deck contract”
- Modes affected: all five modes
- Evidence:
  - sibling `docs/api.md:14-18` requires `poker.CONTRACT_VERSION == "2"`
  - `pbg/games/poker/mc.py:21-24`, contract constants
  - `pbg/games/poker/mc.py:38-60`, `_load_evaluator`
  - `pbg/games/poker/folded.py:20-23`, contract constants
  - `pbg/games/poker/folded.py:82-104`, `_bound_evaluator`
  - `pbg/games/poker/folded.py:283-288`, synthetic v1 metadata on the
    no-folder path
  - `tests/test_p4_folded.py:15,59,122,154`, v1 compatibility expectations
  - `tests/test_p5_completion.py:18,81,151,202`, v1 compatibility expectations
  - `tests/test_p12_eval.py:22,53`, v1 test package
- Current behavior: both production facades advertise expected version `"1"`
  and accept `{"1", "2"}`. Tests deliberately preserve v1. The neutral folder
  result even reports contract version `"1"` without binding the evaluator.
- Required behavior: bind exactly the current versioned sibling production
  contract unless an authoritative, narrowly scoped migration explicitly
  defines compatible versions. The current sibling authority specifies v2.
- Consequence: a stale evaluator installation can silently enter production
  even though v2 added/clarified the minimal completion output, compact deck
  identity, row-local RNG, and production-versus-testing allocation contract.
- Recommended disposition: fix/delete compatibility. Require contract `"2"`,
  update messages and metadata, and delete v1 fixtures and expectations.
- Verification needed: one shared facade contract test that rejects v1 and
  checks all identity fields needed by compact deck, completion, folder
  summaries, and RNG replay.

### POKER-004 — Parallel list-based CPU game/range implementation remains live

- Severity: high
- Confidence: confirmed for non-core reachability; strong for deletion/archive
  disposition pending the coordinator's BR/evaluation classification
- Contract: STORE-008, STORE-010, STORE-015, STORE-016, STORE-023,
  STORE-024; explicit GPU-only and sole-production-path audit requirements
- Modes affected: production surface generally; the five core GPU modes use
  `PokerRolloutBackend`, while evaluation/inference retain a second engine
- Evidence:
  - `pbg/games/poker/engine.py:76-190`, `PublicHistoryEvent`, `_State`, and
    list-backed `StateTable`
  - `pbg/games/poker/engine.py:241-709`, `NLHEEngine`
  - `pbg/games/poker/ranges.py:23-337`, `FoldedPlayer` and `RangeState`
  - `pbg/games/poker/ranges.py:142-161`, repeated `torch.cat` growth
  - `pbg/games/poker/adapter.py:472-710`, `ChanceDraw` and `PokerGame`
  - `pbg/games/poker/adapter.py:547-550,578-581,615-676`, repeated
    device-to-host conversion and Python history construction
  - `pbg/games/poker/rollout_backend.py:91-106`, explicit CPU/CUDA dual path
  - `pbg/core/training.py:2543-2554`, CPU training selection/fallback
  - `pbg/eval/arena.py:1,52-85,187-225`, CPU-only arena consumer
  - `pbg/inference.py:193-229`, list engine/PokerGame consumer
- Current behavior: the tensor-native paged backend is the core training/BR
  path, but a second list-backed game and dynamically growing range store remain
  importable and are exercised by arena and inference code. Training also
  explicitly chooses CPU when configured or when `auto` sees no CUDA.
- Required behavior: NVIDIA GPUs are the sole production target; old production
  implementations must not survive beside the replacement. A small independent
  correctness oracle may remain test-only or archived, but must not be a live
  production/evaluation dependency.
- Consequence: two transition/input implementations can drift; host transfers
  and Python loops are preserved; CPU-only tests can pass while the GPU path is
  broken; the public surface contradicts the declared deployment constraint.
- Recommended disposition: move the oracle and its tests to `archive/` or
  test-only support, migrate any required inference/arena consumers to
  `PokerRolloutBackend`, and make training/build entry points reject non-CUDA
  devices. If arena is not one of the five required modes, delete it.
- Verification needed: a symbol/entry-point gate proving the five modes never
  import `PokerGame`, `RangeState`, or `NLHEEngine`, plus a CUDA-required startup
  test.

### POKER-005 — Evaluator facades introduce repeated CUDA host synchronizations

- Severity: high
- Confidence: confirmed algorithmic performance risk
- Contract: STORE-017, STORE-024, STORE-031, HIST-PERF-02, TEST-STORE-008
- Modes affected: all training modes through profile sampling; normal through
  fixed showdown; both RB modes and BR through completion/profile sampling
- Evidence:
  - `pbg/games/poker/mc.py:107-135`, `_normalize_board` uses Python `bool`
    on device reductions
  - `pbg/games/poker/mc.py:169-190`, `_linear_reach` uses three Python
    `bool` checks on device tensors
  - `pbg/games/poker/mc.py:449-456`, sampled-profile result validation
  - `pbg/games/poker/mc.py:486-518`, showdown validation and conservation
  - contrasting async validation in `pbg/games/poker/folded.py:25-31`
  - `tests/test_aud26_no_host_sync.py:156-165` covers policy/folder forward,
    not profile sampling, completion, or showdown
- Current behavior: each facade call evaluates CUDA scalar predicates through
  Python `bool`, synchronizing the stream. These checks occur around hot
  sampling/completion calls rather than at one declared scheduler boundary.
- Required behavior: preserve device-resident validation with async assertions
  or aggregate validation at an explicit bounded control boundary. Logical row
  results must remain independent of physical microbatching.
- Consequence: latency grows with evaluator microbatches and tree queue packs;
  no current no-host-sync gate covers this boundary, so historical throughput
  preservation cannot be claimed.
- Recommended disposition: optimize after correctness fixes; consolidate the
  common facade validation and use CUDA-safe assertions.
- Verification needed: run the three production evaluator operations under
  CUDA sync-debug mode and add per-operation profiler counts/timing.

### POKER-006 — Sealed learner forwards synchronize on CUDA partition checks

- Severity: medium
- Confidence: confirmed
- Contract: STORE-031, HIST-PERF-01, HIST-PERF-02, TEST-STORE-002,
  TEST-STORE-008
- Modes affected: all five modes during learner/reference/JVP forwards
- Evidence:
  - `pbg/games/poker/rollout_backend.py:1904-1924`, especially
    `bool((pub.street == 0).all())` at line 1913
  - `pbg/games/poker/rollout_backend.py:1925-1933`, especially the postflop
    router/domain `bool` check
  - `tests/test_aud26_no_host_sync.py:156-165`, which does not call
    `policy_forward_partition`
- Current behavior: every sealed physical learner subspan reads CUDA routing
  predicates into Python. Routing and partition identity are already host-known
  from the immutable partition plan, so these synchronizations are avoidable.
- Required behavior: asynchronous/device validation or a once-per-seal
  structural validation outside the repeated learner hot path.
- Consequence: repeated synchronization in reference-cache construction and
  every training subspan can regress the retained backward optimization.
- Recommended disposition: fix/optimize and extend the existing no-sync gate.
- Verification needed: CUDA sync-debug coverage for cached and uncached
  `policy_forward_partition` and `functional_policy_forward_partition`.

### POKER-007 — Allowed `folder_beta_rank=0` is incompatible with the summary cache schema

- Severity: medium
- Confidence: confirmed by static control flow
- Contract: STORE-002, STORE-006, STORE-010
- Modes affected: any 3–6 player run configured with `folder_beta_rank=0`
- Evidence:
  - `pbg/games/poker/folded.py:66-79` explicitly accepts beta rank zero
  - `pbg/games/poker/model/preflop.py:49-91` permits
    `folder_beta_rank in [0,52]`
  - `pbg/games/poker/partition_cache.py:23-35` rejects any schema with
    `beta_rank <= 0`
  - `pbg/games/poker/partition_cache.py:84-108` constructs that schema whenever
    `max_folders > 0`
  - `pbg/games/poker/rollout_backend.py:168-178` constructs the cache eagerly
- Current behavior: zero beta rank is a valid model/facade value and appears in
  checked-in experiment configurations, but `PokerRolloutBackend` construction
  fails for `P>2` because `max_folders=P-2>0` causes a zero-rank schema build.
- Required behavior: one coherent immutable configuration domain. If rank zero
  means the evaluator summary subsystem is disabled, no summary owner should be
  created. If it is invalid for multiplayer production, reject it during root
  config validation.
- Consequence: an accepted active-game configuration fails only during bundle
  construction, and the storage schema is not the single source of validation.
- Recommended disposition: fix and add a 3-player boundary test.
- Verification needed: config/build tests for `P=2,3,6` with ranks zero and
  positive, including exact zero allocation when disabled.

### POKER-008 — Repository instructions point to a missing evaluator authority

- Severity: medium
- Confidence: confirmed
- Contract: audit authority and drift-prevention requirement
- Modes affected: audit/review process; potentially all runtime modes if stale
  documentation is consulted
- Evidence:
  - `AGENTS.md` names
    `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`
  - that path is absent; the sibling has current `docs/api.md` and explicitly
    stale `docs/stale_api_pbg.md`
  - current `docs/api.md` declares contract v2, while production code still
    describes v1 (POKER-003)
- Current behavior: a reviewer following the repository instructions cannot
  read the named binding file and could accidentally select the similarly named
  stale document.
- Required behavior: one existing, unambiguous, read-only authority path.
- Consequence: high risk of contract drift and inconsistent audit conclusions.
- Recommended disposition: update `AGENTS.md` to the current sibling path or
  restore a maintained `api_pbg.md` alias in the sibling repository through its
  owner. This audit is not authorized to modify the sibling.
- Verification needed: a lightweight repository documentation check that the
  named binding path exists.

### POKER-009 — Production CUDA batch planning is permanently worst-width and leaves known packing savings unused

- Severity: optimization
- Confidence: confirmed implementation behavior; performance magnitude needs
  measurement
- Contract: STORE-017, STORE-031, HIST-PERF-01, TEST-STORE-008
- Modes affected: all five modes, particularly turn/river postflop work
- Evidence:
  - `pbg/games/poker/rollout_backend.py:1868-1871`, rollout postflop always uses
    flop compact width
  - `pbg/games/poker/rollout_backend.py:1981`, learner postflop does the same
  - `pbg/games/poker/rollout_backend.py:2072`, functional forward does the same
  - `pbg/games/poker/rollout_backend.py:2120-2125`, CUDA bypasses actual row-cost
    planning
  - `pbg/games/poker/rollout_backend.py:2203-2210`, family maximum is always
    flop width
  - `pbg/games/poker/rollout_backend.py:2319-2349`, the direct non-family path
    already computes street-specific compact size
  - HIST-PERF-01 optimization inventory identifies postflop physical packing as
    live
- Current behavior: CUDA family queues correctly do not split partitions by
  street, but physical execution also pads all turn and river rows to the flop
  compatible-hand width. The implementation equates fixed statistical
  partition identity with fixed physical padding.
- Required behavior: preserve one `post_A` statistical partition while allowing
  bounded, stable physical subspans/packing plans that exploit realized street
  cost without changing row order, logical keys, or sampling law.
- Consequence: avoidable attention tokens/token-pairs and conservative physical
  batch limits, especially in late streets.
- Recommended disposition: benchmark a stable street-aware physical plan after
  POKER-001/002 are fixed; do not create street-specific statistical partitions.
- Verification needed: fixed-row equivalence tests across flop-padded and
  street-aware physical plans, plus per-family/street tokens, time, and reserved
  memory on an idle GPU.

## Preserved contracts observed

- `PokerConfig` enforces exactly the active dials: `P in [2,6]`, stack in
  `[8,128]` at 0.5 bb increments, `R in [4,13]`, and
  `2P+5 < 4R` (`pbg/games/poker/engine.py:41-73`).
- `DeckGeometry` matches the sibling's active global suffix cards and compact
  lexicographic combo order (`pbg/games/poker/cards.py:92-180`), and binding
  validates `evaluator.combo_cards` (`pbg/games/poker/cards.py:242-261`).
- The backend creates one branched root and ordinary sequential root batches
  (`pbg/games/poker/rollout_backend.py:222-236`), consistent with STORE-023 and
  MATH-BRANCH-001.
- Profile and completion sampling use sibling
  `sample_joint_profiles`/`complete`, keyed per row and device-resident
  (`pbg/games/poker/mc.py:400-457`,
  `pbg/games/poker/bundle.py:198-292,373-489`). No local posterior/equity
  implementation was found in the core tensor backend.
- Chance reach updates mask every exact hand incompatible with the realized
  public board (`pbg/games/poker/rollout_backend.py:972-1000`), and player reach
  updates only the acting player's full curried likelihood row.
- Poker partitions are exactly `pre` plus `post_A` for each live count, preflop
  stores `K=R^2`, postflop stores `H`, and no street-specific statistical
  partition exists (`pbg/games/poker/partitions.py:20-147`).
- Decision inputs serialize the complete public/history model input and natural
  reach at node creation (`pbg/games/poker/rollout_backend.py:1048-1144`);
  immutable learner forwards rebuild from these rows without consulting mutable
  state (`pbg/games/poker/rollout_backend.py:1425-1466,1891-1995`).
- Stopped RB leaves retain evaluator-ready public fields and convert the sole
  leaf reach once (`pbg/games/poker/rollout_backend.py:2530-2577`).
- Tensor public states and binary-lift histories use schema-backed recyclable
  pools with explicit row release (`pbg/games/poker/tensor_state.py:23-315`).

## Python-object production reachability inventory

Classification here is relative to the requested five core NVIDIA-GPU modes,
not “imported somewhere in the repository.” Dynamic protocol fallback makes a
few entries tentative and should be resolved by the final whole-repository
reachability pass.

### Core-mode reachable / retain

- `adapter.py`: `PokerPolicyAdapter`, `PokerModelBundle` routing methods used by
  the backend/learner; helper device-policy functions.
- `bundle.py`: `PokerRolloutMemoryPlan`, `estimate_rollout_memory`,
  `_TensorPokerGPUCompletion`, config/model/bundle builders.
- `cards.py`: `CategoryProfile`, `category_profile`, `DeckGeometry`,
  `validate_evaluator_combo_cards`.
- `engine.py`: action/street/status constants, `PokerConfig`, `PotStrata`.
- `folded.py`: `FoldSnapshotBatch`, `PartitionSummaryBatch`,
  `PokerGPUPartitionAdapter`.
- `mc.py`: `EvaluatorContractError`, `ShowdownResult`, evaluator loader and
  metadata helpers, `stratum_levels`, `PokerGPURejectionSampler`,
  `PokerGPUShowdownEvaluator`; `PokerGPUCompletion` remains only as the base of
  the production sealed subclass and should be narrowed.
- `model_inputs.py`: `DeviceHistoryMaterializer`.
- `partition_cache.py`: `poker_folded_summary_schema`,
  `DevicePartitionSummaryCache` (after POKER-001/002 redesign).
- `partitions.py`: `poker_partition_specs`,
  `assert_poker_storage_widths`.
- `policy_planning.py`: `PlannedPolicyBatch`, `plan_policy_batches`.
- `rollout_backend.py`: tensor schemas/capture records and
  `PokerRolloutBackend` production methods.
- `tensor_state.py`: both schema functions and `PokerTensorStateStore`.

### Not reachable from the five core GPU modes / delete, archive, or move test-only

- `engine.py`: `PublicHistoryEvent`, `_State`, `StateTable`, `NLHEEngine`, and
  module-level `assemble_showdown_payoff`. They are used by CPU arena,
  inference, or tests, not the tensor production bundle.
- `ranges.py`: the entire module (`FoldedPlayer`, `RangeState` and all methods).
  Its only non-test consumer is `PokerGame`; it uses `torch.cat` growth and
  Python snapshot lists incompatible with the production lifecycle.
- `adapter.py`: `ChanceDraw`, `PokerGame`, and their methods. Their consumers are
  `pbg/eval/arena.py`, `pbg/inference.py`, and tests. `PokerModelBundle.forward`
  is likewise the mixed CPU/inference route; core learner calls the explicit
  pre/post methods.
- `cards.py`: standard-52 globals/helpers `STANDARD_DECK`, `hand_index`,
  `hand_cards`, `class_of_hand`, `class_members`, `class_size`,
  `suit_perm_apply`, module-level `board_conflict_mask`, and
  `sample_uniform_complement`. Their non-test consumers are the stale CPU
  `PokerGame`/arena path. Production compact geometry uses `DeckGeometry`.
- `mc.py`: the list-engine implementation in `PokerGPUCompletion.complete`,
  `complete_keyed`, `_complete_impl`, plus `_engine`; the production bundle
  subclasses and overrides completion around tensor state/sealed leaves.
- `rollout_backend.py`: `policy_forward`/`_policy_forward_direct` are referenced
  by tests but not the core rollout scheduler, which calls
  `policy_forward_family_actual_cost`. The mutable-generator methods
  `sampled_terminal_reward` have no production caller; non-keyed
  `sample_profiles` and `sample_events` remain only generic fallback targets
  because this backend always exposes keyed variants. Confirm before deletion
  against evaluation tooling.
- `rollout_backend.py`: `capture_probe_leaf` has no caller in current Python
  production code and is a stale-candidate probe helper.

### Outside assigned model subtree but directly exposed by this audit

- `pbg/eval/arena.py` is a CPU-only duplicate-deal evaluator, hardcodes
  standard-deck `N_HANDS=1326`, uses standard-deck hand helpers, and builds
  models without passing compact geometry (`lines 52-85,187-225`). For allowed
  `R<13`, this does not match the configured compact game. Delete/archive it or
  migrate it to the tensor backend; do not treat it as a valid compact-deck
  production mode.
- `pbg/inference.py` permits only CPU/MPS and retains the list engine/PokerGame
  (`lines 193-229,256-265`). It is not one of the five NVIDIA-GPU modes and
  should be handled by the final reachability/deletion decision.

## Tests reviewed and coverage gaps

Reviewed directly relevant portions of:

- `tests/test_p0_config.py`
- `tests/test_p2_cards.py`
- `tests/test_p2_engine.py`
- `tests/test_p3_ranges.py`
- `tests/test_p4_folded.py`
- `tests/test_p5_completion.py`
- `tests/test_p7_adapter.py`
- `tests/test_p7_postflop.py`
- `tests/test_p13a_gpu_rollout.py`
- `tests/test_p13b_integration.py`
- `tests/test_br_probe.py`
- `tests/test_compact_deck_geometry.py`
- `tests/test_history_capacity.py`
- `tests/test_storage_lifecycle_poker_inputs.py`
- `tests/test_aud26_no_host_sync.py`
- `tests/test_refactor_gpu_acceptance.py`

Coverage gaps:

1. No test rejects a neutral `post_P` summary allocation; the existing test
   checks only persistent cache bytes.
2. No test forces one dependency anchor across multiple queue packs and checks
   one summary computation/source row.
3. Existing completion/folder tests preserve v1 instead of enforcing v2.
4. No CUDA sync-debug test covers profile sampling, completion, showdown, or
   sealed learner partition forwarding.
5. No accepted-config test covers multiplayer `folder_beta_rank=0`.
6. CPU-heavy backend/oracle tests do not prove the NVIDIA-only production path.

## Complete file inventory

Fully reviewed in the assigned scope:

- `pbg/games/poker/__init__.py`
- `pbg/games/poker/adapter.py`
- `pbg/games/poker/bundle.py`
- `pbg/games/poker/cards.py`
- `pbg/games/poker/engine.py`
- `pbg/games/poker/folded.py`
- `pbg/games/poker/mc.py`
- `pbg/games/poker/model_inputs.py`
- `pbg/games/poker/partition_cache.py`
- `pbg/games/poker/partitions.py`
- `pbg/games/poker/policy_planning.py`
- `pbg/games/poker/ranges.py`
- `pbg/games/poker/rollout_backend.py`
- `pbg/games/poker/tensor_state.py`
- sibling `/home/pablob/poker-gpu-evaluator/docs/api.md` (read-only)

Cross-scope files inspected only to establish reachability or exact contracts:

- `pbg/core/rollout.py`
- `pbg/core/branching.py`
- `pbg/core/training.py`
- `pbg/eval/arena.py`
- `pbg/eval/br_probe.py`
- `pbg/eval/rao_blackwell_variance.py`
- `pbg/eval/branched_rao_blackwell_variance.py`
- `pbg/inference.py`
- `pbg/games/poker/model/capacity.py`
- `pbg/games/poker/model/preflop.py` (configuration/reachability only; model
  implementation remains assigned to the separate model batch)
