# Model architecture and performance audit

## Scope and method

This batch reviewed `docs/theory/MHA_model.md`, every Python file under
`pbg/games/poker/model/`, `pbg/games/poker/model_inputs.py`,
`pbg/games/poker/policy_planning.py`, the model construction/memory-plan portion
of `pbg/games/poker/bundle.py`, the production model dispatch in
`pbg/games/poker/rollout_backend.py`, and focused model/input tests. Findings are
static unless explicitly labeled measured. No production code or tests were
changed and no GPU benchmark was run in this batch.

The central curried contract is preserved: `PreflopNet.forward` emits one policy
parameter row and actor value for every one of the $R^2$ classes
(`pbg/games/poker/model/preflop.py:541-717`), while `PostflopNet.forward` emits
canonical $H=\binom{4R}{2}$ outputs after compact live-row execution and scatter
(`pbg/games/poker/model/postflop.py:672-916`). Neither network accepts a selected
dealt hand, and neither has a learned per-action Q head. Legal event composition
and masking remain in `PokerPolicyAdapter`, outside the neural heads. Fixed and
dynamic branched modes are both required by configuration to instantiate and
train public $T/C$ heads (`pbg/core/config.py:380-387`); their public-only
readout and sigmoid/exp transforms match the authority
(`pbg/games/poker/model/heads.py:147-169`).

## Findings

### AUD-MODEL-001 — Postflop admission charges turn/river width but executes flop width

- Severity: high
- Confidence: confirmed
- Performance classification: algorithmic regression and hard-memory-accounting defect; measured opportunity exists, current regression magnitude unmeasured
- Contract: STORE-005, STORE-017, STORE-030, STORE-031, STORE-032, HIST-PERF-01
- Modes affected: all five modes at postflop model calls; greatest exposure in both branched modes
- Evidence:
  - `pbg/games/poker/rollout_backend.py:1548`, `_policy_work_rows`
  - `pbg/games/poker/rollout_backend.py:1570-1583`, street-specific combo counts supplied to the physical planner
  - `pbg/games/poker/policy_planning.py:70-111`, admission and packing use those supplied counts
  - `pbg/games/poker/rollout_backend.py:1861-1872`, every `post_A` rollout call sets `compact_size=comb(D-3,2)`
  - `pbg/games/poker/rollout_backend.py:1974-1983`, every learner `post_A` call does the same
  - `pbg/games/poker/model/postflop.py:689-812`, activation and attention dimensions follow `mapping.compact_size`
  - `docs/todo/optimization.md:55-59,82-87`, measured fixed-beta evidence: 97% of representative decisions were turn/river but all paid the 2,554-token flop layout; 18.67 billion padded attention pairs were executed
- Current behavior: the host planner may admit and pack a turn/river batch using
  $\binom{D-4}{2}$ or $\binom{D-5}{2}$ rows, but the actual model call allocates
  and computes at $\binom{D-3}{2}$. Therefore the executed transient can exceed
  the hardware envelope that admitted it. Even where limits are generous, turn
  and river perform unnecessary quadratic attention work.
- Required behavior: retain one logical `post_A` partition, but make the
  game-owned physical forward plan and actual compact width agree. Street is a
  physical execution shape, not a new statistical partition; exact domains and
  loss normalization must remain unchanged.
- Consequence: the advertised token/pair ceiling is not a hard ceiling. This can
  over-admit activation memory and cause OOM, and it preserves a measured
  11–13% attention-pair optimization opportunity from the historical branching
  audit.
- Recommended disposition: fix. Either execute street-homogeneous physical
  subspans at their admitted widths or conservatively admit every current
  `post_A` call at flop width. Prefer the former after proving physical-plan
  invariance.
- Verification needed: assert for every physical call that recorded/admitted
  tokens and token pairs are at least the actual MHA shape; test mixed-street
  `post_A` logical partitions; benchmark equal decisions by street on an idle
  GPU.

### AUD-MODEL-002 — Mandatory folded-summary cache is excluded from the memory bound

- Severity: high
- Confidence: confirmed
- Performance classification: safety/accounting regression; byte impact is statically computable but was not measured here
- Contract: STORE-006, STORE-030, STORE-032
- Modes affected: all modes when a postflop decision follows a fold; all-live `post_P` is correctly exempt
- Evidence:
  - `pbg/games/poker/model/postflop.py:157-160`, `PostflopNet.requires_partition_summary = True`
  - `pbg/games/poker/rollout_backend.py:1700-1721`, every registered model creates a partition adapter/cache service
  - `pbg/games/poker/rollout_backend.py:1723-1750`, cache memory is counted only when `evaluator_summary_dim > 0`
  - `pbg/games/poker/bundle.py:669-680`, preflight uses the same `evaluator_summary_dim > 0` predicate
  - `pbg/games/poker/model/postflop.py:241-462`, semantic folder correction, beta, pressure, and tight fields are consumed independently of `evaluator_summary_dim`
  - `pbg/games/poker/rollout_backend.py:1955-1983`, folded immutable summary anchors are required whenever `live < P`
  - `tests/test_storage_lifecycle_poker_inputs.py:76-100`, coverage proves only that all-live `post_P` allocates no persistent summary owner
- Current behavior: production configurations set `evaluator_summary_dim: 0`, yet
  the model still mandatorily consumes evaluator contract-v1 folded summaries.
  The cache can therefore allocate on folds while both initial preflight and
  `partition_cache_memory_bound()` report zero bytes.
- Required behavior: memory-account summary anchors from the actual
  `requires_partition_summary`/partition schema and folder dependency, while
  retaining zero allocation for `post_P`.
- Consequence: guarded rollout and all-phase estimates omit a live persistent
  owner, undermining fail-before-allocation and potentially allowing allocator
  OOM.
- Recommended disposition: fix the materialization predicate and add a
  folded-path capacity test. Do not solve this by materializing neutral
  summaries for all-live nodes.
- Verification needed: force a fold, seal the summary anchor, compare reported
  capacity bytes with the cache's reserved high-water bytes, and repeat for
  reduced decks and each $P\in[2,6]$.

### AUD-MODEL-003 — Non-NVIDIA model execution is an active parallel implementation

- Severity: medium
- Confidence: confirmed
- Performance classification: production-surface drift, not a measured CUDA regression
- Contract: repository GPU-only requirement; production-only deletion requirement
- Modes affected: BR/inference directly; neural fallback is callable from every mode outside the main CUDA launcher
- Evidence:
  - `pbg/games/poker/model/primitives.py:206-229`, full explicit fp32 attention implementation
  - `pbg/games/poker/model/primitives.py:319-331`, dedicated MPS and generic non-CUDA dispatch
  - `pbg/games/poker/model_inputs.py:36-41`, separate CPU history-width behavior
  - `pbg/games/poker/model/incidence.py:126-138`, separate CPU ragged-size and validation behavior
  - `pbg/eval/arena.py:1,52-72`, active CPU arena builds the poker model on CPU and uses the CPU evaluator
  - `pbg/inference.py:205,261-264`, active CPU/MPS model construction and device contract
  - `tests/test_p6_preflop.py:68-84` and `tests/test_p7_postflop.py:68-84`, CPU forward/autocast are first-class tests
- Current behavior: CPU and MPS are not tiny incidental compatibility branches;
  they have separate attention, shape, validation, inference, and arena paths.
  In particular, the existing CPU arena is not a GPU implementation of the
  requested BR mode.
- Required behavior: the five core modes, including BR, run on NVIDIA GPUs.
  CPU may remain for inexpensive fixture/oracle logic only when it does not
  create a second implementation.
- Consequence: the repository carries parallel execution semantics and tests
  that can mask CUDA-only failures and consume maintenance/context budget.
- Recommended disposition: migrate BR/inference consumers to the CUDA model and
  evaluator path, then delete or archive MPS/CPU serving and model fallbacks.
  Retain small CPU mathematical oracle tests only if they share production
  formulas.
- Verification needed: end-to-end CUDA BR trace and a reachability pass after
  deleting non-CUDA entry points.

### AUD-MODEL-004 — Legacy and ablation configuration remains in the production model schema

- Severity: medium
- Confidence: confirmed
- Performance classification: stale surface; one field also triggers incorrect memory accounting through AUD-MODEL-002
- Contract: MATH-MODEL-001, MATH-MODEL-002, production-only deletion requirement
- Modes affected: configuration/checkpoint loading for all modes
- Evidence:
  - `pbg/games/poker/model/preflop.py:38`, `PokerModelConfig.n_buckets_line`
  - `pbg/games/poker/model/preflop.py:41`, `PokerModelConfig.evaluator_summary_dim`
  - `pbg/games/poker/model/preflop.py:99-105`, compatibility parser for removed `use_moment_heads`
  - repository search finds no production read of `n_buckets_line` outside validation
  - all checked-in production/training configurations that mention `evaluator_summary_dim` set it to zero; nonzero behavior is exercised only by `tests/test_p7_postflop.py`
  - `pbg/games/poker/model/postflop.py:56-110,184,701-745`, arbitrary opaque evaluator fields can enable an extra learned projection outside the named authoritative input channels
- Current behavior: `n_buckets_line` is a wholly dead field; false
  `use_moment_heads` is silently accepted as a legacy checkpoint/config shim;
  and `evaluator_summary_dim` exposes an unused arbitrary-summary ablation while
  the mandatory named folded fields bypass it.
- Required behavior: the production schema contains only inputs and heads used
  by the five modes and named by the architecture authority. Superseded fields
  and migration shims should not remain indefinitely.
- Consequence: stale configurations continue to validate, the model contract is
  broader than the authority, and `evaluator_summary_dim` is mistakenly reused
  as a cache-materialization signal.
- Recommended disposition: delete `n_buckets_line`, remove the
  `use_moment_heads` compatibility parser after explicitly rejecting/migrating
  old checkpoints, and remove the arbitrary summary projection unless the
  authoritative model document is deliberately extended.
- Verification needed: configuration inventory and checkpoint migration list;
  assert unknown removed keys fail validation.

### AUD-MODEL-005 — Test/diagnostic and backward-compatibility objects remain beside the sole production path

- Severity: low
- Confidence: confirmed by static references
- Performance classification: stale-code inventory; dense-beta fallback is a latent performance hazard
- Contract: production-only deletion requirement, MATH-MODEL-001, STORE-007
- Modes affected: none in their intended use; accidental invocation can affect postflop performance
- Evidence:
  - `pbg/games/poker/model/postflop.py:113-154`, `suit_row_permutation` and `permute_suit_representation`, referenced only by tests
  - `pbg/games/poker/model/incidence.py:225-238`, `permute_rows`, reachable only through those suit-ablation helpers
  - `pbg/games/poker/model/primitives.py:168-179`, `MHA._sdpa_attention`, referenced only by `tests/test_p6_primitives.py`
  - `pbg/games/poker/model/primitives.py:294-306`, `MHA.attention_weights`, referenced only by tests
  - `pbg/games/poker/model/heads.py:107-124`, `raise_tick_probs`, referenced only by tests (production uses `dlogistic_tick` through the adapter)
  - `pbg/games/poker/model/postflop.py:280-311`, legacy dense `pair_card_log_ratio` path can run an eigendecomposition during every forward when contract-v1 eigenfields are absent
  - `pbg/games/poker/model/postflop.py:373-376`, backward-compatible `folder_card_marginal` alias
- Current behavior: optional suit-consistency tooling, diagnostic attention
  materialization, a duplicate policy composer, and pre-contract folded-summary
  fallbacks remain importable in production modules.
- Required behavior: only five-mode NVIDIA production objects remain in the
  package; experiments and migration helpers belong under `archive/` or should
  be deleted once their migration window closes.
- Consequence: broader surface and a possibility that malformed/old evaluator
  output silently takes an expensive dense eigendecomposition path rather than
  failing at the facade.
- Recommended disposition: archive the suit ablation as a self-contained
  experiment if still scientifically useful; delete the diagnostic/duplicate
  helpers and legacy evaluator aliases after confirming no supported checkpoint
  depends on them; require contract-v1 beta factors at the boundary.
- Verification needed: repeat symbol reachability after tools/tests are
  classified; ensure the sibling evaluator binding always emits the required
  factor fields for positive `folder_beta_rank`.

### AUD-MODEL-006 — Native biased SDPA and direct-backward compatibility survive statically, but lack a current equal-work benchmark

- Severity: optimization
- Confidence: strong static preservation; performance unverified
- Performance classification: preserved implementation, no current-HEAD measurement
- Contract: HIST-PERF-01, STORE-031, STORE-036
- Modes affected: all training modes
- Evidence:
  - `pbg/games/poker/model/primitives.py:181-204`, native biased SDPA with fp32 additive reach prior
  - `pbg/games/poker/model/primitives.py:231-279`, bounded checkpointed exact fallback
  - `pbg/games/poker/model/preflop.py:623-652` and `postflop.py:783-812`, ordinary differentiable forward without model-side gradient copies
  - `tests/test_p6_primitives.py:72-118`, CUDA SDPA forward/backward precision and throughput coverage
  - `tests/test_aud26_no_host_sync.py:56-104`, static and CUDA sync-debug coverage for attention/compaction
  - `docs/audit/repo_audit_20260721/batches/02_historical_regression_contract.md`, no comparable post-refactor learner run exists
- Current behavior: the core model primitive introduced before/at `537700f` is
  still present, and the KKT change replaced obsolete V/U heads with public
  $T/C$ without adding a Q head. No static model-side change explains a backward
  regression. This does not prove throughput preservation.
- Required behavior: retain exact additive reach bias, content-only temperature,
  bounded activation work, AMP-safe accumulations, and direct flat-gradient
  backward compatibility.
- Consequence: no regression is established, but the requested backward/forward
  efficiency guarantee cannot be closed from code survival or microtests.
- Recommended disposition: benchmark, then gate. Reproduce the historical
  two-RTX-2080-Ti learner geometry and report cache build, forward, backward,
  synchronization, memory, and actual token-pair work separately.
- Verification needed: current-HEAD equal-shape GPU benchmark plus an assertion
  that configured `sdpa` never enters explicit/chunked fallback.

## Production object inventory

### Confirmed production-reachable and contract-aligned

- `PokerModelConfig`, `PokerPublicBatch`, `PackedPublicHistory`,
  `PreflopMasks`, `PostflopMasks`, `PolicyOutput`, `PreflopNet`, `PostflopNet`
- `TypeHead`, `RaiseMixtureHead`, `ValueHead`, `PublicBranchingHeads`,
  `dlogistic_tick`
- `LN`, `FFN`, `MHA` biased CUDA paths
- `IncidenceResidual`, `CompactMap`, `build_compact_map`, `compact_gather`,
  `compact_scatter`
- `DeviceHistoryMaterializer`
- `PlannedPolicyBatch`, `plan_policy_batches`
- `max_player_nodes`, `max_public_history_tokens`,
  `max_policy_sequence_tokens`

### Confirmed stale, test-only, ablation-only, or compatibility-only

- `PokerModelConfig.n_buckets_line`: dead configuration field
- false `use_moment_heads` parser: legacy compatibility shim
- `PokerModelConfig.evaluator_summary_dim` and generic `_summary_fields` learned
  projection: nonproduction ablation in the checked-in configurations
- `suit_row_permutation`, `permute_suit_representation`, `permute_rows`: optional
  suit-consistency ablation/test support
- `MHA._sdpa_attention`, `MHA.attention_weights`: test/diagnostic helpers
- `raise_tick_probs`: duplicate test-only policy composition
- dense-beta eigendecomposition and `folder_card_marginal` alias: evaluator
  contract compatibility paths
- MPS and generic CPU attention/model-width branches: parallel non-NVIDIA
  implementation

### Uncertainty and review limits

- This batch did not deeply audit `PokerPolicyAdapter` probability algebra or
  learner loss weighting; those belong to the math/loss batch. It inspected the
  adapter only to establish that legality is external and `dlogistic_tick` is
  production-reachable.
- The full sibling evaluator API was not reopened in this batch. The conclusion
  that dense beta and old folder aliases are compatibility-only follows the
  checked-in contract-v1 field consumers/tests and should be confirmed against
  `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md` before deletion.
- Public input tensors currently include `legal_action_mask` and `raise_ticks`
  as learned scalar tokens (`preflop.py:492-497`, `postflop.py:607-612`).
  `MHA_model.md` both lists legal status/raise ticks in public trajectory input
  and says legality masks/integer raise ticks are applied outside the model.
  Because that wording is internally ambiguous, this report does not classify
  their tokenization as a confirmed implementation defect; synthesis should
  request an authoritative clarification.

## Recommended order

1. Correct AUD-MODEL-001 before any sizable GPU run, because current admission
   can understate executed postflop attention work.
2. Correct AUD-MODEL-002 and prove folded-summary reserved-byte accounting.
3. Migrate BR to CUDA and retire the CPU/MPS production surface.
4. Remove stale schema fields and compatibility/ablation objects.
5. Benchmark the preserved SDPA/direct-backward path and then pursue
   street-homogeneous physical execution as the highest-evidence model-side
   optimization.
