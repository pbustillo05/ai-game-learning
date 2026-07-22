# Repository reachability, deletion, tests, configuration, tools, and documentation audit

## Scope, method, and conclusion

This batch is the whole-repository cleanup pass. It read batches 00--08, then
inspected every active Python/configuration/tool/documentation surface outside
`archive/` and `runs/`. Vendored/reference-paper trees were excluded. Static
evidence came from imports, direct calls, package entry points, configuration
loaders, CLI roots, tests, documentation links, and
`tools/audit_python_objects.py`; a name search alone was never treated as proof
of runtime reachability.

The repository does **not** yet meet the requested production-only surface.
The five semantic workflows are reachable only through four public concepts:
three training labels plus fixed/dynamic branching, and a `probe` modifier on
sequential RB. BR has no authoritative mathematical contract. Beside those
workflows, the active package still contains a CPU duplicate-deal arena, a
CPU/MPS inference product, two checkpoint-specific variance implementations,
legacy tree/completion/evaluator/checkpoint APIs, and numerous test-only or
uncalled objects. Active configs, tools, tests, installed commands, package
metadata, and documentation deliberately keep many of them alive.

The most dangerous cleanup issue is not merely dead code. `SampledTree` and its
post-hoc compact-topology conversion are currently production-reachable, but
they are a superseded compatibility representation forbidden by the storage
authority. They must be removed by migration, not blessed because imports find
them. Conversely, `occurrence_schema` is currently unused but represents the
authority-aligned direction; it should become the sole rollout schema rather
than simply be deleted.

No source, test, configuration, documentation, or user-owned TODO was changed.
This report is the only file written by this batch. No tests or GPU work were
run; the task was static classification and cleanup evidence.

## Classification vocabulary

- **Core production**: reached by normal, sequential RB, fixed branched RB, or
  KKT branched RB on NVIDIA GPUs.
- **BR-required**: reached only by the approximate BR train/evaluation workflow;
  retain pending an authoritative BR contract and CUDA-only acceptance.
- **Shared support**: operational support required by at least one core mode
  (for example checkpointing and the benchmark after repair).
- **Experiment-only**: reachable, but only through research launchers/configs
  outside the five workflows; move to `archive/` with its evidence.
- **Test-only**: called only by active tests; move the formula/fixture under
  `tests/` if still valuable, otherwise delete it and its stale test.
- **Stale/unreachable**: no active runtime caller; delete.
- **Compatibility-reachable**: called today only to preserve a superseded
  representation/schema/platform; migrate consumers, then delete.
- **Uncertain dynamic**: a protocol or `getattr` fallback prevents proof from
  static calls. With NLHE as the sole game, these should be made explicit and
  then reclassified.

## Findings

### AUD-RD-001 — Three complete non-core execution products remain active and installed

- Severity: high
- Confidence: confirmed
- Contract: NVIDIA-only requirement; sole-production-path requirement;
  `STORE-008`, `STORE-010`, `STORE-021`, `STORE-022`
- Modes affected: repository surface; BR/evaluation adjacent
- Evidence:
  - `pbg/eval/arena.py:1,52-85,187-225,264-299` is a CPU-only, serial,
    full-deck duplicate-deal evaluator.
  - `pbg/inference.py:1,189-235,256-273,275-299,468-576` is a 938-line
    CPU/Apple-MPS state, range, cache, model, and batching implementation; its
    hand and summary dictionaries have no release/eviction boundary.
  - `pbg/games/poker/engine.py:76-190,241-709`,
    `pbg/games/poker/ranges.py:23-337`, and
    `pbg/games/poker/adapter.py:472-710` form the parallel list-backed engine
    used by those products.
  - `pyproject.toml:17-21` installs both inference commands and the macOS extra;
    `docs/inference_on_mac.md` is a complete non-NVIDIA deployment guide.
  - `tests/test_inference_api.py`, `tests/test_p12_eval.py`,
    `tests/test_p2_engine.py`, `tests/test_p3_ranges.py`, and portions of
    `tests/test_p7_adapter.py` preserve these paths.
- Current behavior: users can run supported-looking CPU/MPS products with
  independent public-state, reach, folded-summary, and inference scheduling
  semantics. The CPU arena is also wrong for supported compact decks because
  it hardcodes the 1,326-hand standard-deck domain.
- Required behavior: the five production workflows use the tensor-native
  NVIDIA backend only. An inexpensive oracle may remain under tests or archive,
  not as an installed runtime.
- Consequence: two implementations can drift, CUDA breakage can be masked by
  CPU tests, and unbounded non-production ownership remains a package promise.
- Recommended disposition: archive `pbg/inference.py`,
  `pbg/inference_artifact.py`, both inference CLIs, `pbg/eval/arena.py`, their
  tests/docs/entry points, and the list-engine/range facade after any required
  pure test oracle is relocated. Move `_student_t_critical` from the arena to a
  tiny shared statistics helper because BR uses it.
- Verification needed: prove the four training workflows and BR do not import
  `PokerGame`, `RangeState`, or `NLHEEngine`; add CUDA-native evaluation only if
  checkpoint-versus-checkpoint evaluation is explicitly retained.

### AUD-RD-002 — Checkpoint-specific variance implementations remain parallel active algorithms

- Severity: high
- Confidence: confirmed
- Contract: `STORE-001`, `STORE-007`, `STORE-014` through `STORE-022`;
  production-only requirement
- Modes affected: none; historical sequential/branched variance experiments
- Evidence:
  - `pbg/eval/rao_blackwell_variance.py` manually reconstructs completion,
    JVP rows, dense coefficients, and target moments for one HU 32 bb study.
  - `pbg/eval/branched_rao_blackwell_variance.py` calls raw branched rollout,
    copies whole topology to CPU, and independently reconstructs branched
    moments, leave-one-out derivatives, and functional coefficients.
  - Their CLI modules expose separate schemas and CPU fallback; sequential CLI
    choices at `pbg/cli/rao_blackwell_variance.py:307-347` include advertised
    phases that always call `_unimplemented_phase`.
  - `configs/experiments/` contains 30 active YAMLs: three custom variance
    schemas and 27 historical optimizer/tuning/smoke presets. Twenty-four still
    contain the retired `use_moment_heads: false` key.
  - `tools/run-rao-blackwell-variance`,
    `tools/run-branched-rao-blackwell-variance`, `tools/branching-sweep.py`,
    `tools/run-branching-sweep-resume.sh`, and
    `tools/plot-branching-geometry.py` are experiment-specific launch/analysis
    surfaces.
- Current behavior: roughly 2,300 lines of parallel variance mathematics plus
  geometry-censoring utilities remain under active package/tool/config paths.
- Required behavior: historical experiments live with historical run evidence;
  future probes consume sealed sole-path production outputs and do not duplicate
  target algorithms.
- Consequence: the repository can produce results from stale representations
  and formulas while appearing to exercise production.
- Recommended disposition: move both variance module/CLI/config/test/launcher
  bundles and the fixed-beta geometry tools into a dated archive beside their
  run evidence. Move all 27 tuning/probe configs to the corresponding run or
  archive location. General plotting utilities may remain only if explicitly
  treated as operational analysis support rather than production code.
- Verification needed: package import and entry-point tests find no variance
  module under `pbg`; archived manifests retain enough paths to interpret old
  results.

### AUD-RD-003 — Production still depends on a forbidden legacy tree handoff

- Severity: blocker
- Confidence: confirmed
- Contract: `STORE-012`, `STORE-013`, `STORE-014`, `STORE-021`, `STORE-030`;
  `AUD-RSQ-001`, `AUD-RSQ-002`
- Modes affected: normal, sequential RB, fixed branched RB, KKT branched RB,
  and BR through sequential storage
- Evidence:
  - `pbg/core/tree.py:355-587`, `PagedTreeSource.finalize`, seals the full
    `SampledTree` node/edge representation.
  - `pbg/core/rollout.py:879-952` and `pbg/core/branching.py:888-1013` return it
    alongside policy, leaf, and input owners.
  - `pbg/core/training.py:1725-1826` only afterward builds
    `SequentialDecisionCSR` or `BranchedTreeSpan`, allocating whole-tree reorder
    maps outside the hard seal transaction.
  - `docs/architecture.md:71-75` and `docs/core_algorithms.md:52-59` claim
    direct exact compact sealing, contradicting this control flow.
- Current behavior: `SampledTree`, `RolloutDecisionRows`, `StoppedLeafRows`,
  and `PagedTreeSource` are core-reachable but are compatibility owners, not an
  authority to preserve.
- Required behavior: reserve and seal directly into the immutable
  decision-only sequential CSR or six-column branched span, then delete the
  larger handoff and its conversion scratch.
- Consequence: simple unused-code deletion would miss the principal stale path;
  documentation and migration gates currently give false confidence that it is
  already gone.
- Recommended disposition: migrate all training/BR/evaluation consumers to
  direct compact seal, then delete `SampledTree`, compatibility decision/leaf
  wrappers, full legacy node/edge schema, and post-hoc reorder conversion.
- Verification needed: allocation-hook tests prove no full-tree destination or
  whole-tree conversion maps exist, including multi-page and refill cases.

### AUD-RD-004 — Test-only and genuinely dead objects remain in production modules

- Severity: medium
- Confidence: confirmed for the objects listed as immediate deletion
- Contract: production-only/no-parallel-API requirement
- Modes affected: none
- Evidence:
  - `pbg/core/distributed.py:11-72` is imported only by
    `tests/test_execution_foundations.py`; production uses learner log masses and
    exact alphas.
  - `pbg/core/logging_util.py:10-41` is imported only by
    `tests/test_p0_config.py`; production writes JSON directly.
  - Static call searches find no `pbg` callers for
    `ReusableStagingSlabs`, `logical_categorical`,
    `allocate_statistical_counts`, `DiagnosticTreeBuilder`,
    `build_decision_alias`, `occurrence_schema`,
    `_memory_operating_ceiling`, `_sealed_input_block`,
    `DECK_GEOMETRY_VERSION`, `raise_tick_probs`,
    `permute_suit_representation`, or `assert_poker_storage_widths`.
  - `CompletionStats` is constructed only by unused
    `BatchedCompletionStats.row`; `ModeLoss`,
    `distributed_fixed_rollout_update`, the four algorithm-style target aliases,
    and three observability recording methods have no runtime caller.
- Current behavior: stale formula APIs and test conveniences look like supported
  production objects. Several have names close to the actual production
  implementation and are especially easy to reuse accidentally.
- Required behavior: useful test formulas live under tests; runtime modules
  expose only the sole implementation.
- Consequence: audit ambiguity and continued maintenance of alternate
  normalization, logging, staging, alias, and model-composition paths.
- Recommended disposition: apply the explicit per-object table below. The one
  exception is `occurrence_schema`: wire it as the sole immutable rollout schema
  in the storage repair, rather than deleting the authority-aligned definition
  while retaining the handwritten schema.
- Verification needed: repeat symbol reachability after migration; add an
  import-surface gate based on actual console roots, not substring bans.

### AUD-RD-005 — Legacy compatibility is an active product feature, contrary to the sole-path rule

- Severity: high
- Confidence: confirmed
- Contract: sole-production-path/no-indefinite-compatibility requirement;
  `STORE-001`, `STORE-007`, `MATH-MODEL-002`
- Modes affected: all modes through config/checkpoint/evaluator loading and
  fixed branched benchmarking
- Evidence:
  - `pbg/core/checkpointing.py:24-120` accepts checkpoint schemas v2--v5 and
    migrates retired allocation metadata; `:315-394` retains legacy config/run
    fingerprint comparisons; `:425-430` adds private aliases that training still
    imports.
  - `tests/test_phase3_migration.py:32-76` deliberately requires v2/v3/v4
    compatibility, and `docs/training_and_evaluation.md:94-102` advertises it.
  - `pbg/games/poker/model/preflop.py:95-111` accepts
    `use_moment_heads: false` and translates `p_max`; model fields
    `n_buckets_line` and `evaluator_summary_dim` preserve dead/ablation schema.
  - `pbg/games/poker/mc.py:21-60` and `folded.py:20-104` accept evaluator
    contract v1 even though the current sibling authority requires contract v2;
    v1 fixtures in `tests/test_p4_folded.py`, `test_p5_completion.py`, and
    `test_p12_eval.py` enforce this.
  - `pbg/benchmark.py:202-225` permits missing T/C heads and times randomly
    initialized replacements; `tests/test_aud32_performance_gates.py:171-227`
    requires that behavior.
  - `pbg/games/poker/model/postflop.py:280-311,373-376` keeps the dense-beta
    eigendecomposition and `folder_card_marginal` aliases for old inputs.
- Current behavior: current launch/config paths are strict only after several
  silent legacy translations, while checkpoints/evaluator/model inputs accept
  multiple superseded contracts.
- Required behavior: migrate retained production artifacts once, then require
  one current config, checkpoint, evaluator, and folded-summary schema. Any
  migration window must be explicitly approved and time bounded.
- Consequence: stale representations constrain refactors and can silently enter
  timed/current runs.
- Recommended disposition: inventory checkpoints intentionally required for
  resume; convert them offline if necessary; then delete old schema constants,
  migrations, fingerprint compatibility, private aliases, parser shims, v1
  evaluator acceptance, model fallbacks, and permissive benchmark load.
- Verification needed: strict current-schema negative tests and one documented
  offline conversion procedure if historical artifacts must remain readable.

### AUD-RD-006 — The active tests preserve obsolete platforms, algorithms, and false architecture gates

- Severity: high
- Confidence: confirmed
- Contract: structural tests `TEST-STORE-001` through `TEST-STORE-008`;
  production-only requirement
- Modes affected: all
- Evidence:
  - `tests/test_storage_lifecycle_migration_gates.py:27-100` bans a short list of
    retired names and `torch.cat`, but does not reject `SampledTree`, post-hoc
    conversion, schema-on-first-write, unaccounted queues/seal, preflop
    exact-width model rows, or unbounded branched child waves. It therefore
    passes while batches 04/05 show the authoritative lifecycle is violated.
  - `tests/test_execution_foundations.py:124-137,462-475` tests alternate
    distributed helpers and test-only staging rather than proving the production
    learner/rollout use them.
  - `tests/test_p8_tree.py:27-43,192-209` entrenches
    `DiagnosticTreeBuilder`, `AliasTable`, and `build_decision_alias`.
  - End-to-end CPU tests preserve the unsupported training/BR/benchmark paths:
    `test_br_probe.py`, `test_aud32_performance_gates.py`, and large portions of
    model/backend tests.
  - `test_inference_api.py`, `test_p12_eval.py`, all three variance test files,
    and variance CLI tests exist solely for surfaces that should move to
    archive.
  - No tests establish direct compact seal, source+destination hard admission,
    queue byte accounting, fixed-beta scientific safeguard, total occurrence
    lifetime, physical-plan invariance, CUDA BR, or five-mode benchmark identity.
- Current behavior: test count and migration-gate names overstate readiness;
  some tests actively prevent removing stale objects.
- Required behavior: tests assert authority-defined ownership and lifecycle on
  the real production objects, and every one of the five semantic workflows has
  NVIDIA acceptance.
- Consequence: cleanup and architectural regressions can both be blocked by the
  wrong tests while the intended contracts remain untested.
- Recommended disposition: archive/delete tests with retired products; move
  useful pure formulas under tests; rewrite evaluator v2 and CUDA-only tests;
  replace textual migration gates with structural allocation/ownership tests.
- Verification needed: a traceability matrix from every `TEST-STORE-*` contract
  to an actual production test and one tiny GPU gate for each semantic mode.

### AUD-RD-007 — Active documentation describes three modes and contradicts both policy and implementation

- Severity: high
- Confidence: confirmed
- Contract: five-mode audit requirement; authority/drift-prevention rule
- Modes affected: all, especially BR and KKT
- Evidence:
  - `README.md:11-12,82-90`, `CONTRACTS.md:27,64,240`,
    `docs/architecture.md:5-13`, `docs/core_algorithms.md:1-14`,
    `docs/file_reference.md:8,95`, `docs/README.md:11-12`, and
    `benchmarks/README.md:3-8` present a three-mode product. Fixed and KKT are
    not consistently named as distinct production semantics; BR is described as
    outside the release gate.
  - `docs/architecture.md:52-75` says production owns reusable staging, charges
    queue storage, and seals compact topology directly. Batch 04 proves staging
    is test-only, queues bypass accounting, and a full `SampledTree` is sealed
    first.
  - `CONTRACTS.md:98-128,142-159,198-216` similarly claims every occurrence is
    DONE, all execution allocation is charged before mutation, target builders
    stream from exact storage, and hard branched continuation stops cleanly;
    batches 04/05 identify exceptions to each claim.
  - `docs/poker.md:52-57` says folded summaries live once at sparse dependency
    anchors; batch 06 shows they are duplicated per pack then deduplicated at
    seal. `:75` calls contract v1 installed even though current authority is v2.
  - `docs/file_reference.md:31,40-43` labels test-only `distributed.py` and
    `logging_util.py` as active production and describes tree/target boundaries
    more strongly than implemented.
  - `HARDWARE.md:25,40` permits CPU rollout workers and uses CPU correctness as
    a launch gate, contrary to the NVIDIA-only requirement. It also treats
    scheduler integration as configurable although `HardwareProfileConfig`
    scheduler fields have no runtime consumer.
- Current behavior: current-looking docs describe the desired post-refactor
  architecture as already implemented and normalize unsupported CPU/MPS and
  three-mode scope.
- Required behavior: authoritative math/pseudocode remain distinct from an
  honest implementation-status guide; public docs name the five semantics and
  mark unresolved BR authority and storage drift explicitly.
- Consequence: reviewers and implementers can use false implementation claims
  to approve drift or preserve stale code.
- Recommended disposition: after remediation, rewrite README/contracts/
  architecture/algorithm/file-reference/poker/training docs from the verified
  mode traces. Until then, mark violated claims as target-state rather than
  current-state. Move `docs/inference_on_mac.md` with the archived subsystem.
- Verification needed: documentation links/commands test plus a generated
  five-mode/config/object matrix checked against the strict loader.

### AUD-RD-008 — Operational docs and benchmark workloads contain broken paths and retired commands

- Severity: high
- Confidence: confirmed
- Contract: evidence integrity; strict configuration contract
- Modes affected: training and performance validation
- Evidence:
  - `docs/training_and_evaluation.md:72-79` names missing
    `configs/train/nlhe_hu_32bb_sequential_rb_candidate.yaml` and
    `configs/hardware/local_2x2080ti_12h.yaml`.
  - `docs/training_and_evaluation.md:175-181` and
    `benchmarks/workloads.yaml:4-17` name missing
    `configs/benchmarks/nlhe_hu_short.yaml`.
  - `benchmarks/workloads.yaml:7-13` passes retired `--n-trees` and
    `--node-budget`; `pbg/cli/benchmark.py` exposes neither. It also names
    `readiness_gate: AUD-005`, which is not a stable current audit identifier.
  - `docs/file_reference.md:111` names missing `tests/test_p9_branching.py` and
    `:126-127` names archive paths that need not match the current tree.
  - `docs/references/CONFIG_COMPARISON.md:5-7` is repository-authored rather
    than vendored reference material, but its subject config is missing and its
    final preflight analysis describes the superseded capacity-sized storage
    path. It should be archived with the historical candidate study rather than
    indexed as current configuration evidence.
  - `docs/README.md:31-39` points to archived evaluator IO as binding while
    `AGENTS.md` points to a missing sibling `docs/api_pbg.md`; the actual sibling
    authority inspected by batch 06 is `docs/api.md`.
  - `docs/training_and_evaluation.md:16` says the launcher verifies contract v1,
    while `tools/poker-python:20` correctly requires `CONTRACT_VERSION == "2"`.
- Current behavior: copy-pasted official commands fail or invoke retired
  configuration concepts; benchmark coverage cannot be executed as documented.
- Required behavior: every active command/path loads and dry-runs under the
  current strict schema; benchmark workloads cover all five semantics and use
  current GPU presets.
- Consequence: performance regression evidence cannot be reproduced from the
  checked-in workload matrix.
- Recommended disposition: delete the obsolete workload matrix and recreate it
  from canonical five-mode configs after benchmark repair; fix or remove every
  broken operational example.
- Verification needed: CI link/path check and parser-only execution of every
  documented command/config without allocating a model.

### AUD-RD-009 — Configuration, package, and tool surfaces are larger and less canonical than production

- Severity: medium
- Confidence: confirmed
- Contract: strict/minimal configuration; production-only requirement
- Modes affected: all
- Evidence:
  - No committed YAML represents normal, KKT, or BR; fixed and sequential RB are
    the only canonical presets (`configs/train/`).
  - `configs/nlhe6_h100.yaml` duplicates `configs/train/nlhe6.yaml` but changes
    minibatch size from 128 to 6; tests still consume the root-level duplicate.
  - Six of eight hardware files are historical optimizer/SDPA probes.
  - `HardwareProfileConfig.scheduler`, `.worker_count`, and `.slurm`
    (`pbg/core/config.py:132-152`) are validated but never read by runtime;
    `configs/hardware/quest_slurm.yaml` suggests launcher behavior that does not
    exist.
  - `pbg/benchmark.py:28,68-100` supports only three labels, forces generic
    branched to fixed, and omits BR; comparison can silently omit missing modes
    or metrics (`:123-129,430-461`).
  - `pyproject.toml:8-21` describes generic PBGs, installs `pytest` as a runtime
    dependency, and exposes unsupported inference commands/macOS support.
  - `raise_distribution.html` is a generated top-level artifact; its generator
    is a model visualization experiment, not a production entry point.
- Current behavior: users must mutate configs to reach three of five semantics,
  while dead hardware/config/package features look supported.
- Required behavior: one canonical validated NVIDIA preset (or one tested typed
  generator) per semantic workflow, one current hardware schema, strict
  five-mode benchmark identity, and deployment dependencies only.
- Consequence: unsupported combinations and misleading benchmark comparisons
  are easy to create.
- Recommended disposition: add canonical normal/KKT/BR configs after BR
  authority; choose one H100/fixed preset; archive probe profiles/configs and
  generated visualization; remove dead scheduler fields, inference scripts/mac
  extra, and move `pytest` to a development extra.
- Verification needed: load/classify every retained YAML as exactly one semantic
  workflow and assert package console scripts equal the approved production
  surface.

### AUD-RD-010 — Historical ledgers and TODOs remain mixed into the active guidance surface

- Severity: medium
- Confidence: confirmed
- Contract: authority hierarchy; production-only active documentation
- Modes affected: audit/maintenance
- Evidence:
  - `IMPLEMENTATION_NOTES.md:22-193` records retired synthetic games, CPU
    engines, old P09/P10 paths, old evaluator facades, and legacy configs. The
    root README correctly calls it historical, but it remains top-level.
  - `NOTATION.md:268-340` describes three training labels, CPU `RangeState`, and
    old evaluator/archive correspondence alongside mathematical notation.
  - `TODO.md` points to BR probe notes, while many `docs/todo/` files describe
    experiments or pre-refactor paths. `docs/todo/optimization.md` remains
    useful only after its live ideas are translated to current owners, as batch
    02 explains.
  - Numerous older reports remain directly under `docs/audit/` despite
    `docs/file_reference.md` saying the directory contains current readiness
    only; several document now-superseded architectures or missing artifacts.
- Current behavior: historical evidence is valuable, but directory placement
  makes it look like current implementation authority.
- Required behavior: only current contracts/readiness stay in active docs;
  historical ledgers, concluded experiment plans, superseded audits, and old
  optimization file/line mappings live under dated archive directories.
- Consequence: context and searches surface obsolete guidance beside the
  authoritative pseudocode.
- Recommended disposition: preserve rather than delete evidence, but move
  `IMPLEMENTATION_NOTES.md`, concluded TODOs, superseded audit generations, and
  experiment notes into `archive/`. Keep a rewritten optimization backlog that
  references only current owners and measured/open ideas from batch 02.
- Verification needed: documentation index labels every retained file as
  authority, current implementation guide, open work, or history.

## Explicit Python-object disposition inventory

The following table consolidates every suspect named by batches 03--08 plus
additional static candidates found here. Line numbers identify the definition or
owning surface, not every method in a grouped class.

### Immediate delete, archive, or move to tests

| File/object | Evidence-based classification | Disposition |
|---|---|---|
| `pbg/core/distributed.py`: `WeightedDecisionStats`, `weighted_decision_stats`, `global_mass_objective`, `stratum_objective_scale` | Test-only alternate objective helpers; no production import | Move any useful direct formula into `tests/`; delete module |
| `pbg/core/logging_util.py`: `MetricsLogger` | Test-only; production has separate metrics writer | Move fixture to tests/delete module |
| `pbg/core/learner.py:153` `allocate_statistical_counts` | Test-only predecessor of production `allocate_fixed_statistical_counts` | Delete and update test |
| `pbg/core/learner.py:878` `ModeLoss`; `:1483` `distributed_fixed_rollout_update` | Uncalled compatibility aliases | Delete |
| `pbg/core/targets.py:748-751` `BuildNormalTargets`, `BuildSequentialRBTargets`, `BuildBranchedRBTargets`, `FinalizeTrainingStorage` | Uncalled design-name aliases | Delete; docs use canonical snake-case names |
| `pbg/core/execution.py:782` `ReusableStagingSlabs` | Test-only despite production needing a real bounded staging owner | Delete test object; implement the authoritative production owner separately |
| `pbg/core/execution.py:928` `logical_categorical` | Test-only mutable/generic RNG helper; production uses keyed valid-probability routine | Delete/update RNG tests |
| `pbg/core/tree.py:240` `DiagnosticTreeBuilder` | Tests/variance diagnostics only, Python/CPU construction | Archive with diagnostics or move a tiny fixture to tests |
| `pbg/core/tree.py:595,673` `AliasTable`, `build_decision_alias` | Test-only legacy tree sampler; production builds immutable aliases in targets | Delete and update legacy tree tests |
| `pbg/core/completion.py:19` `CompletionStats`; `BatchedCompletionStats.row` | No consumer outside their mutual construction | Delete |
| `pbg/core/training.py:832` `_memory_operating_ceiling`; `:1449` `_sealed_input_block` | No caller | Delete, unless the former is deliberately wired into the new controller contract |
| `pbg/core/observability.py:235,300,316` `record_cache`, `record_completion`, `record_rank_wait` | Test-only/zero-valued metrics; no production call | Delete methods/fields and update metrics test |
| `pbg/games/poker/cards.py:35` `DECK_GEOMETRY_VERSION`; `:207` `STANDARD_DECK` | No runtime caller (`STANDARD_DECK` no Python caller at all) | Delete |
| Module-level `class_of_hand`, `class_members`, `class_size`, `suit_perm_apply` | Standard-deck test/oracle helpers only | Move compact/general assertions to `DeckGeometry` tests; delete globals |
| Module-level `hand_cards`, `board_conflict_mask`, `sample_uniform_complement` | Non-test consumers are stale CPU arena/game; geometry methods cover production | Delete after arena/game archival |
| `pbg/games/poker/model/heads.py:107` `raise_tick_probs` | Duplicate test-only composer; production uses `dlogistic_tick`/adapter | Delete/update test |
| `postflop.py:113-154` `suit_row_permutation`, `permute_suit_representation`; `incidence.py:225` `permute_rows` | Suit-ablation/test-only chain | Archive ablation or delete chain/tests |
| `MHA._sdpa_attention`, `MHA.attention_weights` | Test/diagnostic-only model methods | Delete or move test calculation under tests |
| `pbg/games/poker/partitions.py:150` `assert_poker_storage_widths` | Test-only wrapper | Inline assertions in test/delete |
| `pbg/games/poker/rollout_backend.py:2590` `capture_probe_leaf` | No lexical caller; intended dynamic use was not found | **Uncertain dynamic**: confirm probe hook resolution, then either wire explicitly as BR-required or delete and flag BR fold capture bug |
| `pbg/cli/branched_rao_blackwell_variance.py:51` `_load_geometry` | No caller even inside experiment | Delete with archived experiment |
| `pbg/eval/br_probe.py` unused imports `DrawPurpose`, `PROBE_FOLD`, `sequential_leaf_logical_rng_keys` | No use in module | Delete imports |

### Archive complete non-core modules

| Module and objects | Classification | Disposition |
|---|---|---|
| `pbg/eval/arena.py`: `ArenaReport`, `_poker_config`, `_build_model`, `_load_model`, `_profile_runouts`, `_linear_ranges`, `_policy_probabilities`, `_uniform_for`, `_draw_event`, `_settle_showdown`, `_replay`, `arena` | CPU-only/full-deck evaluation | Archive all; relocate `_student_t_critical` for BR |
| `pbg/inference.py`: `HandHandle`, `PokerAction`, `Decision`, `FullStrategy`, `_FoldRecord`, `_HandState`, `parse_card`, `_move_public`, `_move_summary`, `PokerInferenceSession` | CPU/MPS serving outside five workflows; `seat_values_bb` fields never populated | Archive all |
| `pbg/inference_artifact.py`: `INFERENCE_ARTIFACT_SCHEMA_VERSION`, `sha256_file`, `_validate_artifact`, `load_inference_source`, `export_inference_artifact` | Inference-only and legacy-checkpoint coupled | Archive all |
| `pbg/cli/benchmark_inference.py`: `_prepare_hands`, `main`; `pbg/cli/export_inference.py:main` | Inference-only installed CLIs | Archive/remove scripts |
| `pbg/eval/rao_blackwell_variance.py`: all top-level objects from `SCHEMA_VERSION` through `ensure_manifest` | Checkpoint-specific experiment/parallel targets | Archive complete module |
| `pbg/eval/branched_rao_blackwell_variance.py`: all top-level objects from `SCHEMA_VERSION` through `variance_table`, including `BranchedVarianceRunner` | Checkpoint-1110 experiment/parallel targets | Archive complete module |
| Both variance CLI modules: every top-level helper and `main` | Experiment-only, CPU fallback, incomplete phases | Archive complete modules |
| `pbg/core/branching.py:26,53` `BranchingCutoffReached`, `BranchingDiagnosticLimits` | Evaluation/geometry-censoring only; not a production memory safeguard | Move with archived branching experiments; remove from core sampler API |

### Compatibility-reachable: migrate first, then delete

| Object/surface | Current reachability | Required final disposition |
|---|---|---|
| `SampledTree`, `PagedTreeSource` full node/edge handoff, `RolloutDecisionRows`, `StoppedLeafRows` | All training/BR rollouts | Direct compact seal, then delete compatibility representation/API |
| `PagedRowAccumulator` | Production policy/action/leaf streams | Replace schema-on-first-write with immutable schema-bound arenas, then delete |
| `ReachView`, `MCCompletion.complete`, list-engine methods of `PokerGPUCompletion` | Old evaluator/tests; tensor production subclasses the class | Narrow tensor completion boundary; delete old mutable-tree/list-engine API |
| `PokerCompletionService.complete_selected_keyed` | Only `tests/test_br_probe.py`; production BR computes all players | Build compact selected-owner sealed completion, then delete helper |
| `PublicHistoryEvent`, `_State`, `StateTable`, `NLHEEngine`, module `assemble_showdown_payoff` | CPU arena/inference/oracle/tests | Archive implementation; keep only minimal test oracle if explicitly useful |
| Entire `ranges.py`: `FoldedPlayer`, `RangeState` and methods | `PokerGame`, inference, tests | Archive/delete with CPU products |
| `ChanceDraw`, `PokerGame`; `PokerModelBundle.forward` mixed CPU route | CPU arena/inference/tests | Archive/delete after consumers move |
| Non-keyed `RolloutBackend.sample_profiles`, `.sample_events`, `.sampled_terminal_reward` and poker implementations | Generic `getattr` fallback/test doubles; poker production exposes keyed methods | Remove fallback and protocol members for sole NLHE keyed path; update test doubles |
| `PokerRolloutBackend.policy_forward` / `_policy_forward_direct` | Tests only; scheduler uses family actual-cost route | Delete after equivalent tensor-boundary tests use production partition call |
| `PokerModelConfig.n_buckets_line`, `evaluator_summary_dim`; false `use_moment_heads`; `p_max` alias | Dead/ablation/legacy config | Migrate retained configs/checkpoints; reject/delete fields and parsers |
| Dense beta eigendecomposition; `folder_card_marginal` alias | Contract-v1 folded-summary fallback | Require evaluator v2 at facade; delete fallbacks |
| CPU/MPS/generic attention and ragged-width branches | CPU/MPS serving and first-class CPU model tests | Delete after non-NVIDIA products archive; retain device-agnostic formulas only where line cost is negligible |
| Checkpoint v2/v3/v4 constants/migration, legacy fingerprints, private function aliases | Resume compatibility | Offline-convert approved checkpoints, then require v5/current and delete compatibility |
| Benchmark missing-T/C `strict=False` load | Historical sequential checkpoint comparison | Require current checkpoint or explicit offline conversion; delete fallback |

### Retain/refactor production or BR-required objects

| Object/surface | Classification | Required change |
|---|---|---|
| `pbg/eval/br_probe.py`: `BRProbeReport`, `_sha256`, `_selected_reward`, `evaluate_br_probe`; `pbg/cli/br_eval.py:main` | BR-required | Retain pending BR authority; require CUDA and correct work/provenance counters |
| `pbg/benchmark.py` and `pbg/cli/benchmark.py` objects | Shared production regression support | Retain, but implement strict five-semantic-mode identity, required metrics, current checkpoints, and CUDA-only execution |
| `occurrence_schema` | Currently dead but authority-aligned | Make it the single immutable allocation/forecast/validation schema; delete handwritten duplicate |
| `PokerConfig`, `PotStrata`, action/street/status constants actually imported by tensor backend | Core production | Retain; split them from archived list engine |
| `DeckGeometry`, evaluator combo validation, tensor state/history, natural partitions, model classes/heads, keyed evaluator facades | Core production | Retain subject to batches 06/07 fixes |
| `CompletionSpec`, `BatchedCompletionStats.for_learning`, tensor completion service | Core production | Retain/narrow after old completion boundary deletion |

## File/configuration/tool disposition

### Retain after repair

- `pbg/config.py`, `pbg/runtime.py`, `pbg/cli/train.py` and core/tensor poker
  production modules, after GPU-only and sole-schema migration.
- `pbg/eval/br_probe.py` and `pbg/cli/br_eval.py`, only after BR authority and
  CUDA acceptance.
- `pbg/benchmark.py`, `pbg/cli/benchmark.py`, `benchmarks/`, only after a strict
  executable five-mode redesign.
- `tools/poker-python`, `tools/run-training.sh`, `tools/run-status.sh` as shared
  GPU operational support. `run-training.sh` should be checked against the
  terminal-failure/checkpoint contract but is not itself stale.
- Canonical current GPU hardware profiles and one validated config/preset per
  semantic workflow.

### Move to archive or delete

- CPU/MPS inference, artifact, arena, list-engine/range product and their docs,
  tests, entry points, and macOS package extra.
- Both variance experiment bundles, all 30 experiment YAMLs after attaching
  them to their evidence, their tests, and their launchers.
- Branching sweep/geometry plot tools and evaluation-only cutoff classes.
- Six optimizer/SDPA probe hardware profiles; duplicate root-level
  `configs/nlhe6_h100.yaml`; generated `raise_distribution.html` and its viewer
  unless explicitly retained as archived design evidence.
- `IMPLEMENTATION_NOTES.md`, completed experimental TODOs, and superseded audit
  generations should be moved, not destroyed, because they are historical
  evidence.

### Rewrite, not archive

- `README.md`, `CONTRACTS.md`, `HARDWARE.md`, `NOTATION.md`, `docs/README.md`,
  `architecture.md`, `core_algorithms.md`, `file_reference.md`, `poker.md`, and
  `training_and_evaluation.md` are needed current guides, but their claims must
  be reconciled with verified implementation after repair.
- `docs/todo/optimization.md` should become a current-owner backlog retaining
  the still-live measured ideas from batch 02; stale line references and
  censored-tree suggestions belong in archive.
- `benchmarks/workloads.yaml` must be rebuilt rather than patched around missing
  configs and retired CLI flags.

## Stale-test disposition map

| Test surface | Why stale or misleading | Disposition |
|---|---|---|
| `test_inference_api.py`, `test_p12_eval.py` | Preserve non-NVIDIA products | Archive with product |
| `test_rao_blackwell_variance.py`, `test_rao_blackwell_variance_cli.py`, `test_branched_rao_blackwell_variance.py` | Preserve historical parallel algorithms | Archive with experiment |
| `test_p3_ranges.py`; CPU-game portions of `test_p2_engine.py`, `test_p7_adapter.py`, `test_p5_completion.py` | Preserve list engine/range/old completion APIs | Move minimal tensor parity oracle under tests if desired; remove runtime API expectations |
| `test_p8_tree.py` builder/alias cases | Preserve legacy diagnostic tree/alias objects | Replace with direct compact-seal structural tests |
| `test_execution_foundations.py` alternate mass/staging/RNG cases | Exercise objects production does not use | Point tests at real learner/scheduler owners or move formula locally |
| `test_phase3_migration.py` legacy checkpoint case | Requires indefinite v2--v4 migration | Replace with current-schema strictness after approved conversion window |
| `test_p4_folded.py`, `test_p5_completion.py`, `test_p12_eval.py` v1 fixtures | Require obsolete evaluator contract | Rewrite against exact v2 contract and reject v1 |
| `test_aud32_performance_gates.py` CPU/three-mode/missing-head cases | Preserves unsupported benchmark and permissive checkpoints | Rewrite strict five-mode CUDA artifact/gate tests |
| `test_br_probe.py` | Valuable semantics but end-to-end CPU-only and calls stale selected completion | Retain pure equations; add CUDA train/eval/resume and compact selected completion |
| `test_storage_lifecycle_migration_gates.py` | Textual absence tests pass despite central lifecycle drift | Replace with real allocation, ownership, direct-seal, and last-consumer assertions |

## Broken-reference and documentation coverage summary

Confirmed broken or stale active references include:

- three missing configs used by official training/benchmark commands;
- one missing candidate config used by the repository-authored literature
  comparison under `docs/references/`;
- retired `--n-trees` and `--node-budget` benchmark flags;
- missing `tests/test_p9_branching.py`;
- contradictory evaluator contract versions and authority paths;
- active docs/directories describing archived/nonexistent config categories;
- three-mode headings throughout public docs despite five requested semantics;
- claims that production uses staging, exact compact seal, fully accounted
  queues, sparse-once summaries, and global terminal failure behavior that are
  disproved by batches 04--06.

A repository link/path checker would catch missing files, but not semantic
claims. The latter require generated traceability from the strict config/mode
matrix and structural production tests.

## Coverage limitations and uncertainty

- Static Python cannot prove every dynamic protocol dispatch. The non-keyed
  rollout methods and `capture_probe_leaf` are explicitly marked uncertain
  where `getattr`/duck typing could hide a caller. With one active game, the
  correct resolution is to remove indirection and trace the concrete NLHE path.
- Reflection, serialized qualified names, and external users of undocumented
  Python APIs were not considered production authority. Removing installed
  inference commands may affect an external consumer; current repository policy
  nevertheless makes archival the default unless the user explicitly expands
  scope.
- A class listed for archival may contain small constants or pure formulas used
  by tensor production. Those must be split first; the table identifies
  `PokerConfig`, `PotStrata`, status/action constants, and the BR Student-t helper
  as known examples.
- This pass did not dynamically import every module, run coverage, load every
  historical checkpoint, or execute CLI help. Batch 03 did validate current
  root-shaped YAMLs; batch 08 ran evaluation-adjacent tests. Those successes do
  not change the architectural dispositions above.
- No deletion should precede the required direct-seal, selected-BR-completion,
  evaluator-v2, and CUDA-evaluation migrations. The desired end state is a sole
  path, not a temporarily broken repository.

## Recommended cleanup sequence

1. Specify BR and freeze the canonical five-semantic-mode matrix.
2. Repair the blocker production path: direct compact seal, immutable schemas,
   hard byte admission, terminal failure agreement, and CUDA-only entry points.
3. Migrate evaluator/checkpoint/config consumers to one current contract.
4. Add canonical configs and five CUDA acceptance/benchmark gates.
5. Archive complete non-core products and experiments with their tests/configs/
   docs/tools.
6. Delete immediate dead/test-only objects and replace stale tests with
   structural production tests.
7. Rewrite current docs and executable workloads from the verified traces.
8. Run a final import/symbol/config/path audit; only then claim the repository
   contains the five NVIDIA production workflows and their required shared
   support.
