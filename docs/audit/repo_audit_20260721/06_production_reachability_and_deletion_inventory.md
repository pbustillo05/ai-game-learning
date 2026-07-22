# Production reachability and deletion inventory

## Verdict

The active package is not production-only. It contains installed CPU/MPS
products, historical variance algorithms, indefinite compatibility, and
test-only/dead objects. Cleanup must follow migrations: reachability does not
justify retaining forbidden owners, and deletion must not break the current
sole path before its replacement exists.

The exhaustive evidence and per-object line references are in
[batch 09](batches/09_reachability_deletion_docs_tests.md). This document is the
canonical condensed disposition; `capture_probe_leaf` is corrected here as
dynamically BR-reachable.

## Immediate move to tests, archive, or delete

| Surface / Python objects | Disposition |
|---|---|
| `pbg/core/distributed.py`: `WeightedDecisionStats`, `weighted_decision_stats`, `global_mass_objective`, `stratum_objective_scale` | Move useful formulas to tests; delete test-only module |
| `pbg/core/logging_util.py`: `MetricsLogger` | Move fixture/delete module |
| `learner.py`: `allocate_statistical_counts`, `ModeLoss`, `distributed_fixed_rollout_update` | Delete predecessors/aliases and update tests |
| `targets.py`: `BuildNormalTargets`, `BuildSequentialRBTargets`, `BuildBranchedRBTargets`, `FinalizeTrainingStorage` | Delete uncalled aliases |
| `execution.py`: `ReusableStagingSlabs`, `logical_categorical` | Delete test-only implementations; separately implement the authoritative production slab |
| `tree.py`: `DiagnosticTreeBuilder`, `AliasTable`, `build_decision_alias` | Archive diagnostics or move tiny fixture; delete legacy sampler APIs |
| `completion.py`: `CompletionStats`, `BatchedCompletionStats.row` | Delete mutually unconsumed objects |
| `training.py`: `_memory_operating_ceiling`, `_sealed_input_block` | Delete unless the ceiling is deliberately wired into the repaired controller |
| `observability.py`: `record_cache`, `record_completion`, `record_rank_wait` | Delete unused methods/fields and update tests |
| `cards.py`: `DECK_GEOMETRY_VERSION`, `STANDARD_DECK`; module `class_of_hand`, `class_members`, `class_size`, `suit_perm_apply`, `hand_cards`, `board_conflict_mask`, `sample_uniform_complement` | Move minimal geometry oracles to tests; delete after CPU product archival |
| `heads.py`: `raise_tick_probs` | Delete duplicate composer/update test |
| `postflop.py`: `suit_row_permutation`, `permute_suit_representation`; `incidence.py`: `permute_rows` | Archive/delete suit-ablation chain |
| `MHA._sdpa_attention`, `MHA.attention_weights` | Delete diagnostic methods or move calculation to tests |
| `partitions.py`: `assert_poker_storage_widths` | Inline in test/delete |
| variance CLI `_load_geometry`; unused BR imports `DrawPurpose`, `PROBE_FOLD`, `sequential_leaf_logical_rng_keys` | Delete with experiment / remove imports |

`capture_probe_leaf` is **not** in this list: dynamic `getattr` dispatch reaches
it in `chance_close`. It is BR-required and needs C04 repair.

## Archive complete non-core products

- `pbg/eval/arena.py` and all its objects; relocate `_student_t_critical` for
  BR if still needed.
- `pbg/inference.py`, `pbg/inference_artifact.py`, inference/export CLIs,
  entry points, macOS extra, docs, and tests.
- `pbg/eval/rao_blackwell_variance.py`,
  `pbg/eval/branched_rao_blackwell_variance.py`, both CLI modules, all their
  top-level objects, configs, tests, launchers, and run-linked tools.
- `BranchingCutoffReached` and `BranchingDiagnosticLimits` with the branching
  geometry/censoring experiment; they are not a production safeguard.
- The list engine/range product (`PublicHistoryEvent`, `_State`, `StateTable`,
  `NLHEEngine`, module `assemble_showdown_payoff`, `FoldedPlayer`, `RangeState`,
  `ChanceDraw`, `PokerGame`) after splitting retained tensor constants/oracles.

This archival is contingent only on the user's external-consumer decision in
C13; these are not sixth core modes.

## Compatibility-reachable: migrate, then delete

| Surface | Required end state |
|---|---|
| `SampledTree`, `PagedTreeSource`, `RolloutDecisionRows`, `StoppedLeafRows` | Direct compact seal, then delete |
| `PagedRowAccumulator` | Replace first-write inference with schema-bound arenas |
| `ReachView`, `MCCompletion.complete`, mutable/list completion APIs | Narrow to tensor completion boundary |
| `PokerCompletionService.complete_selected_keyed` and all-player BR completion | Build selected-owner sealed completion; delete obsolete helper/all-player route |
| Non-keyed rollout protocol and poker methods | Require sole keyed NLHE path; update test doubles |
| `PokerRolloutBackend.policy_forward` / `_policy_forward_direct` | Point tests at production family partition route, then delete |
| `PokerModelConfig.n_buckets_line`, `evaluator_summary_dim`, false `use_moment_heads`, `p_max` alias | Convert approved artifacts; reject/delete fields |
| Dense-beta eigendecomposition and `folder_card_marginal` alias | Require evaluator v2; delete v1 fallback |
| CPU/MPS/generic attention and ragged-width branches | Delete with non-NVIDIA products, retaining negligible shared formulas only |
| Checkpoint v2/v3/v4 migration/constants, old fingerprints, private aliases | Offline-convert approved artifacts, then require current schema |
| Benchmark missing-T/C `strict=False` loading | Require current checkpoint or explicit offline conversion |

## Retain and repair

- BR: `BRProbeReport`, `_sha256`, `_selected_reward`, `evaluate_br_probe`, and
  `pbg.cli.br_eval.main`, pending C03/C04/C08/C16.
- Benchmark objects, rebuilt as strict five-semantic CUDA gates.
- `occurrence_schema`, made the sole schema rather than deleted.
- `PokerConfig`, `PotStrata`, active action/street/status constants,
  `DeckGeometry`, tensor state/history, natural partitions, model/heads, keyed
  evaluator facades, `CompletionSpec`, `BatchedCompletionStats.for_learning`,
  and tensor completion, subject to the canonical fixes.

Archive stale product tests with their products; replace tree, migration,
evaluator-v1, CPU benchmark, and textual lifecycle tests with structural tests
on the real production path. The detailed test map is in batch 09.
