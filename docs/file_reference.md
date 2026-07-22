# Active file reference

## Runtime roots

- `pbg/config.py` — public strict NLHE configuration loader.
- `pbg/runtime.py` — direct NLHE bundle construction and training entry point.
- `pbg/cli/train.py` — `pbg-train` command.
- `pbg/benchmark.py`, `pbg/cli/benchmark.py` — three-mode profiling of the
  single production implementation.
- `pbg/eval/br_probe.py` — accelerator-native paired best-response evaluation.
- `pbg/inference.py`, `pbg/inference_artifact.py` — stateful CUDA/MPS tensor-native
  serving and compact model-only artifacts.

## Generic core

- `config.py` — typed schema parser, game-independent validation, hardware
  profile composition, and explicit random generators.
- `game.py` — event DTO, rollout protocol, owner constants, and `GameBundle`.
- `execution.py` — byte accounting, occurrence arena, work queues, stable
  family routing, reusable staging, and logical RNG.
- `rollout.py` — family-scheduled sequential normal/RB trajectories.
- `branching.py` — family-scheduled labeled randomized-rounded branching.
- `partition.py` — immutable exact/natural domains and opaque model/completion boundaries.
- `storage.py` — versioned schemas, append-only arenas, exact sealed blocks,
  and immutable rollout/training stores.
- `topology.py` — decision-only sequential CSR and contiguous-child branched spans.
- `targets.py` — streaming v1 targets, posterior deduplication, and sampling metadata.
- `learner.py` — bounded stratified update and transactional reference caches.
- `tree.py` — rollout-time CSR topology and device construction buffer; no
  production learner owns it.
- `completion.py` — completion DTOs, learning-field filtering, and backend
  protocol.
- `losses.py` — loss-name/mode registry and entropy schedule shared with the
  streamed learner.
- `distributed.py` — exact decision-mass scaling helpers.
- `optimization.py` — flat gradient SUM, common finite decision, clipping,
  optimizer mutation, and scaler transition.
- `checkpointing.py` — versioned atomic payloads, deterministic v2 migration,
  per-rank RNG/controller state, scientific identity, and metrics-prefix validation.
- `observability.py` — execution, transfer, synchronization, evaluator, and
  memory counters.
- `training.py` — iteration/learner orchestration, precision contexts, memory
  preflight, and metrics.
- `logging_util.py` — JSONL and optional W&B support.

## NLHE package

- `cards.py` — compact deck/combo/class geometry and masks.
- `engine.py` — independent CPU public NLHE navigation and pot accounting.
- `tensor_state.py` — recyclable paged public-state and refcounted binary-lift
  history storage.
- `model_inputs.py` — bounded batched device history materialization.
- `partition_cache.py` — sparse paged folded-summary dependencies.
- `policy_planning.py` — hardware-bounded family packing and declared scheduler
  control transfer.
- `rollout_backend.py` — tensor transitions, family inputs/forwards, reach
  updates, sampling, beta, terminal rewards, and evaluator inputs.
- `ranges.py` — CPU/evaluation reach and fold snapshot support.
- `folded.py` — sibling partition/folder DTO validation.
- `mc.py` — sibling completion, profile, and showdown API boundary.
- `adapter.py` — factored policy adapter, model bundle, and CPU arena facade.
- `bundle.py` — NLHE validation, rollout memory plan, and runtime construction.
- `model/` — attention primitives, compact incidence, action/value heads, and
  preflop/postflop networks.

## Configuration

- `configs/train/` — production and conservative smoke/gate training presets.
- `configs/experiments/` — profiling, smoke, and optimization
  workloads.
- `configs/hardware/` — local RTX 2080 Ti and Quest execution envelopes.

Every active scientific YAML uses direct `game.n_players`, `stack_bb`, and
`rank_count`, plus the neutral rollout fields documented in `CONTRACTS.md`.

Checkpoint-specific research implementations, including their configs and
tests, live under `experiments/` and are excluded from the installed `pbg`
package and default test suite.

## Documentation

- `docs/theory/core_math.tex` — mathematical abstraction.
- `docs/theory/MHA_model.md` — poker network design.
- `docs/theory/pseudocode/storage_lifecycle.tex` — authoritative storage
  object, field, indexing, ownership, lifetime, and phase-specific memory
  contract.
- `docs/theory/pseudocode/family_scheduled_rollout.tex` — rollout allocation,
  routing, write, topology-finalization, and rollout-workspace release law.
- `docs/theory/pseudocode/backward_targets.tex` — normal, sequential-RB, and
  branched-RB target construction, compaction, and release algorithms.
- `docs/theory/pseudocode/distributed_update.tex` — partition-local sampling,
  aligned minibatch materialization, distributed update, and final storage
  release law.
- `docs/theory/pseudocode/shared_response_probe.tex` — selected-seat response
  rollout, finite-budget training, and paired lower-bound evaluation overlay.

Together these five sources (and their tracked PDFs) are the decision-complete
implementation specification: storage ownership is defined once in
`storage_lifecycle.tex`, while the other three documents govern consecutive
lifecycle phases and must use its exact future Python object names; the response
overlay reuses those sequential-RB contracts without defining another storage
family.
- `docs/core_algorithms.md` — implementation mapping for all three modes.
- `docs/poker.md` — NLHE/evaluator/model details.
- `docs/training_and_evaluation.md` — operational guide.
- `docs/inference_on_mac.md` — Apple Silicon installation, export, and serving API.
- `docs/audit/` — current readiness and reachability inventory only.

## Tests

- `test_p0_*` — strict configuration and import boundaries.
- `test_execution_foundations.py` — execution transactions, RNG, global mass,
  and device primitives.
- `test_p2_*` through `test_p7_*` — NLHE rules, compact deck, ranges, evaluator,
  model, and adapter.
- `test_p8_*`, `test_p9_branching.py`, and `test_refactor_storage_targets.py` —
  topology, rollout, branching, and streamed target laws.
- `test_distributed_learner.py`, `test_phase3_migration.py`, and memory-control
  suites — loss, checkpoint, reference-policy, epoch, and memory contracts.
- `test_p12_eval.py` — duplicate-deal arena.
- `test_p13a_gpu_rollout.py` — tensor/CUDA rollout, completion, memory, and
  all-mode integration.
- `test_p13b_integration.py`, `test_p13b_ddp.py` — runtime composition,
  training/resume, unequal distributed mass, and finite gradient reduction.
- `test_inference_api.py` — artifact, stateful batch, folded CPU boundary, and
  real MPS parity contracts.
- `test_c09_*` and adaptive-branching suites — corrected branching, dynamic
  $T/C$, queue-correction, and protected-root laws.

## Archive

- `archive/synthetic_games/` — inert retired Kuhn/Leduc/synthetic snapshot.
- `archive/branching_experiment/` — retired offline branching controller.
- `archive/family_scheduler_promotion/` — migration plans, raw executor
  comparisons, and one-off scripts; legacy names are intentional history.
- `archive/audits/` — superseded readiness and validation generations.
- `IMPLEMENTATION_NOTES.md` and `archive/prompts/` — historical implementation
  ledger, not active contracts.

Archived Python is outside package discovery and the configured `tests/` CI
path. Nothing in active runtime imports it.
