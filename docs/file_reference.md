# Active file reference

## Runtime roots

- `pbg/config.py` — public strict NLHE configuration loader.
- `pbg/runtime.py` — direct NLHE bundle construction and training entry point.
- `pbg/cli/train.py` — `pbg-train` command.
- `pbg/benchmark.py`, `pbg/cli/benchmark.py` — three-mode profiling of the
  single production implementation.
- `pbg/eval/arena.py` — offline duplicate-deal checkpoint evaluation.

## Generic core

- `config.py` — typed schema parser, game-independent validation, hardware
  profile composition, and explicit random generators.
- `game.py` — event DTO, rollout protocol, owner constants, and `GameBundle`.
- `execution.py` — byte accounting, occurrence arena, work queues, stable
  family routing, reusable staging, and logical RNG.
- `rollout.py` — family-scheduled sequential normal/RB trajectories.
- `branching.py` — family-scheduled labeled randomized-rounded branching.
- `tree.py` — CSR topology, device buffer, policy/reach caches, reach replay,
  and decision alias tables.
- `completion.py` — completion DTOs, learning-field filtering, and backend
  protocol.
- `values.py` — sole streamed posterior-mass/value reduction and RB targets.
- `normal.py` — compact sampled-profile normal targets and minibatches.
- `minibatch.py` — aligned RB row sampling and gathering.
- `losses.py` — PPO/RB-PPO, value, entropy, reference KL, and metrics.
- `distributed.py` — exact decision-mass scaling helpers.
- `optimization.py` — flat gradient SUM, common finite decision, clipping,
  optimizer mutation, and scaler transition.
- `checkpointing.py` — atomic payloads, per-rank RNG, scientific identity, and
  metrics-prefix validation.
- `observability.py` — execution, transfer, synchronization, evaluator, and
  memory counters.
- `training.py` — iteration/learner orchestration, precision contexts, memory
  preflight, and metrics.
- `logging_util.py` — JSONL and optional W&B support.

## NLHE package

- `cards.py` — compact deck/combo/class geometry and masks.
- `engine.py` — independent CPU public NLHE navigation and pot accounting.
- `tensor_state.py` — preallocated device public-state and binary-lift history
  storage.
- `model_inputs.py` — bounded batched device history materialization.
- `partition_cache.py` — fixed device-indexed folded-summary cache.
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
- `configs/acceptance/` — tiny one- and two-rank all-mode acceptance presets.
- `configs/benchmarks/` — production profiling workloads.
- `configs/hardware/` — local RTX 2080 Ti and Quest execution envelopes.

Every active scientific YAML uses direct `game.n_players`, `stack_bb`, and
`rank_count`, plus the neutral rollout fields documented in `CONTRACTS.md`.

## Documentation

- `docs/theory/core_math.tex` — mathematical abstraction.
- `docs/theory/MHA_model.md` — poker network design.
- `docs/theory/pseudocode/family_scheduled_rollout.tex` — sole rollout
  algorithm.
- `docs/theory/pseudocode/distributed_update.tex` — distributed learner law.
- `docs/core_algorithms.md` — implementation mapping for all three modes.
- `docs/poker.md` — NLHE/evaluator/model details.
- `docs/training_and_evaluation.md` — operational guide.
- `docs/audit/` — current readiness and reachability inventory only.

## Tests

- `test_p0_*` — strict configuration and import boundaries.
- `test_execution_foundations.py` — execution transactions, RNG, global mass,
  and device primitives.
- `test_p2_*` through `test_p7_*` — NLHE rules, compact deck, ranges, evaluator,
  model, and adapter.
- `test_p8_*`, `test_p9_branching.py`, `test_p10_values.py` — topology,
  rollout, branching, and streamed target laws.
- `test_p11_*` — losses, checkpointing, reference policy, epochs, and memory.
- `test_p12_eval.py` — duplicate-deal arena.
- `test_p13a_gpu_rollout.py` — tensor/CUDA rollout, completion, memory, and
  all-mode integration.
- `test_p13b_integration.py`, `test_p13b_ddp.py` — runtime composition,
  training/resume, unequal distributed mass, and finite gradient reduction.
- `test_c09_*` through `test_c11_*` — corrected branching/value/loss laws.

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
