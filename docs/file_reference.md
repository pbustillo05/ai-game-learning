# File reference

This is an implementation-oriented map of the current repository. “Historical”
or “research design” means the file is not part of production training.

## Root authorities and operations

- `README.md` is the project entry point and command overview.
- `AGENTS.md` defines repository layout and shared-GPU safety rules.
- `CONTRACTS.md` freezes interfaces, shapes, estimator invariants, evaluator
  ownership, precision/DDP requirements, and acceptance conditions.
- `NOTATION.md` maps mathematical objects to code names and tensor layouts.
- `HARDWARE.md` records local/Quest hardware policy and launch gates.
- `IMPLEMENTATION_NOTES.md` is the historical P00–P13 implementation ledger;
  it is not a current readiness report.
- `pyproject.toml` defines the package, dependencies, pytest configuration, and
  `pbg-train`, `pbg-benchmark`, and `pbg-branching-experiment` entry points.
- `proxy_gpu_pbg.md` is a copied evaluator `pbg-v2` API reference. The binding
  consumer contract remains `archive/POKER_GPU_EVALUATOR_IO.md`.
- `tools/run_nlhe6_10bb_normal_seq_4h30.sh` is the local timed launcher for the
  streamed normal and sequential-RB 10bb experiments. It pins one GPU and uses
  the shared poker virtualenv.

## Configuration

General presets:

- `configs/nlhe6_h100.yaml` — six-player NLHE scientific configuration intended
  for an H100 hardware profile.

Hardware profiles:

- `configs/hardware/local_3x2080ti.yaml` — validated 11GB RTX 2080 Ti limits:
  precision, token/token-pair caps, 2GB reserve, and at most three devices.
- `configs/hardware/quest_slurm.yaml` — opt-in Quest/Slurm envelope; its larger
  hardware still requires staged measurement.

Poker training presets:

- `configs/train/nlhe6.yaml` — hardware-independent 100bb branched base.
- `configs/train/nlhe6_10bb.yaml` — current longer branched 10bb candidate.
- `configs/train/nlhe6_10bb_gpu_smoke*.yaml` and
  `nlhe6_10bb_gpu_gate.yaml` — branched calibration/exploration/gate variants.
- `configs/train/nlhe6_10bb_normal_gpu_{smoke,gate,tune}.yaml` — resident
  normal-mode calibrations.
- `configs/train/nlhe6_10bb_sequential_rb_gpu_{smoke,gate,tune}.yaml` — resident
  sequential-RB calibrations.
- `configs/train/nlhe6_10bb_{normal,sequential_rb}_4h30.yaml` — resident
  128-trajectory timed candidates.
- `configs/train/nlhe6_10bb_{normal,sequential_rb}_streamed_gpu_smoke.yaml` —
  chunked 256-trajectory/two-epoch smoke configurations.
- `configs/train/nlhe6_10bb_{normal,sequential_rb}_streamed_4h30.yaml` — logical
  1024-trajectory, resident-128, one-epoch timed candidates.

## Root documentation

- `docs/README.md` is the implementation-doc index.
- `docs/architecture.md` describes packages, actual chunk lifetimes, device
  movement, retained objects, and memory gates.
- `docs/core_algorithms.md` describes all three rollout/target/loss pipelines.
- `docs/poker.md` describes tensor NLHE, evaluator ownership, networks, policy
  packing, synchronization, and memory.
- `archive/synthetic_games/` is the inert historical synthetic-game snapshot.
- `docs/training_and_evaluation.md` is the separate operational guide; it is not
  part of this documentation rewrite.
- `docs/file_reference.md` is this map.

Theory and research material:

- `docs/theory/core_math.tex` is the mathematical source of truth;
  `core_math.pdf` is its rendered artifact.
- `docs/theory/MHA_model.md` is the detailed poker model specification.
- `docs/theory/core_math_audit.md` records theory review, not implementation
  readiness.
- `docs/theory/folder_impact.md` is evaluator-owned card-removal background.
- `docs/theory/preamble/` contains the LaTeX preamble and project macros.
- `docs/references/` contains cited paper PDFs and extracted source trees; none
  are imported by training.
- `docs/todo/POKER_GPU_EVALUATOR_COMPACT_DECK_REQUEST.md` is the required
  sibling `pbg-v3` contract change request.
- `docs/todo/branching.md` is a research specification for alternative
  branching controllers, not a production training contract.

Audit records:

- `docs/audit/README.md` indexes the standing readiness audit.
- `findings.md` and `resolved_findings.md` retain numbered findings.
- `mode_storage_compute.md`, `traceability.md`, and `validation_matrix.md`
  describe expected storage, implementation mapping, and gates.
- `training_readiness_2026-07-14.md` and `nlhe6_10bb_smoke_2026-07-14.md` record
  the corresponding launch/smoke evidence.
- `rollout_efficiency_2026-07-15.md` audits current queueing, physical batching,
  memory planning, chunk/epoch semantics, and measured 2080 Ti behavior.

## Historical archive

- `archive/POKER_GPU_EVALUATOR_IO.md` is the binding sibling-evaluator contract.
- `archive/PREAMBLE.md` is the shared historical prompt preamble.
- `archive/prompts/P00_*.md` through `P13B_*.md` and `C00_*.md` through
  `C11_*.md` are implementation/correction history. Current code, contracts,
  and docs supersede them.

## Generic core

- `pbg/core/game.py` defines owner constants, `EventBatch`, `RolloutBackend`,
  `GameBundle`, and the scalar/oracle `PBGGame` protocol.
- `pbg/core/config.py` performs strict YAML-to-dataclass loading, game-owned
  validation, mode/loss validation, and explicit rank/device RNG construction.
- `pbg/core/policy_adapter.py` defines the common policy/value payload contract.
- `pbg/core/completion.py` defines completion DTOs, learning-field filtering,
  and the completion protocol.
- `pbg/core/tree.py` owns validated CSR topology, CPU/device builders, policy
  cache, reach storage/replay, decision reach retention, and Walker alias
  sampling.
- `pbg/core/rollout.py` implements synchronized sequential frontier rollout and
  public RB dispatch.
- `pbg/core/branching.py` implements fixed randomized-rounded,
  duplicate-preserving labeled branching. Its optional controller seam is used
  only by offline experiments.
- `pbg/core/normal.py` converts sequential samples into scalar normal-PPO data
  and gathers/samples flat minibatches.
- `pbg/core/posterior_values.py` implements streamed `P/M` reduction,
  conditionalization, sibling baselines, lambda bootstrap, byte preflight, and
  optional Q/auxiliary targets.
- `pbg/core/values.py` re-exports only the corrected posterior-value API.
- `pbg/core/minibatch.py` samples/gathers aligned RB decision rows.
- `pbg/core/losses.py` implements normal PPO and posterior-weighted RB-PPO,
  clipped/MSE value terms, entropy, exact reference KL, and metrics.
- `pbg/core/training.py` owns model snapshots, chunking, completion/target
  lifetime, logical-to-physical optimization, AMP, explicit DDP reduction,
  memory planning/calibration, metrics, and resumable checkpoints.
- `pbg/core/logging_util.py` supplies generic JSONL and optional W&B logging;
  the main training loop writes its append-only JSONL directly.

## Poker package

- `pbg/games/poker/cards.py` owns card/combo/class representations, masks,
  permutations, and evaluator-map validation.
- `pbg/games/poker/engine.py` is the independent CPU public NLHE oracle.
- `pbg/games/poker/tensor_state.py` is the preallocated device public-state
  structure-of-arrays with parent-linked histories.
- `pbg/games/poker/rollout_backend.py` implements batched public transitions,
  profile/event sampling, dense reach updates/replay, policy batch planning and
  street routing, fold-summary caching, beta buckets, pot strata, and completion
  inputs.
- `pbg/games/poker/ranges.py` is the CPU/test likelihood and immutable-fold
  representation.
- `pbg/games/poker/folded.py` validates evaluator partition/folder DTOs.
- `pbg/games/poker/mc.py` validates evaluator completion, joint profiles, and
  fixed-profile showdown calls.
- `pbg/games/poker/adapter.py` composes the legal factorized policy, owns the
  two-network model bundle, and supplies the CPU `PokerGame` facade.
- `pbg/games/poker/bundle.py` estimates rollout storage and constructs the
  tensor training bundle/completion adapter.

Poker model files:

- `model/primitives.py` — normalization, FFN, biased MHA, FlexAttention and
  checkpointed query chunks.
- `model/incidence.py` — incidence residuals and compact combo gather/scatter.
- `model/heads.py` — action type, raise mixture/binomial ablation, value, and
  optional moment heads.
- `model/preflop.py` — 169-class network, tokenization, history, incidence, and
  policy/value readout.
- `model/postflop.py` — live-player board-compatible combo network, folded
  features, compact execution, and 1326-row scatter.
- `model/__init__.py` — lazy model exports.

## Evaluation, benchmarks, and experiments

- `pbg/eval/arena.py` runs duplicate-deal, seat-rotated poker comparisons using
  evaluator showdown shares.
- `pbg/benchmark.py` runs repeatable mode profiles, restores model/optimizer/RNG
  state between samples, and compares baselines.
- `pbg/cli/benchmark.py` is the benchmark CLI.
- `benchmarks/README.md` describes performance-gate procedure;
  `workloads.yaml` is its workload matrix.
- `benchmarks/results/branching_smoke_nlhe6_10bb.json` is the recorded fixed-
  policy branching experiment result currently kept in the repository.
- `pbg/experiments/branching.py` implements offline fixed-policy controller and
  size/cost experiments. It does not change the default training sampler.
- `pbg/cli/branching_experiment.py` exposes those read-only checkpoint
  experiments.
- `pbg/cli/train.py` loads configuration/hardware profiles and invokes training.

## Tests

The suite is organized by contract rather than by package:

- `test_p0_*` — strict configuration, logging/import boundaries.
- `test_p2_*` — card representations and CPU NLHE engine.
- `test_p3_ranges.py`, `test_p4_folded.py`, `test_p5_completion.py` — range and
  sibling evaluator boundaries.
- `test_p6_*`, `test_p7_*` — poker attention, heads, pre/postflop models, and
  policy adapter.
- `test_p8_*`, `test_p9_branching.py`, `test_p10_values.py` — topology,
  rollout/reach, branching, and value targets.
- `test_p11_*` — losses, training, checkpoints, reference KL, chunked epochs,
  and memory metrics.
- `test_p12_eval.py` — duplicate-deal evaluation.
- `test_p13a_gpu_rollout.py` — tensor NLHE parity, routing/planning, residency,
  and CUDA behavior. Tests using “queue” in their name validate one-call routing,
  not persistent scheduler queues.
- `test_p13b_integration.py` and `test_p13b_ddp.py` — presets, factories,
  rollout memory estimates, CPU smoke, DDP metrics, and scaling.
- `test_c09_*` through `test_c11_*` — labeled branching, posterior moments,
  alignment, and minimal losses.
- `test_aud26_no_host_sync.py` and `test_aud32_performance_gates.py` — audit regressions. The no-host-sync test
  covers isolated alias/compact-map/attention primitives, not the full poker
  rollout scheduler.
- `test_compact_deck_geometry.py` — compact mappings, category counts/orders,
  configuration boundaries, memory scaling, and the evaluator dependency gate.
