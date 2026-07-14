# File reference

This is an inventory of every tracked file. “Historical” means retained for
provenance or compatibility, not part of the current execution sequence.

## Repository root

- `.gitignore` excludes Python/build caches, virtual environments, LaTeX
  products, editor files, training outputs, checkpoints, and experiment logs.
- `AGENTS.md` gives repository layout and shared-GPU-server safety rules to
  coding agents.
- `README.md` is the project entry point: scope, architecture, setup, presets,
  training, evaluation, tests, and documentation links.
- `pyproject.toml` defines the `pbg` 0.1.0 setuptools package, Python 3.10+
  requirement, runtime/test dependencies, package discovery, and pytest root.
- `CONTRACTS.md` freezes package layout, tensor interfaces, invariants,
  evaluator ownership, precision/DDP rules, and acceptance gates. Corrections
  incorporated into its current P8–P11 sections supersede the old prompts.
- `NOTATION.md` maps the authoritative mathematics to concrete data structures
  and algorithms, including the corrected labeled branching, posterior-moment
  recurrence, and RB-PPO weighting.
- `IMPLEMENTATION_NOTES.md` is a historical chronological implementation
  ledger from P00 through P13B. For current readiness, use
  `docs/audit/findings.md` and `docs/audit/resolved_findings.md`.

## Configuration presets

- `configs/kuhn_smoke.yaml` is the small sequential synthetic smoke/training
  configuration.
- `configs/leduc.yaml` trains Leduc with fixed stage-dependent branching.
- `configs/synthetic_dial_example.yaml` demonstrates a larger three-player
  synthetic game and decision-only reach caching.
- `configs/train/nlhe6.yaml` is the hardware-independent six-player 100bb
  configuration. `configs/hardware/` contains local and Quest execution
  ceilings. Quest access is available, but execution remains explicitly
  deferred until local readiness review and staged H100 measurement.

## Implementation documentation

- `docs/README.md` indexes the implementation documentation and authorities.
- `docs/architecture.md` explains dependency direction, runtime flow, core data
  structures, ownership, and design decisions.
- `docs/core_algorithms.md` explains rollout, branching, reach, completion,
  target propagation, minibatching, and RB-PPO.
- `docs/synthetic_games.md` documents the configurable game family, exact
  tools, model, rollout, and completion.
- `docs/poker.md` documents public NLHE, range tracking, evaluator adapters,
  neural models, and tensor rollout.
- `docs/training_and_evaluation.md` documents setup, configuration, presets,
  training, DDP, checkpoints, evaluation, and tests.
- `docs/file_reference.md` is this per-file inventory.
- `docs/MHA_model.md` is the detailed mathematical specification of the poker
  transformer, tokenization, reach-biased attention, policy heads, and values.
- `docs/audit/README.md` indexes the repository-readiness audit, traceability,
  validation probes, storage/compute requirements, active findings, and
  resolved findings.
- `docs/theory/folder_impact.md` derives folded-range card-removal mathematics. It is
  evaluator-owned background, not code to implement locally in this repo.

## Mathematical LaTeX sources

- `docs/theory/core_math.tex` is the mathematical source of truth for PBG
  decision laws, Rao–Blackwellization, sampled branching, completion, targets,
  losses, and research extensions.
- `docs/theory/core_math_audit.md` records the review of that source, resolved
  corrections, and deferred theoretical questions. It is not an implementation
  readiness audit.
- `docs/theory/preamble/preamble.tex` loads document packages and global LaTeX
  formatting.
- `docs/theory/preamble/project-macros.sty` defines project probability,
  analysis, neural-network, and poker notation.
- `docs/theory/preamble/mathenv.sty` defines custom IEEE-style equation/proof
  environments and helpers.
- `docs/theory/preamble/thmenv.sty` defines theorem, definition, remark, and
  framed-environment styles.

## Archive

- `archive/POKER_GPU_EVALUATOR_IO.md` is the binding cross-repository API
  contract for combo representation, completion, partition summaries, exact
  profiles, and fixed showdowns.
- `archive/PREAMBLE.md` is the historical common instruction block used while
  applying implementation prompts. Its older DTO excerpt is provenance, not the
  current API; use `CONTRACTS.md` and code for current interfaces.
- `archive/prompts/` preserves the historical implementation and correction
  prompts. It is not a current execution queue.

## Generic core package

- `pbg/__init__.py` identifies the top-level package.
- `pbg/core/__init__.py` identifies the game-neutral core namespace.
- `pbg/core/config.py` implements strict YAML-to-dataclass loading, semantic
  validation, and explicit rank/device Torch and NumPy generators.
- `pbg/core/game.py` defines owner constants, opaque `EventBatch`, the batched
  `RolloutBackend` protocol, typed `GameBundle`, and legacy/scalar `PBGGame`
  protocol used by exact and oracle paths.
- `pbg/core/policy_adapter.py` defines the minimal generic policy output and
  selected-log-probability/entropy protocol.
- `pbg/core/completion.py` defines game-neutral single/batched completion
  statistics, opaque `ReachView`, and the completion protocol.
- `pbg/core/tree.py` implements `TreeBatch` invariants, CPU and preallocated
  device builders, device-local categorical decision sampling, stable frozen-policy rows, and
  full/decision/recomputed reach caches.
- `pbg/core/rollout.py` implements the device-resident sequential sampler and
  public sequential/fixed dispatch.
- `pbg/core/branching.py` implements randomized-rounded, fixed-intensity,
  duplicate-preserving labeled branching with uniform child mass. The private
  legacy entry remains isolated for historical tests.
- `pbg/core/posterior_values.py` implements posterior `P/M` backward
  propagation, conditionalization, O(K) sibling baselines, lambda bootstrap,
  optional Q/auxiliary emission, and whole-tree memory chunking.
- `pbg/core/values.py` re-exports only the corrected posterior-value API so
  callers cannot accidentally select the obsolete direct-value path.
- `pbg/core/minibatch.py` samples the empirical decision law and gathers every
  target/cache/legal-event tensor in stable row order.
- `pbg/core/losses.py` implements policy-gradient, PPO-clipped, value,
  value-clipped, and entropy primitives plus the registered minimal `rb_ppo`
  composite.
- `pbg/core/training.py` resolves game factories, freezes rollout models,
  constructs targets, optimizes minibatches, handles AMP/DDP and explicit
  reproducibility profiles, and atomically saves/restores identity-checked
  resumable checkpoints with per-rank RNG state.
- `pbg/core/logging_util.py` provides an append-only JSONL metric logger and an
  optional guarded Weights & Biases sink. The main loop currently writes JSONL
  directly with the same append-only convention.

## Synthetic game package

- `pbg/games/__init__.py` identifies the game implementation namespace.
- `pbg/games/synthetic/__init__.py` identifies the synthetic package.
- `pbg/games/synthetic/family.py` implements the configurable colex card game,
  deduplicated public state table, betting/chance/reward rules, reach and
  posterior sampling, public encoding, and generic rollout backend.
- `pbg/games/synthetic/presets.py` builds Kuhn, Leduc, and the exact two-axis
  card/action dial without named-game branches in the engine.
- `pbg/games/synthetic/model.py` implements the default scaled MHA poker model,
  explicit MLP baseline, categorical policy adapter, and reach-consistent
  exact-policy-table exporter.
- `pbg/games/synthetic/exact.py` enumerates feasible public trees/profiles and
  computes float64 posteriors, policy values, information-set best responses,
  and exploitability under hard guards.
- `pbg/games/synthetic/bundle.py` constructs the model, adapter, rollout
  backend, and synthetic completion backend as a `GameBundle`.

## Poker package

- `pbg/games/poker/__init__.py` identifies the poker namespace and its public
  navigation/evaluator-boundary role.
- `pbg/games/poker/cards.py` owns frozen combo/class representations, suit-row
  permutations, board compatibility masks, evaluator-map validation, and
  uniform complement sampling.
- `pbg/games/poker/engine.py` is the independent CPU public NLHE state machine,
  including raises/reopening, chance transitions, folds, side-pot strata, and
  affine payoff assembly from supplied showdown shares.
- `pbg/games/poker/tensor_state.py` preallocates and prefix-allocates the
  device-resident structure-of-arrays poker state store.
- `pbg/games/poker/rollout_backend.py` implements batched CPU/CUDA public NLHE
  transitions, chance labels/payloads, profile/event sampling, reach updates,
  policy queue routing, fixed beta buckets, pot strata, and completion inputs.
- `pbg/games/poker/ranges.py` tracks unnormalized combo log likelihoods,
  compatibility masks, row cloning/stabilization, immutable fold snapshots,
  normalization, and 169-class aggregation.
- `pbg/games/poker/folded.py` defines fold/partition DTOs and the lazy,
  validating adapter to evaluator-owned partition summaries.
- `pbg/games/poker/mc.py` defines typed errors and lazy validating adapters for
  evaluator-owned completion, exact joint-profile rejection, and fixed
  showdown shares.
- `pbg/games/poker/adapter.py` composes legal-event policy laws, routes the
  street models, and supplies the CPU `PokerGame` facade for navigation,
  ranges, chance, and evaluator-mediated settlement.
- `pbg/games/poker/bundle.py` constructs the single owning poker model and
  tensor-native rollout/completion bundle with explicit residency selection.

## Poker model package

- `pbg/games/poker/model/__init__.py` lazily exposes model classes to avoid
  adapter/model import cycles.
- `pbg/games/poker/model/primitives.py` implements layer norm, GELU FFN, and
  reach-biased multi-head attention with SDPA and explicit reference paths.
- `pbg/games/poker/model/heads.py` implements contextual action-type logits,
  discretized-logistic/all-in raise ticks, a binomial ablation, actor values,
  and dormant public moment heads.
- `pbg/games/poker/model/preflop.py` defines strict model/public/mask/output
  DTOs and the 169-class pre-LN preflop transformer.
- `pbg/games/poker/model/incidence.py` provides differentiable padded compact
  gather/scatter maps and supplied row permutations without private poker math.
- `pbg/games/poker/model/postflop.py` implements compact board-valid combo
  tokenization, supplied partition features, suit representation transforms,
  the postflop transformer, and fixed-width output scatter.

## Evaluation package

- `pbg/eval/__init__.py` identifies the evaluation namespace.
- `pbg/eval/exploitability.py` loads synthetic checkpoints read-only, exports
  policy tables, computes exact exploitability, and writes iteration-sorted
  curves.
- `pbg/eval/arena.py` runs paired duplicate-deal CPU poker matches, rotates
  positions, crosses the evaluator for all showdowns, and reports paired
  Student-t bb/100 intervals and seat breakdowns.

## Historical implementation prompts

- `archive/prompts/P00_skeleton.md` specified package skeleton, strict config, logging,
  seeding, and import-direction tests.
- `archive/prompts/P01_synthetic.md` specified the synthetic family, presets, exact
  analysis, model, and adapter.
- `archive/prompts/P02_engine.md` specified card representations and the public NLHE
  engine.
- `archive/prompts/P03_ranges.md` specified poker reach and fold-snapshot bookkeeping.
- `archive/prompts/P04_folded.md` specified the evaluator-owned partition-summary
  boundary.
- `archive/prompts/P05_completion.md` specified generic completion DTOs and the three
  poker evaluator operations.
- `archive/prompts/P06_preflop.md` specified attention primitives, poker output heads,
  and the preflop network.
- `archive/prompts/P07_postflop.md` specified compact postflop modeling and the poker
  public-tree adapter.
- `archive/prompts/P08_rollout.md` specified the first sequential rollout/cache design;
  C08 supersedes its production boundary.
- `archive/prompts/P09_branching.md` specified the old quotient-tree Poisson/Neyman
  design; C09 supersedes its production algorithm.
- `archive/prompts/P10_values.md` specified the old direct conditional-value backward
  pass; C10 supersedes it.
- `archive/prompts/P11_training.md` specified the broader old loss/training plan; C11
  supersedes it.
- `archive/prompts/P12_eval.md` specified the current read-only synthetic and poker
  evaluation layer.
- `archive/prompts/P13A_gpu_poker_rollout.md` specified the current tensor-native poker
  state and rollout backend.
- `archive/prompts/P13B_integration.md` specified current factories, presets,
  residency, sibling integration, and acceptance tests.
- `archive/prompts/C00_math_contract.md` aligned the mathematical contract before the
  corrected implementation.
- `archive/prompts/C08_device_rollout.md` corrected rollout to the typed,
  device-resident backend and pre-terminal semantics.
- `archive/prompts/C09_labeled_branching.md` corrected branching to randomized-rounded
  fixed intensities and duplicate-preserving labeled children.
- `archive/prompts/C10_posterior_moments.md` corrected values to propagate posterior
  and weighted-return mass with leave-one-branch baselines.
- `archive/prompts/C11_minimal_training.md` corrected row alignment, configuration,
  checkpointing, and the minimal posterior-weighted RB-PPO objective.

## Tests

- `tests/test_p0_config.py` covers complete strict loading, dotted unknown-key
  errors, deterministic explicit generators, and JSONL logging.
- `tests/test_p0_imports.py` AST-checks the permanent package dependency rule.
- `tests/test_p1_family.py` checks standard Kuhn public structure, legal deals,
  Leduc completion cuts, multi-size raise events, zero-sum rewards, and reach
  numerators.
- `tests/test_p1_exact.py` checks exact exploitability/equilibria, posterior
  sampling, feasibility guards, exact dial anchors, and the two-axis grid.
- `tests/test_p1_adapter.py` checks categorical normalization, entropy, and
  exported exact policy tables.
- `tests/test_p2_cards.py` checks all combos/classes, suit bijections, board
  masks, and complement sampling.
- `tests/test_p2_engine.py` checks action order, raises, short-all-in reopening,
  round transitions, side pots, payoff assembly, and absence of private
  settlement.
- `tests/test_p3_ranges.py` checks Bayesian action numerators, compatibility
  masks, frozen folds, class aggregation, branch independence, and gauge
  stability.
- `tests/test_p4_folded.py` checks lazy facade mapping, contract failures,
  tensor residency, absence of local private math, and real sibling smoke.
- `tests/test_p5_completion.py` checks completion mapping/diagnostics, whole
  exact profiles, partial rejection, showdown shares, facade versioning, and
  forbidden-implementation scans.
- `tests/test_p6_primitives.py` checks SDPA/reference attention equivalence,
  untempered key bias, and normalized raise-tick mixtures.
- `tests/test_p6_preflop.py` checks forward/autocast shapes, class equivariance,
  seat masks, gradient coverage, and strict model config.
- `tests/test_p7_postflop.py` checks ragged differentiable incidence maps,
  postflop scatter/features/autocast, absent summary defaults, and suit
  relabeling.
- `tests/test_p7_adapter.py` checks the fully composed event law, exact
  entropy/KL, profile row selection, complement chance draws, and evaluator
  showdown use.
- `tests/test_p8_tree.py` checks CSR/depth/path invariants and categorical
  decision sampling.
- `tests/test_aud26_no_host_sync.py` gates device-local decision sampling,
  fixed-street compact maps, and explicit biased attention under CUDA
  synchronization-debug mode.
- `tests/test_p8_rollout.py` retains sequential law, reach-mode, cache, and
  node-budget compatibility coverage.
- `tests/test_p9_branching.py` retains sequential byte determinism and terminal
  law checks against the corrected labeled implementation.
- `tests/test_c08_device_rollout.py` checks device-resident rollout, corrected
  Leduc cuts, loud capacity errors, sole dispatch, and no explicit hot-path host
  transfers.
- `tests/test_c09_labeled_branching.py` checks randomized rounding, duplicate
  labels/descendants, uniform edges, strict fixed beta, overflow, and no
  dedup/host transfer.
- `tests/test_c10_posterior_moments.py` checks P/M aggregation, leave-one-out,
  single-child baselines, capture order, chunking/caps, precision, and CUDA
  parity.
- `tests/test_c11_minimal_losses.py` checks posterior-weighted treatment of
  duplicate abstract children.
- `tests/test_c11_minimal_training.py` checks fixed-beta training, typed bundles,
  removed config fields, and row-index minibatching.
- `tests/test_p10_values.py` checks the corrected public API, padding, lambda
  endpoints, opt-in Q, chunking, and precision rejection.
- `tests/test_p11_losses.py` checks direct-advantage PPO clipping, masks, and
  identity metrics.
- `tests/test_p11_training.py` checks the Kuhn smoke path and exact checkpoint
  RNG restoration.
- `tests/test_p12_eval.py` checks uniform/fresh Kuhn evaluation, sorted curves,
  zero self-play duplicate arenas, seat rotation, evaluator showdowns, and the
  real sibling combo table.
- `tests/test_p13a_gpu_rollout.py` checks tensor storage, CPU-engine parity,
  side pots/reopening, chance labels, policy queues, batched forwards,
  completion inputs, beta buckets, hot-path residency, and CUDA outputs.
- `tests/test_p13b_integration.py` checks all presets, one-model bundle ownership,
  real sibling signatures, and a short checkpointed CPU poker training smoke.
- `tests/test_p13b_ddp.py` checks per-rank sizing and comparable one-/two-rank
  Kuhn learning curves.
