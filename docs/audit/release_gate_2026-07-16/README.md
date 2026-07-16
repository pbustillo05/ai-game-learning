# Production release gate audit

Audit date: 2026-07-16

Audited revision: `dde65f6d470cd4d890ec1b19293c22e83139cc24`

## Verdict

**NO-GO for the requested production shipment.**

The fixed full-deck implementation has a strong, well-tested core. Normal,
sequential Rao--Blackwell (RB), and branched RB each completed real CUDA
training and two-rank distributed training in this audit. The implemented
fixed-intensity estimator is substantially faithful to the relevant mathematics
in `docs/theory/core_math.tex`, and the poker networks substantially implement
`docs/theory/MHA_model.md`, including evaluator-supplied folded-player statistics
after a fold in games with more than two players.

The repository nevertheless does not satisfy its advertised production
contract:

1. Every reduced deck, `rank_count != 13`, is explicitly rejected by the
   production rollout backend. The documented `R in {4,...,13}` domain is not
   implemented end to end.
2. Public-history embeddings default to 256 tokens, but legal configured games
   can have far longer histories. The shipped 6-player 100 bb preset can reach a
   documented maximum of 519 player nodes; the 6-player 128 bb boundary reaches
   659. The model raises once history exceeds 256.
3. The common distributed finite-gradient result is computed but does not gate
   clipping or the optimizer step. This is unsafe when the FP16 scaler is
   disabled, including FP32 and BF16 execution.
4. `linear_decay` entropy uses the nonexistent `cfg.train` field and therefore
   decays against one iteration instead of `cfg.training.iters`.
5. Configuration loading is type-strict but not value-strict for several
   scientific and optimizer fields. Invalid values documented as parse-time
   errors are accepted and fail late, behave incorrectly, or silently do no
   work.
6. Device-residency and synchronization claims are stronger than the code and
   tests. History packing and folded-summary caching perform unaccounted
   device-to-host scalar/vector reads and Python-side work in model hot paths.

These are implementation defects and contract gaps, not requests for later
optimization. They should be closed before a production run is called
supported.

## What is confirmed now

- NLHE is the only active game. There is no active game registry or importable
  Kuhn/Leduc/synthetic implementation.
- The active game exposes the intended public configuration fields `n_players`,
  `stack_bb`, and `rank_count`; `PokerConfig` correctly enforces player, stack,
  half-big-blind, rank, and card-capacity bounds.
- Full-deck (`R=13`) normal, sequential RB, and branched RB training execute on
  one GPU and two GPUs.
- Branched mode has one distinguished root per rank, applies ordinary
  randomized rounding at that root, and preserves duplicate labeled children.
- Posterior mass/moment reduction, leave-one-out sibling baselines, owner-clock
  lambda bootstrap, path-mass sampling, and distributed gradient SUM match the
  implemented fixed-intensity mathematical design.
- The two-network poker model has class-native preflop rows, compact compatible
  postflop live-player rows, pre-LN reach-biased MHA, exact incidence residuals,
  factorized type/raise heads, and actor-private-type values.
- Postflop folded players have no attention rows. The sibling evaluator supplies
  unary/pairwise partition corrections and per-folder mass, card, seat, and line
  statistics; the model consumes them and tests show that changing them changes
  output.
- The only registered composite losses are `ppo` for normal mode and `rb_ppo`
  for both RB modes. Both can use MSE or clipped value loss, exact legal entropy,
  and optional exact legal-action `KL(new || reference)`.
- The complete test suite passed on an idle RTX 2080 Ti, and direct one- and
  two-GPU training smokes produced finite metrics and checkpoints.

## Report set

- [Scope and method](01_scope_and_method.md)
- [Findings and required fixes](02_findings.md)
- [Mathematics and model traceability](03_math_and_model_traceability.md)
- [Runtime, configuration, and losses](04_runtime_configuration_and_losses.md)
- [Validation evidence](05_validation_evidence.md)
- [File-by-file inventory](06_file_inventory.md)
- [Release checklist](07_release_checklist.md)

This report supersedes the release conclusion in the older sibling audit
`docs/audit/production_training_readiness_2026-07-16.md`. That audit remains
useful performance evidence, but it did not detect the configuration-domain and
optimizer-gating defects above.
