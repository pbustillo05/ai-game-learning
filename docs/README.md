# Implementation documentation

This directory documents the repository as it exists now. The implementation
prompts in `../archive/prompts/` are retained as development history; they are
not a setup or execution sequence.

Start here:

- [Architecture](architecture.md) explains package boundaries, runtime data
  flow, ownership, and the main design decisions.
- [Core algorithms](core_algorithms.md) describes rollout, labeled branching,
  reach tracking, completion, posterior-moment propagation, minibatching, and
  the RB-PPO loss.
- [Synthetic games](synthetic_games.md) covers the configurable finite game
  family, exact solvers, neural model, and completion backend.
- [Poker implementation](poker.md) covers public NLHE navigation, range
  bookkeeping, the preflop/postflop models, GPU rollout, and the external
  evaluator boundary.
- [Training, evaluation, and configuration](training_and_evaluation.md) covers
  configuration, checkpoints, DDP, exploitability, duplicate-deal arenas, and
  the shipped presets.
- [File reference](file_reference.md) gives a purpose and status for every
  tracked file in the repository, including tests and historical prompts.
- [Audit](audit/README.md) records repository-readiness findings, validation
  probes, and resolved items.

The mathematical and interface authorities remain:

- [`theory/core_math.tex`](theory/core_math.tex) — mathematical source of
  truth.
- [`../NOTATION.md`](../NOTATION.md) — mapping from the mathematics to code.
- [`../CONTRACTS.md`](../CONTRACTS.md) — frozen interfaces and invariants.
- [`../archive/POKER_GPU_EVALUATOR_IO.md`](../archive/POKER_GPU_EVALUATOR_IO.md) — binding
  contract with the sibling poker evaluator.

Two existing research documents have narrower scopes:

- [Poker Player Model](MHA_model.md) specifies the poker network in more
  mathematical detail.
- [Card Removal from Folded Ranges](theory/folder_impact.md) is background mathematics
  owned by the evaluator side of the repository boundary. It is not a local
  implementation specification.
