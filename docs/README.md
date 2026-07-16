# Implementation documentation

These pages describe the current working implementation. Historical prompts in
`../archive/prompts/` explain how the repository evolved, but they are not an
execution plan and should not be used as an implementation reference.

Read the pages in this order:

- [Architecture](architecture.md) defines package ownership, the actual
  iteration/chunk lifecycle, device residency, and the main retained objects.
- [Core algorithms](core_algorithms.md) describes normal, sequential
  Rao--Blackwell, and branched Rao--Blackwell rollout and optimization.
- [Poker implementation](poker.md) covers tensor NLHE navigation, the two poker
  networks, policy batch packing, evaluator calls, and the single declared
  mixed-row scheduler-control transfer.
- [`../archive/synthetic_games/`](../archive/synthetic_games/) preserves the
  retired Kuhn/Leduc/synthetic implementation as an inert historical snapshot.
- [File reference](file_reference.md) maps the implementation and validation
  files to their current responsibilities.
- [Training, evaluation, and configuration](training_and_evaluation.md) covers
  commands, presets, checkpoints, DDP, and evaluation. It is maintained as a
  separate operational guide.

Current readiness and the cleanup reachability inventory are indexed in
[`audit/README.md`](audit/README.md). Superseded performance investigations and
promotion comparisons are preserved under `../archive/`.

The binding authorities are:

- [`theory/core_math.tex`](theory/core_math.tex) for the mathematics;
- [`../NOTATION.md`](../NOTATION.md) for math-to-code names and shapes;
- [`../CONTRACTS.md`](../CONTRACTS.md) for interfaces and invariants; and
- [`../archive/POKER_GPU_EVALUATOR_IO.md`](../archive/POKER_GPU_EVALUATOR_IO.md)
  for the sibling poker-evaluator boundary.

Two theory documents have narrower scopes:

- [Poker player model](theory/MHA_model.md) specifies the preflop/postflop
  architecture and attention equations in detail.
- [Card removal from folded ranges](theory/folder_impact.md) is evaluator-owned
  background mathematics, not a local implementation specification.
