# Poker folded statistics

Status: implementation proposal.

## Contract

The engine's only state inputs are:

```text
board
ordered folded r_pi rows resolved from FoldedRanges[pre|post]
```

Partition tags and immutable deck/class maps describe representation; they are not additional game-state inputs.
`FoldedRanges` contains only `r_pi`. Folded statistics are written directly into the destination
`FamilyWork[post_A].retained.natural_input`.

`r_pi[K]` stores total probability mass per model class. It may be expanded uniformly across each class orbit in
bounded workspace or consumed by a `K`-native kernel. `r_pi[H]` is consumed directly. Neither representation is
retained twice outside preflop `natural_input` and `FoldedRanges[pre]`.

## Refresh

- A fold appends its `r_pi`, updates hot references, and recomputes statistics for resulting postflop player rows.
- A chance event recomputes statistics against the resulting board.
- Other transitions copy unchanged statistics.
- Tree completion releases `FoldedRanges` and all folded references; statistics remain in `natural_input`.

The implementation is repository-local under the proposed ownership boundary.
