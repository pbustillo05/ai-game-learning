# Poker folded statistics

Status: implementation proposal.

## Contract

For each `folded_type` in `{pre, post}`, folded row $f$, postflop family `post_A`, and player node $\sigma$:

```text
FoldedRanges[folded_type] <folded_cap[folded_type]>
  r_pi[f] <folded_size[folded_type]>

FamilyWork[post_A].retained
  model_input[sigma] <size[post_A]>
```

The statistics kernel reads only the board and ordered folded `r_pi` rows resolved from the references in
`FamilyWork.hot`. Partition tags and immutable deck/class maps describe representation; they are not game-state
inputs. The kernel writes directly into the destination `model_input`.

`folded_size[pre] = K`; each entry stores total mass for one model class.
`folded_size[post] = H`; all `post_A` families share this exact-hand partition.
A statistics kernel may expand a pre row uniformly across legal members of each class orbit in bounded workspace.

A completion-bound pre row expands once into the $H$-wide `Leaf.range_weights` required by the evaluator. This
transient evaluator input is not retained learner storage. No other owner retains both representations.

## Refresh

- A fold appends its `r_pi`, updates hot references, and recomputes statistics for resulting postflop player rows.
- A chance event recomputes statistics against the resulting board.
- Other transitions copy unchanged statistics.
- Queue processing does not release hot references.
- `align-and-seal` releases `FoldedRanges` and `FamilyWork.hot`; compact statistics remain in
  `FamilyWork.retained.model_input`.

The implementation is repository-local under the proposed ownership boundary.
