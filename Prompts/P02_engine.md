# P02 — Public NLHE engine and representation utilities

**Depends on:** P00.
**Files:** `pbg/games/poker/engine.py`, `pbg/games/poker/cards.py`;
`tests/test_p2_engine.py`, `tests/test_p2_cards.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Implement navigation of the public no-limit hold'em betting tree (cash,
2–9 players, default 6-max/100bb, no ante/straddle/rake) and only the
deterministic card representations needed by the model and evaluator boundary.
This prompt must not implement poker hand strength, hand categories, winner
selection, tie resolution, equity, or showdown payoff logic.

## `cards.py`

Card ids and combo ids follow `POKER_GPU_EVALUATOR_IO.md`. Frozen API:

```python
def hand_index(a: int, b: int) -> int
def hand_cards(h: int) -> tuple[int, int]
def class_of_hand(h: int) -> int
def class_members(k: int) -> list[int]
def class_size(k: int) -> int
def suit_perm_apply(perm: tuple[int, int, int, int], h: int) -> int
def board_conflict_mask(board) -> BoolTensor
def sample_uniform_complement(
    unavailable_cards: Tensor,  # int16 [B,U], -1 padded; supplied by evaluator/profile
    n_cards: Tensor,            # int32 [B], normally 1 or 3
    gen: torch.Generator,
) -> Tensor:                    # int16 [B,max(n_cards)], -1 padded
N_HANDS, N_CLASSES = 1326, 169
```

`sample_uniform_complement` samples an ordered draw without replacement,
uniformly from `{0..51} \ unavailable_cards[b]`. It knows no street or private
assignment law. The caller supplies every unavailable card; in branching this
set is constructed from the profile returned by `poker_gpu_evaluator` plus the
current board. For a three-card flop, the ordered draw is only a sampling
mechanism: `deal` canonicalizes the three cards as one unordered public event,
so its induced law is uniform on the combinations specified by
`Docs/latex/core_math.tex`. Turn and river remain single-card events. Vectorized tensor
variants of the deterministic ID maps are allowed.

Forbidden in this module: `evaluate7`, `hand_rank`, category/kicker tables,
rank comparisons, winner masks, showdown shares, equity, or side-pot winners.

## `engine.py`

```python
@dataclass
class PokerConfig:
    n_players: int = 6
    stack_bb: float = 100.0
    sb: float = 0.5
    bb: float = 1.0
    n_ticks: int = 64

class NLHEEngine:
    def __init__(self, cfg: PokerConfig): ...
```

The SoA state table contains street, board, stacks, street/total commitments,
status (live/folded/all-in), actor, last full raise, current bet, acted/reopen
state, button, pot, hand-over, and betting-closed fields.

Implement public rules exactly:

- blind posting and button-relative pre/postflop action order;
- fold/check/call and raise-to events;
- the affine legal raise-tick grid, with the final tick exactly all-in;
- minimum full raises and the all-in-under-raise no-reopen rule;
- betting-round closure, street transitions, fold-win terminal nodes, and
  showdown pre-terminal chance nodes (the state before the remaining
  runout/chance outcome is drawn);
- validation/application of board cards supplied through `deal`.

The engine does not enumerate chance outcomes or choose cards.

## Pot stratification

The engine owns public chip accounting and exposes:

```python
@dataclass
class PotStrata:
    stratum_players: Tensor  # bool [B,K,P], showdown eligibility
    stratum_amount: Tensor   # float32 [B,K]
    payoff_baseline: Tensor  # float32 [B,P], normally -committed_total

def pot_strata(state_refs) -> PotStrata: ...
def assemble_showdown_payoff(
    strata: PotStrata,
    showdown_share: Tensor,  # float32 [B,K,P], evaluator output
) -> Tensor:                 # float32 [B,P]
```

`assemble_showdown_payoff` is only affine arithmetic:
`baseline + sum_k amount[k] * share[k,p]`. It may validate shapes, nonnegative
shares, eligibility, and conservation, but may not inspect cards or compare
hands. Fixed-showdown shares come only from
`poker_gpu_evaluator.pbg.evaluate_showdowns` through P05.

Fold-win payoff assembly is local because no hand comparison is involved.

## Tests

- Complete ID-map, class-size, suit-action, conflict-mask, and complement-sampler
  tests. Statistically test complement sampling against the uniform law.
- Script public betting fixtures: blind/action order, limp-fold, multiway
  all-ins, minimum raise, under-raise no-reopen, check-down street transitions,
  and button rotation.
- Verify nested strata and amounts on unequal-stack contribution fixtures.
- Feed stipulated fake evaluator shares into `assemble_showdown_payoff`; verify
  conservation and affine assembly, including ties and multiple strata.
- Static forbidden-symbol scan rejects local hand evaluators, hand categories,
  kicker ordering, and rank comparisons.

Do not test known poker-hand winners here. Those tests belong to
`poker-gpu-evaluator`.
