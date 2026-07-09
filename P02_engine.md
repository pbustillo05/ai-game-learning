# P02 — NLHE engine (`engine.py`) and card utilities (`cards.py`)

**Depends on:** P00.
**Files:** `pbg/games/poker/engine.py`, `pbg/games/poker/cards.py`;
`tests/test_p2_engine.py`, `test_p2_cards.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

A vectorized 6-max no-limit hold'em state machine (cash, 100bb, no
ante/straddle/rake, chip-EV) and exact card utilities. This module knows
nothing about types/ranges/models — it is a rules engine over *public* state
plus a showdown evaluator that takes hands as arguments.

## `cards.py`

Deck ids 0..51; `rank(v)=v//4`, `suit(v)=v%4`. Frozen API:

```python
def hand_index(a: int, b: int) -> int          # unordered pair -> h in [0,1326)
def hand_cards(h) -> tuple[int,int]            # inverse; vectorized variants *_t for tensors
def class_of_hand(h) -> int                    # 169-class id (pair / suited / offsuit by ranks)
def class_members(k) -> list[int]              # 4, 6 or 12 hands (assert sizes)
N_HANDS, N_CLASSES = 1326, 169
def class_size(k) -> int                       # n_k
def suit_perm_apply(perm: tuple, h) -> int     # S4 action on hands; vectorized variant
def board_conflict_mask(board) -> BoolTensor   # [1326] True where hand disjoint from board
def evaluate7(board5: Tensor, hands: Tensor) -> Tensor
    # int32 strength, higher wins; vectorized [B] boards x [B, n] hands -> [B, n]
def class_overlap_stats() -> Tensor            # [169,169] expected |h ∩ g| over class orbits (fp32, precomputed)
```

`evaluate7`: implement a standard 7-card evaluator (rank categories
high-card..straight-flush with correct kicker ordering). A lookup-table or
bit-trick implementation is fine; correctness gates below are strict.

## `engine.py`

```python
@dataclass
class PokerConfig:
    n_players: int = 6           # 2..9 supported
    stack_bb: float = 100.0
    sb: float = 0.5; bb: float = 1.0
    n_ticks: int = 64            # raise grid resolution knob, see tick rule
class NLHEEngine:
    def __init__(self, cfg: PokerConfig): ...
```

SoA StateTable over int32 `state_refs` (preallocate, grow by doubling):
`street (0=pre,1=flop,2=turn,3=river)`, `board (5×int8, -1 pad)`,
`stacks[P] fp32`, `committed_street[P]`, `committed_total[P]`,
`status[P] (0 live,1 folded,2 allin)`, `to_act int8`, `last_full_raise fp32`,
`current_bet fp32`, `n_actions_street`, `button int8`, `pot fp32`,
`hand_over bool`, `betting_closed bool`.

Rules to implement exactly (document each with a comment):

- Blinds posted at root; preflop action starts left of bb; postflop left of
  button. Standard NLHE min-raise rule: a raise must be ≥ last full raise;
  all-in below min-raise does not reopen action for players who already acted.
- **Event space at a player node** (this is the engine's public action
  alphabet; the model's continuous head maps onto it in P07):
  `fold` (only facing a bet), `check_call`, and raise-to amounts on the tick
  grid `N(σ)`: legal integer ticks `n ∈ {n_min..n_ticks}` mapping affinely to
  raise-to sizes in `[min_raise_to, all_in_to]`, with `n = n_ticks` exactly
  all-in. `legal_events` returns event ids and mask; event id layout frozen:
  `0=fold, 1=check_call, 2+t = raise tick t`. `E_max = 2 + n_ticks`.
- Chance nodes: one event per street deal; the *outcome space* (which cards)
  is not enumerated by the engine — `step` on a chance node takes the dealt
  cards via a side channel `deal(state_refs, cards)`; `owner_of` returns
  CHANCE when street must advance. (The poker adapter, P07, samples the cards;
  the engine only validates and applies them.)
- Street advance when betting closed and >1 live non-allin player can still
  act on later streets; run-out semantics: when all remaining players are
  all-in (or exactly one can act but betting is closed), the state is
  **pre-terminal** — `is_pre_terminal` True; remaining board cards are dealt
  by completion (P05), not by the rollout. Also pre-terminal: betting closed
  on the river; and fold-wins are TERMINAL directly (reward needs no showdown).
- `payoffs(state_refs, hands[P,2], ) -> fp32 [B, P]`: side-pot-exact showdown
  settlement using `evaluate7` (folded players lose `committed_total`). Chips
  are zero-sum per hand (assert).

## Tests

- `test_p2_cards.py`: `hand_index`/`hand_cards` bijection over all 1326;
  class sizes sum: Σ n_k = 1326 with multiset {4:13*? ...} — assert 13 classes
  of size 4 (pairs... careful: pairs have 6 combos) — implement the assert from
  first principles in the test: pairs 13 classes × 6 combos, suited 78 × 4,
  offsuit 78 × 12; `suit_perm_apply` is a group action (identity, composition,
  class invariance). `evaluate7` fixtures: ≥ 30 hand-vs-hand cases covering
  every category boundary (wheel straight, flush vs straight, full house
  kickers, board-plays ties, quads vs boat) — hardcode expected winners.
- `test_p2_engine.py`: scripted hand fixtures (write ≥ 8): limp-fold preflop;
  3-bet all-in call with exact side pots for 3 players of unequal stacks;
  min-raise legality including the all-in-under-raise no-reopen rule; check
  down to river → pre-terminal flag; button rotation & blind posting; total
  chips conserved on 10k random-legal-action playouts (drive the engine with a
  seeded random legal policy and random deals).

## Do not

- No ranges, types, models, or reach logic (P03+).
- No PBGGame implementation here (P07's adapter owns it).
- Do not enumerate chance outcomes as engine events.
