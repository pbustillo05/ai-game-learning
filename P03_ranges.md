# P03 — Ranges & Bayesian machinery (`ranges.py`)

**Depends on:** P00, P02.
**Files:** `pbg/games/poker/ranges.py`; `tests/test_p3_ranges.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Reach caches over 1326-hand ranges: multiplicative Bayes updates on actions,
board masking on deals, live/folded bookkeeping, fold-time snapshots for the
folded-stats module (P04), and 169-class aggregation for the preflop model.

## Normalization conventions (binding; from folder_impact §1.5)

- Raw reach `r[p, h]` is unnormalized; stored in log space fp32.
- Board reveals **mask** conflicting hands (`log r = -inf`) — they never
  renormalize per-hypothesis sets. Normalized views are computed on demand.
- Folded players' ranges are **frozen at fold time**: snapshot the raw f_k
  weights *evaluated at the fold decision* (already conditioned on the board
  visible then); subsequent board cards do not update a folder's snapshot
  (P04 handles later-board conditioning through its own removal algebra).

## Spec

```python
class RangeState:
    """One table for a batch of rollout nodes; row-indexed by (node_slot, player)."""
    log_r: Tensor            # fp32 [S, P, 1326]; -inf = impossible
    live: Tensor             # bool [S, P]
    board: Tensor            # int8 [S, 5], -1 pad
    def clone_rows(self, src_slots) -> new_slots            # branching support: O(P*1326) copy
    def update_action(self, slots, players, event_ids, policy_log_rows) -> None
        # log_r[slot, p, h] += policy_log_rows[slot, h, event]  (actor rows only)
    def update_board(self, slots, new_cards) -> None
        # -inf mask for every player's conflicting hands; folded players' rows too
        # (their snapshot lives separately, see below)
    def fold(self, slots, players) -> list[FoldedPlayer]
        # marks folded; returns snapshots taken BEFORE any later masking
    def normalized(self, slots, players) -> Tensor          # tilde r: softmax over valid hands
    def to_class(self, slots) -> Tensor                     # [S, P, 169] sum over class members of normalized r

@dataclass
class FoldedPlayer:
    f_weights: Tensor        # fp32 [1326] raw unnormalized f_k at fold time (linear space)
    board_at_fold: Tensor    # int8 [5]
    line_meta: Tensor        # fp32 [d_line]: street of fold, n_raises faced, pot_ratio, position — frozen layout, d_line=8, zero-pad
```

- All methods vectorized over slots; no Python loops over 1326.
- `update_action` consumes the model's per-hand log-policy row for the taken
  event (the rollout, P08, provides it) — ranges.py itself never calls a model.
- Numerical hygiene: keep an fp32 running max per row and renormalize the log
  table (subtract rowmax) whenever it drifts below −30 to avoid underflow;
  document that only *differences* are meaningful.

## Tests (`test_p3_ranges.py`)

- Bayes correctness: drive a scripted 3-player toy sequence with hand-written
  2×2 policies embedded into the 1326 table (mass on 4 chosen hands); compare
  `normalized` against an fp64 brute-force posterior numerator (atol 1e-6).
- Board masking: after dealing a flop, every hand intersecting it has −inf in
  all rows; `normalized` sums to 1 over survivors.
- Fold snapshot: snapshot equals raw pre-fold weights; later `update_board`
  does not modify the snapshot; `line_meta` layout populated.
- Class aggregation: `to_class` sums match per-class brute sums; preflop (no
  board) class masses over a uniform range equal n_k/1326.
- `clone_rows` independence: mutating the clone leaves the source unchanged.

## Do not

- No folded-stats math (Φ, α_B, β_B) — that is P04, which consumes
  `FoldedPlayer`.
- No model calls, no engine stepping.
- No per-hypothesis renormalization anywhere.
