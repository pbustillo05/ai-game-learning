# P01 — Synthetic PBG family ("the dial"), Kuhn/Leduc presets, exact tools, tabular model

**Depends on:** P00.
**Files:** `pbg/games/synthetic/family.py`, `presets.py`, `exact.py`,
`model.py`; `tests/test_p1_family.py`, `test_p1_exact.py`, `test_p1_adapter.py`.
**May touch:** `configs/kuhn_smoke.yaml` (fill real values), append to
`IMPLEMENTATION_NOTES.md`.

## Goal

A parametrized family of small card games implementing `PBGGame`, whose dials
independently scale (a) public-tree size, (b) private-state size, (c) player
count; with Kuhn and Leduc poker as exact presets; plus exact posterior /
best-response / exploitability tools and a small model + adapter so the family
can be trained end-to-end by later prompts.

## Game family spec (`family.py`)

```python
@dataclass
class SyntheticDialConfig:
    n_players: int            # >= 2
    deck_size: int            # D distinct cards, ranks 0..D-1 (rank r = r // n_suits)
    n_suits: int              # D must be divisible by n_suits
    hand_size: int            # m cards per player, dealt WITHOUT replacement
    n_rounds: int             # betting rounds
    board_per_round: tuple[int, ...]  # public cards revealed BEFORE round i (board_per_round[0]==0)
    max_raises: int           # raise cap per round
    bet_size: tuple[int, ...] # fixed bet size per round (limit betting)
    ante: int = 1
```

- **Private types.** `x_sizes[p] = C(deck_size, hand_size)` for all p; type
  index = colex index of the hand's sorted card set. μ = uniform over profiles
  with pairwise-disjoint hands. `sample_mu` by sequential draws without
  replacement.
- **Public tree.** Rounds proceed: (chance: deal `board_per_round[i]` cards as
  ONE chance event whose outcome is the sorted set of new cards; skip the
  chance node when the count is 0) then a limit betting round: players in seat
  order act with events from {`fold`, `check_call`, `raise`} where `raise` is
  legal while the round's raise count < `max_raises`, `fold` is legal only
  facing a bet. Betting round ends by standard limit rules (all live players
  matched, everyone acted). Game ends when one player is live (fold win) or
  after round `n_rounds` betting closes (showdown).
- **Chance rule.** `chance_probs(state, xprofile)` = uniform over card-sets
  disjoint from all dealt hands and prior board (this makes chance depend on
  the full profile x, as in the PBG definition — do not approximate).
- **Pre-terminal.** Follow `Docs/latex/core_math.tex`/NOTATION §1 exactly:
  τ^pre-c is the public prefix at the final chance node, before that chance
  outcome is drawn. Thus once betting has closed for the final time,
  `is_pre_terminal` is true before the remaining board/runout chance event; the
  state must remain steppable through that chance event and showdown to a
  terminal state. A fold-win that is already terminal is classified TERMINAL,
  not a nonterminal PRETERM leaf.
- **Reward (showdown).** Zero-sum. Pot = antes + committed bets. Hand strength:
  `leduc_rule(hand, board)`: a hand containing a card whose rank pairs a board
  rank outranks any unpaired hand; ties broken by best rank in hand
  (descending), then split pot. This reduces to standard Kuhn/Leduc rules for
  the presets and generalizes smoothly with the dials. Losers lose their
  committed chips; winners split the pot equally on exact ties. `reward`
  returns chips won/lost per player (float32), Σ_p = 0.
- **StateTable.** SoA over int32 state_refs, vectorized `step`; store: round,
  raises_this_round, to_act, committed[P], live[P], board (int8 padded),
  history hash for `encode_public`. `encode_public` returns int64 node-key
  tensor (perfect hash of public history) — the synthetic model embeds it.
- **`reach_update`.** log 𝔯 rows [P, X_max]: on a player event, add
  `log policy_rows[p, x, event]` to that player's row; on a chance/board
  event, add `-inf` to every type of every player whose hand intersects the
  revealed cards. (No renormalization; consumers normalize.)
- **`sample_profiles`.** Exact joint posterior draws for tests and branching:
  enumerate feasible disjoint profiles when `prod(x_sizes) <= 2e6` and sample
  from the normalized product of reach weights × disjointness; otherwise
  rejection-sample (draw each player's type ∝ its reach row, accept if hands
  pairwise disjoint and board-disjoint). Return int32 [B, n, P].

## Presets (`presets.py`)

- `kuhn()` = Config(n_players=2, deck_size=3, n_suits=1, hand_size=1,
  n_rounds=1, board_per_round=(0,), max_raises=1, bet_size=(1,), ante=1).
- `leduc()` = Config(2, deck_size=6, n_suits=2, hand_size=1, n_rounds=2,
  board_per_round=(0,1), max_raises=2, bet_size=(2,4), ante=1).
- `dial(public_scale: int, private_scale: int, n_players: int)` — a documented
  helper mapping small integers to configs that grow the public tree
  (max_raises, n_rounds) and private space (deck_size, hand_size) respectively.

## Exact tools (`exact.py`) — numpy float64, CPU only

```python
MAX_EXACT_NODES = 2_000_000
class FeasibilityError(Exception): ...
def enumerate_public_tree(game) -> PublicTreeIndex   # nodes, children, owners; guard on size
def exact_posterior(game, tree_index, node) -> ndarray      # [P, X] joint-marginal, exact
def best_response_value(game, policy_tables) -> ndarray     # [P] value of BR_p vs pi_{-p}
def exploitability(game, policy_tables) -> float            # sum_p (BR_p - J_p(pi))
def uniform_policy_tables(game) -> PolicyTables
```
`PolicyTables = dict[int_node_key, ndarray[X, E]]` (rows valid-type-masked,
normalized over legal events). `exact_posterior` sums μ × reach products over
disjoint profiles (exact enumeration; use the feasibility guard).
Best response: backward induction over infosets (x_p, public node) against
fixed opponents, correctly marginalizing opponents' types with the exact
posterior at each infoset. Exploitability of an exact Kuhn equilibrium must
evaluate to 0 within 1e-10.

## Synthetic model + adapter (`model.py`)

CONTRACTS delta (signed off): the synthetic family owns a small model so the
generic training loop can run on it.

```python
class SyntheticNet(nn.Module):
    # embeds public node-key (hash bucket table, cfg 'n_buckets') and a learned
    # per-type embedding; two-layer MLP -> logits [B, X_max, E_max] and V [B, X_max].
    def forward(self, pub_keys, legal_mask) -> PolicyOutput   # raw = masked logits
class SyntheticPolicyAdapter:   # implements PolicyAdapter
    # categorical over legal events: exact log_prob / entropy / kl from raw logits
```
Also `export_policy_tables(net, game, tree_index) -> PolicyTables` (evaluate the
net at every public node; used by exploitability tests and P12).

## Tests

- `test_p1_family.py`: Kuhn public tree has exactly the known node/terminal
  counts (compute and assert: 2-player Kuhn has 9 non-root decision states and
  6 terminal histories per deal ordering — derive counts in the test from an
  independent hand-rolled enumeration of Kuhn, not from the family code);
  ν/legality invariants; `is_pre_terminal` correct on hand-written Leduc
  histories including fold-endings; reward zero-sum on 10k random playouts;
  `reach_update` matches brute-force posterior numerators on random Leduc
  paths (fp64, atol 1e-12).
- `test_p1_exact.py`: exploitability(uniform) > 0 on Kuhn; exploitability of
  the known Kuhn equilibrium family (alpha-parametrized; implement the closed
  form in the test for alpha ∈ {0, 1/6, 1/3}) equals 0 within 1e-9;
  `exact_posterior` rows sum to 1 and match Monte Carlo playout frequencies on
  Leduc (chi-square, seeded); FeasibilityError raised on a large dial config.
- `test_p1_adapter.py`: adapter log_prob normalizes (logsumexp over legal
  events == 0 per (row, type)); entropy matches numerical −Σp log p; kl(out,
  out) == 0.

## Do not

- Do not implement TreeBatch, rollout, or training here.
- Do not special-case Kuhn/Leduc inside `family.py` — they must be pure configs.
- Do not use string game histories as runtime state (only for debugging repr).
