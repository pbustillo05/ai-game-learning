# P12 — Evaluation (`eval/exploitability.py`, `eval/arena.py`)

**Depends on:** P01, P07, P11.
**Files:** `pbg/eval/exploitability.py`, `pbg/eval/arena.py`;
`tests/test_p12_eval.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## `exploitability.py` (synthetic games)

```python
def exploitability_of_checkpoint(ckpt_path, cfg) -> float
    # rebuild game+model, export_policy_tables (P01), exact exploitability (P01)
def exploitability_curve(cfg, ckpt_glob) -> list[dict]
    # sorted by iter; writes out_dir/exploitability.jsonl; returns rows
```
Feasibility guard inherited from P01 (`FeasibilityError` surfaces with a clear
message pointing at MAX_EXACT_NODES).

## `arena.py` (poker)

```python
@dataclass
class ArenaReport:
    bb100_mean: float; bb100_ci95: float; n_hands: int
    per_seat_bb100: list[float]          # button-relative
    seat_pairing: str                    # "duplicate"
def arena(ckpt_a, ckpt_b, cfg, n_hands, torch_gen, np_gen) -> ArenaReport
```
- Loads two `PokerModelBundle`s; plays full hands through `PokerGame`
  (sequential rollout machinery from P08 driven to TERMINAL — for evaluation
  the runout IS dealt: drive chance through pre-terminal states by uniform
  residual-deck deals; do not use MC completion here).
- **Seat assignment:** model A occupies one seat, B the remaining seats
  (report says which); iterate the A-seat over all positions.
- **Duplicate variance reduction (mandatory):** every deal (all hole cards +
  full runout, indexed by seat position) is replayed with A and B's seat
  assignments swapped; score = paired difference. CI via t-interval on paired
  per-deal differences.
- Actions at play time: sample from the model's composed policy at the TRUE
  dealt hand's row (evaluation is standard play, not RB).

## Tests (`test_p12_eval.py`)

- Exploitability of a freshly initialized (near-uniform) SyntheticNet on Kuhn
  is within 10% of exploitability(uniform tables) — links model export to P01
  exact tools.
- Curve function: run on three checkpoints produced by a 3-iteration training
  stub; jsonl rows well-formed and iter-sorted.
- Arena self-play null: same checkpoint as A and B, 2k duplicate deals →
  |bb100_mean| < 3·ci95 (seeded) and exactly 0 when the policy is
  deterministic (temperature 0 path: add `argmax_actions: bool` flag).
- Chip conservation and per-seat report shape on a 100-hand smoke.

## Do not

- No training logic; read-only over checkpoints.
- No MC completion in the arena path.
- No new model code.
