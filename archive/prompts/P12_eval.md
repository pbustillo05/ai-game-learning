# P12 — Evaluation after corrected minimal training

**Depends on:** P01, P07, C11.
**Files:** `pbg/eval/exploitability.py`, `pbg/eval/arena.py`;
`tests/test_p12_eval.py`.
**May touch:** `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, and `CONTRACTS.md`;
inspect the completed C11 bundle/checkpoint API, P01 exact tools, P07 poker
adapter, evaluator facade contract, the shipped sibling API documented in
`../poker-gpu-evaluator/docs/api_pbg.md`, and existing evaluation stubs. The
local contract already reflects evaluator package 0.3.0: `COMBO_CARDS` is
public and the removed partition query/seed arguments must not be used.
Dependencies describe code already present, not prompt text to paste. Run
focused baseline tests, then implement both evaluators, tests, and notes before
focused and full suites.

## Goal

Provide read-only checkpoint evaluation. Evaluation uses exact/CPU control
flow and is not part of the device-resident training hot path. Do not depend on
Neyman state, public moment heads, stored Q targets, or reference snapshots.

## Synthetic exploitability

```python
def exploitability_of_checkpoint(ckpt_path, cfg) -> float: ...
def exploitability_curve(cfg, ckpt_glob) -> list[dict]: ...
```

- Rebuild the C11 synthetic `GameBundle`, load `bundle.model`, export policy tables, and
  call the P01 exact exploitability tools.
- Preserve the P01 feasibility guard and include `MAX_EXACT_NODES` guidance in
  the error.
- Sort checkpoints by stored iteration, write
  `out_dir/exploitability.jsonl`, and return the same rows.

## Poker arena

```python
@dataclass
class ArenaReport:
    bb100_mean: float
    bb100_ci95: float
    n_hands: int
    per_seat_bb100: list[float]
    seat_pairing: str = "duplicate"

def arena(ckpt_a, ckpt_b, cfg, n_hands, torch_gen, np_gen,
          *, argmax_actions=False) -> ArenaReport: ...
```

`n_hands` counts physical replayed hands, must be a positive multiple of
`2*n_players`, and therefore contains `n_hands/2` duplicate pairs allocated
equally across singled-out seats. Rebuild `PokerModelBundle` and
`PokerPolicyAdapter` directly from P07/config here; P13B's training bundle does
not exist yet.

- The canonical CPU `NLHEEngine` drives evaluation; do not use the training
  GPU rollout backend or MC completion to play a hand.
- Obtain private profiles through the sibling joint-profile API. Sample one
  full public runout uniformly from its complement and reuse it for the paired
  replay. Rely on the existing adapter's all-1326-row `COMBO_CARDS` compatibility
  check; do not construct an alternate evaluator combo map in arena code.
- Rotate the singled-out seat `s` through every button-relative position. In
  replay AB, A occupies `s` and B every other seat; record A's payoff at `s`.
  In replay BA, B occupies `s` and A every other seat; record B's payoff at
  `s`. The paired difference is `payoff_AB[s] - payoff_BA[s]`. By chip
  conservation this is A's aggregate payoff over the two physical hands.
  Convert each pair to A bb/100 as `100*difference/(2*bb)`; use these paired
  observations for the mean, CI, and button-relative per-seat report.
- For stochastic policies, use seat/history-indexed common random uniforms in
  both replays so action randomness is maximally coupled. For
  `argmax_actions=True`, the same checkpoint paired with itself is exactly zero.
- Send every fixed showdown through
  `poker_gpu_evaluator.pbg.evaluate_showdowns`; local code only assembles public
  pot-stratum payoffs. Fold wins remain local.
- Compute the confidence interval from independent pair-level bb/100 values,
  using a Student-t interval when there is more than one pair.

## Tests

- Fresh synthetic model exploitability agrees with exported uniform policy
  within 10% on Kuhn.
- Three stub checkpoints produce iteration-sorted identical return/JSONL rows.
- Deterministic self-play duplicate arena is exactly zero.
- Stochastic self-play lies within three reported standard errors with fixed
  common-random-number seeds.
- Chip conservation, seat rotation and per-seat report shapes hold.
- Fake sibling facade proves every showdown crossed the public boundary and
  its stipulated stratum shares determined rewards.
- Real-sibling boundary coverage confirms the defensive `COMBO_CARDS` export
  agrees with local `hand_index`/`hand_cards` for all 1326 rows.

Required commands:

```bash
pytest -q tests/test_p1_exact.py tests/test_p7_adapter.py
pytest -q tests/test_p12_eval.py
pytest -q
```

## Do not

- No training, local poker evaluator, MC completion, moment/Neyman logic, or
  reference-policy machinery.
