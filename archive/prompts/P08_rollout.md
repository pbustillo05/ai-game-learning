# P08 — TreeBatch, ReachCache plumbing, sequential rollout (`tree.py`, `rollout.py`)

> **Historical prompt — do not launch on the current repository.** C08 is the
> self-contained replacement contract for the rollout boundary and device model.

**Depends on:** P00, P01 (test bed), P05 (Protocols).
**Files:** `pbg/core/tree.py`, `pbg/core/rollout.py`; `tests/test_p8_tree.py`,
`test_p8_rollout.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

The stored rollout tree (SoA/CSR) with alias-table minibatch support, the
reach-cache plumbing with its three storage modes, and the sequential rollout
path: `beta_mode="sequential"` produces single-path trees ending at
pre-terminal leaves. (P09 generalizes the sampler to branching; the growth
loop you write here must already be frontier-based so P09 only changes the
child-count and event-draw rule — structure the code accordingly with a
`_draw_children(...)` hook.)

## `tree.py`

- `TreeBatch` exactly per PREAMBLE, plus:
  - `TreeBuilder`: append-only frontier construction (python-side lists of
    numpy arrays are fine during growth), `finalize() -> TreeBatch` producing
    contiguous CSR, computing `depth_p`, `depth_player`, `path_weight`
    (∏ ν(e|ρ) along root→node — parent-edge ν, matching the source's
    branched-terminal law and NOTATION §4).
  - `validate()`: ν rows sum to 1 (atol 1e-6) on non-leaves; leaves are
    TERMINAL/PRETERM; parents/depths consistent; path_weight of a child =
    parent's × its edge ν.
  - `AliasTable.build(weights) / sample(n, gen)`: Walker alias method, exact;
    `build_decision_alias(tree) -> AliasTable` over player-owned nodes with
    weights `path_weight` (NOTATION §4: this single table implements the
    length-biased tree choice + within-tree decision choice in one step).
- `pbg/core/minibatch.py` is P11's file — do NOT create it; alias tables live
  here and are consumed there.

## Reach plumbing (extend `core/completion.py`'s `ReachView` — May touch? NO:
put the cache class in `tree.py` to respect file ownership)

```python
class ReachCache:
    mode: str                       # "full" | "decisions_only" | "recompute"
    log_r: Tensor                   # fp32 [N_stored, P_max, X_max]
    node_index: Tensor              # int32 [N_stored]
    def rows_for(self, tree, node_ids, game, policy_cache) -> Tensor  # [B, P, X]
```
- "full": every node stored. "decisions_only": player nodes + every
  PRETERM/TERMINAL leaf.
  "recompute": store root only; `rows_for` walks parent chains (batched by
  depth) replaying `game.reach_update` using the `PolicyCache` (below).
- `PolicyCache` (also this file): per player-node storage of the rollout
  policy's per-type log-probs for **stored children only**:
  `logp_child fp32 [Nd, E_max, X_max]`, `v_old fp32 [Nd, X_max]`, plus row
  index. This is the `OldOutCache` of CONTRACTS; frozen name here:
  `PolicyCache`. It serves (a) recompute-mode reach replay, (b) P11 ratio/
  advantage terms.

## `rollout.py`

```python
def sample_trees(game, policy_fn, adapter, cfg_rollout, completion_reach_hook,
                 intensity, torch_gen, np_gen) -> tuple[TreeBatch, ReachCache, PolicyCache]
```
- Frontier loop over ALL trees simultaneously (one queue; the game adapter
  batches model calls internally — poker splits by street): pop a frontier
  slice, classify {player, chance, pre-terminal, terminal}; for player nodes
  call `policy_fn` once per unique state batch; draw children via
  `_draw_children(node_kind, policy_out, ...)`.
- Sequential mode (`beta_mode == "sequential"`, the only mode this prompt
  implements): exactly one child per node. Player event: draw one **private
  profile** via `game.sample_profiles(n=1)` and then one event
  a ~ π(· | x_actor, σ) from the policy rows (NOTATION §5 public-mixture rule
  at K=1; W1: profile is drawn fresh at this node, used for this draw only,
  and discarded — never persisted or reused downstream). For poker every
  profile comes from the sibling exact joint-profile API. Chance event: same
  profile draw, then a uniform outcome from the complement of that supplied
  profile plus board. No generic/local code constructs or repairs poker
  profiles. ν(edge) = 1.
- Stop expanding at pre-terminal states—the final chance node before its
  outcome is drawn; do not sample that chance edge. Mark PRETERM and store
  reach rows there regardless of mode (completion needs them).
- At a TERMINAL leaf, also store reach rows regardless of mode so the caller
  can obtain its collision-conditioned private-type posterior. The terminal
  reward may be exact/public; the posterior is still joint-prior conditioned.
- `intensity` argument accepted but must be `None` in sequential mode (assert).
- Reach maintenance: after each expansion, update log 𝔯 rows per mode; player
  updates use the policy's per-type log-prob of the TAKEN event (all types,
  one event) — the Rao–Blackwell object; never the sampled profile's row only.
- Budget: hard stop with a clear error if any tree exceeds
  `cfg.rollout.node_budget` (should be impossible in sequential mode for
  finite games — still enforce).

## Tests

- `test_p8_tree.py`: builder→finalize round-trip on hand-built trees;
  validate() catches (a) ν rows not summing to 1, (b) wrong path_weight;
  alias table: empirical frequencies of 1e6 draws match weights (chi-square,
  seeded); decision alias excludes chance/leaf nodes.
- `test_p8_rollout.py` (on Kuhn + Leduc from P01, with the P01 SyntheticNet
  at random init as policy_fn):
  - every tree is a single path root→PRETERM/TERMINAL; ν ≡ 1;
  - **distributional check**: 50k sequential rollouts' terminal-history
    frequencies match exact ℙ^π computed from exported policy tables (P01
    `export_policy_tables` + fp64 tree walk; chi-square p > 1e-3, seeded);
  - reach rows at PRETERM leaves equal fp64 replayed posterior numerators in
    all three storage modes (atol 1e-5), and the three modes agree exactly
    with each other;
  - PolicyCache rows reproduce `adapter.log_prob` recomputed on the spot
    (atol 1e-6).

## Do not

- No branching (K>1), no Poisson, no intensity logic (P09).
- No loss/minibatch logic (P11) beyond building alias tables.
- Never persist or condition on sampled private profiles (W1).
