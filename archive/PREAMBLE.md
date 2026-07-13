# PREAMBLE — paste this above every prompt, in a fresh context window

You are implementing one module of a research codebase for policy-gradient
self-play in Public-action Bayesian Games (PBGs), with no-limit hold'em poker as
the first instance. The mathematical source of truth is
`Docs/latex/core_math.tex`; its implementation mapping is `NOTATION.md`.
Interfaces and evaluator ownership are frozen by `CONTRACTS.md` and
`POKER_GPU_EVALUATOR_IO.md`. This preamble
reproduces the parts you need in every prompt; the prompt that follows it
reproduces the module-specific parts. **You never need information beyond what
is pasted; if something seems missing, it is a signal to re-read, not invent.**

## Rules of engagement (binding)

- **R1.** Never rename, re-sign, or "improve" any interface given here. If a
  contract seems wrong, implement it as written and add a
  `# CONTRACT-QUESTION: <one line>` comment at the site.
- **R2.** Only create/modify files listed under **Files** in the prompt. Never
  touch other modules' files except those under **May touch**.
- **R3.** Finish with the full test suite green: `pytest -x -q`. Tests from
  earlier prompts must still pass. Write the new tests named in the prompt.
- **R4.** No abstractions beyond spec: no metaclasses, no plugin frameworks, no
  clever indirection. Plain functions, dataclasses, `nn.Module`s, Protocols.
- **R5.** Math semantics come from `Docs/latex/core_math.tex` as transcribed in
  the quoted NOTATION excerpts, not from memory of standard RL conventions. If
  a quoted excerpt appears to conflict with the source, report it; the source wins.
- **R6.** Every public function/class gets a docstring with tensor shapes.
  Every tensor-returning line of nontrivial shape gets a `# [dims]` comment.
- **R7.** All poker-private combinatorics are owned by the sibling
  `poker-gpu-evaluator` repository: collision-conditioned private laws and
  samples, partition/folded-card summaries, completion/equity coefficients,
  hand ranking, ties, and per-pot-stratum winner shares. Follow
  `POKER_GPU_EVALUATOR_IO.md`; never implement a local CPU "reference", copy a
  kernel, compare hand ranks/categories, or import sibling internals.

## Global conventions

- Python 3.11, torch ≥ 2.2, numpy for CPU-side sampling/exact solvers. Type
  hints everywhere; `from __future__ import annotations`.
- Naming: snake_case functions, CamelCase classes. Symbol map: `sigma → node`,
  `tau → traj`, `x_p → xtype` (int type index), `pi → policy`, `nu → nu`,
  branching intensity `beta → beta`. Poker partition-kernel field names are
  owned and versioned by the evaluator facade.
- Shape letters: `N` nodes, `C` child edges (CSR), `G` trees, `P` players
  (padded `P_max`), `X` private types (padded `X_max`), `E` events (padded
  `E_max`), `B` minibatch, `T` action types, `M` raise-mixture comps,
  `H`=1326, `K`=169.
- dtypes: indices `int32`; tree/statistics/accumulators `float32` (never fp16
  accumulators); model forward under `torch.autocast(torch.float16)` when
  `cfg.train.amp`; attention via `F.scaled_dot_product_attention`.
- Masks are explicit booleans; no float sentinels; additive `-inf` only inside
  attention-logit construction.
- RNG: every sampling function takes an explicit `torch.Generator` or
  `numpy.random.Generator`. `seed_all(seed, rank)` lives in `pbg/core/config.py`;
  per-rank seed = `seed*1000 + rank`. Never use global RNG state.
- Fail loudly. Shape/mask invariant `assert`s guarded by `cfg.debug_asserts`.
  No silent broadcasting over `P`/`X` dims — reshape explicitly.

## Repository layout (create only your files; the rest exists or will)

```
pbg/core/        config.py game.py tree.py rollout.py branching.py values.py
                 completion.py policy_adapter.py losses.py minibatch.py
                 training.py logging_util.py
pbg/games/synthetic/   family.py presets.py exact.py model.py
pbg/games/poker/       engine.py cards.py ranges.py folded.py mc.py adapter.py
                        # public navigation + DTO/adapters; no private poker math
pbg/games/poker/model/ primitives.py preflop.py postflop.py incidence.py heads.py
pbg/eval/        exploitability.py arena.py
configs/  tests/  CONTRACTS.md  NOTATION.md  POKER_GPU_EVALUATOR_IO.md
```

Import direction (enforced by `tests/test_p0_imports.py`): `pbg.core` imports
nothing from `pbg.games` or `pbg.eval`; `games.synthetic` imports only `core`;
`games.poker` imports `core`; `eval` may import both.

## Core constants and datatypes (defined in P0/P1/P8/P10; reproduced for reference)

```python
# pbg/core/game.py
CHANCE, TERMINAL, PRETERM = -1, -2, -3          # owner codes

class PBGGame(Protocol):
    n_players: int
    x_sizes: tuple[int, ...]                    # |X_p|; X_max = max(x_sizes)
    def sample_mu(self, g, gen) -> Tensor: ...              # int32 [g, P]
    def root_states(self, g) -> Tensor: ...                 # int32 [g] state refs
    def owner_of(self, state_refs) -> Tensor: ...           # int32 [B]
    def legal_events(self, state_refs) -> tuple[Tensor, Tensor]:  # ids [B,E_max], mask [B,E_max]
    def chance_probs(self, state_refs, xprofile) -> Tensor: ...   # [B,E_max]
    def step(self, state_refs, event_ids) -> Tensor: ...          # int32 [B]
    def is_terminal(self, state_refs) -> Tensor: ...
    def is_pre_terminal(self, state_refs) -> Tensor: ...    # core_math tau^{pre-c}; NOTATION §1
    def reward(self, xprofile, terminal_state_refs) -> Tensor:    # float32 [B,P]
    def reach_update(self, log_r_rows, state_refs, event_ids, policy_rows) -> Tensor: ...
    def sample_profiles(self, state_refs, reach_rows, n, gen) -> Tensor:  # int32 [B,n,P] joint posterior draws
    def encode_public(self, state_refs) -> Any: ...

# pbg/core/tree.py
@dataclass
class TreeBatch:
    parent: Tensor        # int32 [N], -1 at roots
    child_ptr: Tensor     # int32 [N+1]  CSR
    child_node: Tensor    # int32 [C]
    child_event: Tensor   # int32 [C]
    nu: Tensor            # float32 [C], sums to 1 per non-leaf node
    tree_id: Tensor       # int32 [N]
    owner: Tensor         # int32 [N]  in [0,P) or CHANCE/TERMINAL/PRETERM
    depth_p: Tensor       # int32 [N, P_max]   |sigma|_p
    depth_player: Tensor  # int32 [N]
    state_ref: Tensor     # int32 [N]  index into game-owned StateTable
    path_weight: Tensor   # float32 [N]  prod of nu along root->node
    def validate(self) -> None: ...

# pbg/core/completion.py
@dataclass
class CompletionStats:    # all float32
    p_hat: Tensor         # [P_max, X_max] normalized posterior (0 on invalid types)
    v_hat: Tensor         # [P_max, X_max] E[R_p | X_p=x]; 0 where p_hat==0
    vvec_hat: Tensor      # [P_max] E[R_p]; folded deterministic payoff retained, padded seats 0
    u_hat: Tensor         # [] E[||R||^2]
    n_eff: Tensor         # [] effective sample size

class MCCompletion(Protocol):
    def complete(self, game, tree, node_ids, reach, n_samples, gen) -> "BatchedCompletionStats": ...

# pbg/core/values.py
@dataclass
class NodeTargets:        # one row per stored player-decision node
    node_ids: Tensor      # int32 [Nd]
    p_hat: Tensor         # [Nd, X_max]      posterior over ACTOR types
    q_hat: Tensor         # [Nd, E_max, X_max]  per-stored-child lambda targets
    v_hat: Tensor         # [Nd, X_max]
    vvec_hat: Tensor      # [Nd, P_max]
    u_hat: Tensor         # [Nd]
    nu_row: Tensor        # [Nd, E_max]  zero-padded
    child_event_row: Tensor  # int32 [Nd, E_max], -1 padded

# pbg/core/policy_adapter.py
@dataclass
class PolicyOutput:
    raw: Any              # game-owned payload
    v: Tensor             # [B, X_max]
    # Future extensions may add public moment outputs; v1 does not require them.

class PolicyAdapter(Protocol):
    def log_prob(self, out, selected_event_ids, legal_event_ids,
                 legal_event_mask) -> Tensor: ...  # [B,E_sel,X_max]
    def entropy(self, out, legal_event_ids,
                legal_event_mask) -> Tensor: ...   # [B,X_max]
    def kl(self, out_new, out_ref, legal_event_ids,
           legal_event_mask) -> Tensor: ...        # deferred [B,X_max]
```

## Config (pbg/core/config.py, built in P0)

Nested dataclasses `RootConfig(game, rollout, completion, model, loss, train,
debug_asserts)` loaded by `load_config(path)`; unknown YAML keys are a hard
error. Field names you may rely on are `rollout.n_trees`, `rollout.beta_mode`
(`"sequential"|"fixed"`), `rollout.beta_fix` (a public-stage mapping),
`rollout.node_budget`, `rollout.reach_storage`
(`"full"|"decisions_only"|"recompute"`), `completion.n_samples`, and
`completion.backend` (`"auto"|"cpu"|"cuda"` through the sibling facade).
The active loss fields are `name`, `lam`, `clip_eps`, `clip_eps_v`, `c_v`,
`c_h`, `entropy_schedule`, and `use_suit_consistency`. Training fields are
`iters`, `gd_steps_per_iter`, `minibatch_size`, `lr`, `adam_betas`,
`grad_clip`, `amp`, `seed`, `checkpoint_every`, and `out_dir`.

## Testing conventions

pytest; test files `tests/test_p{N}_*.py`. Statistical tests: fixed seeds,
documented tolerance, and a comment stating the failure probability under H0.
Exact numerical references computed in float64 on CPU. Never mark a failing
test `xfail` to get green — fix or raise a `CONTRACT-QUESTION`.

## Definition of done (every prompt)

1. All named files created; nothing else modified beyond **May touch**.
2. All new tests written and passing; previous suite green; `T12` import test green.
3. No TODOs except `CONTRACT-QUESTION` comments.
4. A short `IMPLEMENTATION_NOTES.md` appendix entry (append-only file at repo
   root): 5–15 lines on decisions taken, plus every CONTRACT-QUESTION repeated.
