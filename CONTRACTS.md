# CONTRACTS.md — Frozen Interface Contract

**Status: FROZEN once signed off. Version 0.2 (draft for Arka/Pablo review).**

The P09--P11 branching, value-target, and loss portions are superseded by the
C00 alignment recorded here and in `NOTATION.md`. The active v1 contract is
fixed-intensity PPO; explicitly marked research extensions are not requirements.

The cross-repository tensor contract for poker completion and rejection is
`archive/POKER_GPU_EVALUATOR_IO.md`. It is normative and wins over older "user
kernel slot" wording in this file.

This document is the single source of truth for repository layout, naming, tensor
conventions, and every public interface. Implementation prompts receive excerpts of
this file verbatim. Rules for the implementer (GPT 5.6):

- **R1.** Never rename, re-sign, or "improve" anything specified here. If a contract
  seems wrong, implement it as written and add a `# CONTRACT-QUESTION:` comment.
- **R2.** Never modify files owned by another prompt except where explicitly listed
  under "May touch".
- **R3.** Every prompt must leave the full test suite green (`pytest -x -q`),
  including tests from earlier prompts.
- **R4.** No abstractions beyond spec. No metaclasses, no dynamic dispatch beyond the
  Protocols defined here, no clever registry magic beyond `losses.py`'s explicit dict.
- **R5.** Mathematical semantics come from `docs/theory/core_math.tex` through
  its implementation transcription in `NOTATION.md`, not from memory of RL
  conventions. If the two documents appear to conflict, `core_math.tex` wins
  and the discrepancy must be reported rather than silently implemented.

---

## 1. Repository layout

```
pbg/
  __init__.py
  core/                    # GENERIC. Must not import from pbg.games.
    config.py              # dataclass configs + YAML loading
    game.py                # PBG Protocols, event/owner constants
    tree.py                # TreeBatch (SoA rollout tree), alias tables
    rollout.py             # optimized SamplePreTerminal (one profile per root)
    branching.py           # labeled SampleTree, fixed beta_fn
    values.py              # BranchedValue backward propagation
    completion.py          # CompletionStats, MCCompletion Protocol
    policy_adapter.py      # PolicyAdapter Protocol (game-specific action factorization)
    losses.py              # loss primitives + composite registry
    minibatch.py           # decision-law sampling over stored trees
    training.py            # train loop, optimizer, AMP, DDP, checkpointing
    logging_util.py        # jsonl metrics logger, optional wandb
  games/
    synthetic/             # P1: parametrized PBG family ("the dial")
      family.py            # SyntheticPBG generator (public-tree / private-state / player dials)
      presets.py           # kuhn(), leduc(), plus dial presets
      exact.py             # exact posteriors, best response, exploitability (with feasibility guard)
    poker/
      engine.py            # P2: public NLHE state machine and pot accounting
      cards.py             # IDs/combo maps/classes + complement sampling; no ranking
      ranges.py            # P3: action-likelihood reach bookkeeping only
      folded.py            # P4: evaluator partition-summary DTOs/adapters
      mc.py                # P5: poker-gpu-evaluator adapters only
      adapter.py           # PokerGame: PBG Protocol impl; PokerPolicyAdapter
      model/
        primitives.py      # P6: LN, MHA(+key bias, content temperature), FFN
        preflop.py         # P6: 169-class net
        postflop.py        # P7: combo-row net, evaluator-summary channels
        incidence.py       # P7: generic transforms of supplied summary fields
        heads.py           # P6/P7: policy (type + raise mixture), V, Vvec, U heads
  eval/
    exploitability.py      # P12: exact BR on synthetic family
    arena.py               # P12: checkpoint-vs-checkpoint poker matches
  configs/
    kuhn_smoke.yaml
    leduc.yaml
    synthetic_dial_example.yaml
    nlhe6_2080ti.yaml
  tests/
    test_p1_*.py ... test_p13_*.py   # one file prefix per prompt
  CONTRACTS.md
  NOTATION.md
  archive/POKER_GPU_EVALUATOR_IO.md
```

**Import direction (enforced by a test):** `core` imports nothing from `games` or
`eval`. `games.synthetic` imports only `core`. `games.poker` imports `core` (and
torch); only poker boundary modules may import the public
`poker_gpu_evaluator.pbg` facade. The model may consume opaque evaluator-summary
and reach DTOs. `eval` may import both.

---

## 2. Global conventions

- **Python 3.10+, torch ≥ 2.2.** No other DL frameworks. numpy allowed for CPU-side
  sampling and exact solvers. Type hints everywhere; `from __future__ import annotations`.
- **Naming.** snake_case functions/modules, CamelCase classes. Mathematical symbols
  map to code as: `sigma → node`, `tau → traj`, `x_p → xtype` (integer type index),
  `pi → policy`, `nu → nu`, `beta (intensity) → beta`. Poker partition
  field names are owned and versioned by the evaluator facade.
- **Shapes in comments.** Every tensor-returning function documents shapes with
  bracket comments: `# [N, P, X]`. Dimension letters (fixed vocabulary):
  - `N` nodes in a TreeBatch, `C` child edges (CSR), `G` trees in batch,
  - `P` players (padded to `P_max`), `X` private-type count (padded to `X_max`),
  - `E` events at a node (padded to `E_max`), `B` minibatch size,
  - `T` action types (poker: 3), `M` raise-mixture components, `H` = 1326, `K` = 169.
- **dtypes.** Indices `int32`. Tree/statistics/accumulator tensors `float32`
  (MC accumulation in fp32 is mandatory — never fp16 accumulators). Model forward
  under `torch.autocast(dtype=torch.float16)` (sm_75: no bf16), fp32 master weights,
  `torch.cuda.amp.GradScaler`. Attention via `F.scaled_dot_product_attention`.
- **Masks.** Padding uses explicit boolean masks, never sentinel values in float
  tensors. Additive `-inf` only inside attention-logit construction.
- **RNG.** Every sampling function takes an explicit `torch.Generator` (GPU) or
  `numpy.random.Generator` (CPU). Global seeding: `seed_all(seed, rank)` in
  `core/config.py`; per-rank seed = `seed * 1000 + rank`. No use of global RNG state.
- **Errors.** Fail loudly: `assert` on shape/mask invariants in debug paths, guarded
  by `cfg.debug_asserts`. No silent broadcasting across `P`/`X` dims — reshape explicitly.
- **Determinism for tests.** Statistical tests use fixed seeds and documented
  tolerances (see §9). Exact tests use fp64 on CPU.

---

## 3. Config system (`core/config.py`)

Plain nested dataclasses, loaded from YAML by `load_config(path) -> RootConfig`.
Unknown YAML keys are a hard error. Structure (fields may grow; names frozen):

```python
@dataclass
class GameConfig:
    name: str                    # "synthetic" | "poker"
    synthetic: SyntheticDialConfig | None
    poker: PokerConfig | None    # n_players (2..9, default 6), stacks_bb=100.0, ...

@dataclass
class RolloutConfig:
    n_trees: int                 # G per iteration (per rank: G // world_size)
    beta: dict[str, float] | None # per-bucket intensity; branched RB only
    node_budget: int             # hard cap on nodes per tree; sampler raises past it
    reach_storage: str           # "full" | "decisions_only" | "recompute"

@dataclass
class CompletionConfig:
    n_samples: int | None        # null for normal; MC budget for RB modes
    backend: str                 # sibling execution backend: "auto" | "cpu" | "cuda"

@dataclass
class LossConfig:
    name: str                    # key into losses.COMPOSITES, default "rb_ppo"
    lam: float                   # lambda, GAE horizon (general; default 0.95)
    clip_eps: float; clip_eps_v: float
    c_v: float; c_h: float
    entropy_schedule: str        # "const" | "linear_decay"
    use_suit_consistency: bool   # default False

@dataclass
class TrainingConfig:
    mode: str                    # normal | rao_blackwell | branched_rao_blackwell
    iters: int; gd_steps_per_iter: int; minibatch_size: int
    lr: float; adam_betas: tuple; grad_clip: float
    amp: bool; seed: int; checkpoint_every: int; out_dir: str

@dataclass
class RootConfig:
    game: GameConfig; rollout: RolloutConfig; completion: CompletionConfig
    model: dict                  # game-owned; poker model config parsed in games/poker
    loss: LossConfig; training: TrainingConfig
    debug_asserts: bool = False
```

---

## 4. Core data structures

### 4.1 Constants (`core/game.py`)

```python
CHANCE: int = -1        # owner value for chance nodes
TERMINAL: int = -2      # owner value for terminal nodes
PRETERM: int = -3       # owner value for pre-terminal leaves of a stored tree
```

### 4.2 `TreeBatch` (`core/tree.py`) — the SoA rollout tree

One object holds `G` trees, concatenated. All tensors CPU or GPU per
`cfg` (rollout on GPU where the game supports vectorized stepping; synthetic family
may run CPU). Frozen fields:

```python
@dataclass
class TreeBatch:
    # topology (CSR)
    parent:      Tensor  # int32 [N], -1 at roots
    child_ptr:   Tensor  # int32 [N+1]
    child_node:  Tensor  # int32 [C]
    child_event: Tensor  # int32 [C]   event id taken on this edge
    nu:          Tensor  # float32 [C] empirical transition weights, sum to 1 per node
    # node metadata
    tree_id:     Tensor  # int32 [N]
    owner:       Tensor  # int32 [N]   in [0,P) or CHANCE/TERMINAL/PRETERM
    depth_p:     Tensor  # int32 [N, P_max]  |sigma|_p  (player-decision counts along path)
    depth_player:Tensor  # int32 [N]         |sigma|_player
    # game payload
    state_ref:   Tensor  # int32 [N] index into the game's StateTable (opaque to core)
    # decision-law support (player nodes only; see NOTATION §4)
    path_weight: Tensor  # float32 [N]  prod of nu along root->node (log stored? NO: fp32 linear, trees are shallow)
```

Invariants (asserted in `TreeBatch.validate()`):
`nu[child_ptr[i]:child_ptr[i+1]].sum() == 1` for every non-leaf `i`; leaves have
`owner ∈ {TERMINAL, PRETERM}`; `depth_*` consistent with parent chain.

`build_alias_tables(tree: TreeBatch) -> AliasTable` — one alias table over all
player-decision nodes, weights `path_weight[i] * 1[owner[i] ∈ players]`, used by
`minibatch.py` for `(P, Σ)` sampling per NOTATION §4. `AliasTable.sample(n, gen) -> int32 [n]`.

### 4.3 `ReachCache` (`core/completion.py`)

Per-node, per-player unnormalized posterior weights over private types (the 𝔯 of
`core_math`), maintained by the game adapter during sampling:

```python
@dataclass
class ReachCache:
    log_r: Tensor       # float32 [N_stored, P_max, X_max]; -inf = impossible type
    node_index: Tensor  # int32 [N_stored] maps rows to TreeBatch node ids
    mode: str           # cfg.rollout.reach_storage
def reach_at(cache, tree, node_ids, game) -> Tensor  # float32 [B, P_max, X_max]
```

In `"recompute"` mode `reach_at` walks parent pointers and calls
`game.reach_update`; in `"full"`/`"decisions_only"` it gathers stored rows.
`"decisions_only"` includes every player node and every PRETERM/TERMINAL
leaf, because collision-conditioned leaf posteriors require the reach state. Consumers
never care which mode is active.

### 4.4 `CompletionStats` (`core/completion.py`)

Output of MC completion at one pre-terminal node (fp32 everywhere):

```python
@dataclass
class CompletionStats:
    p_hat:    Tensor  # [P_max, X_max]  posterior over types, normalized over valid types
    v_hat:    Tensor  # [P_max, X_max]  E[R_p | X_p = x], 0 where p_hat==0
    vvec_hat: Tensor  # [P_max]         E[R_p], including folded deterministic payoff; padded seats 0
    u_hat:    Tensor  # []              E[||R||_2^2]
    n_eff:    Tensor  # []              effective sample size (diagnostics)
```

### 4.5 `NodeTargets` and `MinibatchSample` (`core/values.py`, `core/minibatch.py`)

`BranchedValue` writes, for every stored **player** node, the training targets
(NOTATION §7 defines their values):

```python
@dataclass
class NodeTargets:            # aligned with player-node ids
    node_ids: Tensor          # int32 [Nd]
    parent_p_hat: Tensor      # [Nd, X_max] posterior over actor types at parent
    child_p_hat: Tensor       # [Nd, E_max, X_max] branch-conditioned posterior
    a_hat: Tensor             # [Nd, E_max, X_max] leave-one-branch advantage
    v_hat:    Tensor          # [Nd, X_max]
    child_event_row: Tensor   # int32 [Nd, E_max]
    child_mask: Tensor        # bool [Nd, E_max]
    child_count: Tensor       # int32 [Nd]
    q_hat: Tensor | None      # optional, present only when emit_q=True

@dataclass
class MinibatchSample:        # produced by minibatch.sample(...)
    decision_rows: Tensor     # int32 [B], sampled directly by alias table
    targets:  NodeTargets     # gathered rows, [B, ...]
    old_out:  PolicyOutput    # frozen rollout-policy outputs at these nodes (cached at rollout time)
```

### 4.6 `PolicyOutput` and `PolicyAdapter` (`core/policy_adapter.py`)

The generic core never sees poker's action factorization. Games expose:

```python
@dataclass
class PolicyOutput:           # everything the model emits at a batch of nodes
    raw: Any                  # game-owned payload (poker: type logits, raise mixture, etc.)
    v: Tensor                 # [B, X_max] per-type value

class PolicyAdapter(Protocol):
    def log_prob(self, out, selected_event_ids, legal_event_ids,
                 legal_event_mask) -> Tensor:
        """[B,E_sel,X_max]: log pi(selected event | xtype,node)."""
    def entropy(self, out, legal_event_ids, legal_event_mask) -> Tensor:
        """[B, X_max]. Exact where tractable; poker: exact over the factored head."""
    def kl(self, out_new, out_ref, legal_event_ids, legal_event_mask) -> Tensor:
        """Deferred extension, [B,X_max]."""
    def centered_logits(self, out, selected_event_ids,
                        legal_event_ids, legal_event_mask) -> Tensor:
        """NeuRD support; may raise NotImplementedError in v1."""
```

---

## 5. Module contracts by prompt

### P1 — `games/synthetic` + exact tools

```python
class PBGGame(Protocol):                       # core/game.py — the generic interface
    n_players: int
    x_sizes: tuple[int, ...]                   # |X_p| per player; X_max = max
    def sample_mu(self, g: int, gen) -> Tensor:            # int32 [g, P] type indices
    def root_states(self, g: int) -> Tensor:                # state_ref column for g fresh roots
    def owner_of(self, state_refs) -> Tensor
    def legal_events(self, state_refs) -> tuple[Tensor, Tensor]  # event ids [B,E_max], mask [B,E_max]
    def chance_probs(self, state_refs, xprofile) -> Tensor  # [B, E_max]; rows only for chance nodes
    def step(self, state_refs, event_ids) -> Tensor         # new state_refs [B]
    def is_terminal(self, state_refs) -> Tensor
    def is_pre_terminal(self, state_refs) -> Tensor         # core_math tau^{pre-c}; NOTATION §1
    def reward(self, xprofile, terminal_state_refs) -> Tensor    # float32 [B, P]
    def reach_update(self, cache_row, state_refs, event_ids) -> Tensor  # posterior ⊙-update
    def encode_public(self, state_refs) -> Any               # model-input payload (game-owned type)
```

`SyntheticPBG(SyntheticDialConfig)` implements `PBGGame` with dials:
`n_players`, `deck_size`, `hand_size` (→ `x_sizes`), `actions_per_node`,
`n_rounds`, `chance_branching`, `reward_kind` ∈ {"zero_sum_matchup", "table"}.
`presets.kuhn()` and `presets.leduc()` return configured instances whose equilibria
are known. `exact.py`:

```python
def exact_posterior(game, node) -> ndarray            # [P, X] fp64; raises FeasibilityError if tree too big
def best_response_value(game, policy_tables) -> ndarray
def exploitability(game, policy_tables) -> float
MAX_EXACT_NODES = 2_000_000                            # feasibility guard
```

`policy_tables`: dense `dict[node_key] -> [X, E]` arrays exported by
`export_policy_tables(model_or_uniform, game)` (P12 provides the model exporter;
P1 provides the uniform/tabular one).

### P2 — `games/poker/engine.py`, `cards.py`

```python
class NLHEEngine:
    def __init__(self, cfg: PokerConfig): ...          # n_players, stacks in bb, blinds, tick grid
    # StateTable: SoA of packed states; all methods vectorized over int32 state_refs
    # fields: street, board(5 x int8, -1 pad), pot, stacks[P], committed[P], status[P] (live/folded/allin),
    #         to_act, last_raise, n_raises, parent-linked lossless public-history event
```
Engine exposes the `PBGGame` public-tree surface except private-state semantics,
which live behind `adapter.py`/`mc.py`. `cards.py` exposes
`hand_index(a,b) -> h in [0,1326)`, `hand_cards(h)`, `class_of_hand(h)`,
`class_members(k)`, `suit_perm_apply(perm,h)`, board-conflict maps, and uniform
sampling from a supplied complement card set. It contains no hand rank,
category, kicker, winner, tie, or showdown-payoff implementation.

The engine constructs nested `stratum_players`, `stratum_amount`, and
`payoff_baseline` from public commitments/status. Fixed showdowns call
`poker_gpu_evaluator.pbg.evaluate_showdowns`; local arithmetic only multiplies
returned shares by stratum amounts and adds the baseline. Acceptance covers
public hand-history replay, legal actions, pot strata, and fake-facade payoff
assembly—not poker ranking fixtures.

### P3 — `games/poker/ranges.py`

```python
class RangeState:
    log_r: Tensor            # [P, 1326] fp32; folded players frozen at fold-time range
    live_mask: Tensor        # [P] bool
    board_mask: Tensor       # [1326] bool (hands compatible with board)
    def update_action(self, p, event_id, policy_row) -> None     # ⊙ Bayes update
    def update_board(self, new_cards) -> None                     # mask + renormalize per §1.5 conventions
    def normalized(self, p) -> Tensor                             # tilde r
def to_class_ranges(log_r) -> Tensor    # [P, 169] preflop aggregation
```
Owns action-likelihood multiplication and fold-event snapshots handed across
the evaluator boundary. It never samples a private hand or evaluates a
collision-conditioned partition.

### P4 — `games/poker/folded.py` (evaluator-summary boundary)

```python
class PartitionSummaryBatch:
    # Opaque, versioned evaluator output carried to model inputs. Local code
    # validates fields/shapes/devices but derives none of them from poker cards.
    fields: Mapping[str, Tensor]
    metadata: Mapping[str, object]
class PokerGPUPartitionAdapter:
    def summarize(self, board, reach_rows, folded_snapshots) -> PartitionSummaryBatch: ...
```
Acceptance: fake-facade field mapping, version/device/shape checks, and sibling
integration tests. The adapter verifies the facade's defensive `COMBO_CARDS`
copy and calls the evaluator 0.3.0+ partition signature without the removed
query/seed arguments. No local Φ/α/β/Λ implementation or Monte Carlo oracle is
permitted.

### P5 — `games/poker/mc.py`

```python
class MCCompletion(Protocol):              # core/completion.py owns the Protocol
    def complete(self, game, tree, node_ids, reach, n_samples, gen) -> CompletionStats_batched
class PokerGPUCompletion:                  # thin adapter to poker_gpu_evaluator.pbg.complete
class RejectionSampler(Protocol):          # branching support: sample x ~ posterior at node
    def sample(self, reach_rows, board, n, gen) -> Tensor  # int32 [B, n, P] profiles
class PokerGPURejectionSampler: ...        # thin adapter to sample_joint_profiles
class PokerGPUShowdownEvaluator: ...       # thin adapter to evaluate_showdowns
```
`backend="poker_gpu"` resolves only to the public
`poker_gpu_evaluator.pbg` facade specified in
`archive/POKER_GPU_EVALUATOR_IO.md`.
The sibling CPU backend is the acceptance oracle. There is no local poker
reference implementation, generic user-kernel registry, or locally implemented
CUDA path.

### P6/P7 — `games/poker/model/*`

`primitives.py` (frozen signatures; semantics in `docs/MHA_model.md` "Definitions of
Layer Primitives"):

```python
class MHA(nn.Module):
    def forward(self, q, kv, key_bias, key_mask) -> Tensor
    # per-head learned content temperature tau_m applied to content term ONLY;
    # key_bias (log-reach prior) added un-tempered; masks -> -inf. Uses SDPA where possible.
class FFN(nn.Module): ...     # affine -> GELU -> affine, hidden 4x unless cfg overrides
```

`preflop.py` / `postflop.py`:

```python
class PreflopNet(nn.Module):
    def forward(self, pub: PokerPublicBatch, class_ranges) -> PolicyOutput   # rows: [B,169]
class PostflopNet(nn.Module):
    def forward(self, pub, range_state_batch, partition: PartitionSummaryBatch) -> PolicyOutput
    # COMPACT TOKENIZATION IS THE DEFAULT: rows are board-valid combos H_B
    # (1176/1128/1081); carries compact_to_combo/combo_to_compact int32 maps and
    # scatters outputs back to [B, 1326] before returning. Fixed-shape-per-street batching.
```
`heads.py`: `TypeHead`, `RaiseMixtureHead` (`DLogisticTick` per `docs/MHA_model.md`,
binomial ticks behind a flag as ablation), and `ValueHead`. Existing
`MomentHeads` code is a dormant research extension and is not wired into v1.
Poker-private collision and partition features are evaluator-produced. Learned
tensor transformations over supplied fields may live in the model, but
`incidence.py` must not enumerate or derive legal/conflicting private
assignments.
`PokerPolicyAdapter` in `adapter.py` implements `PolicyAdapter` over these heads
(exact entropy over the factored type×tick distribution).

### P8/P9 — `core/rollout.py`, `core/branching.py`

```python
StateBatch = Tensor  # int32 [B] opaque handles into a backend-owned state store

@dataclass
class EventBatch:
    public_id: Tensor
    payload: dict[str, Tensor]

class RolloutBackend(Protocol):
    device: torch.device
    n_players: int
    x_sizes: tuple[int, ...]
    max_rollout_steps: int
    def root_states(self, n) -> StateBatch: ...
    def classify(self, states) -> Tensor: ...
    def legal_events(self, states) -> tuple[Tensor, Tensor]: ...
    def policy_forward(self, frozen_model, states, reaches) -> PolicyOutput: ...
    def sample_profiles(self, states, reaches, gen) -> Tensor: ...
    def sample_events(self, states, profiles, policy_out, adapter, gen) -> EventBatch: ...
    def step(self, states, events: EventBatch) -> StateBatch: ...
    def reach_update(self, reaches, states, events, policy_out) -> Tensor: ...
    def beta_fn(self, states) -> Tensor: ...       # fp32 [B], fixed public-stage values

def sample_trees(bundle, cfg: RolloutConfig, gen) -> tuple[TreeBatch, ReachCache, OldOutCache]
# K = floor(beta) + Bernoulli(beta-floor(beta)). Every abstract child is stored,
# including repeated public labels, with edge probability exactly 1/K.
# training.mode="rao_blackwell" uses the profile-carrying one-child sampler;
# "fixed" calls backend.beta_fn and draws child-local posterior profiles.
```
`OldOutCache` stores rollout-policy `PolicyOutput` rows at player nodes (for
policy ratios) so minibatching never re-runs the frozen policy. Production
rollout tensors remain on their owning GPU; host transfers are forbidden in the
hot path. Buffer overflow raises rather than changing the sampled law.

### P10 — `core/values.py`

```python
def branched_value(tree, leaf_stats, reach, value_heads, lam, game, *,
                   configured_node_budget,
                   emit_q=False, auxiliary_reducers=()) -> NodeTargets
# The core_math BranchedValue algorithm, transcribed in NOTATION §7:
#   1. MC completion at PRETERM leaves -> CompletionStats
#   2. Backward pass: uniform-child aggregation of posterior mass P and
#      posterior-weighted return mass M.
#   3. At player nodes: conditionalize M/P, compute O(C) total-minus-child
#      baselines, emit A and V, then lambda-mix owner M before parent consumption.
# Q is transient unless emit_q=True. Public moment channels are deferred.
# Backward-MC posterior is the ONLY production path (signed-off decision).
```

### P11 — `core/losses.py`, `core/training.py`

```python
# active primitives (NOTATION §10 formulas): l_pg, l_clip, l_v, l_v_clip, l_h:
# (new_out, old_out, targets, adapter, cfg) -> ([B] or [B,X], metrics).
# Adapter KL/centered-logit methods remain extension points; KL snapshots,
# moment losses and NeuRD/R-NaD are out of v1 scope.
COMPOSITES: dict[str, LossFn] = {"rb_ppo": ...}
# Policy terms consume a_hat and child_p_hat with uniform abstract-child weights;
# value/entropy terms consume parent_p_hat.
def train(cfg: RootConfig) -> None       # torchrun entry; DDP; phase-alternating:
# each rank samples n_trees/world locally -> branched_value locally -> gd_steps with
# DDP-allreduced grads; GradScaler; checkpoint {model, opt, scaler, iter, cfg, rng}.
```

Game construction returns a typed bundle rather than a positional tuple:

```python
@dataclass
class GameBundle:
    model: nn.Module
    rollout_backend: RolloutBackend
    policy_adapter: PolicyAdapter
    completion_backend: MCCompletion
```

### P12/P13 — `eval/`, integration

`exploitability_curve(cfg, checkpoints) -> jsonl`; `arena(ckpt_a, ckpt_b, n_hands, gen) -> ArenaReport`
(mean bb/100 ± CI, per-seat). P13 owns the cross-cutting tests T2/T5/T11 (§9) and the
four YAML presets, including `nlhe6_2080ti.yaml` sized for 2×11GB fp16.

---

## 6. External evaluator boundary

All poker-private combinatorics are implemented by the sibling repository. Its
CPU implementations are the reference oracles:

| Operation | Local implementation | Sibling owner |
|---|---|---|
| completion/equity coefficients | adapter only | `poker_gpu_evaluator.pbg.complete` |
| exact joint profiles | adapter only | `poker_gpu_evaluator.pbg.sample_joint_profiles` |
| fixed showdown by pot stratum | adapter only | `poker_gpu_evaluator.pbg.evaluate_showdowns` |
| partition/blocker summaries | adapter/DTO only | sibling public partition facade |

Backend selection is passed explicitly to the sibling (`"cpu"` or `"cuda"`).
Adapters live only in `games/poker/mc.py`/`folded.py`, import only its public
facade, and perform packing/scattering and error translation. Fake-facade tests
are local; mathematical CPU/CUDA agreement tests belong to the sibling.

---

## 7. Precision & distributed (2×2080Ti baseline)

- fp16 autocast + GradScaler; fp32 master weights, fp32 reductions and MC accumulators.
- No FlashAttention (sm_75); SDPA math/mem-efficient kernels.
- DDP over 2 ranks, phase-alternating (rollout → train → rollout). Rollout sharded by
  tree; no parameter-server / actor-learner split in v1.
- Memory levers already in contract: compact postflop tokenization (default),
  `reach_storage="recompute"`, `node_budget`, gradient accumulation via
  `minibatch_size` vs `gd_steps_per_iter`.
- Nothing outside `games/poker` and `configs/` may mention poker or a GPU model.

---

## 8. Logging & checkpoints

`logging_util.MetricsLogger.log(step, dict)` → `out_dir/metrics.jsonl` (always) and
wandb (optional flag). Mandatory per-iteration metrics: loss components, entropy,
approx-KL, ratio clip fraction, value-head explained variance, tree stats
(nodes/tree, mean β, completion `n_eff` histogram summary), throughput
(nodes/s, GD steps/s). Checkpoints: `out_dir/ckpt_{iter:07d}.pt`, schema in §5-P11.

---

## 9. Test matrix (acceptance gates; IDs referenced by prompts)

| ID | Property | Method / tolerance |
|---|---|---|
| T1 | TreeBatch invariants; ν rows sum to 1 | exact |
| T2 | β≡1 SampleTree ≡ SamplePreTerminal path law | seeded statistical, chi-square p>1e-3 on toy game |
| T3 | Terminal-distribution lemma: E[P^{β,π}] = P^π | chi-square on synthetic preset |
| T4 | `branched_value` vs fp64 brute-force enumeration on a ≤200-node toy tree | atol 1e-10 (fp64 path) |
| T5 | RB gradient vs naive sampled gradient agreement on Kuhn | cosine > 0.99 at large N, seeded |
| T6 | MC posterior vs `exact_posterior` on synthetic presets | TV dist < tol(n_samples) |
| T7 | partition-summary adapter maps sibling fields without recomputation | exact shape/device/field mapping |
| T8 | no local poker-private collision/partition implementation | static forbidden-symbol/import audit |
| T9 | poker policy head normalization; adapter entropy vs numeric | atol 1e-5 |
| T10 | end-to-end exploitability decreases on Kuhn and Leduc | monotone-ish gate: final < 0.3 × initial |
| T11 | 2-rank DDP ≡ 1-rank double-batch (loss curves) | statistical overlay, documented seeds |
| T12 | import-direction rule (§1) | static test |

---

## 10. Mathematical authority

`docs/theory/core_math.tex` is authoritative. `NOTATION.md` §12 records only
symbol-level normalizations needed to transcribe internally cross-referenced
source equations and pseudocode; it does not define alternative mathematics.
In particular, `is_pre_terminal` identifies a completion cut after which no
player node remains; zero or more chance draws may remain. Public moment heads
and their allocation rule are deferred research extensions.
