# C11 — Minimal RB-PPO loss and training correction

**Depends on:** P11, C08–C10.
**Supersedes:** `Prompts/P11_training.md` for the active objective and bundle API.
**Files:** `pbg/core/losses.py`, `pbg/core/minibatch.py`,
`pbg/core/training.py`, `pbg/core/config.py`, `pbg/core/game.py`,
`pbg/core/values.py`, `pbg/core/posterior_values.py`, `pbg/core/rollout.py`,
`pbg/core/branching.py`, `pbg/core/tree.py`;
`pbg/games/synthetic/bundle.py`, `tests/test_c11_minimal_losses.py`,
`tests/test_c11_minimal_training.py`, `tests/test_p11_losses.py`,
`tests/test_p11_training.py`, `tests/test_p10_values.py`.
**May touch:** `pbg/core/policy_adapter.py`, synthetic model wiring,
`pbg/games/synthetic/model.py`, `pbg/games/poker/adapter.py`,
`tests/test_p1_adapter.py`, `tests/test_p7_adapter.py`,
`tests/test_p8_rollout.py`, `tests/test_p9_branching.py`,
`configs/kuhn_smoke.yaml`, `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`, and
the branched minibatch-loss section of `docs/latex/core_math.tex`; inspect all
listed sources plus P11 tests and current configs. P11 is historical behavior
only. Run focused baseline tests, then implement code, tests, config migration,
and notes before the focused and full suites.

## Goal

Ship only the first-run objective
`L_clip[A_hat] + L_H + L_V[V_hat]` over corrected labeled-tree targets.
Keep explicit extension seams without implementing deferred algorithms.

Atomically switch the public value API: `pbg/core/values.py` becomes a thin
re-export of the corrected C10 types/functions from `posterior_values.py`.
Remove the obsolete direct-conditional-value implementation, migrate P10 tests
to the corrected independent oracle, and leave no `V2`/`legacy` public names.
Likewise make `sample_trees(bundle,cfg,gen)` the sole rollout entry: sequential
mode uses C08's device sampler and fixed mode uses C09's labeled sampler.
Delete the temporary positional rollout/branching compatibility functions and
migrate every caller/test in the same change.

## Bundle and row alignment

Use this exact C08 bundle boundary:

```python
@dataclass
class GameBundle:
    model: nn.Module
    rollout_backend: RolloutBackend
    policy_adapter: PolicyAdapter
    completion_backend: MCCompletion

def build_game_bundle(cfg: RootConfig, device: torch.device) -> GameBundle: ...
```

Preserve core import direction with a string factory registry:

```python
GAME_FACTORIES = {
    "synthetic": "pbg.games.synthetic.bundle:build_bundle",
}
```

Resolve `module:function` with `importlib`; C11's synthetic factory constructs
the family, model, adapter, CPU rollout backend and synthetic completion
backend. P13B extends the registry with poker. Core contains no static import
from `games`.

Training calls
`bundle.rollout_backend.policy_forward(frozen_model, states, reaches)` and never
the raw model directly. `model` is the sole optimized module.

All decision-oriented objects use one row order. The alias table samples
decision-row indices directly; target and old-policy caches gather with tensor
indexing. Do not rebuild Python node-id dictionaries in minibatch steps.

```python
@dataclass
class PolicyCache:
    node_ids: Tensor          # int32 [Nd], same order as NodeTargets.node_ids
    old_logp_child: Tensor    # fp32 [Nd,E,X], duplicate child labels retained
    old_v: Tensor             # fp32 [Nd,X]

@dataclass
class MinibatchSample:
    decision_rows: Tensor     # int32 [B]
    state_refs: Tensor        # int32 [B]
    reach: Tensor             # fp32 [B,P,X], reconstructed/gathered per cache mode
    targets: NodeTargets      # tensor-gathered B rows
    old_logp_child: Tensor    # [B,E,X]
    old_v: Tensor             # [B,X]
    valid_type_mask: Tensor   # bool [B,X]
    legal_event_ids: Tensor   # int32 [B,E_legal]
    legal_event_mask: Tensor  # bool [B,E_legal]
```

At cache construction assert exact equality of policy/target `node_ids`.
Build the alias table over decision rows with weights equal to each abstract
node's `path_weight`; its outcomes are row indices, not global node ids.
Minibatch construction obtains reach through the configured C08 cache mode and
legal rows through `rollout_backend.legal_events(state_refs)`. The current
forward is
`rollout_backend.policy_forward(bundle.model, state_refs, reach)`.

Normalize the generic adapter interface in `pbg/core/policy_adapter.py`:

```python
class PolicyAdapter(Protocol):
    def log_prob(self, out: PolicyOutput, selected_event_ids: Tensor,
                 legal_event_ids: Tensor, legal_event_mask: Tensor) -> Tensor:
        ...  # fp32 [B,E_selected,X]
    def entropy(self, out: PolicyOutput, legal_event_ids: Tensor,
                legal_event_mask: Tensor) -> Tensor:
        ...  # fp32 [B,X]
```

Update synthetic and poker adapters to this signature. Probabilities are
normalized over the complete legal-event row; duplicate selected ids simply
gather duplicate log-probability columns.

## Loss

For batch row b with K_b abstract children:

```text
L_b = sum_a child_mask[b,a]/K_b
          * sum_x child_p_hat[b,a,x] * L_clip(ratio[b,a,x], a_hat[b,a,x])
    + sum_x parent_p_hat[b,x]
          * (L_V(v_new[b,x], v_hat[b,x]) + L_H(policy_new[b,x]))
L = mean_b L_b
```

- Ratio uses cached rollout log probability for each abstract child event.
- PG/PPO consumes `targets.a_hat` directly.
- Policy weighting:
  `(1/K) * child_p_hat[b,a,x]` over valid children/types.
- Value and entropy weighting: `parent_p_hat[b,x]`.
- Duplicate public action labels remain separate child terms.
- Active primitives: `l_pg`, `l_clip`, `l_v`, `l_v_clip`, `l_h`.
- Adapter-level `kl` and `centered_logits` may remain, but there is no reference
  snapshot, KL component, moment loss, NeuRD or R-NaD registry entry in v1.
- Loss registrations declare required target fields so optional Q consumers can
  be added later without changing default targets.

```python
@dataclass(frozen=True)
class LossSpec:
    fn: Callable[..., tuple[Tensor, dict[str, float]]]
    required_targets: frozenset[str]

COMPOSITES: dict[str, LossSpec] = {
    "rb_ppo": LossSpec(rb_ppo,
        frozenset({"parent_p_hat", "child_p_hat", "a_hat", "v_hat"}))
}
```

Before rollout target construction, resolve the selected `LossSpec`; pass
`configured_node_budget=cfg.rollout.node_budget` and
`emit_q=("q_hat" in required_targets)` to C10, and fail immediately if any
declared target field is unavailable. Thus v1 allocates no Q while a future
NeuRD-style registration can request it without changing the target API.

## Training

- Freeze theta_k on the rank's GPU; rollout, completion, backprop targets,
  minibatches and current-model optimization remain on that GPU.
- Each DDP rank owns independent trees and device-local RNGs. DDP synchronizes
  gradients only; do not communicate rollout trees.
- CPU use is restricted to configuration, asynchronous scalar logging and
  scheduled checkpoint serialization.
- Checkpoints store model, optimizer, scaler, iteration, config and explicit RNG
  state. No intensity-controller or reference-policy state exists.

## Configuration

Freeze these active fields (besides unchanged game/model/train fields):

```python
@dataclass
class RolloutConfig:
    n_trees: int
    beta_mode: Literal["sequential", "fixed"]
    beta_fix: dict[str, float]       # requires "default"
    node_budget: int
    reach_storage: Literal["full", "decisions_only", "recompute"]

@dataclass
class CompletionConfig:
    n_samples: int
    backend: str                     # validated by the selected game bundle

@dataclass
class LossConfig:
    name: str
    lam: float
    clip_eps: float
    clip_eps_v: float
    c_v: float
    c_h: float
    entropy_schedule: Literal["const", "linear_decay"]
    use_suit_consistency: bool = False
```

Remove `beta_avg`, `beta_max`, `warmup_iters`, `use_moment_heads`, `c_u`,
`c_vvec`, `ref_period`, and `c_kl` from the strict schema and update
`kuhn_smoke.yaml`. Their presence is an unknown-key error; `neyman` is an
invalid enum value, never a silent fallback.

## Tests

Migrate obsolete P11 fixtures from `q_hat-old_v`, parent-posterior policy
weighting, moment/KL config fields, positional bundles, and controller state.
Preserve valid clipping, masking, RB-gradient, checkpoint and training coverage.
Also migrate P10 tests to the now-public C10 engine while preserving their
lambda-boundary, coupled-prior, fold, padding, chunking and oracle checks.

- Analytic two-action gradients verify direct A targets and child posterior
  weights inside/outside clipping.
- A duplicate-label fixture contributes two independent abstract-child terms.
- Padding has exactly zero loss and gradient.
- At equal new/rollout parameters, ratio is one and clipping is inactive.
- RB gradient direction agrees with exact synthetic policy gradients and has
  lower empirical variance than sampled-private-state PG.
- A guard proves no dict lookup is constructed per minibatch.
- Kuhn fixed-beta smoke improves exploitability; save/load/resume is reproducible.
- Single-rank and two-rank runs improve similarly after sample-count scaling.

Required commands:

```bash
pytest -q tests/test_p11_losses.py tests/test_p11_training.py
pytest -q tests/test_p10_values.py tests/test_c10_posterior_moments.py
pytest -q tests/test_p1_adapter.py tests/test_p7_adapter.py
pytest -q tests/test_c11_minimal_losses.py tests/test_c11_minimal_training.py
pytest -q
```

## Do not

- No poker imports in core, reference KL, moment heads/losses, Neyman controller,
  or NeuRD implementation.
