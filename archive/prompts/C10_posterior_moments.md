# C10 — Posterior-moment backpropagation and leave-one-branch advantages

**Depends on:** P10, C00, C09.
**Supersedes:** `Prompts/P10_values.md`.
**Files:** `pbg/core/posterior_values.py`;
`tests/test_c10_posterior_moments.py`.
**May touch:** `pbg/core/completion.py`, `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`, and
the `BranchedValue` section of `docs/latex/core_math.tex`; inspect
`pbg/core/values.py`, completion DTOs, and P10 tests. P10 is a description of
the implementation being corrected, not an additional contract. Run focused
baseline tests, then implement code, tests, and notes before the focused and
full suites.

## Goal

Implement the corrected engine alongside the still-live P10 module so the full
suite remains green between C10 and C11. `posterior_values.py` propagates
posterior mass P and posterior-weighted return mass M, emits A and V by default,
and makes Q optional. C11 promotes this module to the public `core.values` API
and removes the obsolete implementation.

## Interfaces

```python
@dataclass
class HeadEvals:
    node_ids: Tensor       # stable decision-row order
    v: Tensor              # fp32 [Nd,X]
    aux: dict[str, Tensor] = field(default_factory=dict)

@dataclass
class NodeTargets:
    node_ids: Tensor
    parent_p_hat: Tensor   # [Nd,X]
    child_p_hat: Tensor    # [Nd,E,X]
    a_hat: Tensor          # [Nd,E,X]
    v_hat: Tensor          # [Nd,X]
    child_event_row: Tensor
    child_mask: Tensor
    child_count: Tensor
    q_hat: Tensor | None = None
    aux: dict[str, Tensor] = field(default_factory=dict)

def branched_value(tree, leaf_stats, reach, heads, lam, game, *,
                   configured_node_budget: int,
                   emit_q=False, auxiliary_reducers=(),
                   max_resident_rows=2_000_000) -> NodeTargets: ...

@dataclass(frozen=True)
class AuxiliaryChannelSpec:
    name: str
    leaf_field: str
    head_field: str | None = None
    bootstrap: Literal["none", "every_player_node"] = "none"
    emit_field: str | None = None
```

An auxiliary channel is a dense tensor with leaf values taken from
`leaf_stats.<leaf_field>`. Core uniformly averages it over abstract children.
When `bootstrap="every_player_node"`, `heads.aux[head_field]` must exist and
the channel is lambda-mixed at every player node after target emission. If
`emit_field` is set, the captured tensor is returned in
`NodeTargets.aux[emit_field]`. Build the channel list once before the batched
depth loop; never dispatch Python callbacks per node/edge. V1 passes an empty
tuple and allocates no auxiliary tensors.

## Binding recurrence

- Leaf: `P=leaf.p_hat`; `M=P*leaf.v_hat`. Their product must be formed from
  statistics whose bucket count and bucket reward sum share the same samples;
  define zero-count bucket values as zero. A future completion DTO may supply
  an equivalent direct `m_hat`, but v1 does not add another required field.
- Internal node: uniform average of child P and M over abstract children.
- Player node owned by p:
  - `V=Cond(M_parent[p],P_parent[p])`;
  - transient `Q_a=Cond(M_child_a[p],P_child_a[p])`;
  - if K>1, sibling conditional value uses total-minus-child P/M sums;
  - `baseline_a=lam*sibling+(1-lam)*old_V`, or `old_V` when K=1;
  - `A_a=Q_a-baseline_a`;
  - emit parent P, each child P, A, V and edge metadata before bootstrapping;
  - propagate `M_parent[p]=lam*M_parent[p]+(1-lam)*P_parent[p]*old_V`.
- Chance nodes aggregate only. Non-owner M rows are not bootstrapped.
- `Cond(0,0)=0` with an explicit valid-mass mask; do not create NaN/Inf.
- Accumulate in fp32 in production and fp64 in the independent oracle. Reject
  fp16/bf16 leaf or working accumulators.
- Q is transient unless `emit_q=True`.

Factor the implementation into bulk `aggregate`, `conditionalize`, `bootstrap`
and `emit` helpers. Optional auxiliary reducers may add tensor channels later,
but v1 allocates no Vvec/U state.

For memory chunking, compute
`trees_per_chunk=max(1, max_resident_rows//(configured_node_budget*P_max))` and iterate
fixed contiguous tree-id ranges. Select each range with device masks and process
whole trees only; never split a dependency chain or build Python node lists.
If one configured tree can exceed the row cap, raise a configuration error.

## Tests

All new tests import `pbg.core.posterior_values`; historical P10 tests continue
to exercise the untouched old module during this one-step migration window.
C11 migrates those historical tests when it switches the public API.

- A hand-computed posterior-shift tree rejects direct averaging of conditional V.
- Repeated simulations verify unbiased P/M bucket sums and demonstrate finite-
  sample ratio bias plus convergence.
- Total-minus-child baselines match a slow O(K²) fp64 oracle.
- Equal-labeled siblings receive distinct leave-one-out values.
- K=1 and lambda in `{0, .5, .95, 1}` match direct recurrences.
- `emit_q=False` stores no Q; `emit_q=True` matches transient Q.
- Chunked GPU/CPU results match an independent recursive fp64 oracle.
- Padding, zero-mass buckets, coupled priors and public fold rewards are covered.
- A dummy scalar `AuxiliaryChannelSpec` verifies uniform aggregation,
  capture-before-bootstrap ordering, and zero allocation when no spec is passed.

Required commands:

```bash
pytest -q tests/test_p10_values.py
pytest -q tests/test_c10_posterior_moments.py
pytest -q
```

## Do not

- No edits to `pbg/core/values.py`, training, losses, or P10/P11 tests in this
  staging prompt; C11 performs the atomic public-API switch.
- No model calls, moment heads, KL, or NeuRD implementation.
- Never average conditional V directly.
