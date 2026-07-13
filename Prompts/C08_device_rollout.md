# C08 — Device-resident rollout foundation

**Depends on:** P08, C00.
**Files:** `pbg/core/rollout.py`, `pbg/core/tree.py`, `pbg/core/game.py`,
`pbg/core/config.py`, `pbg/games/synthetic/family.py`;
`tests/test_p8_tree.py`, `tests/test_p8_rollout.py`,
`tests/test_c08_device_rollout.py`.
**May touch:** `pbg/core/completion.py`, `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`, and
the rollout/branching portions of `docs/latex/core_math.tex`; inspect every
listed source file and the P08/P09 tests. Dependencies describe expected code
already present in the repository, not extra instructions to paste. Run the
focused baseline tests first, then implement code, tests, and the notes handoff
in this session, followed by the focused and full suites.

## Goal

Build a batched, device-neutral rollout protocol alongside the still-live P08
entry point. The new public staging entry is
`sample_trees_device(bundle, cfg, gen)`. Keep the old positional `sample_trees`
behavior isolated and deprecated so the full suite stays green until C11
performs the atomic switch and removes it. The synthetic backend remains the
CPU mathematical oracle; poker supplies a tensor backend in P13A.

## Public interfaces

`StateBatch` is an opaque `int32[B]` tensor of handles into a backend-owned
state store on the same device. `TreeBatch.state_ref` stores these handles.
Synthetic CPU handles index its existing state table; P13A supplies a
device-resident poker tensor store. Core never dereferences a handle.

```python
StateBatch = Tensor  # int32 [B], same device as the backend state store

@dataclass
class EventBatch:
    public_id: Tensor  # int32 [B], stored in TreeBatch.child_event
    payload: dict[str, Tensor] = field(default_factory=dict)
    # Every payload tensor has leading dimension B and shares public_id.device.

@runtime_checkable
class RolloutBackend(Protocol):
    device: torch.device
    n_players: int
    x_sizes: tuple[int, ...]
    max_rollout_steps: int
    def root_states(self, n: int) -> StateBatch: ...
    def classify(self, states: StateBatch) -> Tensor: ...   # int32 [B]
    def legal_events(self, states: StateBatch) -> tuple[Tensor, Tensor]: ...
    def policy_forward(self, frozen_model: nn.Module, states: StateBatch,
                       reach: Tensor) -> PolicyOutput: ...
    def sample_profiles(self, states: StateBatch, reach: Tensor,
                        gen: torch.Generator) -> Tensor: ... # int32 [B,P], one per row
    def sample_events(self, states: StateBatch, profiles: Tensor,
                      policy_out: PolicyOutput | None,
                      adapter: PolicyAdapter,
                      gen: torch.Generator) -> EventBatch: ...
    def step(self, states: StateBatch, events: EventBatch) -> StateBatch: ...
    def reach_update(self, reach: Tensor, states: StateBatch, events: EventBatch,
                     policy_out: PolicyOutput | None) -> Tensor: ...
    def beta_fn(self, states: StateBatch) -> Tensor: ...    # fp32 [B]

@dataclass
class GameBundle:
    model: nn.Module
    rollout_backend: RolloutBackend
    policy_adapter: PolicyAdapter
    completion_backend: MCCompletion
```

The backend owns public encoding and pre/postflop routing. Training must not
discard or bypass it by calling the raw model directly.
`EventBatch.public_id` is the stable public label used by generic topology and
policy caches. Synthetic events need no payload. Poker chance events put padded
dealt cards in `payload["cards"]` while assigning a deterministic int32 label to the
sorted dealt card set; player events use the existing action id and no payload.
Core may batch, mask, `index_select`, and `repeat_interleave` every payload
tensor by its leading dimension but never interprets payload keys or values.

Sequential C08 expands every active row once: sample one fresh profile, sample
one event, step once, and discard the profile. C09 obtains variable offspring
by repeating parent handles/reach rows first and then calling these same
one-sample-per-row methods.
Within each frontier wave, partition rows on device into player and chance
batches. `policy_forward` is called only for player rows; chance rows pass
`policy_out=None` to `sample_events`/`reach_update`. TERMINAL/PRETERM rows are
never expanded.

## Device and storage contract

- `TreeBatch`, frontier state, reach, cached policy rows, RNG draws and output
  topology stay on one device throughout rollout/training.
- Replace the Python-list `TreeBuilder` in production with a preallocated
  `DeviceTreeBuffer` containing node arrays (`parent`, `tree_id`, `owner`,
  `state_ref`, depths, path weight) of capacity `n_trees*node_budget` and edge
  arrays (`edge_parent`, `child_node`, `child_event`, edge weight) of capacity
  `node_capacity-n_trees`. Roots occupy the first `n_trees` rows. Each frontier
  append uses device prefix sums to reserve contiguous child rows in
  parent-major order. Finalization builds `child_ptr` from device
  `bincount(edge_parent)` and returns views truncated to realized counts.
- Keep the existing simple CPU `TreeBuilder` only as a test-fixture/oracle; the
  production sampler must use `DeviceTreeBuffer` on both CPU and CUDA.
- Production hot paths contain no `.cpu()`, `.numpy()`, `.tolist()`, or scalar
  `.item()` used to drive per-node control flow.
- Run exactly `max_rollout_steps` frontier waves with device masks, so loop
  termination does not require a host read each wave. At the end, assert that
  no live frontier remains; an underestimated bound is an error.
- Use explicit generators created on the rank's device.
- Preallocate topology/payload buffers from `n_trees` and `node_budget`.
  Overflow raises with requested/capacity diagnostics; never change the sampled
  offspring law to fit the buffer.
- Alias tables return decision-row indices so later stages share one stable row
  order and require no Python lookup maps.

## Completion-cut correction

Synthetic `is_pre_terminal` may return true only when no player decision is
reachable afterward. In standard Leduc, roll through the public card and second
betting round; complete after final betting. Fold endings remain terminal.

## Tests

Migrate P08 tests whose call signatures or byte-level RNG assumptions conflict
with the new backend/device boundary. Preserve all still-valid reach, topology,
posterior-law and determinism coverage; do not delete a failing assertion
without replacing the behavior it protected.

- Hand-written Leduc paths cover the state before the board, after the board,
  after final betting, and folds.
- Sequential public-history frequencies match exact synthetic enumeration.
- A deterministic toy backend gives identical CPU/CUDA topology when supplied
  identical draws (CUDA test skipped when unavailable).
- AST/static test rejects host-transfer methods in designated production hot
  functions.
- Buffer overflow raises and never returns a truncated tree.
- Legacy P08 callers remain behaviorally unchanged and emit a deprecation
  warning only when `debug_asserts` is enabled; ordinary tests/logs stay quiet.

Required commands:

```bash
pytest -q tests/test_p8_tree.py tests/test_p8_rollout.py
pytest -q tests/test_c08_device_rollout.py
pytest -q
```

## Do not

- Do not remove or behaviorally change the legacy P08 entry yet; C11 owns its
  deletion after all downstream consumers have migrated.
- Do not implement branching, values, poker tensor stepping, or losses here.
- Do not require the CPU synthetic oracle itself to avoid Python control flow.
