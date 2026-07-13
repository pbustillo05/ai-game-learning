# C09 — Fixed-intensity labeled branching

**Depends on:** P09, C00, C08.
**Supersedes:** `Prompts/P09_branching.md` for production semantics.
**Files:** `pbg/core/branching.py`, `pbg/core/rollout.py`,
`pbg/core/config.py`; `tests/test_c09_labeled_branching.py`.
**May touch:** `pbg/core/tree.py`, `tests/test_p9_branching.py`,
`IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`, and
the labeled-tree section of `docs/latex/core_math.tex`; inspect the current
rollout/tree/branching implementation and P08/P09 tests. Treat P09 as historical
behavior only—this prompt contains every superseding decision. Run focused
baseline tests, then implement code, tests, and notes before running focused and
full suites.

## Goal

Replace spine-Poisson quotient trees with the labeled `SampleTree` in
`core_math.tex`. V1 supports only sequential and fixed public-stage intensity.

## Fixed intensity

- `beta_mode` is exactly `"sequential" | "fixed"`.
- Sequential mode uses `beta=1`; fixed mode calls the game backend's batched
  `beta_fn(states) -> fp32[B]`.
- `RolloutConfig.beta_fix` is a `dict[str,float]` with required key
  `"default"`. The backend maps each public state to a stable bucket string,
  uses an exact-key override when present, and otherwise uses `default`.
- Poker buckets are preflop/flop/turn/river plus game-chosen chance-transition
  buckets. Synthetic buckets are betting round/chance phase.
- Reject nonfinite beta or beta below one. Remove `IntensityController`,
  `neyman`, bucket means, warmup state and moment-output inputs.
- Strict config loading rejects `beta_mode="neyman"`; legacy `beta_avg`,
  `beta_max`, and `warmup_iters` are removed by C11's schema migration rather
  than silently consumed.

Public entry point:

```python
def sample_branched_labeled(bundle: GameBundle, cfg: RolloutConfig,
                            gen: torch.Generator
                            ) -> tuple[TreeBatch, ReachCache, PolicyCache]: ...
```

This is the C08 device path used by new tests and later by C11. Preserve the
old positional P09 callable under a private `_legacy_sample_branched_trees`
name solely so untouched downstream callers remain green during the migration
window. It must not be called by the new implementation and C11 deletes it.

## Offspring and labels

For each active parent, draw on device:

```text
base = floor(beta)
K = base + Bernoulli(beta - base)
```

- Flatten abstract children with `repeat_interleave(K)`.
- Preserve flattened parent-major/sample-major order in CSR and caches; no
  sort by label is permitted.
- Draw one fresh posterior profile and event for every abstract child.
- Never reuse a hidden profile across siblings or descendants.
- Never deduplicate equal public events. Repeated `child_event` entries and
  repeated state payload references are valid and expected.
- Store every outgoing edge weight as exactly `1/K(parent)`.
- `path_weight[child]=path_weight[parent]/K(parent)`; decision alias tables
  sample the resulting abstract-node occupancy law.
- Equal-labeled children expand independently and receive independent leaf
  completion seeds.
- Batch profile sampling, action/chance sampling, stepping, classification,
  reach updates and append offsets. No Python loop per parent/child in the
  production backend.

`node_budget` is preallocated capacity and a loud safety boundary. Never inspect
sampled labels and then clamp/deduplicate to fit it.

## Tests

Rewrite obsolete P09 Poisson/dedup/controller/budget-clamp assertions into the
corresponding randomized-rounding/labeled-tree/fixed-beta/overflow assertions.
Preserve its terminal-law and fresh-profile statistical coverage.

- Integer beta gives deterministic K; fractional beta frequencies match the
  Bernoulli fraction with fixed seed/tolerance.
- A rigged one-action game produces repeated equal-labeled siblings which remain
  distinct nodes with equal `1/K` weights.
- A rigged duplicate-label game verifies independent descendant and completion
  draws across those siblings.
- Every nonleaf row has exactly K edges and uniform weights summing to one.
- `beta=1` matches the sequential public law.
- Averaged induced leaf measures match exact synthetic terminal probabilities.
- Overflow raises before returning a partial tree.
- Static hot-path device-residency checks from C08 remain green.

Required commands:

```bash
pytest -q tests/test_p8_tree.py tests/test_p8_rollout.py tests/test_p9_branching.py
pytest -q tests/test_c08_device_rollout.py tests/test_c09_labeled_branching.py
pytest -q
```

## Do not

- No Neyman controller or moment heads.
- No value propagation, loss changes, or event deduplication.
- Do not route new bundle callers through the legacy implementation.
