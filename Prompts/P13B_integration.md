# P13B — Corrected poker integration, presets and acceptance

**Depends on:** C11, P12, P13A.
**Files:** `pbg/core/training.py`, `pbg/games/poker/bundle.py`;
`configs/kuhn_smoke.yaml`, `configs/leduc.yaml`,
`configs/synthetic_dial_example.yaml`, `configs/nlhe6_h100.yaml`;
`tests/test_p13b_integration.py`, `tests/test_p13b_ddp.py`.
**May touch:** config schema and `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`, and
`POKER_GPU_EVALUATOR_IO.md`; inspect the completed C11, P12 and P13A public
interfaces plus current configs and checkpoint code. Dependencies describe the
expected repository state only. Run focused baselines, then complete wiring,
presets, tests, acceptance commands, and notes in this session.

## Goal

Wire the corrected minimal RB-PPO pipeline through the GPU poker backend and
ship first-run fixed-intensity presets. This prompt integrates existing
components; it does not add estimators or allocation research.

## Bundle construction

Core must preserve its import direction: it contains no static poker import.
Select the game factory through an explicit string registry and `importlib`,
then call the game-owned `build_bundle(cfg, device)`:

```python
GAME_FACTORIES = {
    "synthetic": "pbg.games.synthetic.bundle:build_bundle",
    "poker": "pbg.games.poker.bundle:build_bundle",
}
```

The poker factory constructs and returns:

```python
GameBundle(
    model=PokerModelBundle(preflop, postflop),
    rollout_backend=PokerRolloutBackend(...),
    policy_adapter=PokerPolicyAdapter(...),
    completion_backend=PokerGPUCompletion(backend=cfg.completion.backend),
)
```

Construct `NLHEEngineConfig`/tensor backend from `cfg.game.poker`, build
`PreflopNet` and `PostflopNet` from `cfg.model`, wrap them in
`PokerModelBundle`, and give the rollout backend the same adapter, sibling
joint-profile sampler, fixed beta mapping, and model-input builders. There is
one owning model instance; the backend receives it only through
`policy_forward(frozen_model, ...)` and must not retain a second trainable copy.

The completion backend is the sibling facade execution choice
`"auto" | "cpu" | "cuda"`; `cpu` is the sibling mathematical oracle, not a
local fallback. Missing/incompatible sibling packages fail with exact install
and API-version guidance.

## Presets

- `kuhn_smoke.yaml`: `backend: exact`, sequential beta, `n_trees: 512`,
  completion 256, 300 iterations, minibatch 256, actor V only.
- `leduc.yaml`: `backend: exact`, fixed beta with `default: 1`,
  `round_0: 1`, `chance_round_1: 2`, `round_1: 2`; completion 512,
  `n_trees: 256`, minibatch 256, 300 iterations. The state before its public
  board is not PRETERM because round 1 actions remain.
- `synthetic_dial_example.yaml`: `backend: mc`, fixed beta with explicit
  round/chance keys, `n_trees: 128`, completion 1024, and a comment that exact
  exploitability is disabled when P01's feasibility guard is exceeded.
- `nlhe6_h100.yaml`: six players, 100bb; `d_model: 192`, 6 heads, 3 preflop
  layers, 4 postflop layers, 4 raise-mixture components, 32 ticks;
  `beta_mode: fixed` with `default: 1.0`, `preflop: 1.0`, `flop: 1.05`,
  `turn: 1.10`, `river: 1.25`, and each `chance_to_*: 1.5`;
  `node_budget: 8192`, `reach_storage: recompute`, sibling `backend: cuda`,
  completion 2048, `n_trees: 256` per rank, minibatch 128, 32 GD steps,
  lambda .95, policy clip .2, AMP on, suit consistency off. Use 10,000
  iterations and checkpoint every 100. Mark model/batch/tree/completion values
  `# TUNE` for one H100; DDP keeps these values per rank.

Do not retain `beta_avg`, `beta_max`, `warmup_iters`, `use_moment_heads`,
`ref_period`, `c_kl`, `c_u`, or `c_vvec` in these active presets.

## Device-residency acceptance

- Rollout states, tree topology, reaches, cached policy rows, completion
  statistics, targets, minibatches and optimizer state remain on the rank GPU.
- Only asynchronous scalar metrics and scheduled checkpoint serialization may
  reach CPU during training.
- Each DDP rank rolls out independently; only gradients synchronize.

## Tests

- Three-iteration two-player poker smoke on sibling CPU backend: finite loss,
  labeled branching, finite targets, chip conservation and checkpoint roundtrip.
- Same smoke on CUDA when available; assert no tensor device copies at sibling
  boundaries and no host transfers in the rollout hot path.
- Duplicate-label branches remain distinct through completion and loss.
- Policy-cache rows normalize for every valid private row; target masks respect
  board/type masks.
- All presets load under strict schema and contain no deferred settings.
- One-rank versus two-rank Kuhn curves both improve and have comparable final
  exploitability after sample-count scaling.
- Profiling command records preflop/postflop batch histograms and GPU utilization
  on an H100/SXM node without making a fixed CI performance assertion.

## Final acceptance commands

1. Full test suite, including slow tests when dependencies/hardware exist.
2. Kuhn and Leduc training runs with first/last exploitability recorded.
3. Shrunken poker CPU-sibling smoke with metrics excerpt.
4. Shrunken poker CUDA smoke with device-residency and batch-size report.

Required test commands:

```bash
pytest -q tests/test_c11_minimal_training.py tests/test_p12_eval.py
pytest -q tests/test_p13a_gpu_rollout.py tests/test_p13b_integration.py
pytest -q tests/test_p13b_ddp.py
pytest -q
```

## Do not

- No Neyman allocation, moment heads/losses, reference KL, NeuRD, local poker
  oracle, silent backend fallback, or cross-GPU tree movement.
