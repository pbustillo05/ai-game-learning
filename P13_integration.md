# P13 — Integration: poker wiring, presets, cross-cutting tests

**Depends on:** all previous prompts.
**Files:** `configs/leduc.yaml`, `configs/synthetic_dial_example.yaml`,
`configs/nlhe6_2080ti.yaml`; `tests/test_p13_integration.py`,
`tests/test_p13_ddp.py`; `pbg/core/training.py` poker branch of
`build_game_bundle` (this is the ONE core file you may edit, at the marked
NotImplementedError only).
**May touch:** `configs/kuhn_smoke.yaml` (final values), append
`IMPLEMENTATION_NOTES.md`.

## Goal

Wire poker into the generic training loop, ship the four presets, and land
the cross-cutting acceptance tests (T2 done in P09; here: T5 already in P11 —
this prompt owns T11 and the end-to-end smokes).

## `build_game_bundle(cfg)` poker branch

Construct: `NLHEEngine` → `PokerGame` (P07 adapter) → `PokerModelBundle`
(preflop+postflop nets from `cfg.model`) → `PokerPolicyAdapter` → `policy_fn`
(street-routed, batched, moment heads per flag) → poker `bucket_fn`
(P09 spec: `f"{street}_{n_live}_{potbin}"`). Completion backend from
`cfg.completion.backend`; rejection backend likewise. Return the 5-tuple.
Preflop 169-row outputs are lifted to hand-indexed rows at this boundary
(class broadcast), per the P07 adapter contract.

## Presets

- `kuhn_smoke.yaml`: sequential mode, n_trees 512, completion n_samples 256
  (exact-enum path where P01 supports it), 300 iters, minibatch 256 — sized so
  the P11 T10-lite test passes in minutes on CPU.
- `leduc.yaml`: `beta_mode: neyman` with moment heads ON (this preset is the
  branching showcase); beta_fix default 2.0, beta_avg 2.0, beta_max 6.0,
  warmup 50; document expected exploitability trajectory in a comment.
- `synthetic_dial_example.yaml`: a mid-size dial config demonstrating both
  dials turned up; feasibility-guarded eval disabled (comment why).
- `nlhe6_2080ti.yaml` (values are starting points, marked `# TUNE`):
  6 players, 100bb; d_model 192, heads 6, layers pre 3 / post 4, m_raise 4,
  n_ticks 32, r_beta_max 16; sequential mode for the first run
  (`beta_mode: sequential` with a commented neyman block ready);
  reach_storage: recompute; completion n_samples 2048, backend reference;
  n_trees 256/iter, minibatch 128, gd_steps 32, lam 0.95, clip 0.2,
  amp true; moment heads OFF, suit consistency OFF (first-run
  interpretability, per signed-off plan); KL ref_period null.

## Tests

- `test_p13_integration.py`:
  - **Poker end-to-end smoke (CPU-or-GPU, `@pytest.mark.slow`):** 3 training
    iterations of a shrunken NLHE config (2 players — a config value, not a
    code path; d_model 32; n_trees 8; n_samples 64; n_ticks 8): finite losses,
    targets finite, chips conserved in rollouts, checkpoint round-trip.
    Note: n_players=2 here is purely to make the smoke cheap; the shipped
    preset stays 6-max.
  - Same smoke with `beta_mode: fixed, beta_fix: 2.0`: branching exercised end
    to end (trees have >1 leaf; ν non-degenerate; budget respected).
  - RB plumbing check: PolicyCache rows at poker decision nodes normalize per
    type over legal events; NodeTargets masks consistent with board masks.
  - Preset lint: all four YAMLs load strictly; every documented flag present.
- `test_p13_ddp.py` (**T11**, `@pytest.mark.slow`, skipped unless
  `torch.distributed` + ≥2 visible devices or gloo CPU): 20-iter Kuhn run,
  world_size 2 vs world_size 1 with doubled n_trees/minibatch: overlay the
  loss and exploitability curves; assert final exploitability within 25%
  relative and both strictly improved from init. Use gloo on CPU so CI can run it.

## Final acceptance checklist (run and paste results into IMPLEMENTATION_NOTES.md)

1. `pytest -q` full suite green, slow included, with seeds/durations listed.
2. `python -m pbg.core.training --config configs/kuhn_smoke.yaml` → final
   exploitability printed; attach the number.
3. `python -m pbg.core.training --config configs/leduc.yaml` for 200 iters →
   exploitability curve decreasing; attach first/last values.
4. Poker smoke command line (the shrunken config) → metrics.jsonl excerpt.

## Do not

- No new features, no refactors, no tuning loops. Integration only.
- The only core edit is the marked `build_game_bundle` branch.
