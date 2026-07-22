# Release checklist

## Required before shipment

- [x] Resolve the `rank_count` contract: implement variable hand/class/card
  widths through production, or explicitly restrict v1 to `R=13` everywhere and
  reject reduced decks at config load.
- [x] Derive lossless `max_history_tokens` from `(P,S)`, validate it before
  allocation, and pass adversarial legal-history tests for every boundary player
  count and stack class.
- [x] Gate clipping, optimizer step, parameter mutation, and scaler transition on
  the common reduced finite-gradient decision in FP16, BF16, and FP32.
- [x] Fix `linear_decay` to use `cfg.training.iters` and test start/mid/end in both
  registered composites.
- [x] Add complete numeric and cross-field configuration validation with dotted
  paths, including node budget, lambda, schedules, optimizer values, deck/model
  width, history capacity, and checkpoint interval.
- [x] Require a hardware profile for any CUDA launch, including Python API use.
- [x] Remove or account for full-policy hot-path host synchronization. Extend
  sync-debug/instrumentation tests to history packing and folder-summary caching.
- [x] Rerun the full suite, all three one-GPU acceptance modes, and all three
  unequal-work two-rank modes after the fixes.
- [x] Update README/contracts/poker/theory status and supersede the old accepted
  readiness claim only after the corrected tests pass.

## Minimum acceptance matrix after repair

### Game geometry

- [x] `P=2` and `P=6` at `S=8` and `S=128`.
- [x] Every advertised `R` boundary and at least one interior reduced deck,
  subject to `2P+5 < 4R`; if full-deck-only v1 is chosen, test parse-time
  rejection of every other R.
- [x] Long legal minimum-raise histories beyond 256 at 3--6 players.
- [x] Folded-player continuation with one and multiple folders.

### Modes and distribution

- [x] normal, sequential RB, and branched RB on one GPU;
- [x] all three on two GPUs with unequal local decision mass;
- [x] branched root beta fractional rounding and duplicate children;
- [x] sequential global-start quota boundary at world size;
- [x] checkpoint/resume for each mode and a changed compatible hardware profile.

### Loss/update

- [x] `ppo` with MSE and clipped values;
- [x] `rb_ppo` in both RB modes with MSE and clipped values;
- [x] constant and corrected linear entropy schedules;
- [x] optional reference KL across refresh and resume;
- [x] Adam and AdamW;
- [x] injected finite/nonfinite gradients in FP16, BF16, and FP32;
- [x] rejection tests for every invalid numeric boundary.

### Safety and performance

- [x] idle-device check and explicit device targeting;
- [x] preflight against worst legal history and model token pairs;
- [x] zero unaccounted synchronization in full policy construction, or a revised
  explicit bounded-control contract with exact counters;
- [x] no post-run GPU memory/process residue;
- [x] short soak proving bounded reserved memory;
- [x] one realistic six-player checkpoint run after tiny gates pass.

## Recommended immediately after the minimal release

- [ ] Add the bounded approximate-best-response trend probe from
  `docs/todo/BR_probe.md`.
- [x] Extract optimizer/finite-step and checkpoint lifecycle from
  `pbg/core/training.py`.
- [x] Extract history packing, cost planning, and folder caching from
  `pbg/games/poker/rollout_backend.py`.
- [ ] Establish and store a reproducible performance baseline for the repaired
  full-policy path.
- [ ] Decide whether optional moment heads should receive real losses/control or
  be removed from the production schema until that research is ready.

## Ship decision rule

All required items and the minimum acceptance matrix pass on repaired revision
`5c10366`. The engineering/estimator-correctness release gate is therefore GO.
The unchecked approximate-BR item is M-05, explicitly excluded by the repository
owner. Until it exists, shipment must not make a strategy-quality,
exploitability, improvement, or convergence claim. The remaining unchecked
performance-baseline and moment-head decisions are follow-up research/operations
work rather than blockers for this bounded release.
