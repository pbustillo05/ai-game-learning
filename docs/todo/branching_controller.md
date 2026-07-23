# Finish the two missing branching controls

Work in `/home/pablob/ai-game-learning`. Read `AGENTS.md`, preserve unrelated
changes, and use only physical GPU 2 for CUDA tests
(`CUDA_VISIBLE_DEVICES=2`).

Two controls are already implemented and working:

- `branching_min_nodes`: lower-size refill in fixed and dynamic modes.
- `safe_tree_size`: C-head upper-size pressure in both modes, now gated on
  configurable sustained C accuracy (Gamma deviance < 0.1 and p95
  underprediction < 2x for 8 iterations by default).

Do not redesign those controls. Only add minimal regression coverage for them.
Implement the two missing controls below.

Relevant files are `pbg/core/branching.py`,
`pbg/core/adaptive_branching.py`, `pbg/core/memory_control.py`,
`pbg/core/execution.py`, `pbg/core/training.py`, and
`pbg/core/checkpointing.py`. The failure evidence is in
`runs/experiments/branching_hu32_ckpt1110_20260722/REPORT.md`.

## 1. Accounting-based OOM retry

Add a last-resort OOM safeguard for branched rollout in both fixed and dynamic
modes, modeled on sequential memory accounting.

- The current structural `ByteAccountant` is insufficient because it excludes
  model/evaluator workspace and CUDA allocator pressure. Reuse the sequential
  ledger/forecasting machinery to make a conservative admission decision before
  risky allocation or forward work, including C-head queue forecasts.
- On a memory rejection, abort the whole rollout, release every rollout owner,
  delete all nodes, and restart from the root with a new independent tree and a
  fresh attempt RNG address.
- Fresh-tree rejection/resampling is explicitly approved. Do not block on bias
  concerns and do not replay the rejected tree.
- Rejected attempts must not affect optimization, C calibration, ordinary
  completed-tree controller observations, or completed-rollout statistics.
- Before the fresh retry, apply a small configurable emergency backoff using the
  existing dynamic control:
  - in fixed mode, reduce the effective beta scale;
  - in dynamic mode, reduce kappa;
  - synchronize the backoff across ranks, clamp it to configured bounds, and
    compound it if several attempts fail.
  This is memory-pressure feedback, not a completed tree-size observation.
  Preserve it for the remainder of the iteration and as evidence for the next
  iteration instead of restoring the pre-OOM pressure after a successful retry.
- Make cleanup idempotent so double release cannot mask the memory failure.
- Use a configurable retry limit. On exhaustion, clean up and save a terminal
  checkpoint for the last completed iteration rather than crashing with CUDA
  OOM.
- Under distributed training, a rejection on any rank must make all ranks
  discard that attempt and advance to the same new attempt number.
- Log retry count, reason/phase, predicted memory, allocated/reserved peak, and
  cleanup status.

Keep the implementation narrow; reuse existing memory-control components rather
than creating a second framework.

## 2. Fixed-beta adaptation between iterations

Dynamic mode already adapts kappa. Do not rework it.

For fixed mode, add a checkpointed controller that adjusts the effective beta
constants before each new iteration using past tree size and memory pressure.

- Add/configure an ideal tree-size target between `branching_min_nodes` and
  `safe_tree_size`, validating the ordering when all are configured.
- Use damped/EMA feedback.
- Preserve the relative shape of the configured per-family beta schedule:
  preferably apply one bounded multiplicative/log-domain scale to all configured
  coefficients.
- Keep every effective beta finite and at least 1.
- Tree sizes above target and high memory pressure must reduce the next
  iteration's beta scale; sizes below target with safe memory headroom may
  increase it.
- Commit updates only after a successfully completed iteration. A rejected OOM
  attempt may supply memory-pressure evidence, but not a tree-size observation.
- Synchronize the observation across ranks and checkpoint/restore the controller
  exactly.
- Log configured/effective beta, scale, target error, memory pressure, and bound
  saturation.

Use the smallest strict configuration/schema change that fits the existing
controller conventions.

## Focused tests

Add tests for:

1. fixed-beta update direction, damping, clipping, and schedule preservation;
2. fixed-beta checkpoint/resume;
3. lower < ideal < upper configuration validation;
4. injected memory rejection causing complete cleanup and a genuinely fresh
   tree with lower fixed-beta scale or kappa;
5. rejected attempts not reaching learning, calibration, or ordinary
   completed-tree controller observations;
6. retry exhaustion producing a clean terminal checkpoint;
7. mocked distributed common-attempt behavior;
8. monotone, bounded emergency backoff across repeated failures in both modes;
9. small CUDA fixed and dynamic retry paths;
10. minimal regressions for existing tiny-tree refill and calibrated
   `safe_tree_size`.

Run the focused and directly affected existing tests. Run the full non-GPU suite
if shared configuration, checkpointing, or training lifecycle code changes.

For CUDA, use the repository venv/toolkit setup from `AGENTS.md` and run only
with `CUDA_VISIBLE_DEVICES=2`. Check GPU state before and after, and leave GPU 2
idle.

Report the implementation, new hyperparameters/checkpoint state, test results,
and any concrete remaining limitation.
