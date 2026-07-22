# Math, model, and estimator fidelity

## Verdict

No confirmed equation-level defect was established for normal, sequential RB,
fixed branched RB, or KKT. Their sampling/target/loss wiring is substantially
faithful to `core_math.tex`. Certification is nevertheless incomplete because
storage violations can invalidate the intended bounded implementation, BR has
no authoritative mathematics, and important performance-preserving target
algorithms were replaced by scalar CUDA control loops.

## Four training semantics

- **Normal:** independent terminal trajectories, realized private type,
  player-clock reverse $\lambda$ return, scalar PPO/value/entropy/KL. The sole
  target path is `build_normal_targets` (`pbg/core/targets.py:137-200`).
- **Sequential RB:** independent public pre-terminal paths, posterior
  completion, deduplicated completion pools, player-clock recurrence, and the
  sequential loss. See `pbg/core/targets.py:248-312` and the detailed contract
  in [batch 00](batches/00_math_and_modes_contract.md).
- **Branched RB:** exactly one labeled tree per GPU, including ordinary
  randomized rounding at the distinguished root; fixed and KKT share branched
  targets, importance/alias semantics, and Gamma-deviance T/C losses.
- **KKT:** `gamma_deviance`, randomized rounding, clipped log-domain intensity,
  ancestor recurrence, and damped/checkpointed $\log\kappa$ remain in
  `pbg/core/adaptive_branching.py:20-209`; nominal/executed correction reaches
  the controller (`pbg/core/training.py:1027-1064`).

Validated fixed-beta production requires branching heads and positive T/C loss
coefficients (`pbg/core/config.py:380-388`) and rejects missing frozen T/C in
training (`pbg/core/training.py:983-1004`). The broad claim that fixed mode
omits T/C is rejected. The remaining medium issue is that the lower-level
`sample_branched_tree` helper conditionally infers these fields
(`pbg/core/branching.py:438-460,683-699`) instead of receiving an unconditional
branched schema.

## Model and poker semantics

Natural partitions and MHA/biased-SDPA production primitives are retained.
The confirmed model-adjacent violations are storage/ownership issues: preflop
model carriers are widened from natural $K=169$ to exact $H=1326$ before later
compression (C05), and folded summaries are neutralized/duplicated/unaccounted
(C09). Legacy/ablation fields and CPU/MPS model routes belong to C13/C15.

The original model under-admission claim is rejected. Active CUDA planning and
execution both use conservative flop-width `_family_maximum_width`
(`pbg/games/poker/rollout_backend.py:1861-1880,2100-2127,2188-2230`;
`pbg/core/training.py:2339-2370`). C17 is solely a street-aware physical packing
optimization.

## BR authority gap

The current probe is an approximate shared-policy response. `core_math.tex`
does not specify whether the required response is shared or independent per
seat, how seats are sampled/weighted, the optimization objective, stopping
rule, cutoff value, or NashConv/reporting estimator. Until those choices enter
authority and tests, C03 prevents mathematical certification. This does not by
itself prove the implementation's BR equation wrong; C04 is the separate
confirmed implementation defect.

## Static target risk

`build_normal_targets` and `build_sequential_rb_targets` read CUDA topology and
decision scalars inside Python loops (`pbg/core/targets.py:169-182,251-312`).
That is C11: a high-confidence synchronization risk, not a measured numerical
or wall-time regression. Replace it with the authoritative bounded segmented
algorithm, not the deleted legacy `values.py` representation.
