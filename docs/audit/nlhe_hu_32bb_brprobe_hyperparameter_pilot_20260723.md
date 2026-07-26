# HU 32 bb shared-response probe pilot — 2026-07-23

## Scope

This pilot used the final 700-iteration normal target. The final normal and RB
checkpoints were produced as checkpoint v7 while the current tree requires v8.
Metadata-only target copies were made by adding null cost-safeguard fields.
The model-state hashes were verified unchanged; provenance is recorded in
`runs/experiments/nlhe_hu_32bb_rb_brprobe_targets_20260723/manifest.json`.

The workflow is the authoritative shared-seat approximate-response probe. Its
reported sum is an approximate NashConv lower bound, not exact exploitability.

## Baseline mismatch

The original steady-state runs used 3,072 roots, minibatch 256, and 116
optimizer steps, giving 29,696 learner presentations per iteration.

| Run | Fresh decisions/iter | Presentations/iter | Reuse |
|---|---:|---:|---:|
| Original normal, last 50 | 10,282.6 | 29,696 | 2.89x |
| Original RB, last 50 | 9,913.6 | 29,696 | 3.00x |
| Probe literal baseline | 5,111.2 | 29,696 | 5.81x |
| Probe matched reuse | 5,144.5 | 15,360 | 2.99x |

The probe retains only decisions owned by the selected response seat. In HU,
this approximately halves the trainable rows, so copying 116 steps doubles
within-batch reuse.

## Completed comparison against the normal target

Both completed arms used target initialization, 3,072 roots, six iterations,
1,024 completion samples, minibatch 256, AdamW at $5\times10^{-5}$, and all
original loss coefficients.

| Setting | Steps/iter | Mean iter s | Six-iter wall s | Final ratio max | Final ref KL | Clipped/nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| Literal baseline | 116 | 101.51 | 609.06 | 7.27 | 0.0884 | 0 / 0 |
| Matched reuse | 60 | 68.93 | 413.59 | 4.32 | 0.0379 | 0 / 0 |

Matched reuse was 32.1% faster, preserved the original approximately 3x reuse,
and had milder policy movement. A 512-hand paired evaluation at iteration six
was statistically inconclusive:

| Setting | Mean probe gain, bb/hand | 95% half-width | Approx. NashConv lower bound |
|---|---:|---:|---:|
| Literal baseline | -0.487 | 1.105 | -0.975 |
| Matched reuse | -0.449 | 0.986 | -0.898 |

Negative finite-budget probe gains are valid lower bounds. Six iterations and
512 evaluation hands are not enough to estimate target exploitability.

A preliminary 2,048-root/40-step arm also matched 3x reuse and averaged 44.73
s/iteration, but it was stopped after the user requested an immediate result.
Its three-iteration checkpoint was too early and noisy to select over 3,072
roots. A separate 4,096-root geometry calibration succeeded at 6.69 GB peak
reserved memory with 3.31 GB hard-cap headroom, but its planned 80-step learning
arm was not run.

## Recommendation

Use the conservative measured setting for both targets, sequentially:

- target initialization;
- sequential RB with 1,024 completion samples;
- 3,072 roots/GPU fixed, family batch 84, at most 64 active trajectories;
- minibatch 256 and 60 optimizer steps per iteration;
- AdamW, learning rate $5\times10^{-5}$, betas $(0.9,0.999)$, gradient clip
  0.5, and AMP;
- unchanged loss coefficients and 172.8M effective-hand KL refresh threshold.

At the measured rate, 700 probe iterations are about 13.4 hours on one 2080 Ti.
About 1,400 iterations match both the original runs' total optimizer-step budget
and approximately their total number of fresh learnable decisions; that is a
roughly 26.8-hour probe per target. Budget exhaustion remains the only stopping
rule.

This is a provisional throughput/statistical recommendation, not a claim that
60 steps is globally optimal. A final probe must be evaluated with substantially
more than 512 paired hands.
