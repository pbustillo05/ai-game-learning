# HU 32 bb normalized-loss smoke — 2026-07-21

## Outcome

Stack-normalized utility, deterministic value-head checkpoint migration, and
an explicit entropy-decay horizon passed CPU and two-GPU smoke gates.  No
production continuation was started.

The migration from checkpoint 1110 multiplied only these four tensors by
$1/32$:

- `preflop.value_head.affine.{weight,bias}`;
- `postflop.value_head.affine.{weight,bias}`.

All policy tensors remained byte-identical.  The fresh optimizer and KL
reference were constructed after loading the converted model.  The source
checkpoint path, SHA-256, iteration, source/target utility conventions,
conversion factor, and converted tensor names are present in `run.json` and
every new checkpoint.

## Test matrix

Both CUDA arms used physical GPUs 1 and 2, checkpoint 1110, 512 roots/GPU,
256 completion samples per leaf, 128 rows/GPU per synchronized step, and 16
steps per iteration.  They differed only in $c_V$.

| Arm | Iterations | $c_V$ | PPO | value | entropy | reference KL | final-step grad norm | clipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| preferred probe, first matched iteration | 1 | 0.05 | 0.001963 | 0.002953 | -0.000733 | 0.000013 | 0.03780 | no |
| continuity control | 1 | 0.016 | 0.001967 | 0.000945 | -0.000733 | 0.000013 | 0.03737 | no |

The three-iteration $c_V=0.05$ arm remained finite throughout. Its per-step
terms were:

| Iteration | PPO | value | entropy | reference KL | final-step grad norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.001963 | 0.002953 | -0.000733 | 0.000013 | 0.03780 |
| 1 | 0.000931 | 0.001772 | -0.000716 | 0.000029 | 0.02199 |
| 2 | -0.000060 | 0.001672 | -0.000695 | 0.000054 | 0.08554 |

All reduced gradients were finite, no optimizer step was skipped, and none of
the recorded final steps was clipped. Peak allocated memory was 3.35 GiB/GPU.
The three iterations took 71.55, 65.93, and 68.71 seconds. GPUs 1 and 2
returned to their display-only footprints afterward.

The maximum unweighted importance ratios were 10.81--14.33. These are outlier
diagnostics rather than posterior-weighted behavior KL and do not by themselves
show divergence, but weighted log-ratio/clip telemetry remains necessary before
a long continuation.

## Decision

Use $c_V=0.016$ as the initial continuation/control setting: it is the exact
unit-continuity conversion and produced a value term about half the matched PPO
term. Retain $c_V=0.05$ as the stronger value-learning arm; it was stable but
its matched value term was about 1.5 times PPO.

Keep the other tested settings:

```yaml
clip_eps: 0.2
clip_eps_v: 0.00625
c_h: 0.001
c_kl: 0.00625
entropy_schedule: linear_decay
entropy_decay_iters: 500
utility_normalization: stack
```

This short test cannot calibrate the proposed 37-iteration KL refresh cadence:
the reference was fresh and raw KL was still small. A longer cadence pilot
should follow only after all-step and component-gradient telemetry is present.
