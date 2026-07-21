# HU 32 bb training-parameter tuning — 2026-07-17

> Historical tuning evidence: these measurements used the pre-restoration
> rollout data plane. Scientific settings remain useful context, but memory and
> throughput numbers are not directly comparable to current tiny acceptance.

## Outcome

The balanced production setting is 2,560 sequential roots/GPU initially,
guarded growth to 3,072 roots/GPU, 64-row rollout queue packs, and a
128-row/GPU statistical learner minibatch for 116 optimizer steps. This keeps
the old total of 14,848 learner presentations/GPU/iteration while reducing
synchronization/update overhead and increasing rollout diversity. The learning
rate remains $5\times 10^{-5}$.

`configs/train/nlhe_hu_32bb_rb.yaml` contains these settings.

## Measured matrix

Every successful row used two RTX 2080 Ti GPUs, full-deck HU 32 bb sequential
RB, SDPA, AMP, and initialization from
`runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0000020.pt`. Learner-presentation
throughput counts both ranks. Distinct-decision throughput uses the global
retained decision count, not repeated minibatch draws.

| Roots/GPU | Batch x steps | LR | Iteration s | Optimization s | Learner rows/s | Distinct decisions/s | Reuse/GPU | Peak alloc/reserved GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 64 x 232 | $5\times10^{-5}$ | 93.13 | 69.43 | 318.9 | 123.4 | 2.58x | 4.45 / 6.80 |
| 2,048 | 128 x 116 | $5\times10^{-5}$ | 86.03 | 62.69 | 345.2 | 133.6 | 2.58x | 4.45 / 6.80 |
| 2,048 | 256 x 58 | $5\times10^{-5}$ | 85.64 | 62.14 | 346.8 | 134.2 | 2.58x | 4.45 / 6.80 |
| 3,072 | 128 x 116 | $5\times10^{-5}$ | 104.17 | 68.02 | 285.1 | 166.3 | 1.70x | 5.04 / 7.37 |
| 2,048 | 128 x 116 | $1\times10^{-4}$ | 88.10 | 64.60 | 337.1 | 130.5 | 2.58x | 4.45 / 6.80 |

The 2,560-root production starting point is between the two measured root
widths. Linear interpolation is only a planning estimate, but suggests about
95 seconds/iteration before checkpoint overhead, about 152 global distinct
decisions/s, and approximately 2.0x retained-row reuse.

## Production duration and checkpoint cadence

The production config uses 1,110 iterations. The measured 3,072-root
steady-state time is 104.1704 seconds/iteration. Accounting for eight estimated
2,560-root iterations and eight intermediate-width iterations before reaching
the ceiling gives approximately 32.09 wall-hours before checkpoint I/O, or
64.18 GPU-hours on two GPUs.

`checkpoint_every: 37` corresponds to approximately 62 minutes for the first
checkpoint during the faster controller ramp and 64 minutes thereafter. Since
$1{,}110=30\times37$, the thirtieth checkpoint is written after the final
iteration. Actual duration can move with policy-dependent rollout geometry,
thermal conditions, and filesystem checkpoint latency.

## Decisions

### Rollout size

Larger rollouts do not materially simplify queue management. At 2,048 roots,
only 320 of roughly 10,400 rollout model rows were forced partial drains and
queue age stayed at six or less. At 3,072 roots there were only 284 forced
drain rows. The reason to enlarge the rollout is statistical diversity, not
queue correctness.

Starting at 2,560 avoids spending roughly 24 controller iterations growing
from 2,048 to the old ceiling. The tested 3,072 ceiling is retained for guarded
growth. The 3,072 probe improved distinct-decision throughput by 24.4% over the
2,048-root/batch-128 run, but lowered repeated learner-row throughput by 17.4%,
so starting directly at the ceiling is not the balanced choice.

### Rollout queue packs

Keep `family_batch_rows` and `max_active_trajectories` at 64. A staged 128-row
probe completed rollout but stopped at the explicit memory gate because no
guarded completion/backward workspace remained. It did not OOM, and both GPUs
returned cleanly to idle. Statistical learner minibatches are physically split
independently, so this does not prevent batch 128 training.

### Learner minibatch

Batch 128 reduced the fixed-work iteration by 7.10 seconds (7.6%) and optimizer
time by 6.74 seconds (9.7%) while leaving peak memory unchanged. Batch 256
saved only another 0.39 seconds end-to-end (0.45%) and halves synchronized
parameter-update count again. Batch 128 is therefore the throughput knee with
the more conservative optimization statistics.

Use 116 steps so $116\times128=14{,}848$ presentations/GPU, matching the
validated $232\times64$ schedule. The 160,080-step frozen-reference cadence is
preserved by changing `kl_ref_interval` from 690 to 1,380 iterations. This
interval does not trigger during the configured 90-iteration run.

### Learning rate

Do not linearly scale the learning rate. The $10^{-4}$ probe remained finite,
but its maximum PPO importance ratio increased from 6.69 to 47.03 and its last
gradient was clipped to 0.350 of its pre-clip norm, versus 0.751 at
$5\times10^{-5}$. Retaining $5\times10^{-5}$ is the safer handoff.

## Artifacts

- `runs/experiments/nlhe_hu_32bb_tuning_queue128_1step/` (expected guarded failure)
- `runs/experiments/nlhe_hu_32bb_tuning_mb128_116step/metrics.jsonl`
- `runs/experiments/nlhe_hu_32bb_tuning_mb256_58step/metrics.jsonl`
- `runs/experiments/nlhe_hu_32bb_tuning_roots3072_mb128_116step/metrics.jsonl`
- `runs/experiments/nlhe_hu_32bb_tuning_mb128_lr1e4_116step/metrics.jsonl`

After every probe, GPUs 0 and 1 returned to their 6 MiB display-only footprint.
GPU 2 retained another user's process and was not used.
