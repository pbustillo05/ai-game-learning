# Workload history and N/B/S screen

This note records a read-only workload audit for the 2026-09-28 campaign. The
symbols below mean roots per rank $N$, learner batch per rank $B$, and
synchronized optimizer steps per rollout iteration $S$. The current campaign
uses three ranks and the experiment-local `global_mean` gradient wrapper, so
global batch and root counts are three times the configured per-rank values.

## Historical comparison

The successful July run that produced checkpoint 1110 reached 3,072 roots per
rank after guarded growth, hence 6,144 global roots. It used $B=128$ per rank
(256 global) and $S=116$. That is 14,848 learner presentations per rank and
29,696 global presentations per iteration. At the initial 2,560 roots per
rank, the saved metrics show 14,848 presentations over 7,222 retained fresh
decision rows on rank 0, about 2.06x reuse. At the mature 3,072-root width,
the historical timing audit reports about 1.7x reuse; the exact value moves
with the retained row mix.

The failed `ab_rb_day_20260722` continuation was launched with
`--nproc-per-node=1`, despite using a three-GPU hardware profile. Its effective
values were therefore $N=512$ global/local, $B=128$ global/local, and $S=16$:
2,048 learner presentations per iteration. The first two metrics records show
2,048 presentations over 2,566 and 2,523 retained fresh decisions, or about
0.80x and 0.81x. Its Rao–Blackwell completion used 256 samples per leaf with
completion Kish ESS near 256; that is estimator precision, not fresh training
data.

| run | world | $N$ local/global | $B$ local/global | $S$ | global presentations | observed reuse |
|---|---:|---:|---:|---:|---:|---:|
| July checkpoint-1110 run, mature | 2 | 3,072 / 6,144 | 128 / 256 | 116 | 29,696 | about 1.7x |
| July checkpoint-1110 run, initial | 2 | 2,560 / 5,120 | 128 / 256 | 116 | 29,696 | about 2.06x on first record |
| Failed `ab_rb_day_20260722` | 1 | 512 / 512 | 128 / 128 | 16 | 2,048 | 0.80–0.81x |
| Current three-GPU control | 3 | 512 / 1,536 | 128 / 384 | 24 | 9,216 | measure from retained rows |

The failed run is confounded by the pre-`0a09496` completion target
permutation. Its result cannot establish that $S=16$, its smaller rollout, or
its lower presentation budget caused the quality failure. Increasing $S$
reuses the current rollout storage and adds optimizer updates; it does not
create independent public data. Increasing $B$ generally lowers per-update
sampling noise, while changing $N$ changes fresh public exposure and the
resulting reuse ratio. Completion ESS and posterior breadth must remain
separate counters.

Sources: [campaign history audit](../hyperparameter_search_20260928/history.md),
[distributed semantics](../hyperparameter_search_20260928/distributed_semantics.md),
[successful July metrics](../../../runs/nlhe_hu_32bb_full_sequential_rb/metrics.jsonl),
[failed-run command](../../../runs/experiments/nlhe_hu_32bb_ab_rb_day_20260722/command_20260722T080832.txt),
[failed-run config](../../../runs/experiments/nlhe_hu_32bb_ab_rb_day_20260722/config.yaml),
and [failed-run metrics](../../../runs/experiments/nlhe_hu_32bb_ab_rb_day_20260722/metrics.jsonl).

## Selected three-GPU joint screen

The campaign should retain the upper fresh-data signal requested by the user:
$N\in\{512,1024\}$ roots per rank, $B\in\{128,256\}$ learner rows per
rank, and $S\in\{24,48\}$ optimizer steps. This is the eight-cell factorial;
the earlier 256/512-root proposal was a conservative fallback and is not the
selected design.

| $N$/rank | $B$/rank | $S$ | global roots | global batch | global presentations/iteration |
|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 24 | 1,536 | 384 | 9,216 |
| 512 | 128 | 48 | 1,536 | 384 | 18,432 |
| 512 | 256 | 24 | 1,536 | 768 | 18,432 |
| 512 | 256 | 48 | 1,536 | 768 | 36,864 |
| 1,024 | 128 | 24 | 3,072 | 384 | 9,216 |
| 1,024 | 128 | 48 | 3,072 | 384 | 18,432 |
| 1,024 | 256 | 24 | 3,072 | 768 | 18,432 |
| 1,024 | 256 | 48 | 3,072 | 768 | 36,864 |

At fixed $B,S$, the $N$ contrast measures fresh rollout exposure versus reuse.
At fixed $S$, the $B$ contrast tests gradient-batch noise together with its
presentation budget. At fixed $B$, the $S$ contrast tests optimizer-update
count together with additional reuse. The matched-presentation pairs
$(B,S)=(128,48)$ and $(256,24)$ both produce 18,432 global presentations;
they are the cleanest batch-versus-update comparison. Report actual retained
rows and `presentations / retained fresh decisions`, rather than inferring
reuse from roots alone.

Run each cell for the same wall-clock screen and compare at fixed elapsed
seconds. Keep initialization, LR $5\times10^{-5}$, loss coefficients,
reference policy, seeds, estimator, completion samples, precision, and
microbatch limits fixed. Do not LR-scale when changing $S$. Record fresh
decisions, retained rows, presentations, optimizer steps, reference copies,
throughput, and progress per wall time; use per-fresh-decision and
per-optimizer-step plots only as secondary views.

The two bridge settings $N=2048,B=128,S=96$ and $N=2048,B=256,S=48$
are now registered following full-iteration CUDA capacity checks. They give
6,144 global roots on three ranks. Both modes share physical learner pack 6
for these bridges; the eight smaller cells use pack 12. This is a measured
execution-capacity adjustment, not a change to the logical statistical batch
or optimizer-step count. The reference-cache budget is 1 GiB throughout.
See [VALIDATION.md](VALIDATION.md) for the failed pack-12 RB probe, the passing
pack-6 probes, and the measured cache throughput gain. These measurements do
not establish stability for every future policy or seed.
