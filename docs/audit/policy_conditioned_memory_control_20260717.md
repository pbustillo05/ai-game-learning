# Policy-conditioned CUDA memory control — implementation audit

> Historical design evidence: the policy-conditioned capacity forecaster and
> compatibility owners discussed below were retired on 2026-07-21. Current
> production uses schema-derived four-phase byte planning and realized page
> observations.

## Accounting identity

At a synchronized boundary the runtime records

$$
M_{\rm device}=T_{\rm CUDA}-F_{\rm CUDA}
=M_{\rm other}+M_{\rm active}+M_{\rm reusable}+M_{\rm nonreleasable}.
$$

`MemoryLedger` takes CUDA allocator statistics and counts unique tensor
storages reachable from named owners. Views and explicit aliases count once.
The unexplained active term is reported rather than hidden in a multiplier.

The configured CUDA limit is now the actual hard comparison. The former
`measured * 1.25` post-iteration exception was not an OOM condition: it killed
a completed update even when the allocator still had physical headroom.

## Retained storage decisions

| Storage | Lifetime and consumer | Decision |
|---|---|---|
| Trainable model | Whole run; rollout source and learner | Required |
| Adam moments and gradients | Whole run/optimizer step | Required; measured separately from parameters |
| KL reference model | Configured reference interval | Required by the selected loss |
| Frozen rollout snapshot | Snapshot through rollout only | Deleted before completion/value reduction |
| Poker state/history store | Whole run, sized to the configured maximum starts | Required by current integer-handle backend; chunking remains conditional on a measured 64 MiB saving |
| Device tree/occurrence arena | One rollout through target compaction | Released when `_build_iteration_data` returns |
| Root and working reaches | Rollout/completion | Full reaches are replaced by compact live-seat decision reaches before optimization |
| Policy log-probabilities and old values | Rollout through all learner reuse | Required by clipped RB PPO and value clipping |
| Completion batches/frontier | Completion/value reduction | Streamed; batches are deleted after reduction |
| Parent/child posterior | Learner reuse | In sequential mode `child_p_hat` aliases `parent_p_hat` after equality validation |
| Advantage and value targets | Learner reuse | Required by `rb_ppo` |
| Compact decision reaches | Learner reuse | Required model input; folded full rows are omitted |
| Decision weights | Learner reuse | Required by family-stratified decision sampling and global objective mass |
| Iteration `AliasTable` | None in the stratified RB path | Removed; per-family sampling builds only its temporary table |

Policy fragments still use list accumulation followed by concatenation. CUDA
memory snapshots must show at least 64 MiB of avoidable overlap before this is
replaced with a more complex chunked store. Scientific fp32 posterior, reach,
old-policy, and value tensors are unchanged.

## Online forecast

The offline experiment is only a prior. `PolicyMemoryForecaster` keeps the
most recent 128 rank-max iteration observations with a 32-iteration half-life.
Inputs include realized nodes/decisions/targets plus inference and training
`pre`/`post_L` rows, padded geometry, queue pressure, and forced drains.

The fitted center is a nonnegative least-squares model of the incremental
allocator peak above the quiescent iteration-start reservation. Using the end
reservation here would erase the workload signal because PyTorch normally
keeps peak-sized cached bins. A one-sided positive-residual bound forms the
soft envelope. Fewer than 40 observations use the maximum positive residual.
Positive surprises widen immediately; tightening requires 40 clean
observations. Eight consecutive positive residuals or excessive envelope
breaches starts a new calibration epoch, freezes growth, and backs starts off.

Decision observations stage a proposal before learning. The successful
iteration's rank-max memory observation commits it. Thus a failed iteration
cannot checkpoint a next-start count that was never validated.

## Recovery boundary

The only full-rollout retry is before the first optimizer step. If the rolling
model cannot certify the optimization phase, all ranks discard their data,
reduce starts by 20%, and draw an independently addressed retry. No trajectory
is selected or removed according to its realized length or postflop content.
The pre-optimizer ceiling is recomputed from current driver-free memory and
current PyTorch reservation, so memory acquired by another CUDA context
contracts admissible headroom before the learner allocation; rank MIN is used
for this ceiling while the predicted demand uses rank MAX.

Completed iterations above the old synthetic margin are retained. Actual CUDA
OOM remains a failed invariant. Learner strata are passed through the same
family-homogeneous hardware token/token-pair planner used by rollout. If a
logical family minibatch needs multiple physical calls, each split loss is
divided by the original logical family size. This preserves objective weights
and accumulated gradients while bounding the actual model calls.

Rejected pre-optimizer rollouts update policy-conditioned workload features.
Because their learner peak is censored (it was deliberately never allocated),
they enter the fit at no less than the rejected upper prediction and therefore
cannot tighten the memory envelope.

After two independent retries, or when the configured minimum starts cannot
be reduced further, all ranks capture RNG state and rank 0 writes an atomic
`ckpt_memory_stop_*.pt` plus `memory_stop.json`. Training then exits through
normal distributed teardown instead of provoking an allocation or losing the
last completed state.

## Diagnostics

`metrics.jsonl` holds common control summaries. `memory_rank_<rank>.jsonl`
holds rank-local workload and allocator data, while `memory_attempts.jsonl`
records discarded attempts. All are labeled by iteration, rank, optimizer
step, retry, and whether they entered learning. Resume trims rank diagnostics
to the same checkpoint prefix as training metrics.
