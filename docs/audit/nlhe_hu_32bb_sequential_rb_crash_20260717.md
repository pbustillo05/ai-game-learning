# NLHE HU 32bb Sequential RB Crash Audit — 2026-07-17

> Historical incident evidence: the allocation and ownership path diagnosed
> below was retired by the 2026-07-21 storage-lifecycle restoration. Retain the
> measurements as motivation, not as a description of current production.

## Executive summary

The run did not fail from an out-of-memory allocation exception, NCCL failure, or
GPU process leak. Rank 1 completed iteration 21, measured its CUDA reserved-memory
peak, and intentionally raised the framework's `MemoryError` calibration guard.
`torchrun` then sent `SIGTERM` to rank 0. The run wrapper consequently reported
rank 0 as `-15`, although rank 1 was the root failure.

The adaptive sequential-start controller was active, but it controlled decision
throughput rather than memory. By the time of the crash it was clamped at its
configured maximum of 2,560 starts per GPU. The failing rollout had a relatively
larger postflop realization, especially on the rank with the highest workload.

## Run and failure facts

- Host: `nova`
- Launch: two `torchrun` ranks on GPUs 0 and 1
- Resume source: `ckpt_0000010.pt`
- Latest completed checkpoint: `ckpt_0000020.pt`
- Last completed metric: iteration 21
- Rank 1 failure time: `2026-07-17 10:09:41`
- Rank 0 termination time: `2026-07-17 10:09:42`
- Rank 1 exception:

  ```text
  MemoryError: measured CUDA peak plus 25% topology/allocator safety margin
  exceeds the configured limit: measured=9399435264,
  safe_limit=10779246592
  ```

- Rank 0 result: `SIGTERM` (`exitcode -15`), sent by torchrun after rank 1 exited
- Final GPU state: approximately 6 MiB used on GPUs 0 and 1; no lingering job

The hardware profile exposed 11,346,575,360 bytes of total VRAM per card. Its
0.5 GiB reserve and 0.95 memory fraction produced a calibration safe limit of
10,779,246,592 bytes. A measured peak could therefore be at most

```text
floor(10,779,246,592 / 1.25) = 8,623,397,273 bytes
```

before the 25% calibration margin would reject it.

Rank 1 measured 9,399,435,264 bytes, exceeding that raw threshold by
776,037,991 bytes. After the margin, its projected usage was
11,749,294,080 bytes, or 970,047,488 bytes above the configured limit.

## What the adaptive controller did

The controller targeted 8,000 rank-max decision nodes per GPU. After each
rollout it computed:

```text
raw starts = round(8,000 / EWMA(decisions per start))
```

It then applied minimum, maximum, and rate-of-change bounds. The relevant
sequence was:

| Iteration | Starts/GPU used | Decision/start EWMA | Raw request | Next starts | Clamp |
|---:|---:|---:|---:|---:|---|
| 0 | 2,048 | 3.1357 | 2,551 | 2,551 | none |
| 1 | 2,551 | 3.1431 | 2,545 | 2,545 | none |
| 6 | 2,551 | 3.1241 | 2,561 | 2,560 | maximum |
| 17 | 2,539 | 3.1079 | 2,574 | 2,560 | maximum |
| 18 | 2,560 | 3.0789 | 2,598 | 2,560 | maximum |
| 19 | 2,560 | 3.0417 | 2,630 | 2,560 | maximum |
| 20 | 2,560 | 3.0091 | 2,659 | 2,560 | maximum |
| 21 | 2,560 | 3.0016 | 2,665 | 2,560 | maximum |

Thus, when the run approached failure, the controller was already using the
maximum allowed starts. Its decreasing decision/start EWMA caused it to request
more starts, not fewer, but the request was clipped to 2,560. The controller did
not consume the CUDA peak as an input, and the memory validation occurred only
after the current iteration had completed. Rank 1 therefore failed before any
next-iteration start count could be applied.

## Decision-node realization

Street-level metrics were emitted from iteration 10 onward. `rank max` is the
larger count on either GPU; `2-GPU total` is the sum across both GPUs. Postflop
means flop plus turn plus river.

| Iter | Starts/GPU | Preflop max | Postflop max | Flop / Turn / River max | Preflop total | Postflop total |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2,519 | 4,884 | 3,187 | 2,244 / 726 / 217 | 9,703 | 6,325 |
| 11 | 2,516 | 4,798 | 3,121 | 2,355 / 624 / 142 | 9,485 | 6,180 |
| 12 | 2,525 | 4,619 | 3,250 | 2,440 / 641 / 169 | 9,230 | 6,417 |
| 13 | 2,535 | 4,606 | 3,348 | 2,484 / 675 / 189 | 9,079 | 6,558 |
| 14 | 2,546 | 4,500 | 3,402 | 2,492 / 709 / 201 | 8,981 | 6,739 |
| 15 | 2,553 | 4,479 | 3,700 | 2,688 / 776 / 236 | 8,954 | 7,327 |
| 16 | 2,544 | 4,337 | 3,741 | 2,718 / 789 / 234 | 8,633 | 7,237 |
| 17 | 2,539 | 4,145 | 3,365 | 2,472 / 676 / 217 | 8,240 | 6,657 |
| 18 | 2,560 | 4,088 | 3,524 | 2,525 / 748 / 251 | 8,164 | 6,790 |
| 19 | 2,560 | 3,985 | 3,433 | 2,434 / 772 / 227 | 7,906 | 6,665 |
| 20 | 2,560 | 3,889 | 3,480 | 2,408 / 818 / 254 | 7,713 | 6,950 |
| 21 | 2,560 | 3,869 | 3,768 | 2,457 / 955 / 356 | 7,681 | 7,313 |

At the failing iteration, the two-GPU total was 14,994 decision nodes:
7,681 preflop and 7,313 postflop. Relative to iteration 20, total decisions
rose only from 14,663 to 14,994 (+0.24%), while postflop decisions rose from
6,950 to 7,313 (+5.2%). The rank-max postflop count rose from 3,480 to 3,768
(+8.3%).

## Memory evidence

The iteration-21 increase was visible in every major transient phase:

| Phase | Iteration 20 allocated peak | Iteration 21 allocated peak | Change |
|---|---:|---:|---:|
| Rollout | 5,121,457,152 | 5,325,796,864 | +204 MiB |
| Completion | 2,674,322,432 | 2,881,657,344 | +198 MiB |
| Backward | 2,368,419,328 | 2,573,902,848 | +196 MiB |
| Optimization | 7,043,638,784 | 7,244,028,928 | +191 MiB |

The total iteration reserved peak rose from 8,201,961,472 to 8,403,288,064
bytes (+201,326,592 bytes). Padded model work also increased:

- Total padded token pairs: `100,153,351,233 → 104,534,280,250` (+4.4%)
- Postflop training padded token pairs: `33,292,963,264 → 35,060,673,500` (+5.3%)
- Retained target bytes: `273,927,837 → 274,596,951` (+0.64 MiB)

The small retained-target increase cannot account for the roughly 200 MiB
phase increases. The evidence points to transient rollout, completion, and
model forward/backward workspace associated with the larger, longer-history
postflop workload. CUDA allocator reservation then preserved the higher
high-water mark used by the calibration guard.

## Conclusion

The run crashed because rank 1's iteration-21 transient CUDA workload produced
a reserved-memory peak of 9.399 GB, whose mandatory 25% safety-adjusted value
exceeded the profile's 10.779 GB safe limit. The decision-based adaptive
controller was functioning as designed for its throughput target, but it was
at its 2,560-start ceiling, had no memory feedback, and could not react before
the post-iteration calibration gate rejected rank 1. Rank 0's SIGTERM was a
secondary distributed-process shutdown, not the cause.
