# Storage, queue, and distributed fidelity

## Verdict

The production data plane does not yet satisfy the pseudocode authority. The
central failure is architectural rather than cosmetic: an oversized legacy
handoff is sealed before normative compact storage, while significant queues,
summaries, completion, and conversion overlaps lie outside a complete hard
ledger.

## Blockers

**C01 — direct seal.** `PagedTreeSource.finalize` gathers full node/edge columns
and constructs `SampledTree` (`pbg/core/tree.py:555-587`). Sequential and
branched sealing then allocate `SequentialDecisionCSR` or `BranchedTreeSpan`
and whole-tree reorder maps (`pbg/core/training.py:1725-1826`). This violates
`STORE-012/013/014/021/030`. `SampledTree`, `PagedTreeSource`,
`RolloutDecisionRows`, and `StoppedLeafRows` are compatibility-reachable stale
owners: migrate to direct compact sealing, then delete them.

**C02 — distributed failure agreement.** Mandatory rank-local feasibility and
execution failures can occur before a common terminal decision while peers
enter mass/gradient collectives. Reorder mandatory feasibility before shared
collectives and add asymmetric NCCL failure injection (`AUD-TL-001`).

## High-severity structural drift

- C05: schemas are inferred on first write or duplicated by hand; preflop
  model rows retain exact width; logical addresses are reserved after model
  work/draws. Wire `occurrence_schema` as the sole immutable schema.
- C06: `pbg/core/branching.py:551-610` materializes all logical children in one
  wave. At full-deck $H=1326$, `child_reach` plus `next_reach` cost about
  21,216 bytes/child heads-up or 63,648 bytes/child six-max; 100,000 children
  are about 1.98 GiB or 5.93 GiB before other tensors. Logical addresses/RNG
  must be fixed first, then execution streamed through bounded physical slabs.
- C07: scheduler order differs from the declared two-level policy;
  `DeviceWorkQueue` clones/concatenates without `ByteAccountant`
  (`pbg/core/execution.py:596-684`); fixed mode and KKT warmup lack the common
  pressure safeguard because suppression is gated by `dynamic_active`
  (`pbg/core/branching.py:474-481`). A diagnostic cutoff that changes the
  sampling law is not an acceptable fix.
- C09: forbidden neutral `post_P`, per-pack folded-summary duplicates, and
  mandatory folded cache bytes omitted when `evaluator_summary_dim == 0`
  violate sparse-anchor ownership.

## What is preserved

Page growth uses grouped rollback before random draws/state mutation for the
owners included in the transaction (`pbg/core/branching.py:351-380`). Phase
owners generally have explicit `finally`/last-consumer release. Learner inputs
are sealed and partitioned; direct flat-gradient accumulation and one SUM are
retained. Tiny CUDA queue/page/RNG and two-GPU exact-mass/SUM tests passed.

Those facts do not establish complete memory closure. Required structural
gates must prove direct compact sealing, source+destination admission, queue
and folded-anchor accounting, bounded child peaks independent of logical work,
all-occurrence lifetime, physical-plan invariance, and common-order failure.
