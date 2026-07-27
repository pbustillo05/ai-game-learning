# Queued rollout transactions

Status: Phase 4 executable contract, validated and audited at
`de6b429ce5c174df1af21e5709bfd9d51e007429`.

## Scope

`pbg.queued.rollout` owns the game-opaque queued path:

```text
stable FamilyWork prefix
  -> model workspace
  -> logical sampling workspace
  -> transition descriptors
  -> core-validated counts, scans, reservations, and atomic publication
```

The hook owns game state interpretation, model input construction, sampling,
and transition descriptors. It does not own FIFO consumption, destination
counts, retained publication, endpoint reservation resolution, retry epochs,
or topology acceptance.

The same transaction serves sequential rows and branched fan-out. Compatible
fixed and KKT lanes may share a physical cohort, but `ScheduledFamily`,
`SampleRequest`, and `TransitionRequest` retain the immutable schema of every
lane. Inactive lanes are absent rather than padded.

## Semantic FIFOs and scheduling

Each declared input partition has one `SemanticFamilyFIFO` over:

```text
FamilyWork[input_type]
QueueMetadata[input_type] = {head, ready_tail, reserved_tail, ...}
```

The committed prefix is `[head, ready_tail)`. Rows never wrap, recycle, or
move. Scheduling is stable: full packs precede partial packs, input preorder
precedes load and age, and a selected cohort uses the largest common prefix
without inventing inactive lanes. Cross-family publication is legal only for
an input family in the same `PartitionContract`; for example, a `pre` source
may atomically publish a `post_A` destination FIFO.

Every `FamilyWork` destination in a commit plan must name exactly one published
queue. Published queues must be unique and must match their destination owner
and input type. This keeps queue metadata an effect of the generic commit,
never a hook-managed side effect.

## Core-owned routing and coverage

A hook returns fixed-shape `CommitCoverage` descriptors. Each valid descriptor
contains:

```text
source_ordinal
destination_request
destination_ordinal
```

The core proves on CUDA that:

- every scheduled source row occurs at least once;
- inactive lanes have no descriptor;
- every realized destination row occurs exactly once;
- destination ordinals form a complete stable prefix;
- source order is nondecreasing within each destination FIFO;
- no descriptor names an undeclared destination.

Destination counts are derived from these descriptors. Hook-supplied counts
are not used for reservation. The derived counts pass through the device
exclusive scan and into the sole public `PublicationTransaction` operation.
Coverage width and temporary hit/scan bytes are checked against bounds derived
from the generated owner set and the rollout-phase workspace admission before
temporary tensors are allocated.

## Atomic `model -> sample -> commit`

`RolloutTransaction` uses distinct model/sample and commit streams.

1. The model stream gathers the stable prefix and produces raw workspace.
2. Logical RNG keys use the scheduled logical root/path, lane, purpose, draw,
   iteration, seed, and the retry epoch read from the complete owner registry.
3. The sample and transition hooks run under retained-owner and failure-record
   version guards.
4. The core validates coverage, owner membership, partitions, queue effects,
   topology descriptors, and manifest bounds.
5. The commit stream waits for model/sample and topology-validation events,
   reserves every destination with device counts, fills fixed-shape prefixes,
   updates queue/source metadata, and publishes one marker.

An exception before publication aborts reservations. An exception after
publication rolls every destination and queue/source metadata back, poisons
the affected destination lanes, and records `transaction_rollback` in the
shared `DeviceFailureState`. `GeneratedOwners.abort_failed_lane()` can then
reset every live slab to the same incremented epoch. Internal/order failures
are distinct from recoverable `capacity_overflow`.

Only `RolloutOwnerRegistry.from_generated()` is a production construction
path. It binds the complete live owner set and immutable manifest. The
`for_testing()` path exists only for closed synthetic fixtures.

## Logical RNG

The logical RNG version is `counter_hash_v3`. A key is a pure function of:

$$
(\text{seed},\text{iteration},\text{root},\text{path},
\text{draw},\text{purpose},\text{lane},\text{retry epoch}).
$$

Placement, subdivision, stream choice, and compatible lane grouping do not
change it. Inactive addresses receive sentinel zero and consume no draw. Child
paths use positive one-based child ordinals, so duplicate action labels remain
distinct logical children.

Retry epochs come from the generated retained owners. All live slabs must
agree on the lane epoch; a divergent epoch is terminal. Chance closure uses
the same authoritative epoch rather than a caller-supplied counter.

## Asynchronous topology acceptance

Topology is replica-local. Drafts and committed packets are keyed by
`(lane, sequence)`, so equal sequence numbers in different lanes are valid and
replica identity is retained in `ReplicaTopologyPacket`.

A draft has two CUDA events:

- `validation_event`: the commit stream waits on this before checking the
  device validation marker and endpoint ordinal validity;
- `ready_event`: CPU acceptance waits on this asynchronous pinned D2H
  topology transfer.

Endpoint patches carry reservation indices and device ordinals, not
precomputed destination rows. After reservation, the core resolves:

```text
leaf target row = lane reservation begin + endpoint ordinal
```

and proves $0 \leq \text{ordinal} < \text{realized count}$ on the ordered
commit stream. A false topology marker or invalid ordinal terminates before
staging, reservation, or publication.

Endpoint rows and commit markers use two preallocated pinned transfer slots.
Ordinary `drain()` polls events without blocking. At sealing,
`drain(block=True)` waits for every validation/transfer dependency on one
acceptance stream and performs one aggregate synchronization. Packet, node,
edge, in-flight, and pinned-transfer extents are immutable bounds; committed
host storage cannot exceed cumulative topology capacities.

## Bounded chance and completion

`BoundedChanceClosure` has a fixed CUDA frontier and a geometry-derived
generation limit. It launches every declared generation without a host
synchronization, guards retained owners and failure records around both hooks,
uses authoritative retry epochs, and terminates if active rows remain after
the bound.

`CompletionFIFO` consumes stable `Leaf` prefixes only after a required CUDA
readiness event. Its cursor observes generated-owner retry epochs and resets
automatically after a failed attempt. Completion coverage uses the same exact
source/destination proof and the completion-phase workspace admission.
Destinations must be exactly `LayerWork.<request.input_type>`; unrelated
families, metadata, leaves, or heads are rejected before reservation. The
bounded cursor workspace has explicit release.

## Acceptance

The final candidate passed:

- 20 focused Phase 4 CUDA tests;
- 91 cumulative Phase 1–4 CUDA tests;
- delayed-producer and permanent-failure topology ordering probes;
- exact stable routing, inactive RNG, collision-at-cap, extreme overflow,
  rollback/reset, cross-family fan-out, chance exhaustion, and completion
  ownership probes;
- 1/2/4/8-pack profiler gates with zero D2H, H2D, `aten::nonzero`, host scalar
  reads, or stream synchronizations in the measured hot path.

The independent executable audit found no remaining critical or high-severity
issue. Full evidence is in
`docs/audit/queued_phase4_report_20260727.md`.
