# Queued storage and lifetime contract

Status: final verified behavior. The executable authority is
`pbg.core.storage` and `pbg.core.capacity`; the final cutover candidate is
`71a92378cf89a81ed5724ba61515fa6c9f441706`.

## Allocation vocabulary

Queued execution has three data-allocation mechanisms:

- `SoASchema` is immutable physical data: ordered fields, dtype, trailing
  shape, field alignment, schema alignment, fill value, and lifetime.
- `MonotoneSlab` owns one fixed-capacity aligned CUDA backing arena plus its
  fixed lane-control tensor. Its operation is
  `reserve -> fill -> publish`.
- `ExactBlock` owns one aligned CUDA backing arena whose realized row extent
  cannot change. It is released explicitly.

`SoAField`, reservations, failure records, and owner plans are metadata for
those mechanisms; they do not provide another store. There is no page, growth,
recycling, free-list, wraparound, inferred schema, arbitrary column store, or
legacy-storage conversion.

Each schema allocates one `uint8` CUDA backing arena. Typed fields are
contiguous views at schema-computed aligned offsets. The arena's exact storage
bytes are:

$$
M_{\rm arena}
=
\operatorname{align}\left(
  \max_f\left(o_f+n_{\rm replica}C\,b_f\right),
  a_{\rm schema}
\right),
$$

where $C$ is row capacity, $b_f$ is field bytes per row, and each $o_f$ is the
next offset aligned to field $f$'s declared alignment. Payload bytes and
alignment padding are therefore separate and exact. No allocator-slack
constant is invented.

Mutable owner metadata is a fixed `[n_replica, 8]` `int64` CUDA tensor:
published tail, reserved tail, attempt high-water, lifetime high-water, retry
epoch, open begin, open end, and poison status. The capacity and replica axis
cannot change after allocation.

## Generated owners

`build_storage_plan` visits the immutable `ModeSchema` row. Every declared
field is converted to exactly one physical `SoAField`; an unknown field fails
planning. Inapplicable fields therefore cannot be allocated.

Partitioned owners have one slab per declared partition. `Leaf` is scalar-row
partitioned. `QueueMetadata` has one row per input-partition instance.
`LayerWork` has one slab per output partition and an explicit leading
two-arena field dimension. `Topology` and `TopologyChild` are recorded as the
two deferred compact CPU owners for Phase 3; they are not CUDA allocations.
Every other owner declared by each of the twelve mode rows is a generated CUDA
owner.

The opaque `PhysicalLayout` supplies only fixed game-side shapes and calibrated
workspace bytes per bounded row. It must cover the input partitions exactly.
It cannot add, remove, or rename a `ModeSchema` field.

## Reservation and publication

One lane may have at most one open reservation in a slab. A positive extent
receives `[epoch, begin, end)`, advances only `reserved_tail`, and updates its
high-water values. Fill requires the schema's complete field tuple with exact
shape, dtype, and device. Only publish advances `published_tail`.

Published addresses are monotone and never wrap. A failed attempt is
discarded across every live owner and increments the lane epoch before retry,
so `(epoch, row)` addresses do not repeat. Lifetime high-water survives retry.

`PublicationTransaction` admits and preflights every destination before
reserving any. All destinations share one lane and one `DeviceFailureState`.
It validates and fills every destination before enqueueing publication on the
commit stream. A consumer synchronized after that commit boundary sees every
destination; an exception before the boundary exposes none. Unexpected
publication failure rolls visible tails back and poisons the lane.

Capacity overflow advances no tail. Each lane's fixed CUDA failure record has:

```text
status, phase, failure_code, input_type, logical_address,
requested_extent, available_extent, sampler_attempt_count
```

Batched reports deterministically retain the minimum logical address per lane.
The failure path writes the preallocated record and does not allocate another
CUDA tensor.

Whole-attempt retry is two-pass. Every live owner first proves that the failed
lane has no open reservation; only then are all of that lane's rows and tails
discarded. Other lanes, including completed lanes, are untouched. The shared
failure record is cleared only after all owners reset.

## Immutable capacity admission

`CapacityManifest` contains exactly the eighteen Phase 1 capacity names:

- player-node, trained-player, action, and folded-range rows by partition;
- leaf, topology-node, and topology-edge rows;
- family, chance, completion, learner-sigma, learner-action, and topology
  packet bounds;
- two target arenas, movement scratch, library workspace, evaluator workspace,
  and memory guard.

The manifest also fixes `root_count`, `n_replica`, `beta_max`, legal game
geometry, hardware memory/token/query limits, and versioned lower bounds for
external workspaces.

Planning occurs before owner allocation and rejects:

- player count outside `[2,6]`, stack outside `[8,128]` or off the `0.5` bb
  lattice, rank count outside `[4,13]`, or failed deck fit
  $2P+5<4R$;
- capacity that does not cover the exact
  $\lceil SP+3P-S+1\rceil$ distinguished-path bound for every root;
- partition keys that differ from the game contract, a non-scalar sampled
  output, or RB output without an explicit endpoint conversion;
- trained/action/leaf/topology/folded/target relationships that cannot cover
  their declared geometry and `beta_max`;
- a pack larger than its backing row capacity;
- model token or token-pair demand beyond the hardware profile;
- an external workspace below its versioned requirement;
- a hardware profile larger than the selected CUDA device; or
- any phase envelope larger than the immutable budget or currently free device
  memory.

Caps do not change after rejection or overflow. Hardware values calibrate
capacity, shape, and launch bounds; they do not select an algorithm or sampling
law.

## Lifetimes and phase admission

`GeneratedOwners` is the sole aggregate lifetime owner. A phase allocates all
new destinations as one rollback-protected operation. Advancing to the next
phase is forbidden until every owner whose last consumer just ran has been
explicitly released.

| Owner | Allocation phase | Last consumer |
|---|---|---|
| `FamilyWork` | rollout | align-and-seal |
| `SigmaHead`, `ActionHead` | rollout | final backward |
| `Leaf` | rollout | completion |
| `FoldedRanges` | rollout | align-and-seal |
| `QueueMetadata` | rollout | queue drain |
| `LayerWork` | completion | target publication |
| `LearnerIndex` | align-and-seal | final backward |
| `SigmaTarget`, `ActionTarget` | targets | final backward |

The failure record lives for the complete owner set. `close` succeeds only
after every generated slab was released exactly once at its declared boundary.

For each phase $\phi$, admission checks:

$$
M_{\rm owners,\phi}
+M_{\rm bounded,\phi}
+M_{\rm external,\phi}
+M_{\rm guard}
\le M_{\rm budget}.
$$

The memory report reconciles the configured budget, schema-analytical live
bytes, exact named CUDA storage bytes, payload high-water, bounded workspace,
and CUDA allocated/reserved current and peak observations. Named allocation
must equal analytical allocation exactly; allocator observations may be
larger, but never smaller.

## Phase boundary

Phase 2 does not implement topology construction, queues, model execution,
sampling, target propagation, learner sampling, or optimization. It supplies
their only storage, transaction, failure, capacity, and lifetime mechanisms.
Later phases may add kernels and consumers but may not add a second allocator,
capacity authority, failure owner, or release path.
