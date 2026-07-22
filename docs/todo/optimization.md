# Performance and optimization program

Status: measured 2026-07-22. This replaces the 2026-07-21 list. It is an
experiment backlog, not authority for training semantics or storage lifetime;
`docs/theory/` and its pseudocode remain authoritative.

The optimization objective is not merely the shortest iteration. At fixed
statistical work and an explicit memory envelope, maximize:

1. fresh player decision nodes generated per second;
2. useful learner rows and backward passes per second; and
3. end-to-end fresh player nodes per second at the intended rollout-to-learner
   reuse ratio.

For sequential training, compare implementations at approximately one fresh
rollout decision to two learner presentations. For branching, benchmark both
1:1 and 2:1 rollout-to-learner ratios. A run that becomes faster only by doing
less learner work is not an implementation speedup.

## Executive diagnosis

The largest opportunity is the rollout hot path. The current learner is about
10% faster per optimizer step than the July 17 historical record, but the
matched two-GPU iteration spends 2.57 times as long in rollout. Git bisection
and a controlled intervention identify the largest known cause: commit
`a399e30` replaced direct structure-of-arrays state indexing with a generic
paged-column facade on every hot poker-state field. On the same commit, seeds,
492 decisions, 24 model calls, and 2.040 billion padded attention pairs,
replacing only that facade with the preceding flat state implementation reduced
rollout from 3.758 to 1.961 seconds. The queue family and 64-row policy pack were
already present in the faster implementation; the queue idea is not the
regression.

The current rollout also repeats legal-action, history, policy-probability,
selected-log-probability, reach, and serialization work around every model
call. A bounded operator profile found about 790 kernel launches and 20
implicit stream synchronizations per fresh decision; attention accounted for
only about 21% of self CUDA time in that diagnostic. The page facade explains a
large part of the indexing, copy, temporary-allocation, and launch pressure;
duplicated pack work and immutable decision capture are additional costs.

The second immediate issue is configuration. Recent sustained A/B runs used
the conservative `local_3x2080ti` profile even when one card was assigned. That
profile admitted only 15 preflop and 3 postflop learner rows per physical
subspan, causing roughly 20 forward/backward subspans per optimizer step. The
10 GiB two-card profile admitted 54/10 rows in the historical-geometry run,
only two subspans per step. Memory admission must describe the card envelope,
not the number of participating ranks.

Larger rollout packs help, but only within the memory contract. Moving from 64
to 96 family rows improved a 512-root iteration by 16.6%, while 128 rows failed
the downstream workspace-closure guard. At 2,048 roots, 96 rows reserved
10.497 GB (9.78 GiB). That is below the current profile's 10.723 GB limit but
above a literal 10,000,000,000-byte hard ceiling. Until the ceiling is made
unambiguous in configuration, treat 96 rows as an experiment rather than a
production default.

## Implementation progress: dense live-state boundary

The first rollout milestone was implemented and measured on 2026-07-22.  The
accepted performance gate is now a production-scale single-GPU sequential-RB
workload with 2,048 trajectory starts, checkpoint 1110 from
`runs/nlhe_hu_32bb_full_sequential_rb`, 64 family rows, 64 active trajectories,
1,024 completion samples, and the configured 256-row append-only page size.
The earlier 128- and 512-root probes remain debugging tools and are not release
performance evidence.

The implementation makes mutable state/history a distinct physical domain.
For sequential rollout it admits one recyclable dense slab of
`roots_per_gpu + family_batch_rows` rows, the exact live-frontier high-water
bound observed and implied by the scheduler's parent/child overlap.  Logical
topology ids remain separate.  Parents still release their slots after all
successors and immutable retained rows commit, history remains refcounted, and
the memory plan charges the full slab capacity.  Append-only topology,
decision, action, and leaf ownership retain their own page geometry.

Within a resolved live slab, `PagedTensorColumn` now delegates ordinary tensor
indexing directly to the dense field.  The generic arena also has direct
single-page gather/write paths, and whole-row replacement no longer gathers
the field that it is about to overwrite.  Multi-page append-only arenas retain
their generic checked implementation.

Production-scale results are:

| 2,048-start sequential-RB measurement | Decisions | Model calls | Padded attention pairs | Rollout | End-to-end decisions/s |
|---|---:|---:|---:|---:|---:|
| Audit matched baseline, 64 rows | about 8,050 | 139.5 | not recorded in the summary | 87.45 s | 90.9 |
| Pre-fix reproduction, repeat seed 101 | 8,087 | 140 | 33.149 billion | 90.27 s | 88.3 |
| Dense live-state implementation, two-repeat median | 8,049.5 | 139.5 | 32.976 billion | 51.65 s | 152.7 |

Against the matched audit baseline, the accepted implementation reduces
rollout time by 40.9% and raises end-to-end decision throughput by about 68%.
The median peak was 3.34 GiB allocated and 6.16 GiB reserved, with a 2,112-row
state high-water and no state row crossing the seal boundary.  The history
high-water median was 719 rows and exactly the sentinel row remained at seal.

A diagnostic one-repeat run that used a global 2,112-row page for every rollout
arena reached 35.90 s for the same 8,087 decisions and 140 model calls.  That is
useful headroom evidence but is not accepted as the live-state result: a global
page changes append-only slack and abstraction boundaries.  The next rollout
work should recover that headroom with explicit bulk topology/retained writes,
then remove duplicated legal/probability/history work without changing page
ownership or statistical work.

Focused CPU gates after this milestone passed 101 tests with seven CUDA-only
skips.  CUDA benchmark teardown left no compute process resident.

### Follow-on rollout width result

The production family-width gate was rerun after the dense-state milestone on
the same live source snapshot, using the schema-migrated copy of checkpoint
1110, 2,048 starts, repeat seed 104830, 1,024 completion samples, and one warmup
per arm.  GPUs 0 and 2 ran the 64- and 88-row arms concurrently; the 84-row arm
then ran on GPU 0.

| Family rows | Decisions | Model calls | Rollout | ms/decision | Peak allocated / reserved |
|---:|---:|---:|---:|---:|---:|
| 64 | 8,233 | 141 | 36.228 s | 4.400 | 3.63 / 6.62 GB |
| 84 | 8,283 | 112 | 33.723 s | 4.071 | 4.60 / 9.05 GB |
| 88 | 8,262 | 107 | 33.118 s | 4.008 | 4.80 / 9.84 GB |

Family 84 is the production choice for the 2080 Ti profile.  It reduces
rollout time per decision by 7.5% and raises end-to-end throughput by 7.6%
relative to 64 rows while retaining about 955 MB below the decimal 10 GB
ceiling.  Family 88 is 1.5% faster than 84 but leaves only about 158 MB and is
therefore retained as an aggressive diagnostic point, not the default.  The
live-state slab forecast now covers the adaptive controller's configured
maximum root frontier plus one family pack, so controller widening cannot
silently fall back to multi-page hot state.

The first production sample of combined public/history, legal-mask, and
selected-log-probability reuse did not improve rollout (36.45 s versus 35.95 s
for identical 8,087 decisions and 33.149 billion padded attention pairs).
Keep that experiment provisional until a settled paired median demonstrates a
gain; do not credit it in the cumulative speedup.

Sequential target construction and especially branched target construction
still contain or recently contained host-driven recurrences. A concurrent
dirty-worktree change began vectorizing sequential targets while these
measurements were being collected, so target timings must be re-baselined from
a settled revision. The architectural conclusion remains: all target payload
and recurrence work belongs on CUDA, with only bounded structural summaries at
declared control boundaries.

## Measurement provenance and limitations

Measurements used RTX 2080 Ti GPUs 0 and 1; GPU 2 was left idle. Each CUDA
process was explicitly assigned a device and teardown was checked. The Python
environment was `/home/pablob/.venvs/poker`, PyTorch was 2.12.1+cu130, and the
cards have compute capability 7.5 and 11,264 MiB physical VRAM.

The working tree was based on commit `b9a42fbca95da10f7f892ef3d79033dcb22ee284`
and was dirty before diagnostics began. Other storage, target, learner, and
poker-evaluator edits appeared while the sweep was running. Consequently:

- the results are diagnostic evidence, not a release performance baseline;
- every artifact records its checkpoint, workload, seeds, and environment;
- comparisons that crossed a live source edit are identified as candidates to
  rerun, not accepted A/B conclusions; and
- no numerical loss from a newly initialized branching head is treated as a
  trained-policy result.

The benchmark harness needed one correctness repair before branching could be
measured. Mode conversion now clears every branching-only control for normal
and sequential-RB modes, preserves native branching configuration, and
requires an explicit finite queue capacity when synthesizing a branched probe
from a sequential config. The selected branching mode and queue capacity are
now part of workload provenance. Six focused benchmark-gate tests pass.

One harness defect remains: median aggregation intersects optional metric keys,
so a tiny branching repeat can erase useful strata that were present in other
samples. Raw samples are authoritative until aggregation uses a declared
metric schema and reports missing strata explicitly.

## Historical comparison

The strongest comparable historical record is the July 17 two-GPU HU 32 bb
sequential-RB run with 2,048 roots per rank, 1,024 completion samples, a
64-row minibatch, and 232 optimizer steps. The current reproduction used the
nearest loadable checkpoint rather than the exact historical model, so rollout
topology is close but not identical.

| Quantity | July 17 record | 2026-07-22 diagnostic | Change |
|---|---:|---:|---:|
| Total iteration | 93.13 s | 115.68 s | 24.2% slower |
| Rollout | 19.51 s | 50.13 s | 2.57x slower |
| Targets | 4.15 s | 2.84 s | 31.6% faster in this full run |
| Learner total | 69.43 s | 62.70 s | 9.7% faster |
| Learner time/step | 299.3 ms | 270.3 ms | 9.7% faster |
| Learner forward | 18.83 s | 16.24 s | 13.7% faster |
| Learner backward | 33.87 s | 30.62 s | 9.6% faster |
| Reference cache | 8.79 s | 7.59 s | 13.6% faster |
| Gradient synchronization | 3.20 s | 3.16 s | approximately flat |
| Peak allocated / reserved | 4.45 / 6.80 GiB | 3.15 / 5.21 GiB | lower |
| Global fresh decisions/s | 123.4 | about 87.3 | about 29% lower |

The current rank realized 5,051 fresh decisions and presented 14,848 learner
rows, a reuse factor of 2.94. That exceeds the intended sequential factor of
about 2. At fixed measured per-step cost, 158 rather than 232 steps would be
near a factor of 2, but this is a workload calibration, not an optimization
claim. Implementation comparisons must hold this ratio fixed.

There is an important target-timing discrepancy to preserve in the baseline:
an older exact 5,051-decision artifact reported 1.055 s for targets, versus
2.836 s in the current two-rank reproduction. The full July 17 report reported
4.15 s. The new vectorized target work should be accepted only after one exact
artifact and timing definition explain this difference.

Recent 60-iteration sustained runs are not directly comparable to the July 17
record because they used 512 roots, 16 learner steps, minibatch 128, and the
`local_3x2080ti` workspace envelope. Their useful diagnosis is structural:

| Mode | Decisions | Total | Rollout | Targets | Learner | Physical subspans/step |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 2,519 | 87.13 s | 29.07 s | 0.26 s | 57.79 s | about 20 |
| Sequential RB | 2,920 | 94.93 s | 34.41 s | 1.32 s | 59.18 s | about 20 |

Those learner times are dominated by an unnecessarily narrow workspace plan,
not by the statistical minibatch size itself.

### Rollout regression bisection

The family scheduler is not new. Commit `181ab70` made it the sole rollout path
on July 16, before the fast July 17 record. Git blame shows that dequeue,
64-row family formation, model forward, action sampling, poker transition,
reach update, and successor classification in the present loop mostly descend
from `181ab70` and `e9e2aff`. The large discontinuity instead begins at
`a399e30` (`refactor(core): add byte-admitted rollout storage`) on July 21.

Archived snapshots of `1cde6de` immediately before that commit and `a399e30`
were run with the same checkpoint, HU 32 bb sequential RB workload, 128 roots,
node budget 43, family rows 64, active rows 32, 16 completion samples, one
16-row learner step, one warmup, and three timed repeats. The pre-refactor
median rollout was 1.965 seconds; unmodified `a399e30` was 3.718 seconds even
though both medians made 24 model calls and nearly identical padded model work.

A stricter two-stage intervention was then made only in a temporary `a399e30`
archive:

| `a399e30` diagnostic | Decisions | Model calls | Padded attention pairs | Rollout | Total |
|---|---:|---:|---:|---:|---:|
| Paged state, new history zero-writes removed | 492 | 24 | 2,040,271,723 | 3.758 s | 4.126 s |
| Prior flat state backend, same history control | 492 | 24 | 2,040,271,723 | 1.961 s | 2.317 s |

Removing the new history initialization writes alone did not improve the
result. Replacing the paged state backend recovered the entire pre-refactor
rollout time across all three paired samples. This isolates the state access
abstraction, rather than model work, queue fill, topology, learner work, or the
new history initialization, as the cause of the approximately 1.9x step at
that commit. The present small-workload regression is smaller because later
changes recovered some time, but the same facade remains in the hot path.

## Current controlled experiments

Unless noted otherwise, these used checkpoint 1110 from
`runs/nlhe_hu_32bb_full_sequential_rb`, fixed benchmark seeds, one warmup, and
three timed repeats on one idle 2080 Ti.

### Mode and phase baseline

At 128 roots, 16 completion samples, family rows 64, active trajectories 32,
and one 16-row learner step:

| Mode | Decisions | Total | Rollout | Targets | Learner | Decisions/s | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 492 | 3.590 s | 3.291 s | 0.048 s | 0.239 s | 129.5 | 2.77 / 5.10 GiB |
| Sequential RB | 495 | 3.730 s | 3.261 s | 0.235 s | 0.212 s | 124.9 | 2.72 / 5.02 GiB |

With deliberately little learner work, rollout owns 87--92% of the iteration.
RB target work is visible, but it is much smaller than rollout at this size.

### Rollout family width and active-window sweep

The 128-root family-width sweep gave:

| Family rows | Median decisions | Rollout | Approx. rollout ms/decision | Model calls | Peak allocated / reserved |
|---:|---:|---:|---:|---:|---:|
| 16 | 494 | 6.905 s | 14.0 | 41 | 1.00 / 1.85 GiB |
| 32 | 494 | 4.591 s | 9.3 | 27 | 1.72 / 3.38 GiB |
| 64 | 495 | 3.261 s | 6.6 | 19 | 2.72 / 5.02 GiB |

The previous document's claim that 32 and 64 rows were tied is stale for the
trained checkpoint and current code. Sixty-four rows reduced rollout time per
decision by about 29% relative to 32 and by about 53% relative to 16. The gain
tracks fewer model calls and kernel-launch groups.

At 512 roots, family 64 with active windows 32 and 64 was effectively tied:
15.35 versus 15.42 s of rollout and 113.4 versus 114.5 end-to-end decisions/s.
Widening the active trajectory window alone is not useful once the family pack
is already full.

At the same 512-root geometry, moving family rows from 64 to 96 reduced model
calls from about 50 to 38, rollout from 15.42 to 12.58 s, and total time from
17.18 to 14.33 s. End-to-end throughput rose from 114.5 to 138.3 decisions/s,
a 20.8% throughput gain (16.6% less time). Median reserved memory rose from
5.57 to 8.49 GiB. Family rows 128 failed cleanly before backward because the
10 GiB plan could not close downstream completion and learner workspace.

A settled, paired production sweep then ran 64 rows on GPU 1 and 96 rows on GPU
0 simultaneously. The processes launched four seconds apart and loaded the
same source snapshot, checkpoint, and repeat seeds before later worktree edits;
GPU 2 remained idle:

| Family rows | Decisions | Model calls | Rollout | Total | Decisions/s | Peak allocated / reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8,050 | 139.5 | 87.45 s | 88.55 s | 90.9 | 3.34 / 6.16 GiB |
| 96 | 8,071 | 98.0 | 70.49 s | 71.59 s | 112.7 | 4.79 / 9.78 GiB |

Ninety-six rows cut rollout time by 19.4%, total time by 19.2%, and increased
end-to-end throughput by 24.0%. The two raw repeat pairs showed the same large
gap. It reserved 10,497,294,336 bytes and allocated 5,145,100,288 bytes at
median peak. This validates the throughput gain, but the point nearly consumes
the current profile and violates a decimal 10 GB limit. The next sweep should
find the largest safe intermediate width, likely in the 80--88 range, rather
than adopting 96 as the 2080 Ti default.

### Branching diagnostic

No loadable checkpoint in the inspected run set has trained current T/C heads.
An older six-player checkpoint predates the present branching heads and compact
postflop schema. The only current branch measurement therefore used the trained
sequential trunk plus newly initialized T/C heads, fixed beta 2.3, a 4,096-row
queue, and a 1,000-node duplication cutoff.

One representative sample produced 2,128 nodes and 1,346 decisions in 15.73 s:

| Phase | Seconds | Share |
|---|---:|---:|
| Rollout | 14.029 | 89.2% |
| Targets/completion | 1.458 | 9.3% |
| Learner | 0.228 | 1.5% |
| Seal | 0.012 | 0.1% |

About 97.5% of its decisions were postflop and 75.5% were on the river. Another
repeat collapsed to five nodes because the branching heads are untrained. This
is useful evidence for phase costs and late-street geometry, not a stable
throughput baseline. A trained T/C checkpoint or fixed replayable topology is
required before accepting branch optimizations.

## Time and memory anatomy

For the current 232-step two-GPU reproduction, the phase shares were:

| Phase/component | Seconds | Share of iteration | Optimization ceiling |
|---|---:|---:|---|
| Rollout | 50.13 | 43.3% | largest end-to-end regression |
| Targets | 2.84 | 2.5% | host recurrence; larger branch share |
| Learner total | 62.70 | 54.2% | large absolute budget |
| └ backward | 30.62 | 26.5% | graph/fusion candidate |
| └ forward | 16.24 | 14.0% | graph/fusion candidate |
| └ reference cache | 7.59 | 6.6% | compute only consumed policy fields |
| └ gradient sync | 3.16 | 2.7% | lower ROI than July estimate |
| └ loss | 2.28 | 2.0% | include in learner graph |
| └ optimizer | 2.10 | 1.8% | leave outside first graph |
| Seal | 0.013 | negligible | not a target |

Manually declared scheduler synchronization was only 198 calls and 4.4 ms in
the full iteration. This does not mean the host is irrelevant. It means queue
control traffic is cheap. Python kernel issuing and implicit synchronization
from dynamic-shape CUDA operations remain expensive and are not counted by the
manual scheduler metric.

Memory is phase-shaped rather than cumulative. In the full 96-row probe:

| Boundary | Allocated | Reserved | Interpretation |
|---|---:|---:|---|
| Iteration start | 0.23 GiB | 2.87 GiB | model/optimizer/reference plus allocator cache |
| Rollout peak | 4.79 GiB | 9.78 GiB | actual iteration peak and limiting phase |
| Pre-optimization live | 0.40 GiB | 9.78 GiB | most reservation reusable after rollout release |
| Learner peak | 2.64 GiB | 2.88 GiB | much lower after cache release |
| Iteration end | 0.40 GiB | 2.88 GiB | no retained rollout owner |

The 4.99 GiB gap between peak allocated and reserved is allocator-held slack,
not live tensors. Admission nevertheless has to respect reserved memory plus
non-Torch occupancy on an 11,264 MiB card. Do not raise a global capacity based
on low learner live memory; preserve last-consumer phase closure and separately
bound rollout, completion, sealing overlap, and learner transients.

Define two hardware envelopes:

- `2080ti_10gb`: conservatively cap Torch peak reservation at
  10,000,000,000 bytes until the user-facing "10 GB" ceiling is explicitly
  defined otherwise, retain a non-Torch/driver guard, and fail before work;
- `h100_throughput`: admit substantially wider physical rows from measured H100
  transient formulas, but retain finite schemas and phase-local closure even
  when memory is plentiful.

## Bounded operator profile

A one-iteration, 32-root sequential-RB profile was intentionally bounded and
disabled shape, stack, and memory recording. Even so it generated about
615,000 events for only 137 fresh decisions, so its absolute wall time is
profiler-distorted. Event counts and self-CUDA proportions are still useful:

| Event | Calls | Diagnostic meaning |
|---|---:|---|
| `cudaLaunchKernel` | 108,254 | about 790 launches per decision |
| `cudaStreamSynchronize` | 2,800 | about 20 implicit syncs per decision |
| `aten::empty` | 23,382 | severe transient-allocation pressure |
| `aten::copy_` | 14,190 | 12.3% of self CUDA time |
| `aten::index` | 7,624 | repeated gather/scatter work |
| `aten::cat` | 748 | 7.8% of self CUDA time |
| async assertions | 6,532 | repeated hot-path validation kernels |
| D2D async copies | 5,528 | device staging/copy pressure |
| efficient-attention forward | 90 | only 21.1% of self CUDA time |

The result argues against treating the model FLOPs as the whole rollout cost.
A full profiler trace is unsafe: earlier complete traces consumed 16--19 GiB
host RAM and minutes constructing event trees. Use CUDA-event component buckets,
Nsight on short scheduled windows, or narrow profiler schedules.

## Optimization leaks in the current design

### Hot state paging is at the wrong abstraction boundary

The byte-admitted storage objective is correct, but `PagedTensorColumn` makes
the page abstraction visible at every ordinary tensor expression. A field read
does not perform the former `column[rows]`. It allocates a zero output, computes
page ids and offsets, gathers a candidate from every allocated page, and merges
each candidate with `torch.where`. A field write first gathers the selected
field into a temporary, mutates it, then calls `write`; `write` constructs a
`[selected rows, page width]` match matrix for every page and copies a whole
page-width field through another `where`. Poker encoding, legality, transition,
classification, history, and state initialization each touch many fields, so
one 64-row dequeue causes these page scans repeatedly.

This is why processing the same queue width no longer launches the same code in
the meaningful sense. The outer policy-forward and poker functions have the
same names and much of the same source, but their apparently tensor-like state
loads and stores expanded into many allocation, division, remainder, clamp,
comparison, `where`, argmax, and copy kernels. The cost also scales with the
number of admitted pages, not just the 64 useful rows.

Keep paging and byte admission at an ownership boundary, but remove the
per-field facade from compute. The first experiment should add one explicit
bulk `read_state_pack(handles)` that materializes all required fields once into
a dense fixed-width `PokerStateBatch`, followed by one bulk successor write
grouped by destination slab. The stronger design is to store queued live states
directly in admitted, recyclable dense queue/slab slots and keep logical node
ids separate; this is phase-local frontier storage, not a topology-capacity
table. Immutable retained decisions and cold sealed data may remain paged.
Compare both approaches against the exact flat-state intervention while
preserving last-consumer release and byte accounting.

### Philosophy-preserving state replacement contract

Removing `PagedTensorColumn` from the hot path does **not** authorize restoring
the old all-generated-node state table. It changes the physical compute
interface, not the storage semantics in
`docs/theory/implementation_notes.md` or the authoritative lifecycle
pseudocode. Keep these invariants:

1. Logical occurrence and topology ids remain stable and separate from
   recyclable physical state slots.
2. Mutable public state and active reach/profile exist only for occurrences
   that are queued, evaluating, or being chance-closed. Their capacity is
   derived from the transactionally admitted live-frontier high-water bound
   $F_{\rm state}$, never from total possible or generated nodes $N$.
3. After a parent has committed all successors, immutable retained rows, and
   leaf data, release its mutable state and active reach/profile immediately.
4. At each realized player spot, write the natural partition input and required
   reach/folded artifacts once. Do not reintroduce rollout replay or reconstruct
   discarded state for targets or learning.
5. Decision, action, leaf, and occurrence tables remain schema-derived,
   byte-admitted, realized-work storage. Growing retained tables may remain
   chunked/append-only and are sealed once into exact contiguous destinations.
6. Admission continues to charge allocated capacity, page/slab slack, graph
   workspaces, seal overlap, completion, and learner closure against the hard
   memory envelope before allocation.

Treat the three storage domains differently:

| Domain | Row domain and lifetime | Required physical behavior |
|---|---|---|
| Hot mutable public state | At most the admitted live frontier; released at the parent's last consumer | Dense recyclable slots or slabs; one bulk pack read and one bulk successor write; never field-wise page scans |
| Growing retained rollout data | Realized decisions, sampled edges, leaves, and compact occurrences | Schema-derived chunked append-only tables with stable logical addresses |
| Sealed training data | Exact realized rows that survive into targets/optimization | One exact dense extent per final column/partition; source chunks released after validated sealing |

The preferred interface is explicit rather than tensor-compatible:

```text
read_live_pack(slot_ids, static_pack_buffers)
    -> dense PokerStateBatch and active reach/profile
process_dense_pack(...)
    -> retained rows, leaves, and dense successor candidates
reserve_live_successor_slots(valid_mask)
commit_live_pack(successor_slots, successor_candidates)
release_live_slots(consumed_parent_slots)
```

Poker encoding, legality, sampling, transition, history, and classification may
operate only on the dense pack representation. They must not receive a
`PagedTensorColumn` or perform storage page resolution. If physical live-state
slabs are retained, resolve all selected slot ids once at `read_live_pack` and
group all destination writes once at `commit_live_pack`; the slab abstraction
must end at those two boundaries. A stronger implementation lets admitted
family queues own the dense recyclable slots directly.

Acceptance requires more than recovering speed:

- no mutable state capacity proportional to $N$, a node budget, or retained
  decision capacity;
- no target/learner public-state handle and no replay of discarded rollout
  state;
- identical keyed RNG addresses, sampled actions, successor states, retained
  natural inputs, reaches, topology, and targets under paired tests;
- parent slots provably released only after every required child/retained/leaf
  write and reusable immediately afterwards;
- schema-ledger prediction agrees with measured owners and peak reservation,
  including CUDA graph buffers, under the byte-exact 2080 Ti ceiling; and
- the eager dense-pack path first recovers near the 1.96-second flat-state
  intervention before compilation or graph capture is credited with a gain.

### Repeated rollout work

For each policy pack, the current orchestration approximately does:

1. encode public state and legal events for the model forward;
2. classify and compute legal probabilities again for keyed sampling;
3. execute the poker transition;
4. recompute legal events and selected full-hand log probability for reach;
5. clone the full `[batch, players, hands]` reach tensor;
6. recompute selected log probability for retained old policy data; and
7. serialize retained inputs by encoding public state and history again.

Reach normalization, preflop 1,326-to-169 reduction, legal masks, and history
materialization are repeated across those calls. The selected full-hand log
probability is computed twice. This work creates many pointwise kernels,
temporary tensors, copies, and validation assertions around one model forward.

The desired unit is a fused family step that computes legal structure once and
returns action, selected log probability, next state, updated reach, routing
summary, and already-natural retained inputs. It should write directly into
preallocated destinations. This is also the right CUDA graph boundary.

### History materialization and recycling

CUDA history is currently materialized at the maximum rollout depth (42) on
every call to avoid a scalar read. It performs six binary-lift levels over the
batch-by-42 slab and gathers multiple fields. Retained input serialization then
does it again. Recycling history rows uses boolean-indexed dynamic tensors,
`unique`, Python recursion, and `numel()`-dependent control.

Fixed width is capture-friendly but needlessly expensive for short histories.
Use immutable 8/16/32/42 history buckets or an incremental compact descriptor,
reuse the packed history between inference and retention, and replace dynamic
recycling with fixed-capacity device worklists. Do not solve this by retaining
a global dense history capacity beyond the last consumer.

### Hot validation and dynamic CUDA shapes

The policy adapter launches repeated device/shape/legal checks for each action
path. Assertions remain essential at ingestion and acceptance boundaries, but
immutable partition and shape facts should be validated once, not four times
inside every pack. Throughput builds should use graph-safe asynchronous checks
only for genuinely data-dependent invariants. A debug mode can retain the full
diagnostic suite.

Manual `execution_undeclared_host_synchronizations == 0` is not sufficient
proof: it records a small known set of dynamic-position operations and misses
implicit synchronization inside many dynamic-shape PyTorch operators. Add an
operator-level sync gate for bounded profiles.

### Target construction

The audited sequential builders used Python trajectory/decision loops with
CUDA scalar extraction. Branched targets copied topology to CPU, traversed
nodes and edges in Python, and issued many small CUDA operations. The branch
diagnostic spent 1.46 s in targets for only 1,346 decisions.

Use sealed CSR and segmented device recurrences. Sequential player-clock
targets can use batched segmented scans. Branched recurrences should use
bounded reverse-depth or frontier passes over sealed topology, keeping payload
on device. A bulk compact pointer/control transfer is acceptable only at an
explicit bounded structural boundary. The in-progress dirty sequential target
rewrite follows this direction but needs exact semantic and performance gates.

### Learner launch structure

The learner has stable parameters, a stable flat fp32 gradient buffer, no
dropout, immutable sealed inputs, and a small physical-shape set. It is a good
CUDA graph candidate. At present it also creates and records many CUDA events
for every physical subspan even outside focused diagnostics, performs a full
finite scan, and converts finite/norm results to host scalars each step.

Gate detailed CUDA timing behind profiling. Fuse finite detection, norm, and
clip scale into one device reduction and one common decision. First capture a
pure subspan graph containing static-index gather, current forward, cached
reference lookup, loss, and scaled backward. Zero the flat gradient buffer once
outside the graph; keep NCCL and optimizer update outside the first version.

### Reference cache computes unused heads

The reference KL consumes raw policy fields, but cache construction retains a
full policy output including value and branching heads. Add a policy-only
reference forward and cache only the consumed natural logits/masks. Its measured
ceiling is 7.59 s in the current full iteration and roughly 72 MB of retained
payload, so it belongs ahead of communication overlap.

### Physical layouts left on the table

- All postflop calls still pay the flop hand width. Exact physical widths are
  1,176 on flop, 1,128 on turn, and 1,081 on river. This saves only a few percent
  on the current sequential street mix, but about 11--13% of attention pairs in
  late-street-heavy branching samples.
- Active preflop reach remains full 1,326-hand form even though retained and
  model inputs are natural 169-class form. Keep it natural until the flop and
  expand once, if the theory mapping and sampling laws remain exact.
- Compatibility masks and other immutable learner descriptors are reconstructed
  for sampled rows. Precompute or bit-pack them in sealed storage when the byte
  cost is lower than repeated kernels.
- Chance sampling sorts all 52 scores to choose a few cards. A deterministic
  fixed-size selection/top-k formulation should be measured after the larger
  rollout leaks are removed.

## ROI-ordered experiment backlog

The order below combines measured ceiling, expected speedup, implementation
risk, and whether the change unlocks CUDA graphs. Each item needs an A/B at
fixed seeds, checkpoint, realized-work accounting, and reuse ratio.

| Rank | Experiment | Why now / expected result | Effort | Acceptance gate |
|---:|---|---|---:|---|
| 0 | Freeze a trustworthy benchmark revision and byte-exact 2080 ceiling | Live source edits and ambiguous GB/GiB currently invalidate final tuning decisions | Low | clean revision; raw samples; 10,000,000,000-byte gate; GPU teardown |
| 1 | Use the 10 GB card envelope independently of rank count | Removes the observed 20-subspan/step configuration failure without semantic code changes | Low | same loss/topology; <=10 GB; materially fewer physical subspans |
| 2 | Remove field-wise paging from hot live state | Exact intervention cut rollout 47.8%; enables a stable dense graph boundary | Medium-high | recover near-flat A/B time; one bulk pack read/write; exact state/lifetime accounting |
| 3 | Fuse rollout legal/probability/sample/logp/reach/retention preprocessing | Removes duplicated pack work and reduces launch/copy pressure after state staging | Medium | >=20% further rollout reduction; identical keyed actions and retained tensors |
| 4 | Reuse and bucket history materialization; fixed-capacity recycling | History is encoded twice and dynamic recycling likely causes many implicit syncs | Medium | fewer launches/syncs and no lifetime/schema drift |
| 5 | Capture/compile the whole dequeue-to-enqueue rollout pack | Kernel launches are about 790/decision; stable dense 64/32/16/tail packs can amortize them | Medium-high | >=15% additional rollout gain; bounded graph variants; logical RNG unchanged |
| 6 | Re-baseline and finish vectorized CUDA target recurrences | Removes scalar CPU work; branch targets already consume about 9% of representative time | Medium | exact target equality/estimator tests; no per-node D2H; >=2x target speedup |
| 7 | Tune family rows under phase-closure admission | 96 rows gave +20.8% at 512 roots, but production peak exceeded decimal 10 GB | Low-medium | select largest safe 64/80/88/96 width with p99 reserved < ceiling |
| 8 | Split postflop physical execution by street | Reduces pair work 11--13% in late-heavy branching and creates better graph shapes | Medium | exact natural mapping; fewer pairs and lower rollout time |
| 9 | Capture learner gather+forward+loss+backward per partition/shape | Forward+backward is 46.86 s and graph prerequisites are already strong | Medium-high | >=15% learner compute gain at fixed rows/step and gradients |
| 10 | Policy-only reference forward/cache | 7.59 s measured ceiling and avoids unused value/T/C outputs | Medium | KL bitwise/tolerance match; lower cache time and bytes |
| 11 | Gate per-subspan CUDA events and fuse finite/norm/clip reductions | Removes instrumentation and two host decisions from every step | Low-medium | unchanged skip/clip semantics; lower launch/sync counts |
| 12 | Precompute/bit-pack immutable learner descriptors | Reduces sampled-row pointwise/index work and benefits learner graphs | Medium | net byte reduction or admitted memory; faster gathered subspan |
| 13 | Natural 169-class active preflop reach | Reduces active reach bandwidth and scatter work before flop | Medium-high | exact reach/sampling equivalence and one bounded flop expansion |
| 14 | Overlap gradient communication with final backward buckets | Only 3.16 s current ceiling, so valuable after local compute work | Medium-high | identical reduced flat gradient; exposed sync time falls |
| 15 | Replace allocator `empty_cache()` boundary with measured reusable-space admission | May avoid release/reallocation churn, but current phase planner relies on it | Medium | no OOM regression across adversarial phase closures |
| 16 | Deterministic small-k chance selection and minor kernel cleanup | Real but below repeated rollout/history work | Low-medium | exact counter-key law or approved equivalent; measured gain |

Rank 0 is measurement infrastructure, not a request to postpone optimization.
Ranks 1--7 can proceed as a short sequence: establish the envelope, restore a
dense hot state boundary, fuse the rollout pack, stabilize history shapes,
capture the whole pack, validate targets, then retune width around the faster
implementation.

## CUDA graph plan

Do not capture the whole training iteration. Queue lengths, realized tree size,
storage phase transitions, and learner work remain deliberately dynamic and
bounded. Within rollout, however, the graph boundary should be the whole
fixed-width processing of one device queue pack: dequeue through direct
successor enqueue. A CPU routing summary is only an implementation stepping
stone, not the target architecture. Do not capture the current
`PagedTensorColumn` implementation: replay would faithfully repeat its page
scans and read-modify-write copies, and its changing page set is not a stable
capture shape.

### Rollout graph boundary

For each `(execution family, street, row bucket, mode)`, one graph replay should
perform:

```text
dequeue 64/96 live-slot handles from the selected device queue
    -> bulk-load each required state/reach/history field exactly once
    -> compact history/public encoding
    -> legal events and policy forward
    -> keyed action sampling
    -> selected full-hand log probability
    -> acting-player reach update
    -> poker transition and chance/posterior work when required
    -> classify successors by execution family
    -> direct write of natural retained inputs
    -> allocate/write successor state and history rows
    -> compact and enqueue successors into device queues
    -> write terminal leaves, counters, and a device failure flag
```

Sequential mode has at most one successor per input row. Branching uses an
immutable `[batch, max_offspring]` candidate slab, a valid-child mask, and
device prefix sums to compact children. Reserve each destination queue once per
pack rather than atomically appending one child at a time. Capacity overflow or
an invalid transition writes a device failure flag that is checked at a
declared control boundary. No dynamic-output `nonzero`, `.item()`, Python loop,
or CUDA allocation may occur inside the captured path.

Start with 64-row steady-state graphs and 32/16/8/tail graphs; separately test
the largest byte-admitted steady width in the 80--96 range. After the
street-specific layout experiment, key postflop graphs by flop/turn/river.
Dummy rows need an explicit inert mask: they must not consume logical RNG keys,
mutate state, reserve storage, or write retained rows. Queue, state, history,
retention, leaf, and scratch addresses must be stable and preallocated from the
immutable admitted schemas.

The preferred hot representation is a structure of dense, recyclable live
slots owned by the admitted family queues or frontier slabs. Queue entries are
slot ids; all columns for the selected slots are bulk-loaded once into static
graph input buffers, and successor columns are written once to slots reserved
in the destination queues. The number of slots is derived from the admitted
live-frontier and closure bounds, not total possible tree nodes. This preserves
realized-work and last-consumer storage: consumed slots are released after the
successor and retained rows have been committed. Paged arenas can still own
cold immutable captures, but no `PagedTensorColumn` operation belongs inside
the graph. This representation and its tests must satisfy every item in the
philosophy-preserving state replacement contract above; “static graph address”
never means an all-$N$ mutable-state allocation.

CUDA graph capture amortizes launches but does not itself fuse kernels. First
express the eager dequeue-to-enqueue pack as one allocation-free tensor
operation over dense pack buffers and prove that it recovers the flat-state
baseline. Use `torch.compile(fullgraph=True, mode="reduce-overhead")` or a few
targeted custom kernels to fuse legal/probability/sample/logp/reach, history
packing, transition routing, and masked compaction where profitable. Then
capture the complete compiled pack with explicit `torch.cuda.CUDAGraph`. Keep
SDPA/Flash Attention and evaluator kernels intact when they are capture-safe.
CUDA 13 in the venv is available; no CPU fallback is permitted.

The host initially selects a nonempty execution-family queue and replays its
graph. It must not receive actions, successor states, histories, or routing
payloads. Occasional compact queue counters and failure flags are sufficient;
device-side or persistent scheduling is lower priority because measured host
queue control is currently only milliseconds per iteration. Double buffering
and secondary streams come only after a single-stream whole-pack graph is
correct and measured; otherwise they obscure the launch problem.

Graph capture owns additional stable workspaces. Recalibrate memory after all
graph variants are warmed: the measured 96-row eager path already exceeds a
literal 10,000,000,000-byte Torch ceiling, so an 80--88-row graph may be the
2080 Ti optimum even though H100 should admit much wider packs.

### Learner graph boundary

For each immutable partition and physical row shape:

```text
static sampled-row index buffer
    -> sealed-input gather / immutable descriptor lookup
    -> current policy forward
    -> policy-only cached reference lookup
    -> PPO/RB loss
    -> scaled backward into stable flat fp32 gradient storage
```

Zero flat gradients once before replaying all subspan graphs. Do not capture
per-subspan metrics or CUDA-event construction. Keep finite/norm/clip, NCCL,
and optimizer update outside initially. Expected current shapes in the 64-row
historical geometry are 54 preflop and 10 postflop; with a 128-row statistical
minibatch they are approximately 64 and 12 under the same card envelope.

Once stable, measure capturing finite/clip and optimizer separately. Capturing
the optimizer first has only a 2.10 s ceiling and is not high ROI.

## Scientific benchmark matrix

Every proposed change should run this minimum matrix on a settled revision.
Use GPUs 0 and 1 concurrently for independent one-card probes or one two-rank
DDP run, and leave GPU 2 idle.

### Sequential matrix

1. Normal and sequential RB, 128 roots, completion 16, one learner step:
   quick semantic/phase smoke and three-repeat launch comparison.
2. Sequential RB, 512 roots, completion 256, family widths 64/80/88/96,
   active windows 32/64, one learner step: packing and memory curve.
3. Sequential RB, 2,048 roots/rank, completion 1,024, minibatch 64, learner
   steps chosen after rollout so learner/fresh ratio is 2.0: production gate.
4. The same production workload with fixed 232 steps only as a link to the
   historical record.

### Branching matrix

Use a trained current T/C checkpoint or a fixed replayable sealed topology.
Stratify natural tiny, natural medium, natural large, and cutoff-closed trees.
Report both 1:1 and 2:1 rollout-to-learner ratios. Never use the median of a
mixture where one random untrained tree collapses to five nodes.

### Required reported metrics

- fresh decision nodes and all player nodes per rank and globally;
- rollout-only and end-to-end nodes/s;
- learner statistical rows/s, physical rows/s, backward subspans/s, and
  optimizer steps/s;
- learner presentations divided by fresh decisions;
- phase wall time and nested CUDA component time;
- model calls, rows/call, tail fraction, padded tokens, and attention pairs by
  street and execution family;
- kernel launches, implicit stream synchronizations, D2D/H2D/D2H bytes;
- peak allocated, peak reserved, non-Torch occupancy, and phase-boundary live
  owners; and
- GPU clocks, power, temperature, and utilization so thermal throttling is not
  mistaken for a code regression.

Use median plus range or bootstrap interval, not only a mean. Run enough warmup
for allocator and compiled/graph state. Compare paired repeat seeds and include
raw samples. A performance change is accepted only when semantics, keyed RNG,
storage lifecycle, finite-gradient behavior, and teardown gates also pass.

## What not to optimize first

- Scheduler control transfers: measured at only 4.4 ms in the full iteration.
- Sealing: about 13 ms and already follows exact last-consumer storage.
- Optimizer capture: only a 2.10 s ceiling before larger learner graph work.
- Wider active windows by themselves: 32 and 64 were tied once 64-row packs
  filled.
- Blindly larger family packs: 128 rows already failed closure, and 96 rows
  exceeded the conservative decimal 10 GB production ceiling.
- Whole-iteration CUDA graphs: they conflict with bounded dynamic realized work
  and would hide, rather than solve, unstable compute boundaries.
- Moving poker transitions to CPU: it would add strict device/host dependencies
  when the measured queue-control CPU cost is already negligible.

The near-term success criterion is a settled two-GPU sequential-RB run at a
true 2.0 learner/fresh ratio that returns global fresh decision throughput to
at least the historical 123.4 decisions/s while remaining below the byte-exact
2080 Ti ceiling. The stretch target is to exceed that record by 25% through
rollout fusion and capture, without counting reduced learner work as speedup.
