# Rollout optimization program

Status: eager-rollout stopping point accepted 2026-07-22. The 2,560-start
production baseline is now established. The next pass should start with
history-shape stabilization and a full dequeue-to-enqueue execution boundary.

This document is the performance decision log and experiment queue. It is not
authority for sampling, estimator, target, or storage-lifetime semantics.
`docs/theory/core_math.tex` and `docs/theory/pseudocode/` remain authoritative.

## Current result

The July 21 paged-state refactor regressed production-scale sequential-RB
rollout from approximately 259 to 90 fresh decisions/s. The accepted fixes
restore it to approximately 257 decisions/s on the historical-continuity gate:

| Frozen 2,048-start checkpoint-1110 gate | Before | Current stopping point |
|---|---:|---:|
| Rollout time | 87.45--90.27 s | 31.39 s median |
| Fresh decisions | about 8,050--8,087 | 8,064.5 median |
| Rollout time/decision | about 10.9--11.2 ms | 3.892 ms |
| Raw rollout throughput | about 90--92 decisions/s | 256.9 decisions/s |
| Model calls | about 140 | 118 |
| Padded attention pairs | 33.149 billion in the reproduction | 31.229 billion |
| Peak allocated/reserved | not comparable | 4.607/9.071 GB |

This is about a 64% rollout-time reduction and 2.8x raw throughput improvement
against the audited regression. It is within about 1% of the pre-refactor raw
record. The first required 2,560-start gate subsequently established a 3.827
ms/decision current baseline, or 261.3 decisions/s. End-to-end two-rank
performance at a fixed learner/fresh ratio has not yet been re-established.

## Mandatory production benchmark contract

Do not accept an optimization using 128 or 512 starts. Those sizes are only for
debugging correctness, crashes, and gross regressions.

### Workload

| Setting | Required value |
|---|---|
| Game/mode | HU NLHE, 32 bb, 13 ranks, sequential Rao--Blackwell |
| Config | `configs/train/nlhe_hu_32bb_rb.yaml` |
| Hardware profile | `local_2x2080ti` (explicit SDPA selection) |
| Device | CUDA on one otherwise idle RTX 2080 Ti for the rollout gate |
| Trajectory starts | 2,560 per GPU, the configured production value; never below 2,500 |
| Checkpoint | Highest-numbered checkpoint in `runs/nlhe_hu_32bb_full_sequential_rb`; currently `ckpt_0001110.pt` |
| Completion samples | 1,024 |
| Family rows | 84 unless that parameter is the experiment |
| Active trajectories | 64 unless that parameter is the experiment |
| Configured storage page rows | 256 unless that parameter is the experiment |
| Warmup/repeats | At least 1 warmup and 3 timed repeats; use 5 repeats before changing a production default |
| Rollout-isolation learner work | One learner step; never present this as end-to-end production throughput |
| End-to-end confirmation | Two ranks, minibatch 128, 116 learner steps/rank, and report the realized learner-presentations/fresh-decision ratio |

Resolve the checkpoint again before every experiment; do not hard-code an older
checkpoint after a newer one appears. Record its path, iteration, and SHA-256
in the benchmark artifact. Checkpoint loading is strict and accepts only the
current schema. If the latest trained checkpoint predates that schema, use a
read-only current-schema benchmark wrapper containing the exact same model
weights and record both identities. Never substitute random initialization.

At this stopping point, `ckpt_0001110.pt` is schema v3 while the runtime accepts
only schema v6. The validated exact-weight wrapper used for these measurements
is `/tmp/pbg_ckpt_0001110_v6_benchmark.pt`. Because `/tmp` is ephemeral, verify
that it exists and that every model tensor equals the source checkpoint before
using it; regenerate an equivalent read-only wrapper if it does not. This is a
benchmark artifact, not permission to restore legacy loading in production.

Example single-GPU rollout gate, after confirming the selected GPU is idle:

```bash
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=1 python -m pbg.cli.benchmark \
  --config configs/train/nlhe_hu_32bb_rb.yaml \
  --checkpoint /tmp/pbg_ckpt_0001110_v6_benchmark.pt \
  --hardware-profile local_2x2080ti \
  --device cuda \
  --mode rao_blackwell \
  --roots-per-gpu 2560 \
  --completion-samples 1024 \
  --family-batch-rows 84 \
  --max-active-trajectories 64 \
  --gd-steps 1 \
  --warmup 1 \
  --repeats 3 \
  --output /tmp/nlhe_rollout_candidate.json
```

The wrapper in this example corresponds to the latest source checkpoint as of
2026-07-22. Replace both source and wrapper when the run directory contains a
newer checkpoint. Do not omit `--hardware-profile`: this training config does
not embed a profile, and omission selects the rejected automatic/FlexAttention
path rather than the production SDPA path.

### Benchmark and commit discipline

For every candidate:

1. Check `nvidia-smi` before CUDA work and select only an idle GPU. Do not touch
   another user's process.
2. Change one performance idea at a time. Preserve fixed seeds, checkpoint,
   workload, statistical work, and logical RNG addressing.
3. Run a small correctness probe if needed, then the mandatory 2,560-start
   gate. Report every raw sample, not just the median.
4. Compare rollout seconds/decision first. Also report decisions, calls, padded
   tokens/pairs, phase times, allocated/reserved bytes, and teardown state.
   Raw samples are authoritative for optional per-family/per-street metrics:
   the current median aggregator intersects optional keys, so a tiny repeat can
   erase a stratum that exists in other samples.
5. Run the relevant semantic tests and CUDA acceptance tests. A speedup with a
   storage-lifetime, target, RNG, or gradient change is not accepted.
6. Confirm the GPU returns to its pre-benchmark idle state.
7. Commit each successful optimization separately. Do not accumulate several
   unmeasured wins in one implementation commit.
8. Immediately update this document with the result and implementation commit
   hash in a small follow-up documentation commit. This avoids the impossible
   circular requirement of putting a commit's own hash inside itself.
9. Rejected code must not remain on the production path. Record the rejection
   and evidence here.

A result should normally improve the production median by at least 3% or have a
clear paired confidence interval before adding structural complexity. Smaller
changes can be accepted when they simplify the hot path or reduce memory with
no semantic cost, but must be labeled as such.

## Times to beat

The primary cross-run gate is time per fresh decision because different seeds
and checkpoints realize different tree lengths.

| Gate | Time to beat | Equivalent throughput | Notes |
|---|---:|---:|---|
| Current eager baseline | **3.827 ms/decision** | **261.3 decisions/s** | Median at the required 2,560 starts after `48ce2a3` |
| Pre-refactor raw rollout record | **3.863 ms/decision** | **258.9 decisions/s** | 19.51 s for 5,051 decisions on July 17 |
| Rollout stretch target | **3.090 ms/decision** | **323.6 decisions/s** | 25% above the pre-refactor raw rate |
| Historical end-to-end record | n/a | **123.4 global fresh decisions/s** | Two ranks; must match learner/fresh ratio before comparison |
| End-to-end stretch target | n/a | **154.3 global fresh decisions/s** | 25% above the historical record at a true 2.0 learner/fresh ratio |
| Memory ceiling | n/a | n/a | Torch peak reserved must stay below 10,000,000,000 bytes on the 2080 Ti profile |

The old 87.45-second paged-state result is a regression floor, not a meaningful
target. Do not claim a win merely for beating it. Do not compare absolute
rollout seconds across different realized decision counts; compare paired
seconds/decision and include the raw counts.

## Optimization project commits

| Commit | Role | Result |
|---|---|---|
| `32adb8c` | Pre-optimization marker | Reproducible starting point for this pass |
| `d95b7d3` | Dense recyclable live-state slab | Removed generic `PagedTensorColumn` page scans from hot mutable state while preserving bounded live-frontier ownership |
| `be0b009` | Safe family width 84 | Improved packing while retaining about 0.95 GB below the decimal 10 GB ceiling |
| `5ecfeb4` | Direct contiguous arena appends | Avoided generic arbitrary-row page matching for already-proven contiguous reservations; about 1.1% in the paired sample |
| `cf6bc34` | Leaner multi-page gathers | Removed the zero-seed/first-page merge; about 0.35% and lower reservation in the paired sample |
| `83aab85` | Finish eager rollout pass | Exact-street execution queues, pack-local public/legal/history and selected-logp reuse, fixed-width transition updates, and the final benchmark record |
| `48ce2a3` | Immediate child commit fast-path | Used host-proven contiguous cursors for the unsplit sequential child commit; 0.26--0.50% paired gain at 2,560 starts |

Commit `f0b049b` occurred during the sweep but is a compatibility-schema
cleanup, not a credited performance optimization.

## ROI-ordered backlog

Priority applies to the next work. Closed items remain in the same table so an
experiment is not accidentally repeated. Status must be exactly `Open`,
`Implemented in <hash>`, or `Rejected`.

| ROI | Idea | Status | Evidence or acceptance gate |
|---:|---|---|---|
| 1 | Bucket history at 8/16/32/42 tokens and replace dynamic recycling with fixed-capacity device worklists | Open | History still runs six binary-lift levels over width 42 and uses dynamic `unique`/boolean compaction; require fewer launches/syncs and unchanged lifetime |
| 2 | Build an allocation-free dequeue-to-enqueue pack and compile/capture it per exact family and row bucket | Open | About 790 launches/decision in the bounded profile; require at least 15% rollout gain, bounded graph workspaces, and unchanged logical RNG |
| 3 | Fuse remaining transition, reach, routing, and retained-write preprocessing | Open | Public/legal/history and selected logp are already reused; require at least 15% for a complex fused implementation |
| 4 | Finish and re-baseline vectorized CUDA target recurrences | Open | Require exact targets, no per-node D2H, and at least 2x target speedup; especially relevant for branching |
| 5 | Use the 10 GB card envelope independently of participating rank count | Open | The conservative multi-card profile caused about 20 learner subspans/step versus two in the historical geometry; require unchanged loss/topology and no more than 10 GB |
| 6 | Capture learner gather + forward + loss + backward by partition/shape | Open | Require at least 15% learner compute gain with identical gradients and fixed rows/step |
| 7 | Add policy-only reference forward/cache | Open | Reference cache had a 7.59 s ceiling and retains unused value/branching outputs; require matching KL and lower bytes/time |
| 8 | Gate detailed CUDA events and fuse finite/norm/clip reductions | Open | Preserve skip/clip semantics while reducing launches and host decisions |
| 9 | Precompute or bit-pack immutable learner descriptors | Open | Accept only with net admitted-byte benefit and faster sampled subspans |
| 10 | Keep active preflop reach in natural 169-class form until one bounded flop expansion | Open | Require exact reach and sampling equivalence |
| 11 | Overlap gradient communication with final backward buckets | Open | Only a 3.16 s measured ceiling; require identical reduced flat gradients |
| 12 | Replace the allocator `empty_cache()` boundary with measured reusable-space admission | Open | Require no OOM regression across adversarial phase closures |
| 13 | Deterministic small-k chance selection instead of sorting all 52 scores | Open | Preserve the exact counter-key sampling law or obtain explicit approval for an equivalent law |
| — | Dense recyclable live-state slab | Implemented in `d95b7d3` | Dominant regression fixed; mutable capacity remains proportional to the admitted live frontier, not total nodes |
| — | Tune production family width | Implemented in `be0b009` | Width 84 selected; width 88 was only 1.5% faster and left about 158 MB below 10 GB |
| — | Direct contiguous append commit | Implemented in `5ecfeb4` | About 1.1% matched gain with unchanged ownership |
| — | Trim generic multi-page gather | Implemented in `cf6bc34` | About 0.35% matched gain; small but essentially free |
| — | Split physical postflop execution by exact street | Implemented in `83aab85` | About 5.6% fewer padded pairs at family 64 and about 4.9% lower rollout in the progression |
| — | Reuse public/history/legal representation and selected policy logp within a pack | Implemented in `83aab85` | Removes duplicate encoding/log-probability work without extending state lifetime |
| — | Remove dynamic fold compaction and its host synchronization | Implemented in `83aab85` | Fixed-width masked update was about 1% faster in the matched sample |
| — | Commit immediate unsplit child batches contiguously | Implemented in `48ce2a3` | Median 3.827 versus 3.846 ms/decision; a small, consistent, essentially free hot-path simplification |
| — | Raise production width to 88 or 96 | Rejected | Faster, but 88 reserved 9.84 GB and 96 reserved 10.497 GB; insufficient 2080 Ti headroom |
| — | Use 4,096-row cold-storage chunks | Rejected | Only about 0.3% faster than the bounded derived chunk |
| — | Flexible/automatic attention dispatch | Rejected | 56.078 s versus about 31.86 s with SDPA |
| — | Widen active trajectory window without widening family packs | Rejected | Active windows 32 and 64 were effectively tied once family packs filled |
| — | Optimize scheduler control transfers or sealing first | Rejected | Scheduler control was 4.4 ms and sealing about 13 ms per full iteration |
| — | Whole-iteration CUDA graph | Rejected | Conflicts with bounded dynamic realized work; capture fixed pack shapes instead |
| — | Move poker transitions to CPU | Rejected | Violates the accelerator-only hot-path contract and adds strict device/host dependencies |

## Accepted benchmark evidence

### Production progression

These are 2,048-start historical-continuity measurements using checkpoint 1110
weights, 1,024 completion samples, and one learner step. They are retained to
explain the accepted commits; new work must use the 2,560-start contract above.

| Implementation | Decisions | Calls | Padded pairs | Rollout | Raw decisions/s |
|---|---:|---:|---:|---:|---:|
| Audit matched baseline, family 64 | about 8,050 | 139.5 | not recorded | 87.45 s | about 92.1 |
| Pre-fix reproduction, seed 101 | 8,087 | 140 | 33.149 B | 90.27 s | 89.6 |
| Dense live slab and efficient chunks, family 64 | 8,233 | 141 | 34.057 B | 35.72 s | 230.5 |
| Exact-street queues, family 64 | 8,233 | 145 | 32.145 B | 33.98 s | 242.3 |
| Exact-street queues, family 80 | 8,255 | 123 | 32.277 B | 32.46 s | 254.3 |
| Final family 84, two-repeat median | 8,064.5 | 118 | 31.229 B | 31.39 s | 256.9 |

The two final family-84 samples were 32.170 seconds for 8,248 decisions and
30.603 seconds for 7,881 decisions. Median total time with one learner step was
32.455 seconds. The final fixed-width fold update later measured 31.856 seconds
for the same 8,248-decision sample that had taken 32.170 seconds. Immutable
event-id caching and the identity-matched cache fast path were CUDA-tested but
not credited separately.

### Required 2,560-start baseline

Commit `48ce2a3` established the first baseline under the mandatory production
contract. It specializes only the immediate unsplit sequential child commit;
branched subspan commits retain arbitrary-row routing. Source checkpoint
`ckpt_0001110.pt` (schema v3, SHA-256
`bced881945512f2a4152d3acd88a0a06395a00130cdf68a61aa2652bc1b5aebb`)
was represented by exact-weight schema-v6 wrapper SHA-256
`b72fb7f313032114f605ac665d1c80741a4bb8b9993dfe68cfc7b48947010603`.
All 366 model tensors were equal. Repeat seeds were 104830, 209559, and 314288
on one RTX 2080 Ti with PyTorch 2.12.1, CUDA 13.0, and explicit
`local_2x2080ti` SDPA execution.

| Repeat | Decisions | Calls | Padded pairs | Baseline rollout | Candidate rollout | Baseline/candidate ms per decision | Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,316 | 147 | 40.625 B | 39.509 s | 39.399 s | 3.830 / 3.819 | 0.28% |
| 2 | 9,745 | 149 | 37.480 B | 37.647 s | 37.550 s | 3.863 / 3.853 | 0.26% |
| 3 | 10,201 | 144 | 39.913 B | 39.238 s | 39.041 s | 3.846 / 3.827 | 0.50% |

Candidate peak allocation was 4.624--4.625 GB and peak reservation was
8.852--8.921 GB. Post-process teardown left 17.4 MB allocated and 130.0 MB
reserved, and `nvidia-smi` showed the GPU back at its 6 MiB display-only idle
state. Focused validation was 31 passed and one skipped in the storage/tree
suite plus the CUDA topology test. The gain is below the ordinary 3% threshold
and is credited only because it removes generic page matching from a
host-proven contiguous write without changing the arbitrary-row path.

An earlier invocation without `--hardware-profile local_2x2080ti` measured
6.635--6.695 ms/decision because it selected automatic/FlexAttention. That is
invalid baseline evidence and led to the explicit-profile contract correction
above.

### Root-cause bisection

Commit `a399e30` introduced the large discontinuity by replacing direct
structure-of-arrays state indexing with generic paged-column access on every
hot poker field. On a controlled 128-start workload with identical 492
decisions, 24 model calls, and 2.040 billion padded pairs:

| `a399e30` intervention | Rollout | Total |
|---|---:|---:|
| Paged state, history zero-write control | 3.758 s | 4.126 s |
| Prior flat state backend, same control | 1.961 s | 2.317 s |

Removing history initialization alone did not help. Replacing paged hot state
recovered the entire pre-refactor time in all paired samples. The queue family
and 64-row packs existed before the regression and were not its cause.

### Width and memory evidence

| Family rows | Decisions | Calls | Rollout | ms/decision | Peak allocated/reserved |
|---:|---:|---:|---:|---:|---:|
| 64 | 8,233 | 141 | 36.228 s | 4.400 | 3.63/6.62 GB |
| 84 | 8,283 | 112 | 33.723 s | 4.071 | 4.60/9.05 GB |
| 88 | 8,262 | 107 | 33.118 s | 4.008 | 4.80/9.84 GB |

Width 84 is the safe 2080 Ti choice. Width 128 failed the downstream workspace
closure guard. Low learner live memory does not justify a larger rollout
capacity: phase admission must account for reserved memory, non-Torch
occupancy, graph workspaces, completion, sealing overlap, and learner closure.

### Operator and phase ceilings

A bounded 32-start profile was wall-time distorted but useful for counts:

| Event | Count | Meaning |
|---|---:|---|
| CUDA kernel launches | 108,254 | About 790 per fresh decision |
| Stream synchronizations | 2,800 | About 20 implicit syncs per decision |
| `aten::empty` | 23,382 | High transient-allocation pressure |
| `aten::copy_` | 14,190 | 12.3% of self CUDA time |
| `aten::index` | 7,624 | Repeated gathers/scatters |
| `aten::cat` | 748 | 7.8% of self CUDA time |
| Efficient-attention forward | 90 | Only 21.1% of self CUDA time |

The older full two-rank reproduction spent 50.13 s in rollout, 2.84 s in
targets, and 62.70 s in the learner. Learner subcomponents were 30.62 s
backward, 16.24 s forward, 7.59 s reference cache, 3.16 s gradient sync, 2.28 s
loss, and 2.10 s optimizer. Those ceilings explain the open ROI order; they
must be re-measured after the rollout changes before accepting learner work.

The comparable historical phase record was:

| Quantity | July 17 record | Pre-fix July 22 reproduction |
|---|---:|---:|
| Total iteration | 93.13 s | 115.68 s |
| Rollout | 19.51 s | 50.13 s |
| Targets | 4.15 s | 2.84 s |
| Learner | 69.43 s | 62.70 s |
| Learner time/step | 299.3 ms | 270.3 ms |
| Peak allocated/reserved | 4.45/6.80 GiB | 3.15/5.21 GiB |
| Global fresh decisions/s | 123.4 | about 87.3 |

The learner was already faster; rollout caused the end-to-end regression. The
historical run used a 2.94 learner-presentation/fresh-decision ratio, so its
total seconds are not an acceptance gate for a new run targeting ratio 2.0.

## Non-negotiable implementation constraints

Performance work must preserve all of these:

1. Production rollout, completion, model, losses, backward, optimizer state,
   posterior work, and showdown evaluation remain on CUDA. No CPU fallback.
2. Logical occurrence/topology ids remain separate from recyclable physical
   live-state slots.
3. Mutable public state and active reach/profile capacity derive from the
   admitted live-frontier high-water bound, never total possible/generated
   nodes or retained decision capacity.
4. Release a parent's mutable slot only after all successors, immutable
   retained rows, and leaf data have committed.
5. Retained decision, action, leaf, and occurrence tables remain schema-derived,
   byte-admitted, realized-work storage and seal once into exact destinations.
6. Do not replay discarded rollout state or retain public-state handles for
   targets/learning. Write natural retained inputs once at the realized spot.
7. Preserve keyed RNG addresses, sampled actions, successor states, reaches,
   topology, targets, and learner tensors under paired tests.
8. Charge slab/page slack, graph workspaces, seal overlap, completion, and
   learner transients against the hard memory envelope before allocation.

The three storage domains deliberately differ:

| Domain | Physical rule |
|---|---|
| Hot mutable state | Dense recyclable slots bounded by live frontier; no per-field page scans |
| Growing retained rollout data | Schema-derived chunked append-only storage with stable logical addresses |
| Sealed training data | Exact realized dense extents; release source chunks after validated sealing |

## Notes for the top open ideas

### History shapes and recycling

History currently materializes maximum width 42 for every new policy pack and
performs six binary-lift levels. Immediate decision retention now reuses that
packed history, but recycling still uses boolean-indexed dynamic tensors,
`unique`, Python recursion, and `numel()`-dependent control. Test immutable
8/16/32/42 buckets or an incremental compact descriptor plus fixed-capacity
device worklists. Do not create a global history capacity beyond the last
consumer.

### Full rollout pack compilation/capture

The desired fixed-shape unit is:

```text
dequeue admitted live-slot handles
  -> bulk-load state/reach/history once
  -> compact history and encode public state
  -> legal events and policy forward
  -> keyed sampling and selected log probability
  -> reach update and poker transition
  -> classify and compact successors
  -> direct retained/leaf writes
  -> reserve, write, and enqueue successor slots
```

Start from the accepted dense live slab and exact flop/turn/river families.
Begin with 84-row steady-state and bounded 42/21/tail variants. Dummy rows must
be inert: no RNG keys, state mutation, storage reservation, or retained writes.
No dynamic-output `nonzero`, `.item()`, Python loop, or CUDA allocation belongs
inside capture. Failure/overflow uses a device flag checked only at a declared
control boundary.

First make the eager pack allocation-free, then try
`torch.compile(fullgraph=True, mode="reduce-overhead")` or targeted kernels,
then capture with `torch.cuda.CUDAGraph`. Do not capture the whole iteration;
realized tree size, phase transitions, and learner work remain intentionally
dynamic and bounded.

### Targets and learner

Sequential and especially branched targets have contained host-driven
recurrences. Use sealed CSR and segmented device recurrences with no per-node
D2H. Branching needs a trained current T/C checkpoint or fixed replayable
topology before performance acceptance; untrained branching heads sometimes
collapse a sample to five nodes. The only representative untrained-head sample
had 2,128 nodes and 1,346 decisions: rollout was 14.029 s, targets/completion
1.458 s, and learner 0.228 s; 97.5% of decisions were postflop and 75.5% river.
Treat it only as phase/geometry evidence.

Target timing also has an unresolved provenance discrepancy: exact artifacts
reported 1.055 s and 2.836 s for apparently similar 5,051-decision work, while
the July 17 full report recorded 4.15 s. Define and reconcile the timing before
crediting a target speedup.

For the learner, graph immutable partition/shape subspans containing sampled
gather, current forward, policy-only reference lookup, loss, and backward into
the stable flat fp32 gradient buffer. Keep NCCL and optimizer outside the first
graph. Do not optimize optimizer capture first: its measured ceiling was only
2.10 seconds.

## Required report for every accepted result

Include:

- commit/source revision, resolved checkpoint path/iteration/SHA, config, seed,
  repeat seeds, GPU model, CUDA/PyTorch versions, and raw samples;
- fresh decisions, all player nodes, rollout seconds and seconds/decision;
- end-to-end decisions/s only with learner rows, steps, and the realized
  learner-presentations/fresh-decision ratio;
- model calls, rows/call, tail fraction, padded tokens, and attention pairs by
  street/execution family;
- rollout, targets, completion, sealing, reference, learner, and optimizer time;
- kernel launches, implicit syncs, and D2D/H2D/D2H traffic when profiling;
- peak allocated/reserved bytes, non-Torch occupancy, phase-boundary owners,
  and post-run teardown state; and
- median plus range or confidence interval, never only a mean.

Validation at the current stopping point was 247 passed and 37 skipped in the
general suite. The full CUDA suite was 272 passed, 10 skipped, and two unrelated
stateful-inference failures from the separately audited preflop class/combo
shape mismatch at `pbg/inference.py:596`. The final focused CUDA rollout gate
was 30 passed and two skipped. Re-check this status on every new optimization;
do not treat known unrelated failures as permission for new failures.
