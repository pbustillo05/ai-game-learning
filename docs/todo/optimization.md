# ROI list of potential optimization

Measured average budget from the completed run:

- Rollout: 44.3 s
- Learner forward: 23.4 s
- Learner backward: 42.7 s
- Reference-cache construction: 14.1 s
- Gradient synchronization: 7.0 s
- Loss: 2.9 s
- Optimizer update: 1.0 s
- Scheduler CPU synchronization: only 0.009 s

Those numbers are the sequential RB baseline. CPU scheduler control is already
negligible there. The leaks are GPU launch overhead, duplicated tensor work,
poor postflop packing, immutable preprocessing, and exposed communication.

## 2026-07-21 bounded packing sweep

This follow-up was measured at commit `b55df976e80b` on one otherwise idle RTX
2080 Ti. Every process was restricted to GPU 0, teardown was checked with
`nvidia-smi`, and the card returned to 6 MiB / 0% utilization after the sweep.
The workload used HU 32 bb sequential RB, 128 roots, 16 completion samples,
one 16-row optimizer update, one warmup, and three timed repeats. The model was
freshly initialized with the benchmark's fixed initialization protocol. These
are throughput diagnostics, not trained-policy loss measurements.

The table reports medians. Changing physical pack width can change borderline
fp16 actions, so the realized topology is close but not bit-identical across
every row. Compare rollout milliseconds per decision and model-call pressure,
not raw iteration seconds alone.

| Family rows | Active trajectories | Decisions | Rollout seconds | Rollout ms/decision | Model calls | Peak allocated / reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 396 | 5.273 | 13.316 | 34 | 1.01 / 1.56 GiB |
| 32 | 64 | 396 | 3.742 | 9.449 | 23 | 1.81 / 2.97 GiB |
| 64 | 16 | 394 | 3.715 | 9.429 | 21 | 1.92 / 3.33 GiB |
| 64 | 32 | 394 | 3.564 | 9.046 | 21 | 1.94 / 3.34 GiB |
| 64 | 64 | 408 | 3.743 | 9.175 | 20 | 2.13 / 3.69 GiB |

The 16- and 32-row/64-active samples performed the same median 958.9 million
postflop padded attention pairs. The 16-row case was nevertheless about 41%
slower per decision and made 34 rather than 23 model calls. This is a real
launch/packing cliff, not extra attention work. A physical family below 32 rows
should therefore be a tail shape, never the steady-state target, on this GPU.

At this small workload, 32 active trajectories with 64-row physical families
was the best stable point. Moving from 32 to 64 active trajectories did not
improve median time per decision and increased peak allocation by about 10%.
This does **not** supersede the production-width calibration at thousands of
roots; it says that wider active windows must be justified by measured pack
fill rather than assumed to help.

A matching normal-mode point at family 64 / active 32 realized 418 decisions
and spent 3.850 of 4.364 seconds in rollout (88.2%), at 96.9 decisions/s. The
sequential-RB point spent 3.564 of 4.191 seconds in rollout (85.0%), at 94.0
decisions/s. Seal plus target construction remained below 0.23 seconds and the
one-update learner path below 0.39 seconds. Scheduler synchronization was only
0.54--0.60 ms. The next optimization should still attack policy/model work
inside rollout rather than scheduler control, sealing, or target propagation.

The bounded branched probe cannot be used as a stable throughput baseline. Its
three repeats produced 66, 74, and 2 nodes because the sequential base model
has randomly initialized branching heads. The median sample used 18 model
calls for only 36 decisions and spent 2.043 seconds in rollout, which is useful
as evidence of small-pack overhead but not as an estimate for a trained
branched policy. Future branched gates must use a trained T/C checkpoint or a
fixed replayable topology and must stratify tiny/natural/cutoff-closed trees.

### Measurement and harness findings

- Full-iteration `torch.profiler` with CPU+CUDA activities, shapes, and memory
  is unsafe here. Two completed 128-root traces each spent minutes constructing
  the event tree, consumed 16--19 GiB of host RAM, and retained a 264 MiB CUDA
  context. Both were terminated explicitly and all memory was recovered. Use
  narrow scheduled windows, disable shape/memory recording unless required,
  or add the CUDA-event buckets proposed below. Do not profile a complete
  warmup-plus-iteration with every operator event.
- `benchmarks/workloads.yaml` has drifted from the current CLI: it references a
  missing `configs/benchmarks/nlhe_hu_short.yaml` and retired `--n-trees` and
  `--node-budget` arguments. Repair that manifest before treating it as an
  automated performance gate. Current bounded commands must use
  `--roots-per-gpu` plus the schema's byte-admitted storage controls.
- The benchmark correctly releases CUDA storage, but process setup and teardown
  are a material fraction of tiny runs. Keep warmups and at least three timed
  repeats, and report per-decision timing alongside iteration totals.

### Ranking corrections from the sweep

1. Preserve 32 rows as the measured steady-state floor and treat 16/8 rows as
   tail graph shapes. Stable compiled/captured shapes remain high ROI.
2. Do not prioritize a wider active-trajectory window until a production-width
   profile shows it increases full postflop packs; 32 and 64 were effectively
   tied here while 64 used more memory.
3. Street-specific postflop compaction remains independent and high ROI: the
   sweep changed call pressure but left padded postflop attention work near
   one billion pairs per sample.
4. Add lightweight component CUDA timing before attempting another general
   operator trace. The profiler failure demonstrates that instrumentation must
   itself be bounded.

## Fixed-$\beta$ branching benchmark

Measured on 2026-07-21 on one idle RTX 2080 Ti using checkpoint 1110 from
`runs/nlhe_hu_32bb_full_sequential_rb` (SHA-256
`bced881945512f2a4152d3acd88a0a06395a00130cdf68a61aa2652bc1b5aebb`). The
trained trunk was loaded exactly. Because that sequential checkpoint has no
T/C branching heads, the 32 missing head tensors were newly initialized. This
is appropriate for compute and memory benchmarking, but their numerical loss
or gradient behavior is not a trained-model result.

The geometry benchmark used fixed $\beta=2.3$, one rooted tree, 64-row family
batches, a hard capacity of 12,000 nodes, a fixed-mode upper cutoff at 2,000
nodes, and an 8.5 GB measured peak gate. Once the upper cutoff activates,
further multiplicity is suppressed to one. The capacity prevents allocation
beyond the finite analytical plan; placing the upper cutoff well below that
capacity leaves room for already queued paths to close. The two tiny trees are
excluded from the analysis below.

Across the six relevant trees:

| Metric | Median | Range |
|---|---:|---:|
| Nodes | 3,756 | 2,948–4,416 |
| Decisions | 2,308 | 1,701–2,817 |
| Leaves | 1,077 | 1,055–1,108 |
| Rollout time | 24.18 s | 17.28–29.22 s |
| Time per decision | about 10.4 ms | about 10.2–10.6 ms |
| Peak allocated VRAM | 3.48 GiB | 3.47–3.49 GiB |
| Peak reserved VRAM | 6.13 GiB | 6.11–6.13 GiB |

A representative 4,494-node/2,866-decision tree took 29.71 seconds. It made 48
postflop model calls for 2,862 rows; 43 calls were full 64-row packs. Queue fill
is therefore already good in branching mode. The same tree made 73 evaluator
calls for 4,493 posterior rows. Its 147 scheduler control synchronizations moved
only 2,952 bytes and consumed 2.8 ms in total, so scheduler synchronization is
not an optimization target.

The important geometry is postflop width. Of the representative tree's
decisions, 0.1% were preflop, 2.9% flop, 24.8% turn, and 72.1% river. Nevertheless,
all postflop rows used the flop-width 2,554-token physical layout. The rollout
therefore executed 18.67 billion padded attention pairs, or essentially one
full flop-width square per postflop decision.

A bounded end-to-end diagnostic used the same checkpoint and $\beta$, a 1,000
node upper cutoff inside the 12,000-node capacity, 64 completion samples, and
one requested optimizer step. It produced 2,591 nodes, 1,627 decisions, and 557
leaves:

| Phase | Seconds | Share of 11.38 s profiled iteration |
|---|---:|---:|
| Rollout | 10.316 | 90.7% |
| Completion + branched targets | 0.316 | 2.8% |
| Seal/compact storage | 0.037 | 0.3% |
| Learner/reference/optimizer path | 0.707 | 6.2% |

Peak allocation was 3.27 GiB and peak reservation was 5.78 GiB. Repeating the
same tree with 16 rather than 64 completion samples changed measured completion
time only from 0.192 to 0.196 seconds. This does not establish scaling to the
production 1,024 samples; it only shows that small-$n$ completion is not the
present bottleneck. The optimizer correctly skipped its update after the
untrained T/C heads produced a nonfinite reduced gradient, so the learner timing
is useful as a shape/compute diagnostic, not as a training-valid result. Every
CUDA process exited and all three GPUs returned idle.

### Branching-specific ROI additions

| Priority | Optimization | Evidence and likely payoff | Effort |
|---:|---|---|---:|
| B1 | Split postflop physical execution by street | 97% of representative decisions were turn/river but paid flop width; about 11–13% attention-pair reduction is available before other compaction | Medium |
| B2 | Segment child sampling by parent without expanding full policy/reach tensors | Every branched parent output is gathered into repeated child-sized structures before profile/action/reach work | Medium–high |
| B3 | Write branched decision caches directly by node id | Removes batch-list clones, `cat`, node sort, and edge remapping; reduces peak coexistence with dense reach storage | Medium |
| B4 | Chunk the remaining node-capacity tensors | Current reach, topology, diagnostics, and queues scale with the hard node budget rather than realized nodes | Medium–high |
| B5 | Make fixed-$\beta$ memory admission a first-class contract | Safety-critical; permits larger useful trees without risking allocator OOM | Medium |
| B6 | Fuse/capture fixed-shape posterior/action/transition blocks | 73 evaluator calls and 50 policy calls in one representative rollout; useful after B1/B2 establish stable shapes | High |
| B7 | Add cutoff-stratified branching performance baselines | Separates natural completion from cutoff closure and normalizes by decisions/nodes | Low |

## ROI ranking

| Rank | Optimization | Likely payoff | Effort |
|---:|---|---:|---:|
| 1 | Fuse rollout action sampling, selected log-probability, reach update, and retained old-policy output | High | Low–medium |
| 2 | Compile/capture fixed-shape rollout policy steps | High | Medium |
| 3 | Compile/capture learner forward + loss + backward | High | Medium |
| 4 | Compile and streamline reference-cache construction | High; 14.1 s ceiling | Medium |
| 5 | Improve postflop rollout batch fill | High | Medium |
| 6 | Remove hot-path validation synchronizations | Medium, very easy | Low |
| 7 | Overlap gradient communication with final backward | Up to 7 s/iteration | Medium–high |
| 8 | Precompute immutable learner/model inputs | Medium | Medium |
| 9 | Keep preflop reach in 169-class form | Medium/high memory and bandwidth gain | Medium–high |
| 10 | Double-buffer rollout graphs across CUDA streams | Medium | Medium–high |
| 11 | Eliminate the rollout→learner `empty_cache()` cycle | Possibly medium | Low–medium |
| 12 | Graph/fix-budget branched rejection sampling | High in branched mode only | Medium–high |
| 13 | Specialized sequential backward plus completion overlap | Small now | Low–medium |
| 14 | Capturable optimizer/scaler/finite gate | Small; ~1 s ceiling | Medium |
| 15 | Metrics, checkpoint, and telemetry overhead | Small | Low |

## 1. Eliminate duplicated rollout policy work

The current rollout performs this sequence in [rollout.py](/home/pablob/ai-game-learning/pbg/core/rollout.py:742):

1. Model forward.
2. Build legal probabilities and sample the action.
3. Execute poker transition.
4. Recompute the selected full-hand log-probability for reach update.
5. Recompute it again for old-policy retention.
6. Recompute legal events in several of those paths.

`reach_update()` even constructs a fresh policy adapter and clones the entire reach tensor in [rollout_backend.py](/home/pablob/ai-game-learning/pbg/games/poker/rollout_backend.py:893).

Make one fused result:

```text
policy_step(
    state,
    reach,
    logical_rng_key
) -> {
    action,
    selected_logp_by_hand,
    value_by_hand,
    next_state,
    next_owner,
    updated_reach
}
```

Then:

- Compute legal events once.
- Compose policy probabilities once.
- Produce both sampled-profile action and full-hand selected log-probability together.
- Use that same log-probability for reach and retained `old_logp`.
- Update the acting player’s reach row in-place.
- Write outputs directly into reusable/preallocated destinations.

This is the natural unit to compile or graph.

## 2. Forward rollout CUDA graphs

Your proposed boundary is good. I would keep the action sampling and poker tensor transition on the GPU initially, rather than returning every action to the CPU.

A graph per execution-family/bucket could contain:

```text
static state/reach/key inputs
    → reach normalization
    → public encoding and masks
    → model forward
    → legal policy composition
    → keyed action sampling
    → selected full-hand logp
    → reach update
    → poker transition/classification
    → tiny routing summary
```

The CPU would only:

- Select the queue/family.
- Copy node/state handles into the static input buffers.
- Replay the graph.
- Read a tiny routing/count block.
- Prepare the next queue batch.

Useful graph shapes:

- Preflop: 64 rows covers about 96% of current calls.
- Postflop: 64/32/16/8 buckets, possibly with masked dummy rows.
- Separate graphs for inference and reference-cache construction.

The graph should accept logical RNG keys as mutable static-buffer contents. That preserves schedule-independent draws without trying to capture Python/PyTorch generator state.

I would prototype this first with `torch.compile(..., fullgraph=True, mode="reduce-overhead")`. It may provide the pointwise fusion and CUDA-graph replay automatically. Move to explicit `torch.cuda.CUDAGraph` if graph breaks or compilation produces too many variants.

### CPU poker logic caveat

The recorded scheduler moves only about 13 KB device-to-host per iteration and spends roughly 9 ms synchronizing. Moving poker transitions to CPU would add hundreds of strict GPU→CPU→GPU dependencies.

It could still win if Nsight shows that poker transition/legal-action kernels create substantial bubbles, but it should be an A/B experiment. The safer first design is:

- CPU owns queue decisions.
- GPU owns profile/action sampling, reach arithmetic, and batched poker transition.
- GPU returns only route counts or compact descriptors.

## 3. Backward CUDA graphs

The learner is already graph-friendly:

- Parameters have stable addresses.
- Gradients write directly into one stable flat fp32 buffer.
- Physical shapes are drawn from a tiny fixed set.
- Reference outputs are cached.
- No dropout is used.

The recent production geometry uses only a handful of physical shapes—typically preflop 64/43 and postflop 12/9. Capture one graph per partition/shape:

```text
static sampled-row indices
    → gather sealed inputs
    → current model forward
    → cached-reference lookup
    → fused PPO loss
    → scaled backward into flat gradient buffer
```

Important details:

- Zero the flat gradient buffer once before the step, outside the subspan graph.
- Do not capture zeroing inside each subspan, or later replays will erase accumulated gradients.
- Sampled indices change by copying into a static index buffer.
- Keep all-reduce and optimizer update outside the graph initially.
- Put the loss scale in a stable device scalar if needed.
- Avoid capturing metrics, validation, Python dataclasses, or CUDA-event creation.

Forward + backward currently totals about 66 seconds per iteration. Graphing cannot remove SDPA FLOPs, but even a 15–25% reduction would be material.

## 4. Reference-cache construction

This is an unexpectedly large target: about 14.1 seconds per iteration.

The cache is scientifically useful, but its build is still hundreds of no-grad forward calls. Apply the same fixed-shape compiled/graph path to cache construction.

Additional opportunities:

- Precompute model-ready immutable public fields and masks before building it.
- Cache only the reference policy information actually required by KL, rather than every `PolicyOutput` field.
- Avoid value/moment head work if reference KL does not consume those outputs.
- Use fixed 64-row preflop and calibrated postflop graph buckets.
- Reuse static graph workspaces across cache partitions.

The reference inputs change each iteration, so the cache cannot generally persist across iterations even when the reference model is unchanged.

## 5. Improve postflop batch filling

Current rollout averages:

- 9,130 postflop rows.
- 221 postflop model calls.
- Only 96 calls are full 64-row batches.

The theoretical minimum is about 143 calls. There are roughly 78 avoidable postflop forward launches per iteration.

Ways to improve:

- Increase simultaneously active trajectories if memory permits.
- Let postflop work accumulate longer before forced draining.
- Bucket partial postflop work into graph shapes instead of exact tail sizes.
- Reuse fixed 64/32/16/8 buffers.
- Reduce reach/storage memory first so a wider active trajectory window becomes safe.

Postflop also constitutes about 96% of rollout padded attention-pair work. Physical street bucketing could help further: the logical family can remain `post_2`, while flop/turn/river are packed into different physical compact widths. The current path explicitly pads turn and river to flop width in [rollout_backend.py](/home/pablob/ai-game-learning/pbg/games/poker/rollout_backend.py:1352).

## 6. Remove graph-blocking synchronizations

Several validations still read CUDA results synchronously inside repeated execution:

- Learner partition street/live-player checks.
- Flat-gradient finiteness every optimizer step.
- `float(grad_norm)` every optimizer step.
- Some model shape/seat assertions.
- Repeated flat-gradient alias validation.

For immutable sealed partitions, validate the full partition once before optimization, then trust subspan gathers. Replace repeated CUDA `bool(...)` checks with:

- One-time validation.
- Asynchronous device assertions in debug/acceptance mode.
- No hot-path check in production throughput mode.

The finite-gradient and norm reads can at least be consolidated into one small device result and one host synchronization per optimizer step.

## 7. Overlap all-reduce with final backward

The single 44.5 MB flat SUM is clean but fully exposed, costing about 7 seconds per iteration.

Partition the flat gradient buffer into stable buckets. During the final physical subspan’s backward:

- Autograd hooks mark each bucket complete.
- Launch asynchronous SUM as soon as a bucket is complete.
- Let communication overlap the remaining backward kernels.
- Wait for all buckets before finite checking/clipping.

This preserves exact SUM semantics. It is preferable to reverting to ordinary DDP averaging.

Capture NCCL inside CUDA graphs only later; first establish overlap with ordinary asynchronous collectives.

## 8. Cache immutable model inputs

Every learner subspan currently reconstructs:

- Public encoding.
- Legal masks and raise targets.
- Board compatibility.
- Live-seat ordering.
- Compact hand mappings.
- Full `[P,H]` weights from natural live-player storage.

These are immutable for the sealed iteration. Precompute or lazily cache them once per stored decision, subject to memory admission.

A practical compromise is to cache compact integer/mask descriptors, not full neural embeddings. This both saves kernels and makes graph inputs simpler.

## 9. Compact preflop reach storage

Preflop has 169 classes but working reach is stored and updated at full width 1,326. Every preflop model call then reduces it back to classes using `scatter_add_` in [rollout_backend.py](/home/pablob/ai-game-learning/pbg/games/poker/rollout_backend.py:1303).

Keep preflop reach as `[P,169]` until the flop transition, then expand once to `[P,1326]`.

Benefits:

- Roughly 7.8× less preflop reach memory/bandwidth.
- No repeated 1,326→169 scatter reduction.
- Cheaper reach graph.
- More active trajectories, improving postflop batch fill.
- Smaller retained preflop decision inputs.

This is structurally valuable but touches storage and transition contracts, so it needs more correctness work than graph capture.

## 10. Double-buffer rollout execution

Once graphs exist, use two static workspaces:

```text
stream 0: model/attention graph for batch N+1
stream 1: action/reach/poker postprocessing for batch N
CPU: prepare routing decision for batch N-1
```

The model is read-only throughout rollout, and different trajectories are independent, making this plausible. Queue/state writes need explicit events to prevent address reuse hazards.

This is likely more useful than trying to run two heavy model forwards concurrently.

## 11. Stop flushing the allocator between phases

`torch.cuda.empty_cache()` fixed an admission bug, but it forces later phases to recreate allocator segments and may add an unprofiled synchronization.

Improve the workspace calculation so it distinguishes:

```text
driver-free bytes
+ PyTorch reserved-but-inactive reusable bytes
- fragmentation/reserve allowance
```

Then retain reusable blocks across rollout, targets, and learner. Measure before changing it; allocator reuse depends on whether rollout blocks can satisfy learner allocation shapes.

## 12. Branched rejection sampling

For branched mode, graph capture is attractive only if rejection work has a bounded static schedule.

A data-dependent rejection loop cannot be captured directly. Convert it into:

- Fixed attempt blocks.
- Masked acceptance.
- Compact survivors between graph replays.
- CPU reads only accepted/remaining counts.
- Repeat the same attempt graph if necessary.

Since the sibling evaluator owns posterior sampling, this must be implemented through its supported interface; the evaluator repository remains read-only.

## 13–15. Lower-priority leaks

- Sequential value propagation can use a direct per-trajectory carry instead of the generic frontier, but only about 0.4 seconds remains after the validator fix.
- Completion batch $N+1$ can overlap value reduction for batch $N$; total ceiling is similarly small.
- Optimizer/scaler capture has only about a one-second total ceiling.
- Per-subspan CUDA timing currently creates thousands of event objects. Aggregate timing or enable it only for profiled iterations.
- JSON metrics and checkpoint writes can be moved off the critical path, but their amortized contribution is small.
- Thermal throttling is real on the 2080 Tis, but reducing wasted work is the actionable fix; fan privileges are unavailable.

## Branching implementation details

### B1. Street-specific postflop physical families

Keep the logical sampling family `post_2`, but split its physical packs into
flop, turn, and river buffers. The current family forward always sets
`compact_size=comb(n_cards - 3, 2)`, even for turn and river. For a full deck,
the compatible two-card domains are 1,176 on the flop, 1,128 on the turn, and
1,081 on the river. With two live players, moving from flop to turn/river width
removes the duplicated player-token rows twice.

The observed street mixes imply about an 11–13% reduction in padded attention
pairs if nothing else changes. This should precede CUDA graph capture because
it defines the useful stable graph shapes. The logical RNG address and tree
semantics remain unchanged; only physical packing changes.

History length can become a secondary physical bucket later. Its ceiling is
small relative to the two compact hand domains, so it should not delay the
street split.

### B2. Segmented parent-to-child expansion

`expand()` currently creates `parent_positions` and then materializes repeated
child-sized states, reaches, and every tensor in `PolicyOutput`. For
$N(u)\in\{2,3\}$, all children of parent $u$ share the same public state and
policy output until their keyed posterior/profile and action draws diverge.

Implement a segmented primitive with parent rows, offspring counts, and child
ordinals:

```text
parent output + parent reach + child RNG keys
    → child profiles/actions
    → selected logp by child
    → next state/reach/owner
```

The primitive should index the parent output inside the sampling kernel rather
than first gathering complete child copies. It can also produce child path keys
and parent-local edge ordinals directly. This is the branching counterpart of
the generic fused rollout policy step in item 1.

### B3. Direct cache writes instead of list/cat/sort

Branched rollout retains cloned reach batches in `policy_node_reaches`, cloned
policy outputs in several more Python lists, and then concatenates and sorts
them by node id at finalization. Edge log probabilities are similarly rebuilt
through a dense node-to-row map and an append-order-to-CSR scatter.

Reserve cache rows once and write them by stable node id or assigned decision
row as each batch expands. Store edge data at its already-known parent-local
ordinal. This removes the final `cat`/sort/scatter wave and, more importantly,
avoids coexistence between list-owned cloned decision reaches and the full
`reach_rows` table. Preflop rows should use the 169-class representation from
item 9.

### B4. Chunk capacity-proportional branching state

The occurrence arena is chunked, but `DeviceTreeBuffer`, `reach_rows`, the T/C
diagnostic arrays, path-factor arrays, and every family queue are allocated at
the complete node budget. That makes a conservative safety budget expensive
even when the realized tree is much smaller.

Move the largest remaining arrays—starting with reach and queue payloads—to
admitted slabs. Grow only after checking the next slab plus the reserved
completion/learner envelope. Stable logical node ids can remain flat while a
small slab table resolves physical storage. Keep a modest number of fixed slab
sizes so allocator fragmentation does not replace capacity waste.

### B5. Fixed-$\beta$ OOM safeguard

Fixed mode must use the same quality of memory control as dynamic mode. The
benchmark path now accepts an explicit fixed $\beta$ and
`branching_upper_cutoff_nodes`. Across the geometry and full-iteration probes,
the safeguards were the analytical plan/preflight, hard node capacity, hardware
reserve, measured peak gate, and teardown check.

For production, make the cutoff part of the validated rollout schema rather
than a tool-injected attribute. Admission should account for:

- Current realized nodes.
- At least one closing descendant for every queued/frontier occurrence.
- The next offspring block before it is materialized.
- Retained completion, target, reference, gradient, and optimizer envelopes.
- Driver reserve and allocator fragmentation slack.

When the forecast crosses the operating ceiling, atomically set future
multiplicity to one before allocating children. If even unit-multiplicity
closure cannot fit, fail before allocation with geometry and memory evidence;
never discover the condition through a CUDA OOM. Record the first cutoff node,
suppressed children, pending frontier, and whether the sample is censored so
performance and scientific reports cannot silently mix regimes.

### B6–B7. Capture only after stratified measurement

After B1/B2, capture fixed street/batch blocks for posterior sampling, action
sampling, transition, reach update, and cache writes. Preserve logical RNG keys
as mutable graph inputs. A fixed attempt budget with masked survivors is still
required for evaluator rejection work as described in item 12.

Report naturally completed and cutoff-closed trees separately. For each,
record seconds per node/decision, street mix, full-pack rate, evaluator calls,
padded token pairs, and peak allocated/reserved bytes. The measured branching
queue already filled 43 of 48 postflop calls to 64 rows, and scheduler control
took only 2.8 ms, so neither extra queue accumulation nor scheduler rewrites
should precede B1–B4.

## Recommended implementation order

I would run the next optimization session in this order:

1. Add bounded aggregate CUDA-event buckets inside rollout for model, policy
   composition, reach update, poker transition, chance, and queue work. Reuse
   fixed instrumentation and enable it only for a short scheduled window; do
   not retain every operator or create thousands of event objects.
2. Fuse legal-policy/action/logp/reach work into one rollout-step function.
3. Compile that function for pre-64 and post-64/32/16/8.
4. Capture learner forward/loss/backward for its four observed physical shapes.
5. Graph reference-cache construction.
6. Remove repeated validation synchronizations.
7. Improve postflop queue fill.
8. Overlap gradient buckets with final backward.
9. Then consider compact preflop reach and multi-stream pipelining.

For branching mode specifically:

1. Promote the fixed-$\beta$ cutoff and memory admission into validated production configuration.
2. Split postflop physical packs by street and remeasure padded attention work.
3. Implement segmented parent-to-child sampling without repeated full outputs/reaches.
4. Replace branched list/cat/sort caches with direct indexed writes.
5. Chunk the remaining budget-sized reach/queue/topology tensors.
6. Capture the resulting fixed street/batch shapes and evaluator attempt blocks.

The relevant execution loops are [rollout.py](/home/pablob/ai-game-learning/pbg/core/rollout.py:698) and [learner.py](/home/pablob/ai-game-learning/pbg/core/learner.py:1276); gradient synchronization is in [optimization.py](/home/pablob/ai-game-learning/pbg/core/optimization.py:113). The existing optimization progression is documented in [nlhe_hu_32bb_optimizer_speed_record_20260717.md](/home/pablob/ai-game-learning/docs/audit/nlhe_hu_32bb_optimizer_speed_record_20260717.md:1).
