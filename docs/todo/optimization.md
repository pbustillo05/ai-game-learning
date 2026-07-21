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

So CPU scheduler control is already negligible. The leaks are GPU launch overhead, duplicated tensor work, poor postflop packing, immutable preprocessing, and exposed communication.

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

## Recommended implementation order

I would run the next optimization session in this order:

1. Add CUDA-event buckets inside rollout for model, policy composition, reach update, poker transition, chance, and queue work.
2. Fuse legal-policy/action/logp/reach work into one rollout-step function.
3. Compile that function for pre-64 and post-64/32/16/8.
4. Capture learner forward/loss/backward for its four observed physical shapes.
5. Graph reference-cache construction.
6. Remove repeated validation synchronizations.
7. Improve postflop queue fill.
8. Overlap gradient buckets with final backward.
9. Then consider compact preflop reach and multi-stream pipelining.

The relevant execution loops are [rollout.py](/home/pablob/ai-game-learning/pbg/core/rollout.py:698) and [learner.py](/home/pablob/ai-game-learning/pbg/core/learner.py:1276); gradient synchronization is in [optimization.py](/home/pablob/ai-game-learning/pbg/core/optimization.py:113). The existing optimization progression is documented in [nlhe_hu_32bb_optimizer_speed_record_20260717.md](/home/pablob/ai-game-learning/docs/audit/nlhe_hu_32bb_optimizer_speed_record_20260717.md:1).