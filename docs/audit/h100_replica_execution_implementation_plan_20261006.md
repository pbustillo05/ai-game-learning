# H100 replica training implementation plan

Date: 2026-10-06

Status: implementation specification, revised to the user's single-pipeline
and shared-model requirements. This document edit changes no production code.

## Binding implementation contract

An instruction to **implement this document** means complete the entire
in-place implementation, integration, cleanup, and locally executable validation
below in one workstream. Do not stop at a prototype, a new optional backend,
an isolated learner benchmark, or a list of follow-up implementation tasks.
Stages are internal construction order, not separately selectable deliverables.
Only hardware-dependent H100 validation and final tuning wait for the rental;
genuine permission/resource blockers must be reported, never disguised as a
successful handoff.

The following requirements override older alternatives elsewhere in the cited
audits:

1. **One modular production pipeline.** Modify the existing learner, model,
   optimizer, replica owner, and queued integration in place. Statistical modes
   select their declared process, fields, objectives, and sampling laws; they
   do not select old versus new trainers. K=1 is the singleton case of the same
   implementation. No legacy trainer, split-network model, compatibility
   facade, demoted module, unused helper, or permanent comparison implementation
   remains after cutover. Git history is the recovery mechanism.
2. **The queued design is the storage and algorithm contract.** Preserve
   depth-organized LayerWork, preallocated fixed-address SoA queue slabs,
   family FIFO rollout batches, sample/commit ownership, topology, seal,
   completion, target recurrences, and publication. This is an under-the-hood
   PyTorch/CUDA execution refactor, not permission to redesign those systems.
3. **The fused pre/post model is mandatory and exclusive.** Replace separate
   trunks/readouts with one registered shared trunk and policy/value readout
   per replica, plus the necessary family-specific encoders/adapters. Use one
   configurable shared depth, initially six blocks for the H100 configuration.
   No architecture selector or old split-model fallback survives. Sharing is
   across streets within a run, never across independent replicas.
4. **Complete before rental.** Ship the shared model, batched forward/backward,
   optimizer, graph ownership, every existing production integration, strict
   new-format persistence, cleanup, tests, and runnable H100 harness together.
   The rental is for running, repairing, profiling, and optimizing that complete
   pipeline, not for implementing deferred core modules.

Small production-used modules are appropriate where they reduce duplication;
do not create parallel `*_v2`, `*_legacy`, `*_fast`, or H100-specific learners
or whole-model definitions. Eager, compiled, and captured execution invoke the
same tensor program. Hardware-specific primitive implementations are allowed
behind one operation contract when required by supported devices. Tests and
benchmark tooling are legitimate non-production consumers, but may not shelter
copies of the retired pipeline.

### Authority and the narrow approved changes

Read every current file in [docs/queued/design](../queued/design/README.md)
before implementation, especially the storage/lifecycle and layered-target
contracts. Preserve their fundamental design. Resolve these specific older
clauses explicitly rather than silently ignoring them:

- `features_to_preserve.md` currently requires separated pre/post parameters.
  The user's mandatory shared-model decision supersedes that one requirement;
  other preservation requirements remain binding.
- `storage_redesign.md` section 5.2 prescribes lane-wise stream execution and
  prohibits summed replica losses. Replace only that execution prescription
  with explicit-K projections and the mathematically independent scalar-loss
  sum described here. Preserve exact sampled rows/action spans, immediate
  backward, FP32 ownership, synchronous logical model versions, and bounded
  workspace. No padded actions or invented learner rows are introduced.
- The September shared-model assessment proposed retaining split architecture
  and old-checkpoint support. Those alternatives are rejected here. Keep its
  representation, ownership, gradient, and adapter requirements.

As part of implementation, reconcile those narrow statements in the queued
design and shared-model assessment, and update active usage/configuration docs
to the sole production path. Do not rewrite the queued design around a new
runtime. Historical audit evidence need not be deleted or mistaken for current
instructions. The README references an absent `implementation_plan.md` in this
checkout; read it if present later, otherwise use the actual design contracts
and this implementation sequence, not an invented missing specification.

The user has selected the shared architecture. Repository rules still require
separate explicit approval before editing `docs/theory/MHA_model.md` or
`docs/theory/core_math.tex`. Do not infer that file-edit approval from this plan;
record the narrowly authorized model change here and request permission if
theory synchronization is undertaken. Do not postpone the approved production
model change or retain the split model on that account.

### Queued invariants that the execution rewrite must prove

| Boundary | Required invariant |
| --- | --- |
| Modes and owners | Preserve the twelve `training_regime × rollout_mode × private_estimator` schema combinations; replicas are independent lanes, not another statistical mode |
| Rollout storage | Fixed-address, preallocated lane-local SoA owners; unchanged hot/retained lifetimes and bounded workspace; no alternate replay database |
| Family scheduling | Only `pre` and `post_A` input families; stable FIFO prefixes, no skipping rows to improve batching, no new street/width queues; preserve history/board-compatible token packing |
| Model → sample → commit | Same RNG addresses, profiles, labels, reservations, dependencies, atomic tail publication, and failure/retry semantics; inactive lanes do no dummy rollout work |
| Seal and topology | Same compact topology, distinct duplicate-child identities, protected root prefixes, trained-row alignment, and last-consumer lifetimes |
| Completion and targets | Sibling contract-v2 evaluator ownership; existing stable Leaf addresses; reverse-depth LayerPlan; two LayerWork arenas and existing slot reuse; ordered per-layer CUDA propagation across replicas without per-layer host synchronization |
| Mathematical reductions | Same target recurrences, child order, post-to-pre mass/moment aggregation, statistical weights, sampling, and logical optimizer boundaries |
| Learner storage | Exact sealed sigma/action gathers, bounded transient workspace and immediate backward; no padded action evaluation, fabricated statistical rows, or retained activation database |

Add tests at these boundaries using fixed model outputs/actions where necessary:
sharing weights intentionally changes predictions and subsequent trajectories,
but it must not change the queue/target algorithm given identical numerical
inputs and logical RNG addresses.

## Decision and first principles targets

Refactor the existing CUDA learner to treat independent experiments as an explicit
replica axis. Execute compatible work step by step across replicas, accumulate
independent gradients, and perform one device-side optimizer transition with
separate state for every replica. Use fixed physical shapes, compile bounded
tensor regions, and replay them with CUDA Graphs. Replace the model with the
mandatory shared pre/post architecture; keep the sampling law, logical batch
size, objective, queued algorithm, and update schedule unchanged.

Build one explicit replica-native execution path: batched GEMM projections,
packed QKV, ordinary fused attention over the combined replica/presentation
batch, compiled feature packing, an independent device-side optimizer, and
CUDA Graph replay. K=1 uses the same implementation with a singleton replica
axis. Correctness compares this same shared model at K=1, K>1, and
eager/compiled/captured execution; retain no separate serial learner.
Do not build competing vmap, stream-wave, and grouped-GEMM trainers.

Complete the implementation, local CUDA correctness coverage, graph lifecycle,
new-format checkpoint round trips, memory accounting, and automated RunPod harness before
renting the H100. The rental is for hardware-specific validation, debugging,
and optimization of this design. Source fixes and targeted kernel tuning on
the rental are expected; an entire new architecture should not be necessary.

The delivery is one integrated implementation, not a sequence of independent
optimization experiments. Local development may use small correctness tests
and incremental commits, but the pre-rental handoff includes the complete
replica learner and its pipeline integration. On RunPod, tune that runtime and
fix its measured bottlenecks rather than choosing among competing trainers.

Plan from hardware and code-derived work counts, not historical runtime
fractions. H100 SXM offers approximately 17–18x the RTX 2080 Ti's dense
mixed-precision Tensor throughput with FP32 accumulation, about 5x scalar FP32
throughput, and 5.4x memory bandwidth. These are separate resource ratios, not
multipliers. For an efficiently supplied, compute-heavy learner, roughly
8–15x cross-device throughput is a defensible conditional planning band, with
Tensor-dominated sections potentially approaching the 17–18x arithmetic ratio.
This is not a promise for the complete poker pipeline.

The initial absolute learner design target is 150–350 useful major-matrix
TFLOP/s on H100 SXM. For the six-block shared architecture and an explicitly assumed
40% postflop presentation mix, that corresponds to roughly 2,000–5,000 learner
presentations/s. This is an engineering utilization target, not a rate implied
by specifications alone. The final section derives the work and assumptions.

Speedup over today's inefficient implementation is a different quantity. It
could exceed the hardware ratios substantially, but first principles cannot
assign it a number without that implementation's achieved throughput. The
previous 2.6–3.5x whole-pipeline forecast is withdrawn as a target for this
redesign. It assumed unchanged phases and was not an H100-versus-2080-Ti estimate.

Every performance report must name its denominator: optimized H100 versus
optimized 2080 Ti, new H100 versus old H100, or new H100 versus existing 2080 Ti.
Report absolute learner presentations/s alongside the ratio, and distinguish
aggregate replica throughput from individual-run latency and fresh-node rate.

This revision resolves the prior H100 proposals as follows:

| H100 audit document | What this design retains or corrects |
| --- | --- |
| [Idea 1](h100_idea_1.md) | Explicit replica state, BMM, SDPA, independent optimizer, graph replay; replace width-256 estimates with the actual architecture |
| [Idea 2](h100_idea_2.md) | Shared semantics, readout selection, absolute FLOP targets; do not silently adopt its no-incidence architecture or 384/2688 token counts |
| [Idea 3](h100_idea_3.md) | Outer backward, step-major scheduling, state lifetimes; do not use its session-peak activation slope for admission |
| [August consolidated plan](h100_plan.md) | Explicit BMM, full-width families, correctness gates, offline preparation; replace its many-executor matrix with one implementation |
| [August learner plan](h100_learner_optimization_plan_20260803.md) | Runtime/dispatch gates and separation of numerical versus execution changes |
| [August execution changes](h100_normal_rb_execution_changes_20260802.md) | Exact pooled sampling and logical update boundaries |
| [July Quest plan](quest_h100_rb_vs_normal_plan_20260724.md) | One process batches independent experiments and reuses batching for rollout/reference work |
| [July launch blockers](queued_issues/quest_h100_launch_blockers_20260729.md) | Dtype/capacity checks, one real learner pass, rollback, progress; not the provisional stream-only prescription |

## Evidence and current implementation

Historical results explain defects and compatibility history. They are not
inputs to the hardware forecast or absolute throughput target.

The retained [H100 optimization archive](../../runs/h100_learner_optimization_20260803/h100_learner_optimization_20260803.tar.gz)
contains `h100_learner_optimization_20260803/RESULTS.md` and
`audit/h100_learner_optimization_20260803.md`. Its full workload is 4,096 roots,
116 optimizer steps, and 256 logical presentations per step.

| Historical measurement | Normal | Rao–Blackwell |
| --- | ---: | ---: |
| Modern-runtime complete iteration | 42.41565 s | 43.13239 s |
| Learner event median | 34.74880 s | 35.79665 s |
| Peak allocated memory | 24.14 GiB | 32.50 GiB |
| Peak reserved memory | 50.85 GiB | 59.32 GiB |
| Older-runtime complete iteration | 69.2 s | 71.3 s |

These are historical measurements, not timings of current HEAD `3870801`.
The archive used PyTorch 2.12.1+cu130 and freshly built SM90 extensions. Its
retained instrumentation/runtime changes must not be assumed to all exist on
the current branch. Rebaseline the exact implementation revision before
reporting a new speedup.

A short Nsight learner range lasted 3.215 seconds, launched 25,573 kernels,
and had 34.12% unioned GPU-kernel activity. The remaining 65.88% had no active
GPU kernel. The device work included elementwise operations, GEMMs, Flash
forward/backward, concatenation, normalization, and reductions. Production
attention already used cuDNN SM90 Flash kernels, with representative shapes
`[B,4,535,32]` and `[B,4,2854,32]`.

Driver sampling showed low Tensor activity but also transient high SM activity.
Those samples perturbed timings and do not establish model FLOP utilization or
per-kernel occupancy. Nsight Compute counters were unavailable because of
`ERR_NVGPUCTRPERM`. Do not change shared driver policy to obtain them.

Matched smaller K=2 runs showed no learner overlap: aggregate learner throughput
was 1.0195x for Normal and 1.0948x for RB. Existing K>1 behavior is not evidence
against genuine replica batching.

Current implementation boundaries are:

- [Replica ownership and inference](../../pbg/core/replica.py): named stacked
  parameters exist, but inference batching does not imply batched learning.
  Lane modules wrap aliased slices in distinct autograd leaves.
- [Learner transaction](../../pbg/core/training.py): prepares lanes, then runs
  each lane's entire learner schedule serially.
- [Learner operation and sampling](../../pbg/core/learner.py): combines model,
  loss, checks, and backward; partitions pooled random indices using
  data-dependent Boolean selection.
- [Dense optimizer](../../pbg/core/optimization.py): obtains finite and norm
  values on the host for each lane update, forcing synchronization.
- [Poker model](../../pbg/games/poker/model/preflop.py) and
  [postflop model](../../pbg/games/poker/model/postflop.py): substantial feature
  packing, residual, incidence, and readout work surrounds fused attention.
- [Learner adapter](../../pbg/games/poker/queued/model.py): Normal computes full
  private-row heads before selecting the sampled private state.

The [replica parity audit](replica_learner_parity_and_rb_determinism.md) explains
the previous failure: duplicate learner semantics, loss-control dtype drift,
and repeated-index BF16 backward accumulation in preflop RB. Preserve the
corrected class-space RB formulation. Reintroducing a second loss implementation
would recreate the original risk.

## Why the H100 advantage is currently hidden

### Temporal gaps and spatial underutilization

There are two different inefficiencies. Temporal underutilization means the
GPU has no kernel to execute. Spatial underutilization means an executing
kernel does not effectively use all available SMs or their execution units.
The trace directly establishes large temporal gaps. Small grids, narrow
matrices, memory dependencies, and launch overhead make spatial inefficiency
plausible, but it must be diagnosed kernel by kernel.

An SM can be active while most Tensor Core capacity is unused. High occupancy
also does not imply high throughput: resident warps may be waiting for data.
Conversely, an efficient kernel need not maximize occupancy. Report useful
work per second, active intervals, executed instruction types, memory traffic,
and stall reasons rather than a single utilization percentage.

### Hardware improvements apply to different portions of the work

| Resource | RTX 2080 Ti reference or Founders Edition | H100 SXM | Ratio |
| --- | ---: | ---: | ---: |
| SMs | 68 | 132 | 1.94x |
| FP32 CUDA cores | 4,352 | 16,896 | 3.88x |
| Scalar FP32 peak | 13.4–14.2 TFLOP/s | 67 TFLOP/s | 4.7–5.0x |
| Dense FP16 Tensor peak with FP32 accumulation | 53.8–56.9 TFLOP/s | About 989 TFLOP/s | 17.4–18.4x |
| Dense BF16 Tensor peak with FP32 accumulation | Not native | About 989 TFLOP/s | No native like-for-like ratio |
| Memory bandwidth | 616 GB/s | 3,350 GB/s | 5.44x |
| Memory capacity | 11 GB | 80 GB | About 7.3x |

RTX figures are from NVIDIA's [Turing architecture whitepaper, Table 1](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf),
including separate FP16- and FP32-accumulation rows. H100 resource counts are
from the [Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/);
use final [H100 product specifications](https://www.nvidia.com/en-us/data-center/h100/)
for throughput. The published 1,979 TFLOP/s BF16/FP16 figure includes sparsity
and must be halved for dense computation.

The 2080 Ti's often-quoted 107.6–113.8 Tensor TFLOP/s is for FP16 accumulation,
not the FP32-accumulation training row above. Comparing H100 BF16 Tensor
throughput with 2080 Ti scalar FP32 throughput also mixes execution paths.
The matched Tensor comparison supports ambition beyond 3x. The SM count alone
does not: H100 has approximately twice as many SMs, and each generation's SM
and Tensor Core have different width.

Turing supports FP16 Tensor operations, whereas native BF16 Tensor support
arrived with Ampere. Consequently, an H100 BF16 versus 2080 Ti comparison
must identify the local precision, accumulation mode, and actual attention
backend. Do not compare an H100 dense training number with a Turing inference
or FP16-accumulation headline. See NVIDIA's
[Turing guide](https://docs.nvidia.com/cuda/turing-tuning-guide/) and
[Ampere format description](https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/index.html).

More Tensor compute cannot accelerate host waits or turn concatenation,
indexing, normalization, or Adam into matrix multiplication. More HBM bandwidth
does not help while the GPU is waiting for the next launch. Extra capacity
helps when it admits useful concurrent work, static buffers, and better cache
lifetimes; filling memory with replicas is not itself a performance objective.

For illustration, if an older GPU spends 80% of a job in hardware-independent
overhead and 20% in GPU work, even making that GPU work 10x faster produces
only $1/(0.8+0.2/10)=1.22$x overall. This is an explanatory example, not a
measurement of the RTX 2080 Ti runs.

### Narrow layers can remain bandwidth limited

For a tall BF16 matrix multiplied by a resident 128 by 128 weight matrix,
the idealized arithmetic intensity based on reading inputs and writing outputs
is about 64 FLOP/byte. The H100 dense BF16 peak-to-HBM ratio is about
295 FLOP/byte. If those activations actually travel to HBM, this layer cannot
reach peak Tensor throughput merely by increasing its row count.

Cache residency and fusion can change that bound. Replicas add independent
tiles and improve scheduling but do not share weights or automatically increase
per-layer arithmetic intensity. Benchmark forward, input gradient, and weight
gradient separately; their tile counts and reduction dimensions differ.
The relevant mechanisms are described in NVIDIA's
[matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
and [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

## Scope and invariants

The primary performance target is single-GPU CUDA, full-deck heads-up 32 bb,
sequential Normal and RB training. This is not the compatibility scope:
preserve all existing statistical modes, valid game geometries, distributed
CUDA/NCCL training, best-response and variance workflows, and supported
CUDA/MPS inference through the same model and pipeline. Runs
may have independent weights, sampled data, seeds, and supported optimizer or
loss coefficients. Cohort only compatible architecture, estimator mode, model
family, and dtype. Hyperparameter independence does not imply that different
architectures must share a kernel.

Keep these invariants throughout implementation:

1. Each lane owns parameters, gradients, moments, counters, references, targets,
   sampling streams, metrics, and checkpoints. No cross-lane gradient reduction.
2. Each logical batch has exactly the existing sampled presentations and
   objective normalization. Hardware widths do not add sampled rows or actions.
3. Every lane completes logical step $s$ before using its updated parameters
   for step $s+1$. All family and microbatch contributions precede its update.
4. FP32 remains authoritative for master weights, gradients, moments, loss
   controls, policy arithmetic, and posterior reductions. BF16 is the initial
   H100 model-compute dtype.
5. Preflop RB stays in model-class space; no repeated-index BF16 expansion
   followed by backward accumulation.
6. Preserve dense zero-gradient semantics for unused parameters. This is not
   equivalent to leaving `.grad=None` and skipping their optimizer transition.
7. Preserve iteration-wide commit and rollback. No failed iteration publishes
   model state, counters, checkpoints, or run metrics as successful.
8. No CPU production execution, implicit backend selection, MPS training, or
   evaluator compatibility facade. Do not change `docs/theory/`.

Branched schedules, best-response ownership, multi-way `post_A` families, and
distributed collectives are integrations of the same runtime, not reasons to
keep an old trainer. Preserve rank-common synchronization and reduction for
matching replica IDs across ranks; never reduce different replica IDs together.
Unsupported device capabilities continue to fail clearly under AGENTS.md.
Performance tuning beyond the primary HU workload can follow on H100, but
functional coverage and removal of duplicate implementations cannot be deferred.

## Execution architecture

```text
lane owned rollout and targets
    -> exact sampled index schedule on CUDA
    -> bounded exact-size preflop and post_A family packs
    -> compatible replica cohort
    -> replica-batched model and authoritative FP32 loss
    -> sum of independent lane losses
    -> outer backward into lane owned gradients
    -> lane local finite check and norm
    -> one masked independent Adam transition
    -> iteration validation and transaction commit
```

The implementation should expose a small replica-runtime protocol rather than
make the training session aware of individual BMM and graph internals:

```python
# Proposed interfaces, not existing APIs.
prepare_schedule(prepared_lanes) -> ReplicaSchedule
pack_family(schedule, logical_step, family, chunk) -> PackedFamily
executor.accumulate(packed_family, controls) -> DeviceLossMetrics
optimizer.transition(enabled_lanes, failure_state) -> DeviceStepMetrics
transaction.validate_and_commit(device_status) -> IterationResults
```

`accumulate` immediately executes backward for one physical pack. It must not
retain graphs across microbatches or across logical optimizer steps. Only
parameters, optimizer state, static buffers, and detached metrics persist.

### Parameter and gradient ownership

Use flat FP32 `[K,Nphysical]` owners for master weights, gradients, and each
Adam moment. Named parameters are leaf views with contiguous matrix dimensions
and a lane stride spanning the physical parameter row. The flat storage
container is not an additional trainable leaf. Bind each named leaf's gradient
to its flat gradient view and test actual accumulation into that owner.
Remove obsolete per-lane autograd wrappers after callers use this owner. Any
necessary lane views are views of this implementation, not another learner.

Maintain a deterministic logical-to-physical parameter map for packed QKV and
zero alignment padding. Serialize the sole shared architecture's logical
parameters through that map in a versioned checkpoint. Reject old split-model
checkpoints clearly; do not merge weights or moments. Padding stays zero,
receives zero gradient, and contributes nothing
to clipping or optimization. Establish the layout once; do not repack FP32
master state each microbatch.

Autocast may create dense BF16 matrices from the strided views; count the cost
and lifetime. If backward needs a temporary gradient layout, preallocate and
account for it explicitly. Resolving layout is a bounded implementation issue,
not grounds for a competing trainer.

Full cohorts use views. Noncontiguous lane selection uses contiguous runs or
explicit lane-indexed primitives, not dummy model rows. Do not `index_select`
every model parameter per pack. Cohort formation cannot reorder a lane's
logical steps or mutate an inactive lane.

PyTorch's [stacked-state contract](https://docs.pytorch.org/docs/main/generated/torch.func.stack_module_state.html)
explicitly creates new optimizable leaves. Aliasing parameter storage does
not make separate leaf objects share `.grad` ownership.

### One authoritative loss boundary

Split `LearnerOperation.forward_loss_backward` into a tensor-only
`forward_loss` operation and one accumulation caller in the existing learner.
Preserve the externally required `LossResult` behavior; delete the replaced
combined implementation and dead callers rather than retaining a serial path.

Return scaled scalar loss plus detached device metrics. Move Python validation
of static configuration outside the compiled region. Replace hot data-dependent
host checks with device status accumulation and the failure protocol below.
Keep policy, entropy, KL, value, and objective controls explicitly FP32.

For lane $r$, the accumulated logical-step objective is

$$
L_r=\frac{1}{B_{\mathrm{logical},r}}
\sum_{f,c,i\,\mathrm{valid}} w_{r,f,c,i}\,\ell_{r,f,c,i}.
$$

The sum visits actual selected rows only, and $w$ denotes the existing objective
weights; it is not a new importance-sampling rule. Preserve all inner RB
posterior and action reductions before this outer presentation reduction.
The current physical-mean-times-objective-weight implementation must be shown
equivalent to this sum for each supported mode. Retain mode-specific stratum
and action-span normalization where required; the schematic outer sum does
not replace those formulas with a universal uniform weight.

For a physical pack, sum losses over replicas and call ordinary backward:

$$
\nabla_{\theta_r}\sum_j L_j=\nabla_{\theta_r}L_r.
$$

Do not average across replicas, divide by padded capacity, or clip before all
contributions to a lane's logical step have arrived. Empty families produce
zero contribution without passing an invalid zero objective weight through an
API that currently requires a positive scalar.

### Exact schedules and static packing

Implement fixed-step and supported shuffled-epoch sequential schedules and
the existing branched stratum/allocation schedules in the same learner.
The fixed-step sequential workload is the first development test, not the
delivered feature boundary. Generate per-lane random calls in the same order and with the
same shapes. Do not replace 116 separate random calls with one differently
shaped call and assume identical RNG consumption.

Store pooled index draws on CUDA. Partition them into preflop and postflop
index buffers while preserving the existing within-family order. Sampling
still occurs over the original pooled retained rows; never impose fixed
preflop/postflop quotas to make batching easier.

Use one bounded scheduling boundary to obtain small family counts if necessary;
CPU orchestration may plan launches but must not own model data or execute
target/loss work. Pack indices and gathered data on CUDA into preallocated
workspace, with exact active views and explicit counts instead of repeated
dynamic Boolean-index allocations in the hot loop.

Keep `pre` and `post_A` families (just `post_2` for HU). Do not subdivide
postflop by street or create padding-based queues. Preserve the design's
history/board-compatible token packing. Give each family a physical capacity
and split each lane's selected rows into chunks. Implement a default resident
target K=16, preflop capacity 32, and postflop capacity 8 per lane. Support
K=32 and one halved/doubled capacity around each default. These are starting
settings, not claimed SM90 optima. These are execution microbatch limits,
not changes to fixed queue-slab capacities or statistical sample allocations.

Use exact-size tail views; never invent rows, dummy actions, or statistical
mass to fill a graph. Batch compatible equal-shape lane chunks, allowing K=1
cohorts for unmatched tails. Capture recurring exact shapes with a bounded
cache. Rare shapes run the same tensor program without capture instead of
creating unbounded graph variants or a fallback learner. Zero alignment in
feature/matrix dimensions and the design's existing token masks are distinct
from forbidden padding of statistical rows or actions.

Branched packs retain `sigma_index`, `action_ptr`, and exact `action_index`
spans, with admission for both sigma count B and action count E. Batch the
model over compatible sigma shapes; evaluate actual flattened action spans
with lane/sigma indices and segmented reductions. Do not allocate or evaluate
a padded `[K,B,max_actions,...]` domain. Preserve sampler allocations, T/C
validity, reference weights, and logical-step grouping. Sequential epochs keep
their existing family update order; shared weights are not permission to
invent a new street mixture.

For each logical step, zero gradients once, visit family chunks in a fixed
order, accumulate immediate backwards, and then update once. Different
microbatch reduction orders can cause bounded floating-point differences;
sampling identity and objective scaling must nevertheless be exact.

### Replica model primitives and backward

Implement explicit replica-aware operations directly, including a singleton
K axis for K=1. Ordinary autograd and the compiler produce backward. No
`vmap(grad)` or alternate per-lane learner belongs in production. Test against
independent K=1 calls to this same shared-model code and small mathematical
primitive references; a separate functional-training prototype is not needed.
Modify the existing primitives to implement:

- Linear projection: `X[K,M,Din] @ W[K,Dout,Din].transpose(1,2)`.
- Embedding lookup with an explicit lane index into `[K,V,D]` tables.
- Layer normalization over the feature dimension only, followed by independent
  `[K,D]` affine parameters; never normalize across replicas.
- Packed QKV and output projections, every FFN, and policy/value heads. Q/K/V
  use the same normalized source here, so use one `[K,3D,D]` projection and
  canonical logical serialization instead of timed weight concatenation.
- Existing incidence contractions with lane-independent geometry broadcast
  appropriately and the replica/presentation dimensions folded when safe.

Ordinary autograd for BMM already provides

$$
dX_r=dY_rW_r,\qquad dW_r=dY_r^TX_r.
$$

There is no need to materialize a full Jacobian or write a whole-model manual
backward. Verify generated layouts and reductions using PyTorch's
[autograd definitions](https://raw.githubusercontent.com/pytorch/pytorch/main/tools/autograd/derivatives.yaml)
and the measured operator trace.

After replica-specific projection, reshape `[K,B,H,N,Dh]` to
`[K*B,H,N,Dh]` for ordinary SDPA. Preserve per-lane/head temperature scaling,
additive key priors, and masks. Never merge token axes across replicas. Do
not materialize a `[K,B,H,N,N]` bias tensor where broadcasting suffices.

Test Q/K/V forward and all parameter/input gradients, including temperature
gradients, against the same shared model's eager K=1 operation. Assert actual fused forward and backward
dispatch on the H100. A new FlashAttention API that cannot express the key
prior is not a compatible replacement.

### Device side optimizer and failures

Implement the current DenseAdam semantics over independent rows. Initially
express the transition as tensor operations and compile it; introduce a custom
kernel only if its measured share justifies maintenance.

The transition consists of a per-lane finite/norm reduction, a small per-lane
control calculation, and elementwise parameter/moment updates. For a tiled
reduction, use one partial-reduction pass and one finishing pass; avoid atomic
accumulation into a single contention point. Keep all decisions on CUDA.

Preserve the current operation ordering exactly at the semantic level:

1. Unscale when the configured scaler is enabled.
2. Check finiteness and compute the norm of the dense gradient.
3. Clip independently by lane.
4. For coupled Adam decay, add weight decay after clipping, as current code does.
5. Update moments and use each lane's step count for bias correction.
6. Form the denominator with epsilon in its current location.
7. For AdamW, apply decoupled parameter decay and then the update.
8. Update lane-specific scaler/counter state according to the existing outcome.

Store learning rate, betas, epsilon, decay, clipping threshold, enabled flags,
and counters as device lane state when they may differ across experiments.
Inactive lanes preserve all state, not just weights. Explicit zero gradients
for unused parameters still participate in the current dense update semantics.

Separate recoverable FP16 overflow from fatal BF16/nonfinite-loss failures.
Preserve the ordinary path's decisions for both. For graph-safe BF16 execution,
latch fatal status on the device, suppress subsequent optimizer mutations as
required, and check status at the transaction boundary before any publication.
Remaining scheduled work after a fatal flag may be discarded, but cannot
publish a partial result. Roll back parameters, buffers, moments, counters,
reference/version state, RNG, scaler, and run state exactly as the current
transaction requires. CUDA execution faults still require immediate error
handling; a device status tensor does not recover a poisoned CUDA context.

### Compilation and CUDA Graph ownership

Compile only the fixed-shape tensor region for a family/cohort, not the entire
Python training session. Begin with `fullgraph=True`, `dynamic=False`, and a
mode without automatic graph capture, so manual graph ownership is explicit.
Let compilation generate forward and backward. Use
`max-autotune-no-cudagraphs` as the intended H100 mode with one explicit graph
owner. Local eager/default execution is a diagnostic mode of the same tensor
program, not a separate architecture.

The current learner calls backward outside autocast. On runtimes exposing
the documented control, set compiled backward's autocast assumption to
`"off"` during compilation/first tracing and verify generated dtypes. An
incorrect assumption can silently alter gradients; see
[compiled backward semantics](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_backward.html).

Capture family accumulation separately from the optimizer transition first:

```text
gradient reset
    -> preflop accumulation graph replay as needed
    -> postflop accumulation graph replay as needed
    -> optimizer graph replay
```

Each accumulation graph includes gather/packing if that code is capture-safe,
forward, authoritative loss, backward, and detached metric/status updates.
Static input addresses and parameter/gradient addresses remain stable. Update
index and existing token-mask buffers in place; never replace objects under a captured graph.
Capture must not freeze an unintended Python scalar such as learning rate,
step count, iteration-dependent entropy, or the current loss scale.

Warm up allocation, compilation, and backward before capture. Snapshot and
restore all state affected by warmup/capture, including RNG, weights, moments,
counters, gradients, buffers, and metrics. Prove the first replay starts from
the intended state rather than an extra warmup update.

Keep graph variants bounded by `(layout version, family, cohort size, exact
shape, dtype, semantic mode)`. Shapes include token extent and actual action
span counts when captured. Bound cache admission and use uncaptured execution
of the identical program for infrequent shapes. Report retained graph-pool bytes. Reuse
pools only when execution order and tensor lifetimes make it safe; concurrent
stream graphs must not overwrite shared scratch. These constraints follow
the [PyTorch CUDA Graph contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

Do not keep an autocast cache containing old cast weights across optimizer
updates. Use the ordinary cache lifetime first. Any explicit BF16 packed-weight
cache must refresh after every applicable update and preserve gradient flow to
FP32 masters; detached cached weights are not a valid training implementation.

Do not build a stream-wave trainer alongside this runtime. Ordinary streams
and events for packing and library dependencies remain appropriate.

## Device work reductions included in the build

### Fusion and dtype boundaries

FP32 embeddings/residuals promote significant activation paths even under BF16
autocast. Preserve those semantics while fusing normalization, side-feature
packing, padding, and the final BF16 cast into fewer passes. Avoid constructing
large FP32 concatenations only to cast them immediately before GEMM. Include
this in the initial compiled tensor program. Inspect generated fusion and fix
broken regions without introducing another model definition.

Do not change the residual stream to BF16 in this build. Keep FP32 residual,
loss/control, master-parameter, and optimizer semantics while fusing temporary
packing and casts. A residual-precision change requires a separate numerical
decision rather than being mixed into this execution rewrite.

### Matrix layouts and repeated context

Factor the shared model's first FFN projection into common dynamic features
and family-specific static features as specified below. Derive each matrix's
logical and aligned physical widths from that layout, not from hard-coded
split-network widths. The former 855-to-864 projection is an alignment example,
not a requirement to retain the old FFN. Fuse zero feature padding with assembly
and count its arithmetic; logical parameter values and outputs remain unchanged.

For Normal, gather the sampled private row after the shared trunk/pooling but
before policy/value readout heads. Retain full trunk computation. Apply the
same selection convention to reference outputs and loss inputs.

For RB readouts, factor the first affine map when shared context is broadcast
to every private row:

$$
W[x_h,c]=W_xx_h+W_cc.
$$

Compute the shared term once per decision, then add before the existing
nonlinearity. This is algebraically equivalent but changes summation order.
Include it and Normal's selected-row readout in the initial build, test their
gradients, and tune their kernels as parts of the one implementation.

### Excluded initial work

Do not build a second grouped executor. If H100 profiling exposes a dominant
BMM or tail inefficiency, replace the offending primitive behind the same
contract. Validate backward and graph capture, not just forward speed.

Consider an incidence-specific kernel only after attributing significant time
to the existing exact contractions. Deterministic accumulation and the earlier
RB gradient problem make an unexamined scatter/atomic rewrite unacceptable.

FP8, alternative attention implementations, sparse attention, reduced token
sets, and removal of incidence/folded features are separate studies. They
must not be credited as unchanged-work execution speedups. In particular,
FP8 scale/amax state must not accidentally couple independent replicas.

## Whole pipeline integration

Do not freeze rollout, reference inference, and completion at current speed
and impose a ceiling from old phase timings. Use the same replica-native model
for training, frozen-reference inference, and rollout inference, with explicit
output requests and no-gradient regions where appropriate.

Preserve stable per-lane FIFO prefixes and lane IDs at the inference boundary.
Pack compatible preflop/postflop work across replicas without sharing state,
consuming another lane's RNG, or changing lane-local decision order. Batch
parallel work across independent trajectories rather than their dependencies.

Make reference caching replica-aware: one total admission budget, per-lane
reference versions, bounded build packs, and last-consumer release. Use views
or fixed slots, not full parameter copies, to select inference cohorts.

Use the existing replica-grouped completion calls and equal-depth target work
with explicit lane/leaf identity and existing RNG ownership. Optimize their
underlying operations without replacing the LayerPlan, two-arena propagation,
or fixed-slab lifecycle. Preserve the corrected FIFO leaf addresses. The
sibling evaluator still owns showdown and contract-v2 completion operations;
do not change its mathematics or add a compatibility facade. Capture bounded
tensor regions where the API permits it; do not assume every dynamic queue can
become one graph. Include these phases in the work and byte ledger.

This integration belongs to the initial build, although the learner is built
first. The rental should begin with a functioning pipeline, not a learner demo.

## Memory admission and cohort sizing

Distinguish resident replicas from simultaneously executing replicas. Keeping
32 independent runs resident need not imply executing all 32 at every physical
shape. Sweep active cohort size independently and use bounded waves if needed.

The historical split configuration had 11,134,238 FP32 parameters, implying
approximately 0.332 GiB per replica under eight-vector FP32 ownership. This
is context, not the new admission budget. Count unique shared-model parameters,
family adapters, gradients, moments, references, rollback and packed caches
from the implemented owner; no duplicated shared tensors may be charged or
allocated. At K replicas persistent owner bytes scale with K, not with the
number of family calls.

A single BF16 postflop hidden activation `[32,64,2854,512]` occupies about
5.6 GiB. One such saved tensor across each of six layers is already about
33 GiB. This is only a sizing example, not a complete autograd allocation
model. FP32 materialization can double the corresponding component.

Measure admission from component lifetimes:

$$
M=M_{\mathrm{persistent}}+M_{\mathrm{retained\ data}}
+M_{\mathrm{references}}+M_{\mathrm{live\ autograd}}
+M_{\mathrm{staging}}+M_{\mathrm{graphs}}+M_{\mathrm{workspace}}.
$$

Reserve at least 10 GiB on the nominal 80 GB device and use actual reported
device bytes, not an assumed 80 GiB. Track allocated and reserved memory
separately, plus device-level memory used by libraries/extensions. Do not
extrapolate activation cost from differences in full-session peak memory.

Release packs and autograd state at their last consumer. Audit reference-cache
and retained-rollout lifetimes without dropping data still required by the
iteration transaction. Do not stage production activations or optimizer state
through CPU to fit more replicas.

Prefer smaller physical microbatches over activation checkpointing initially.
Checkpointing trades extra arithmetic for memory and may be worthwhile only
if the newly admitted concurrency more than compensates. The logical batch
and number of optimizer updates do not change.

## File level implementation sequence

Modify the existing owners below. A small extracted module is acceptable only
if it is used by the sole production pipeline and replaces the corresponding
implementation at its old location. New filenames are not deliverables in
themselves. Keep internal stages reviewable; complete all of them without
asking the user to choose another implementation phase. Preserve unrelated
working-tree changes.

| Stage | Files and responsibility | Required result before proceeding |
| --- | --- | --- |
| 0 | Existing benchmark CLI, tests, configuration and caller inventory | Record starting revision, queue/target fixtures, mode/workflow coverage and work counters; no copied legacy implementation |
| 1 | `pbg/config.py`, `pbg/games/poker/queued/config.py`, `queued/model.py`, existing `model/` modules | Replace split networks with the sole shared trunk/readout and family encoders; one shared depth; remove split selection/type assumptions; eager K=1 correctness |
| 2 | Existing model primitives, `model/capacity.py`, `queued/capacity_compiler.py` and consumers | Explicit-K BMM, packed QKV, embeddings/norms, factored FFN, fused attention, selected Normal/factored RB heads; admission matches actual family shapes and intermediates |
| 3 | `pbg/core/learner.py`, `pbg/core/training.py` | One tensor loss and accumulation owner; step-major independent-lane scheduling for sequential steps/epochs and branched modes; exact sigma/action tails |
| 4 | `pbg/core/replica.py`, `pbg/core/optimization.py` | Unique shared parameter/gradient registry, independent compiled optimizer, scaler/failure/rollback semantics, same-replica distributed reductions |
| 5 | Existing execution owners; extract one small graph-lifecycle module only if useful | Bounded compile/capture cache for the same program, stable addresses, correct warmup/first replay/reset and uncaptured exact tails |
| 6 | Existing queued scheduling, replica, reference and completion owners | Batched rollout/reference calls without altering FIFO/slab/layer contracts; aggregate memory admission; every statistical mode integrated |
| 7 | `pbg/runtime.py`, persistence, inference/export, distributed, evaluation and variance callers | Shared-only checkpoint identity and resume; explicit split-artifact rejection; all current public workflows use the sole model/runtime |
| 8 | Replaced modules, tests, active configs/docs and import/export surfaces | Delete obsolete code/files/flags and stale tests; reconcile narrowly superseded design clauses; no reachable or dormant alternate trainer/model |
| 9 | Existing benchmark CLI plus one thin H100 orchestration harness | Complete local CUDA gates, pinned environment, exact replay artifacts, reproducible command and pending-SM90 report |

Develop eager K=1 shared-model semantics first, then K=2/4 and graph replay,
then complete integrations and K=16/32 admission. These are construction
milestones within one delivery, not alternative trainers left in the tree.
Finish stages 0–9 locally to the extent supported by the available CUDA GPUs.
Do not postpone model replacement, any existing production workflow's
integration, graph lifecycle, checkpoint support, or cleanup until H100 access.

Deliver a runnable candidate, local CUDA correctness tests, exact replay
artifacts, a pinned environment, a work/byte ledger, and one resumable
benchmark/profile command before rental. Explicitly mark BF16/SM90 validation
pending. CPU substitution or untested implementation stubs do not establish
readiness; the local GPU cannot prove H100 dispatch.

### Mandatory cutover and cleanup

- Consolidate pre/post modules into family encoding/geometry responsibilities
  and one shared trunk/readout. Files may keep their names if their remaining
  code is genuinely needed; remove files whose only purpose was the retired
  complete network. Do not keep an unused class because a historical test
  imports it.
- Replace existing learner/optimizer/model call sites and public exports.
  Remove old serial scheduling, duplicate losses, incompatible lane-autograd
  wrappers, architecture switches, dead flags, stale imports and obsolete
  benchmark implementations. Preserve existing feature behavior, not classes.
- Use one canonical shared-depth configuration field (`model.layers` as a
  positive integer); update active configs and strict validators together.
  Remove separate pre/post depth controls. Historical experiment artifacts
  remain historical and must not become a production compatibility layer.
- Version/hash the shared scientific configuration, checkpoints and inference
  exports. Require shared-format artifacts for resume, initialize-from-target,
  and frozen opponents. Reject old split formats with an actionable message;
  no automatic trunk averaging, optimizer-state merging, or hidden conversion.
  Existing user checkpoints and run data are not deleted; recovery uses the
  recorded old revision from Git.
- Keep only small test-local mathematical references and recorded fixtures.
  A K=1 baseline invokes the production shared model. Historical code can be
  inspected from Git, not copied into tests or `legacy/` to evade cleanup.
- Audit imports, CLI/config choices, direct constructors, and all call sites
  after cutover. Give every new helper a real caller, remove dead branches,
  and run affected tests plus full relevant regression coverage. Update stale
  tests to the new shared architecture without discarding their queue/loss/
  ownership guarantees. Never delete unrelated user work to satisfy tidiness.

## Validation gates

Add focused CUDA tests, preferably grouped into
`tests/test_replica_loss.py`, `tests/test_replica_optimizer.py`,
`tests/test_replica_schedule.py`, and `tests/test_replica_graphs.py` if those
names fit the repository test organization.

### Exact semantic tests

- Compare sampled indices, multiplicities, logical step boundaries, RNG
  advancement, target/reference identity, and objective mass exactly.
- Compare eager and compiled/captured K=1, then K=2/4 against independent K=1
  calls to the same shared model with different parameters and data.
- Perturb one lane's inputs, targets, coefficients, and parameters; unrelated
  lanes must not change, allowing only independently characterized numerical
  variation from a different kernel shape.
- Test empty families, single rows, all tails, uneven family counts, full
  capacity, inactive lanes, and lanes ending at different step counts.
- Test dense unused-parameter updates, independent norm/clipping, Adam versus
  AdamW ordering, nonzero moments, independent learning rates, and counters.
- Inject finite-loss and gradient failures in one lane at early and late steps;
  verify the existing rollback/publication contract and supported scaler policy.
- Round-trip shared-model checkpoints, optimizer/reference state and inference
  exports; verify precise rejection of split-format artifacts in every loader.
- Verify that no production state migrates to CPU and unsupported devices fail.
- Cover every existing mode, sequential schedule, best-response ownership,
  supported inference backend and valid game geometry boundary; exercise
  multi-way `post_A`, folded seats, root/profile reuse and exact action spans.
- Compare shared gradients with weighted pre/post gradient sums before one
  update; count each shared parameter exactly once in clipping/Adam/serialization.
- Test distributed initialization, same-replica gradient SUM, global objective
  normalization, failure decisions and checkpoint resume on available CUDA
  devices. Independent replica lanes must never be reduced together.
- Prove the queued invariants listed above, including stable FIFO order,
  addresses/tails, failure atomicity, layered recurrence and completion routing.
  Model-dependent trajectories need not match the retired split architecture.

### Numerical tests

Use FP32 CUDA primitive comparisons to localize algebra or indexing mistakes,
then H100 BF16 tests at actual family shapes. Report absolute and relative
errors, gradient norms and cosine agreement, loss components, clipping
decisions, optimizer deltas, and parameters after a full 116-step replay.

Set per-output acceptance tolerances before optimizing the executor.
Calibrate them against repeated eager executions of the same shared model and known
precision effects, not by widening them until a candidate passes. Near clipping
thresholds require explicit inspection. Identical discrete decisions and
lane ownership are mandatory even where floating-point bitwise equality is not.

Keep dtype-changing residual experiments separate from algebraically equivalent
execution changes. A faster result with altered entropy coefficients, targets,
sampling, or dropped heads is a correctness failure, not acceptable BF16 noise.

### Capture and lifetime tests

Verify first replay after warmup, consecutive replay accumulation, reset exactly
once per logical step, changing device controls, noncontiguous/tail cohorts,
checkpoint load into existing addresses, and invalidation after shape/layout
changes. A graph must not retain stale gradients, weights, entropy controls,
reference versions, or input-buffer addresses.

Run enough complete iterations to exercise reference refresh and high-water
memory. Require no continuing memory growth, graph proliferation, repeated
compilation, or lingering GPU allocation after process exit.

## H100 experiment protocol

### Runtime and resource gate

Use an idle explicitly selected GPU and record its UUID, exact H100 SKU, MIG
status, clocks/power configuration, driver, Torch/CUDA/cuDNN versions, extension
build architectures, repository revision, and experiment patch. H100 SXM peak
numbers must not be applied to a different SKU without adjustment.

Use `/home/pablob/.venvs/poker` and the venv-owned CUDA toolkit described in
`AGENTS.md`. Check `nvidia-smi` before and after every GPU task. Do not use
occupied shared GPUs, change another user's workload, or modify driver policy.

The local RTX 2080 Ti can validate supported FP32/FP16 CUDA semantics, but it
cannot validate native H100 BF16, SM90 fused attention, or Hopper-specific
grouped kernels. Never silently substitute local precision and label the
result an H100 compatibility test.

### Timed boundaries and comparisons

Measure synchronized wall time for learner total and complete iteration.
Also measure CUDA-event subranges, including gather/pack, forward, loss,
backward, optimizer, references, rollout, and completion. Multi-stream event
timing must bracket the full dependency join. Do not sum overlapping event
durations and call them wall time.

Exclude artifact loading, compile/autotune, capture, profiler startup, and
checkpoint serialization from steady-state measurements, but report their
costs separately. Include production schedule generation and packing in learner
total, even if a core-only replay omits them.

Compare batched execution on the same K-lane replay against independent K=1
calls to the same shared model, run sequentially by benchmark orchestration.
That benchmark loop is not a production serial learner or second loss path.
Report both that measured ratio and the normalized $K T_1/T_K$ quantity;
they can differ because individual lanes have different work and cache costs.
Also compare K=1 directly to K=1 to separate single-run latency improvement
from aggregate replica throughput.

For cross-device comparisons run two explicit experiments: matched supported
precision/configuration and best valid configuration on each GPU. Report the
different precision/backend choices in the latter. Do not claim a hardware
speed ratio from unrelated roots, sample counts, model shapes, or profiler runs.

### Rental validation and tuning

1. Install the pinned environment and run the ready-made preflight: correct
   GPU/SKU, extension architecture, native BF16, and real fused attention
   forward/backward. Fix environment or implementation defects on the rental.
2. Run the one implementation's K=1 and K=2 correctness, full-replay, failure,
   and checkpoint gates. Eager diagnostic mode isolates a graph/compiler fault;
   it does not introduce another trainer.
3. Run the complete graph-enabled K=16 pipeline with pre/post capacities 32/8.
   Record useful learner rate, fresh-node rate, memory, and the work ledger.
4. Compare K=32 if admitted, and K=8 if memory or efficiency calls for it.
   Tune one neighboring physical capacity at a time. Do not run a Cartesian
   matrix of executors, compile modes, architectures, and dozens of batch sizes.
5. Profile the dominant gap between the implementation and its operation-level
   roofline. Fix layout, dispatch, fusion, scheduling, or the specific kernel
   responsible. Rerun correctness and the full pipeline after material changes.
6. Rerun shared-model integration gates for branched/best-response/multi-way
   and supported distributed workflows after material H100 changes. Validate
   final Normal and RB settings over multiple iterations, reference
   refreshes, and at least three independent seed sets; export trained checkpoints
   and a final performance/remaining-bottleneck report.

The rental harness must emit atomic progress/results and survive a stopped
benchmark. The agent may change source and recompile on the H100 within the
approved implementation scope. The old prohibition on rental-stage source
edits is not the intended workflow; most coding is offline, final optimization
is on the target hardware.

Warm to stable kernels and memory, then collect at least five unprofiled paired
measurements per final candidate when practical. Interleave baseline/candidate
order to limit drift. Report median, dispersion, and paired speedup confidence
intervals rather than selecting the fastest single run.

Use Nsight Systems first for gaps, API waits, copies, and stream dependencies.
Use short operator traces for shape/dtype/allocation attribution. If counter
access is available, collect Nsight Compute only for dominant kernels and
inspect Tensor instructions, bandwidth, occupancy, tile distribution, and
stall reasons. Do not time the production claim under heavy profiling.

### Selection and stopping rules

Tune the one implementation by complete-iteration useful throughput, using
learner rate and the operation-level roofline diagnostically. Require all
correctness gates, stable memory with reserve, and repeatable improvement.
Keep a kernel change only when its benefit exceeds measurement noise or it
fixes a correctness/capacity defect; do not keep permanent alternate trainers.

Stop increasing K when gains plateau or memory/scheduling costs erase them; K=8
may be optimal even if K=32 fits. If postflop is saturated but preflop is not,
allow different cohort capacities per family without changing logical work.

If graph replay closes gaps but the learner remains slow, optimize dominant
device work rather than adding more replicas. If the learner stops dominating
the iteration, move effort to rollout, reference construction, and completion.
Preserve the sibling evaluator's ownership of its contract-v2 operations.

Save experimental configurations under `configs/experiments/` and artifacts
under `runs/experiments/`, using the repository's NLHE naming convention. Each
result should include machine-readable work counts, timing distributions,
memory, precision, dispatch, and environment data. Preserve successful evidence;
remove failed-run clutter according to repository policy without deleting
unrelated user data.

## First principles throughput model

### Three different comparisons

Keep these questions separate:

1. How much faster is an efficiently implemented learner on H100 than an
   efficiently implemented equivalent learner on one RTX 2080 Ti?
2. How fast can the H100 implementation process a specified amount of learning?
3. How much faster is that than the current implementation on either device?

Hardware resource ratios inform the first question, operation counts and
efficiency assumptions inform the second, and a matched baseline measurement
is indispensable for the third. Old defective timings and old phase fractions
are not inputs to the estimates below.

### Operation level roofline

For an operation $i$, let $F_i$ be Tensor-eligible arithmetic, $U_i$ scalar
arithmetic, $Q_i$ actual memory traffic, and $P_T,P_S,B_W$ the hardware peaks.
A lower bound, with overlap within a kernel treated optimistically, is

$$
t_i\geq\max(F_i/P_T,\ U_i/P_S,\ Q_i/B_W).
$$

A planning model substitutes achievable efficiencies for each peak and adds
unhidden launch/dependency costs. Sum the sequential operations, not just their
FLOPs; account for actual concurrency before summing overlapping work. Include
FP32 policy/loss/reduction work, packing, optimizer traffic, and attention
recomputation. Host latency is a separate term, not scaled by Tensor FLOP/s.

Extra replicas improve the achieved efficiencies and amortize submission, but
cannot multiply one H100's hardware peak by K. Fusion can reduce $Q_i$ and
launch count; selected/factored readout can reduce $F_i$ without changing the
mathematical output required by the objective. These are distinct mechanisms.

### Required shared architecture arithmetic

Use the checked-in [heads-up configuration](../../configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_normal_h100_replica_k1.yaml),
as the source of geometry, then apply the required six-block shared trunk and
factored adapters. Use the incidence/readout code to count work. Do not use the
width-256 plain-transformer approximation from the older idea documents.

The planning geometry is $d=128$, FFN hidden width $h=512$, six shared blocks
executed by each family, 169/1,326 actor rows, two players, and full-width maximum
sequences 536/2,855. There are 150 scalar tokens, up to 42 history tokens,
five anchors, and one preflop or six postflop additional public tokens. The
public/history extent is deliberately conservative.
These are conservative arithmetic envelopes, not instructions to override
the queued design's family/history/board-compatible token packing. Capacity
must bound the actually executed tensors: a full-width operation cannot use
a smaller compact-width admission estimate. Count actual extents for MFU.
Lossless history and model geometry determine sizes; timings do not. Equivalent
logical side-input widths total 600/855 before the shared/family factorization,
and readout width is 928.

Count an FMA as two FLOPs. For forward, dX, and dW, major dense work per
presentation is approximated by the following terms, with $T=2R$ private rows
and $C=52$ cards:

$$
F_{\mathrm{attention}}=12LN^2d,\qquad
F_{\mathrm{QKVO}}=24LNd^2,
$$

$$
F_{\mathrm{FFN}}=6LNh(A+d),\qquad
F_{\mathrm{incidence\ linears}}=30LTd^2,
$$

$$
F_{\mathrm{incidence\ contractions}}=8LTCd,\qquad
F_{\mathrm{three\ readouts}}=18Rh(928+d).
$$

Here $A$ is 600 or 855. The incidence contraction term counts forward and
source-gradient multiplication by fixed geometry, not an unnecessary gradient
for that geometry. Small projections, scalar operations, optimizer work, and
reference/rollout forwards are not included in this subtotal.

| Major operation group | Preflop GFLOP per presentation | Postflop GFLOP per presentation |
| --- | ---: | ---: |
| Attention matrix products | 2.65 | 75.12 |
| QKV and output projections | 1.26 | 6.74 |
| Side-input FFNs | 7.19 | 51.73 |
| Incidence learned projections | 1.00 | 7.82 |
| Incidence fixed-geometry contractions | 0.11 | 0.85 |
| Three unfactored full-row readouts | 1.64 | 12.90 |
| Subtotal | 13.85 | 155.16 |

This is a reproducible algorithmic subtotal, not exact issued instructions.
Flash backward commonly recomputes score products; one extra QK product adds
$2LN^2d$, about 0.44/12.52 GFLOP. Zero alignment padding adds physical work.
The implementation ledger must record these separately from useful model work.

Selected Normal readout reduces the head subtotal to about 0.010 GFLOP per
presentation. Factoring RB's shared 768-dimensional context reduces that
subtotal to about 0.46/3.53 GFLOP. These savings do not eliminate the trunk.
The useful major-matrix subtotal after readout optimization is approximately
12.22/142.26 GFLOP for Normal and 12.67/145.78 GFLOP for RB. Use a conservative
rounded 15/160 GFLOP sizing budget to allow for omitted smaller projections.
Attention recomputation consumes effective throughput; it is not useful work
to add to an MFU numerator. Replace the rounded budget with exact recorded
operation counts when reporting actual MFU.

### Absolute H100 design target

Set an initial target of 150–350 useful major-matrix TFLOP/s, approximately
15–35% of dense H100 SXM peak. That efficiency is an engineering assumption
for narrow matrices, head width 32, FP32 residual/loss work, and optimizer
traffic; hardware specifications alone cannot prove it. Do not claim a high
MFU by including padded work or extra recomputation in the useful numerator.

Using the rounded 15/160 GFLOP shared-model workload as a transparent planning model gives:

| Assumed postflop fraction of learner presentations | Mixed major work per presentation | Aggregate rate at 150–350 useful TFLOP/s | Learner time equivalent for 29,696 presentations |
| ---: | ---: | ---: | ---: |
| 20% | 44 GFLOP | 3,410–7,950/s | 3.7–8.7 s |
| 40% | 73 GFLOP | 2,050–4,790/s | 6.2–14.5 s |
| 60% | 102 GFLOP | 1,470–3,430/s | 8.7–20.2 s |

These fractions are sensitivity assumptions, not imported from old runs.
The rate is learner presentations, not fresh game decisions, optimizer steps,
or tokens. Each lane's 116 logical batches contain 29,696 presentations.
For K lanes, multiply work by K. At the central mix, K=32 means about
198–462 seconds of aggregate learner work at the target rate, not 32 free
learners each receiving the full device throughput. Report total cohort time
and lane-equivalent time separately.

Build a corresponding operation/byte ledger for rollout, references, completion,
and targets before quoting a full-pipeline time. Their new cost must come from
their work and the implementation, not an unchanged residual from old timings.

### Cross device speedup

For comparable implementations and approximately equal fractions of each
resource's peak, Tensor-dominated work scales near 18x while memory/scalar work
scales near 5x. A simple illustrative mixture with fraction $a$ of the optimized
2080 Ti time in Tensor-bound work and the rest scaling at 5.4x gives

$$
S=\frac{1}{a/18+(1-a)/5.4}.
$$

At $a=0.5,0.7,0.9$, the ratios are approximately 8.3x, 10.6x, and 14.6x.
This yields the conditional 8–15x compute-heavy learner planning band without
using historical experiments. It is not a measured workload decomposition.
H100 can have lower relative utilization on narrow operations; graph replay,
fusion, and replicas are intended to counter that. Different valid attention
backends, cache reuse, and precision can also alter the ratio materially.

For the whole pipeline, resource-bound portions plausibly lie on a 5–18x
hardware scale under comparable efficiency, but unhidden host work or
dependency-bound phases can fall outside it. Use the full ledger rather than
promise that the learner ratio applies to everything. A roughly 10x cross-device
learner target is a meaningful ambition; 3x is not a first-principles ceiling.

### Why a speedup against inefficient code is unbounded by the hardware ratio

For Tensor-dominated work, hardware ratio $R$ and achieved efficiencies
$e_H,e_T$ give approximately

$$
S_{\mathrm{new\ H100/old\ 2080Ti}}=R\,e_H/e_T.
$$

For example, $R=18$, $e_H=0.30$, and $e_T=0.10$ would give 54x; $e_T=0.05$
would give 108x. These are conditional illustrations, not evidence that the
current program has either efficiency, and other bottlenecks can prevent the
ratios. Conversely, comparing two already-efficient implementations would not
yield 100x from the Tensor hardware difference alone.

Thus 100x over badly inefficient code is not ruled out in principle, but it
cannot be promised without measuring that code on matched work. The build is
specified by absolute throughput, resource efficiency, correctness, and a
functioning full pipeline, rather than by a multiplier fitted to old defects.

## Mandatory shared preflop and postflop model

After revising the execution plan, the complete
[September 28 assessment](shared_preflop_postflop_model_assessment_20260928.md)
was reviewed against `QueuedPokerModel`, `ReplicaModuleOwner`, and the work
ledger above. Implement its shared-model mathematics as part of this build,
not as a later option. The user's subsequent decision makes this the only
production architecture and supersedes its split-model retention instructions.
The historical forward timings quoted in that assessment are not inputs to
the estimates here.

### One executor, two distinct kinds of sharing

Sharing pre/post weights means using one trunk and readout **within each
replica**. Different replicas still have different weights, gradients, Adam
moments, and random streams. A shared projection therefore still has physical
shape `[K,in,out]`, not `[in,out]` broadcast across independent runs.

Keep separate pre/post physical packs and their native 169/1,326 actor domains.
Sharing a trunk does not require padding preflop to postflop length, concatenating
their attention sequences, or changing any FIFO, target, or statistical weight.
Each family's attention still batches replica and presentation axes; each
projection still uses explicit-K BMM. Graphs remain family/shape-specific even
when they read and accumulate into the same parameter addresses.

Implement all of the following in the existing model and runtime:

1. **Canonical ownership.** Parameter layout is a registry of unique owners,
   with family consumers referencing those owners. Do not assume disjoint
   pre/post parameter ranges. Register a shared trunk/readout once under the
   queued model, not again under each encoder. The current owner's deduplicated
   `named_parameters()` versus potentially aliased `state_dict()` keys makes
   duplicate module registration unsafe for lane checkpoint lookup.
2. **One gradient and update boundary.** Zero each lane's complete gradient
   owner once at the start of a logical step. Backward each weighted family
   chunk immediately into it; after all contributions, compute one lane-local
   norm and perform one optimizer transition. Shared weights receive the sum
   of pre/post gradients, never an average across replicas or an invented
   equal street weighting. Clipping, weight decay, moment updates, rollback,
   and serialization each visit a unique parameter once. Existing `steps`
   grouping already expresses this boundary; no new objective is needed.
3. **Graph and cache identity.** Pre/post backward graphs must accumulate into
   the same stable gradient addresses without zeroing each other's work.
   Include architecture/layout identity in graph and checkpoint compatibility.
   An update to shared weights affects both streets: any prediction cache tied
   to that model version must be invalidated for both. Frozen reference caches
   remain valid until their own reference snapshot is refreshed; refresh both
   families together when that shared snapshot changes.
4. **Shared-only persistence.** A split trunk and two sets of Adam moments
   cannot be silently merged into an exact shared-model resume. Start fresh
   and implement shared-model save/resume. Frozen opponents use the same sole
   architecture constructed from their own compatible scientific config.
   Reject old split artifacts at the loader, including inference, resume and
   initialize-from-target. No old-network implementation survives for loading.

Refactor the existing model into the shared architecture using the explicit-K
primitives and runtime, not a second model implementation. Align public-token
semantics before sharing embeddings, retain street conditioning and the
representation-specific incidence mathematics, and implement the assessment's
per-layer factorization:

$$
W_2\operatorname{GELU}(W_z z+A_su_s+b).
$$

Here the common projection and output map are shared across streets within a
lane; $A_s$ retains the different static pre/post feature domains. This replaces
the existing first affine projection rather than adding another full FFN.
At fixed depth, its matrix-product FLOP sum is essentially unchanged, but two
projections and an addition can introduce extra launches/intermediates. Plan
their accumulation, layout, padding, and fusion explicitly; do not assume that
weight tying itself removes this cost. Recompute capacity bounds for these
intermediates. The existing 600/855-wide projection packing is not a hard-coded
contract for every future architecture.

Give `QueuedPokerModel` one registered trunk and policy/value readout, with
family encoders supplying their native representation. Share attention,
incidence-residual learned weights, compatible public/history embeddings and
the common FFN maps. Preserve preflop class sizes, orbit-averaged incidence,
identical-combo scaling and suit invariance; preserve postflop board masks,
live-seat rows, folded statistics and geometry channels. Family-specific
static-feature adapters and small branching public heads remain legitimate
parts of this one architecture, not independent street networks. All callers
request outputs from this owner, including learner, rollout, references,
frozen opponents, evaluation and supported inference.

### Depth choice and throughput consequences

Use a **six-block shared trunk** for the H100 configuration, retaining the
previous postflop depth and increasing preflop depth from four to six. Other
valid configurations may choose a different single shared depth; they cannot
choose separate depths or separated weights. Sharing is the user's required
model change; six blocks is this plan's concrete depth choice to preserve
postflop capacity, not an allegedly equivalent CUDA optimization.

For arithmetic comparison only, the old 4/6 geometry had trunk subtotals
8.1396/142.2527 GFLOP. The required 6/6 geometry has 12.2094/142.2527;
selected Normal heads add 0.0097 per family, while factored RB heads add
0.4556/3.5266. At an **assumed 40% postflop mix**:

| Architecture | Normal mixed GFLOP/presentation | RB mixed GFLOP/presentation | Compute-limited rate relative to split 4/6 |
| --- | ---: | ---: | ---: |
| Historical split 4/6 arithmetic only | 61.79 | 63.47 | 1.00x |
| Required shared 6/6 | 64.24 | 65.91 | About 0.96x |

These ratios assume unchanged achieved useful FLOP/s. They exclude additional
adapter launch costs, changed cache behavior, padding, and minor operations,
just as the base subtotal does. Sharing with unchanged executed depths would
not materially reduce this arithmetic. The historical row is not a retained
production implementation or an old-timing input. Do not reduce postflop depth
to manufacture an execution speedup.

For the required 6/6 configuration, a conservatively rounded 15/160
GFLOP pre/post budget gives 73 GFLOP per mixed presentation rather than 70.
At the same assumed 150–350 useful TFLOP/s this is approximately **2,050–4,790
aggregate learner presentations/s**, versus 2,140–5,000 for split 4/6. The broad
2,000–5,000 design target therefore remains appropriate; it is not a measured
prediction. Hardware resource ratios and the conditional 8–15x optimized
H100-versus-optimized-2080-Ti band do not gain another multiplier from sharing.

### Memory and learning quality

Shared parameters reduce persistent weights, gradients, optimizer moments,
reference copies, and rollback copies. Under the preceding eight-FP32-vector
accounting, removing $\Delta P$ unique parameters saves approximately
$32K\Delta P$ bytes across K resident replicas, before padding and caches.
Count the actual factored architecture's unique parameters; do not assume a
50% reduction because embeddings/adapters remain family-specific and depths
may change. Lower persistent memory can permit larger cohorts or retained
datasets, but does not make independent activations shareable. A 6/6 model also
has more preflop activation work than 4/6; postflop activation pressure remains.

Positive pre/post transfer might reduce GPU-hours to a given strategy quality.
That is a separate, unproven sample-efficiency effect, not an execution speedup
to multiply into these rates. Preserve the assessment's tests for weighted
gradient sums, one update per step, unique ownership, lane isolation, checkpoint
round trips, suit invariance, and unchanged family shapes. Add captured-versus-
eager mixed-family accumulation and reference-cache version tests. Also
evaluate strategy quality and per-street policy movement across seeds, not just
presentations/s. Distributed SUM parity is part of the existing production
workflow's integration gate, not grounds for retaining a split distributed model.

Learning-quality experiments evaluate the user's selected architecture; they
do not introduce an automatic switch back to a split production model. Keep
the shared implementation even while characterizing transfer and stability.
Any subsequent scientific architecture change requires a separate decision.

## Completion criteria

### Pre-rental implementation completion

All of the following are required before reporting the implementation finished:

1. One shared-model production pipeline serves K=1 and admitted K>1 cohorts,
   all existing statistical modes and supported workflows; no old learner,
   split network, demoted files, duplicate implementation or unused helper
   remains in the delivered code.
2. Queue slabs, family scheduling, sample/commit boundaries, depth-layered
   targets, sampling and objectives satisfy the preserved contracts. The only
   scientific model change is the required shared architecture.
3. Shared-model forward/backward, lane optimizer, graph lifetime, references,
   persistence, failures and resume are implemented end to end. No stubs,
   unconnected modules or TODOs for required integration remain.
4. Locally supported CUDA correctness, all-workflow regression and cleanup
   gates pass. Document tests that genuinely require unavailable hardware;
   never substitute CPU production execution or claim an unrun test passed.
5. Deliver an actual tested command using the repository's CLI/harness, pinned
   dependencies, active shared-model configuration, work/memory accounting,
   and exact instructions for H100 preflight, correctness, benchmark and profile.
   The command's arguments must exist; do not hand off pseudocommands.

Do not claim measured H100 performance at this boundary. If local resource
availability blocks a required gate, identify it and distinguish implemented
code from validated readiness rather than silently deferring implementation.

### H100 validation completion

After rental access, validate native BF16 and fused attention forward/backward,
repair dispatch/compilation/capture defects, tune the sole runtime, and rerun
correctness after changes. Require bounded compilation/memory and report
repeatable complete-iteration throughput for Normal and RB, plus affected
workflow regressions. The 150–350 useful TFLOP/s band is an engineering target,
not permission to weaken mathematical or storage contracts to reach it.

The final report must distinguish measured speedup from the forecasts here,
state the chosen resident/cohort sizes and physical capacities, explain the
remaining bottleneck, and include the same-model K=1 correctness baseline. Remove
temporary diagnostic code after preserving its evidence. Do not leave several
competing production loss/model definitions behind.
