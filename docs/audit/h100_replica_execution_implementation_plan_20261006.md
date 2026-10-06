# H100 replica execution: implementation contract

Date: 2026-10-06. Status: complete implementation specification, not implemented.

## 1. Outcome and non-negotiable boundaries

An instruction to **implement this document** means finish one integrated,
in-place refactor: shared model, replica-batched learning, production integration,
cleanup, local CUDA validation, and a runnable H100 harness. Do not stop at a
prototype, optional fast path, or learner-only benchmark. Internal milestones
are not separate user deliverables. H100-specific validation and final tuning
follow on the rental; most coding must already be complete.

1. **One modular production pipeline.** Modify existing owners, not parallel
   learners/models. K=1 is the singleton case of K>1. Statistical modes select
   different processes, schemas and objectives, not implementation generations.
   Remove replaced functions, unused helpers, demoted files, dead selectors and
   compatibility facades. Git history is recovery; do not keep the old trainer
   as an oracle, even under `tests/` or `legacy/`.
2. **Preserve the queued architecture.** Read all current files in
   [docs/queued/design](../queued/design/README.md). Fixed-address queue slabs,
   stable family FIFO batches, sample/commit ownership, depth-organized LayerWork,
   sampling laws, target recurrences and publication are not being redesigned.
   Change the underlying tensor execution, not the algorithm or retained storage.
3. **One shared pre/post model per run.** Replace separate trunks/readouts with
   one shared trunk and policy/value readout, plus required family encoders and
   adapters. No split architecture selector or fallback survives. Independent
   replicas still have independent parameters and optimizer state.
4. **Preserve supported workflows.** All twelve mode-schema combinations,
   sequential steps/epochs, branched fixed/KKT, self-play/best-response, valid
   game geometries, CUDA/NCCL distributed training, evaluation, variance runners,
   and supported CUDA/MPS inference use the same model and learning machinery.
   The primary performance workload is HU 32 bb sequential Normal/RB, not the
   boundary of functional support. Existing unsupported devices still fail.

Small extracted modules are welcome only when they simplify the sole pipeline
and replace code at its old location. Eager, compiled and captured execution
use the same tensor program. Hardware-specific primitives behind one operation
contract are legitimate; duplicate whole-model or whole-learner implementations
are not. Tests, benchmarks and historical reports remain legitimate artifacts,
but may not shelter a retired production pipeline.

### Protected design documents: stop and ask

**Any modification under `docs/queued/design/` or `docs/theory/` requires
explicit user approval.** If implementation requires such a change, stop the
affected work and ask with the exact conflict, smallest proposed diff and its
semantic impact. Do not silently override a contract, edit first, or treat a
goal's persistence instruction as approval. This plan grants no blanket rewrite.

The anticipated reasonable clarification is that pre/post input types may share
parameters, instead of having disjoint weights. They remain different input
types and queues: full-deck preflop model inputs and natural outputs remain
169-class-wide, postflop 1,326-combo-wide. Normal's selected-row loss interface
does not change those native domains. Parameter sharing does not redesign the
queued abstraction. Even that wording edit must be proposed for approval.

Check `features_to_preserve.md`'s separation requirement and `storage_redesign.md`
§5.2's older stream-only/no-summed-loss prescription during preflight. If their
literal requirements block the planned under-the-hood execution, surface the
conflict and ask rather than declaring them superseded. Protected dtype/schema
contracts and the sibling evaluator interface are not implicitly relaxed either.

Use the [shared-model assessment](shared_preflop_postflop_model_assessment_20260928.md)
for model mathematics, not its rejected split-model retention proposal. Active
unprotected usage docs may be updated normally. The design README references an
absent `implementation_plan.md`; read it if present later, otherwise use the
actual contracts and this plan without inventing its contents.

## 2. Where changes belong—and where they do not

**This is not model-only.** `LearnerTransaction.optimize` currently executes
each lane's complete schedule serially; `LearnerOperation` owns backward;
`ReplicaOptimizerGroup` constructs per-lane optimizers; `DenseAdam` performs
host-visible finite/norm decisions. Model batching alone cannot remove these
execution bottlenecks. Four core modules need substantive but bounded changes:

| Area | Required change | Preserve |
| --- | --- | --- |
| `pbg/games/poker/model/` | Shared trunk/readout, family encoders/adapters, explicit-K projections, packed QKV, fused packing, efficient heads | Native private domains, geometry/incidence, history, legality and output semantics |
| `pbg/core/learner.py` | Separate tensor loss from execution checks/backward; accumulate batched independent losses; exact pack scheduling interfaces | Loss formulas, posterior reductions, statistical samplers/counts, reference semantics |
| `pbg/core/training.py` | Change lane-major execution to logical-step/family/cohort scheduling | Transaction boundaries, RNG ownership, failure publication, run-state accounting |
| `pbg/core/replica.py` | One unique stacked parameter/gradient owner, shared inference/learning views and lane-safe state | Replica independence and same-replica cross-rank SUM |
| `pbg/core/optimization.py` | Device-side vectorized clipping/Adam/scaler decisions over lane rows | Exact update ordering, dense unused-parameter behavior and rollback |
| Poker queued model/replica adapters and capacity consumers | Bind existing family batches to the shared owner; account for actual tensor lifetimes/shapes | FIFO/slab schemas, stable prefixes, completion ownership, layered targets |
| Config, runtime, persistence, inference/export and workflow callers | One shared depth/config identity; shared-only artifacts; remove split assumptions | Public workflow behavior and strict state/config validation |
| Tests and existing benchmark tooling | Same-model K=1/K>1 evidence, capture/failure tests, work counters and thin rental harness | No second production algorithm in the benchmark |

Do not broadly rewrite `pbg/core`. Queue/slab owners, topology/planning,
sample/commit, seal, Leaf routing and target propagation should remain unchanged
except for necessary binding/capacity adjustments. Every edit there must identify
the preserved contract and why the execution integration requires it. Do not add
poker geometry to generic learner/optimizer code. Do not move generic replica
scheduling into a poker-only model wrapper to make the core diff look smaller.

### Queued invariants to verify

- Same preallocated lane-local SoA owners, fixed capacities and hot/retained
  lifetimes; no replacement replay database or retained activation store.
- Only `pre` and `post_A` input families. Stable FIFO prefixes, no skipping rows
  to reduce padding, no street/width queues; preserve existing history and
  board-compatible token packing. Inactive lanes do no dummy rollout work.
- Same `model -> sample -> commit` boundaries, logical RNG addresses, propagated
  profiles, distinct duplicate-child identities, reservations and atomic tails.
- Same compact topology, protected root prefixes, align-and-seal, retained
  sigma/action alignment and last-consumer release.
- Sibling evaluator retains contract-v2 showdown/completion ownership. Preserve
  corrected stable Leaf addresses, reverse-depth LayerPlan, two LayerWork arenas
  and existing slot reuse. Ordered per-layer CUDA work spans replicas without
  per-node Python loops or per-layer host synchronization.
- Same child-order reductions, post-to-pre mass/moment aggregation before
  conditionalization, target recurrences, failure/retry and publication rules.

Use identical fixed model outputs/actions and RNG addresses to test these
contracts. The shared model intentionally changes predictions and trajectories;
trajectory equality with the retired split model is not an acceptance criterion.

## 3. Sole shared model and parameter layout

Refactor existing modules following the shared-model assessment:

- `QueuedPokerModel` registers one trunk and policy/value readout, used by both
  family encoders. Do not register the same module under two paths: deduplicated
  `named_parameters()` and aliased `state_dict()` keys break lane ownership.
- Share attention, incidence-residual learned weights, compatible public/history
  embeddings and common FFN maps. Align public token meanings before tying
  embeddings; retain street conditioning. Family-specific board/folder maps and
  small branching public heads are parts of this model, not separate networks.
- Keep native $R^2$ preflop classes and $\binom{4R}{2}$ postflop combos; never
  expand preflop to postflop width for sharing. Preserve class sizes, orbit-
  averaged incidence, identical-combo scaling, suit invariance, live-seat masks,
  folded statistics and all required geometry channels.
- Use one positive integer `model.layers`, initially six for the H100 config,
  retaining former postflop depth. Remove separate pre/post depth controls and
  update active configs/validators. Width remains 128 and FFN hidden width 512
  for this configuration; derive game-dependent dimensions normally.

The different side-feature widths require factoring each layer's first affine
map, not simply aliasing incompatible FFN weights:

$$
W_2\operatorname{GELU}(W_z z+A_su_s+b).
$$

Share common/dynamic-feature $W_z$, output $W_2$ and bias within each lane;
retain family-specific static-feature $A_s$. This replaces the existing affine
map, not an added full FFN. Fuse accumulation/packing where possible: two small
projections plus an addition can otherwise add launches. Derive aligned widths
from this layout; the old 855-to-864 padding example is not a fixed interface.

Use the precision policy below; its starting state layout has flat FP32
`[K,Nphysical]` owners for masters, accumulated parameter gradients and moments, with
unique named leaf views and contiguous matrix dimensions. The flat container
must not also be a trainable leaf. Bind leaf gradients to owner views and test
actual backward accumulation and pointer stability. Shared family consumers
reference one parameter ID; no duplicate clipping, decay, update or serialization.

Record logical-to-physical mapping for packed QKV and zero alignment padding.
Padding receives zero gradient and contributes nothing to clipping/updates.
Do not repack FP32 masters per microbatch. Cohorts use views/contiguous lane runs
or explicit lane-indexed primitives—not copies of every parameter or dummy lanes.
Account for autocast/gradient layout temporaries and refresh any low-precision
compute-weight cache after an update without disconnecting gradients from masters.

**Persistence:** version/hash the shared config, checkpoint and inference format.
Start fresh; split trunks and moments cannot be merged into an exact resume.
Reject old split artifacts clearly for resume, frozen opponents, inference and
initialize-from-target. Keep their files untouched; Git recovers their runtime.
Compatible frozen opponents use the sole model with their own checkpoint config.
Preserve optimizer/scaler/RNG/controller/reference/counter/metrics state on new-
format round trips. Utility conversion visits the shared value head once.

### Hardware-specific precision, one tensor program

Choose precision for measured throughput and stable learning, not historical
dtype fidelity. Resolve one hardware precision policy at startup and pass it to
shared primitives; do not scatter device-name tests or fork trainers. Record
compute/activation, reduction/state dtypes, scaler and attention backend in the
run/checkpoint manifest. The execution device remains explicitly selected.

| Operation/state | H100 starting policy | RTX 2080 Ti starting policy |
| --- | --- | --- |
| Projections, QKV/FFNs/heads, attention inputs and working activations | Native BF16 Tensor Core path | Native FP16 Tensor Core path |
| Residual streams, embedding outputs, model feature buffers | Prefer BF16 storage; higher-precision accumulation only where justified | FP16 storage where stable; retain FP32 for sensitive paths |
| GEMM accumulation, normalization/softmax statistics | Stable fused kernels, normally FP32 accumulation with low-precision outputs | Same principle using supported FP16 kernels |
| Master weights, accumulated parameter gradients, Adam moments | FP32 starting choice for update accuracy | FP32 starting choice |
| Policy probabilities/logs, losses, posterior/target reductions, clipping | FP32 stability islands; respect explicit existing contracts | Same stability requirements, not identical kernels |
| Loss scaling | Normally unnecessary for BF16 | Dynamic FP16 scaling and overflow handling |

FP32 masters do not force FP32 matrix multiplies. Backward working tensors may
be BF16/FP16 even when parameter gradients accumulate in FP32. Avoid accidental
promotion of whole activation/residual paths by FP32 embeddings, affine weights
or side features; fuse casts at deliberate boundaries. Do not simply call
`.bfloat16()` on every state or assume autocast alone chooses the best storage.

These are conventional starting choices, not a demand for maximum FP32 or a
proof of optimality. Profile casts, memory traffic and FP32 islands too; retain
higher precision where stability warrants its cost. BF16 residuals and other
model-internal dtype improvements are in scope, with numerical/learning tests.
Changing a protected storage/reduction contract still requires approval above.
Fail unsupported profiles clearly; never force BF16 on SM75 or emulate it.
No cross-device bitwise identity, identical loss-scaler behavior or matching
training trajectories is required. Preserve algorithms and test each hardware's
chosen profile for numerical health and learning quality. Record any intentional
profile change on resume; do not describe it as bit-exact continuation.

References: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) and
[NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).

## 4. Replica-native learning in the existing core

### Tensor operations and gradients

Implement the replica axis throughout the existing primitives:

- Projection: `X[K,M,Din] @ W[K,Dout,Din].transpose(1,2)` using BMM.
- Lane-indexed embeddings and feature-only normalization with `[K,D]` affine
  weights; no normalization or parameter sharing across replicas.
- Packed `[K,3D,D]` QKV, all FFNs/heads, and existing exact incidence contractions.
- After lane-specific QKV, reshape `[K,B,H,N,Dh]` to `[K*B,H,N,Dh]` for ordinary
  fused SDPA. Preserve temperature gradients, additive key priors and masks;
  never merge token axes or materialize a quadratic bias unnecessarily.

Ordinary autograd already differentiates independent BMMs:

$$
dX_r=dY_rW_r,\qquad dW_r=dY_r^TX_r,\qquad
\nabla_{\theta_r}\sum_j L_j=\nabla_{\theta_r}L_r.
$$

No whole-model manual backward, Jacobian, `vmap(grad)`, alternate stream trainer
or per-lane production learner is needed. Sum independently normalized lane
losses; never average across replicas. Keep one tensor loss implementation,
with explicit stable reduction/control dtypes from the precision policy. Move
static validation outside capture and hot finite checks into device status.
Retain all mode-specific stratum/action normalization and loss components.

### Exact scheduling

In the existing transaction, prepare lanes, then execute logical steps across
compatible family cohorts: reset once, accumulate every family/chunk immediately,
then one independent update per enabled lane. No lane consumes its next version
early; preserve existing cross-rank and logical version barriers. Shared pre/post
weights receive the weighted sum of their contributions before clipping.

Preserve sequential fixed-step draws, epoch permutations and family update order,
branched allocation/alias sampling, sample multiplicities and RNG call shapes.
Do not replace many random calls with one large draw or impose pre/post quotas.
CPU may orchestrate bounded counts/topology; all bulk data/numerical work stays
on CUDA. Replace hot Boolean-selection allocations with bounded device packing.

Use existing `pre`/`post_A` families and preallocated transient workspace.
Branched packs keep exact `sigma_index`, `action_ptr`, `action_index` and admission
for both sigma count B and action count E. Model each sigma once; evaluate actual
flattened action spans with segmented reductions, not padded maximum-action rows.

Start with resident K=16 and per-lane physical learner capacities pre=32/post=8;
support K=32 and nearby widths. These are launch limits, not queue capacities
or statistical batch sizes. Execute exact-size tails, grouping compatible lane
shapes and using K=1 cohorts when needed. Capture recurring shapes; rare shapes
use the identical uncaptured program. No invented rows/actions to fill graphs.
Feature alignment and existing token masking do not alter statistical rows.

### Optimizer and failure semantics

Refactor DenseAdam/replica ownership into device-side per-lane finite/norm
reductions and compiled elementwise transitions. Keep independent device learning
rates, betas, epsilon, decay, clipping, counters and scaler controls. Preserve:

1. Unscale, finite check and norm; clip per lane over unique parameters.
2. Coupled Adam decay after clipping; existing moment/bias correction and epsilon.
3. AdamW's existing decoupled-decay ordering and outcome-specific counters/scaler.
4. Dense zero gradients for unused parameters—not `.grad=None` skipping updates.
5. Inactive lanes preserve all state. Distributed SUM joins matching replica IDs
   across ranks only, with existing global objective normalization.

Distinguish recoverable FP16 overflow from fatal BF16/loss failures. Device-latch
fatal status and suppress later mutations as required; inspect before transaction
publication. Restore weights, buffers, moments, counters, reference/version state,
RNG, scaler and run state under the existing rollback contract. CUDA context
faults need immediate handling; status tensors cannot repair them.

## 5. Compilation, pipeline integration and memory

Compile fixed-shape tensor regions, not the Python session. Use explicit graph
ownership (`fullgraph=True`, `dynamic=False`, intended H100 mode
`max-autotune-no-cudagraphs`); eager diagnostics call the same code. Backward is
outside autocast: configure/verify compiled-backward assumptions and dtypes.

Capture family accumulation separately from the optimizer initially. Static
input, parameter and gradient addresses must remain stable; update index/token-
mask buffers in place. Learning rate, step, entropy and loss-scale changes must
remain device inputs, not frozen Python scalars. Snapshot/restore warmup and
capture mutations; test first replay, repeated accumulation and reset once.

Bound graph caches by family, exact shape, cohort size, dtype, mode and layout
version, including action spans when captured. Uncommon shapes use the same
uncaptured program. Account for graph pools; share scratch only with proven
ordered lifetimes. No stale weight/gradient/reference cache across updates.

Include these work reductions in the build:

- Fuse normalization, side-feature assembly, feature alignment and casts to the
  hardware compute dtype. Keep working activations/residuals low precision where
  stable; avoid large FP32 concatenations and needless cast round trips.
- Normal: select the sampled private row after trunk/pooling, before policy/value
  heads. Preserve required public heads and reference-output conventions.
- RB readout: factor broadcast context as $W[x_h,c]=W_xx_h+W_cc$, compute its
  context projection once per decision, then apply the same nonlinearity.
  Test forward/gradient agreement despite changed reduction order.

Use this same model for rollout, references, frozen opponents and inference.
Bind existing FIFO prefixes to cohorts without changing lane-local order/RNG.
Retain grouped evaluator completion and equal-depth target operations; do not
force dynamic queues into one graph. Shared-weight version changes invalidate
both streets' applicable caches. Frozen reference caches expire when their own
snapshot changes, not after every learner update.

Admission must use actual tensor shapes and lifetimes, not an underestimated
compact bound for full-width execution. Count unique masters/gradients/moments,
rollback/reference copies, retained data, live autograd, staging, caches, graph
pools and library workspace. Reserve at least 10 GiB using reported device bytes
(80 GB is not 80 GiB); track allocated, reserved and external memory separately.
Shared weights save persistent bytes, not independent activations. Under eight
FP32-vector accounting, removing $\Delta P$ parameters saves $32K\Delta P$ bytes;
compute the actual owner count rather than assuming a 50% saving.

Keep bounded microbatches and release at last consumer. Prefer smaller physical
packs before activation recomputation; never stage production state on CPU.
Admit reference caches against one total budget with per-lane versions. Resident
K and active family cohort size may differ without changing logical work.

FP8, sparse/reduced-token attention, removed incidence/folded inputs, alternative
trainers and rewritten sampling/targets are not part of this build. Replace a
primitive only when correctness and measured benefit justify it; do not retain
both experimental implementations as production options.

## 6. Implementation order, commits and adversarial reviews

Execute these milestones in one workstream:

| Milestone | Integrated result |
| --- | --- |
| 1 | Inventory callers/contracts; shared-only config/model and eager K=1 tests |
| 2 | Explicit-K primitives, unique ownership, exact core scheduling/loss/optimizer; K=2/4 parity |
| 3 | Graph lifecycle, fusion/readout reductions, actual-shape capacity accounting |
| 4 | All production workflows, reference/persistence/inference/distributed integration; queued-contract regression tests |
| 5 | Remove replaced code/config paths, complete reviews/tests, deliver tested rental harness and readiness report |

### Goal execution and Git discipline

- Before editing, record the clean starting commit and inspect current worktree
  changes. Preserve unrelated user work. Continue through milestones without
  routine phase approvals, but stop and ask for protected-document changes or
  another genuine authority conflict; a goal does not bypass that gate.
- Commit as work progresses: small coherent, preferably test-passing slices
  within milestones, not one enormous final commit. Do not leave days of work
  uncommitted or create empty progress commits. Include coupled caller/schema
  changes together so a commit is meaningful and reviewable.
- Commit messages record the implemented contract and validation run/results;
  explicitly label hardware-blocked or intermediate checkpoints rather than
  claiming readiness. Avoid staging unrelated/newly arrived user changes. Never
  amend/rewrite previous user history or push without authorization.
- Use commit history and one concise implementation status report to track
  completed milestones, current blockers and exact next validation commands.
  Do not repeatedly append progress prose to this specification. After crashes
  or context resets, resume from code/tests/commits, not a new parallel rewrite.
- Finish with a clean worktree containing all task changes in commits and a
  final commit ID. Genuine unavailable resources/permissions are blockers to
  report, not permission to mark an incomplete goal complete.

### Independent adversarial subagents

The implementing agent is explicitly instructed to use bounded, read-only
subagent reviews after the first K>1 vertical slice and before final handoff.
Run them alongside useful integration/validation work; reviewers do not build
alternative implementations or edit overlapping files.

1. **Architecture/unification reviewer:** trace actual callers and ownership.
   Challenge duplicate learners/models, leaky poker/core boundaries, unnecessary
   abstractions, unused code, hidden legacy fallbacks and queued-design drift.
2. **CUDA execution reviewer:** inspect launch/synchronization paths, layouts,
   attention forward/backward eligibility, dtype traffic, optimizer reductions,
   graph/address lifetimes, tails and memory accounting. Distinguish measured
   bottlenecks from hypotheses; do not promise TFLOP/s from source inspection.
3. **Correctness reviewer:** challenge replica isolation, shared gradient sums,
   statistical schedules/action spans, transactions, checkpoint/version handling,
   distributed SUM and preserved queue/target contracts.

Require concrete file/line evidence, severity and a minimal fix—not stylistic
wish lists or blanket approval. The main agent reconciles findings, fixes valid
issues, tests and commits fixes, and obtains targeted re-review of high-severity
findings. Record unresolved/rejected findings with reasons. Reviews do not
replace CUDA tests and must not grow into a committee redesigning the scope.

## 7. Validation and definition of done

Use small test-local mathematical references and independent K=1 calls to the
same production shared model. No copied legacy network/trainer. Establish
tolerances before optimization using stable references and each hardware profile;
do not widen them until a candidate passes. Cross-profile floating results and
model-dependent trajectories need not match. Separate numerical/learning-quality
tests from exact structural and fixed-input sampling invariants.

- Primitive and whole-model output/input/parameter-gradient comparisons,
  including temperature, incidence, adapter and selected/factored readout paths;
  verify actual compute/activation dtypes, finite gradients, residual stability
  and learning behavior for H100 BF16 and SM75 FP16 separately.
- K=1 versus K=2/4 with different parameters/data; perturb one lane and prove
  others unchanged. Shared pre/post gradients equal the weighted contribution
  sum with exactly one norm/update per logical step.
- Exact sampled indices, multiplicities, RNG advancement, objective mass,
  action spans and boundaries across steps/epochs and branched schedules.
  Empty families, uneven counts, tails, inactive lanes and different end times.
- Dense unused parameters, clipping thresholds, Adam/AdamW order, independent
  controls, FP16 overflow, fatal failures, complete rollback and publication.
- First/repeated graph replay, capture restoration, changing device controls,
  address stability, checkpoint load, reference refresh and bounded cache growth.
- Shared checkpoint/export round trips and resume state; all old split loaders
  reject explicitly. Supported inference, evaluation, variance and best-response
  callers use the sole model. Multi-way/folded-seat and valid geometry boundaries.
- All queued invariants in section 2; concentrated/zero posterior mass, duplicate
  children, unary/unbalanced trees, stable completion addresses and target parity.
- Same-replica distributed initialization/SUM, global normalization, failure
  decisions and resume on available CUDA hardware; no cross-replica reduction.
- Import/config/caller audit: no replaced model/learner, dead helper, stale flag,
  demoted file or unused new abstraction. Update obsolete tests without dropping
  their invariant coverage; do not delete unrelated user experiments/artifacts.

Follow AGENTS.md: prescribed venv/toolkit, `nvidia-smi` before/after, explicit idle
GPU selection, no interference with other jobs. Local SM75 validates supported
FP32/FP16, not native BF16 or SM90 dispatch. Report exact tests and untested gates;
never replace production CUDA with CPU. Multi-iteration tests must show stable
memory and no lingering GPU workload after process exit.

**Pre-rental done:** all required code/integrations and cleanup complete; local
gates pass where hardware permits; reviews resolved; clean committed tree; pinned
environment; work/memory ledger; shared-model config and actual tested commands
for preflight, correctness, benchmark and profiling. No required-code TODOs or
unconnected modules. Identify genuine hardware-blocked tests explicitly.

**H100 done:** native BF16/fused attention backward verified, capture/dispatch
defects repaired, the sole pipeline tuned, regression gates rerun, and repeated
end-to-end results reported. Neither milestone guarantees a predicted speedup.

## 8. First-principles performance expectations

Historical H100 failures establish reasons for change, not forecast inputs:
serial K>1 learning, duplicate loss semantics, dtype drift and repeated-index
BF16 preflop backward were real problems. A prior trace showed 25,573 kernels in
3.215 s and only 34.12% unioned kernel activity. This does not measure current
MFU or prove every kernel is spatially underutilized. Active SMs are not the
same as busy Tensor Cores; diagnose gaps and device work separately.

| Hardware resource | RTX 2080 Ti | H100 SXM | Approximate ratio |
| --- | ---: | ---: | ---: |
| SMs / FP32 cores | 68 / 4,352 | 132 / 16,896 | 1.94x / 3.88x |
| Scalar FP32 | 13.4–14.2 TFLOP/s | 67 TFLOP/s | 4.7–5.0x |
| Dense FP16 Tensor, FP32 accumulation | 53.8–56.9 TFLOP/s | 989 TFLOP/s | 17–18x |
| Bandwidth / capacity | 616 GB/s / 11 GB | 3,350 GB/s / 80 GB | 5.44x / 7.3x |

Use NVIDIA's [Turing table 1](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf),
[Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).
H100's 1,979 headline includes sparsity; 2080 Ti's ~114 Tensor headline uses
FP16 accumulation. Neither is the matched dense FP32-accumulation comparison.
2080 Ti lacks native BF16; report actual supported precision/backend on each GPU.

For each operation, model time from Tensor FLOPs, scalar work, bytes and unhidden
submission/dependencies—not core counts alone. A resource lower bound is
$t_i\geq\max(F_i/P_T,U_i/P_S,Q_i/B_W)$; use achievable efficiencies for planning
and sum dependent operations. K improves utilization, not the device's peak.
A tall BF16 128-wide projection has roughly 64 FLOP/byte if activations traverse
HBM, below H100's ~295 peak FLOP/byte: narrow layers still need cache/fusion.

### Reproducible work ledger

For the six-block shared HU/full-deck geometry, use $d=128$, $h=512$,
169/1,326 actor rows, conservative maximum sequence lengths 536/2,855,
equivalent FFN input widths 600/855 and readout width 928. These are arithmetic
envelopes, not instructions to discard board/history-compatible packing.
For $L$ blocks, $N$ tokens, $R$ actor rows, $T=2R$, $C=52$, and FMA=2 FLOPs:

$$
F_{attn}=12LN^2d,\quad F_{QKVO}=24LNd^2,\quad
F_{FFN}=6LNh(A+d),\quad F_{inc.lin}=30LTd^2,
$$
$$
F_{inc.fixed}=8LTCd,\qquad F_{3heads}=18Rh(928+d).
$$

These count major forward/dX/dW work; fixed incidence has no geometry gradient.
Before readout reduction, totals are about 13.85/155.16 GFLOP per pre/post
presentation. Selected Normal heads give 12.22/142.26; factored RB heads give
12.67/145.78. Flash score recomputation can add ~0.44/12.52 GFLOP and alignment
adds issued work; neither belongs in useful model FLOPs for MFU. Count actual
shapes/minor projections and scalar work in the implementation ledger.

Target **150–350 useful major-matrix TFLOP/s** initially: an engineering
utilization assumption, not a hardware guarantee. A rounded 15/160 GFLOP budget
gives this transparent sensitivity model:

| Assumed postflop presentation fraction | Mixed GFLOP | Aggregate learner presentations/s | Time equivalent for 29,696 presentations |
| ---: | ---: | ---: | ---: |
| 20% | 44 | 3,410–7,950 | 3.7–8.7 s |
| 40% | 73 | 2,050–4,790 | 6.2–14.5 s |
| 60% | 102 | 1,470–3,430 | 8.7–20.2 s |

At the central mix, K=32 needs roughly 198–462 s for 32 times that work, not
32 free learners. Report aggregate throughput separately from per-run latency,
fresh decisions, reused presentations and optimizer steps. Build rollout,
reference, completion and target work/byte ledgers before forecasting whole-
pipeline time; do not import unchanged residuals from flawed old experiments.

Under comparable implementations/efficiency, a Tensor-bound fraction $a$ with
other work scaling at 5.4x gives $S=1/(a/18+(1-a)/5.4)$: about 8–15x for
$a=0.5$–$0.9$. That is a conditional optimized-H100/optimized-2080-Ti learner
band, not a whole-pipeline guarantee. Speedup over inefficient code may be much
larger: $S\approx18e_H/e_T$ permits 54x at efficiencies 30%/10%, or 108x at
30%/5%, but those are illustrations, not measured efficiencies. The former
~3x estimate is not a ceiling. Every ratio must name its denominator.

Shared weights save memory and may improve learning per GPU-hour; positive
transfer is unproven and not a throughput multiplier. At an assumed 40% post
mix, shared 6/6 costs about 4% more major arithmetic than historical split 4/6
because preflop gains two blocks. Do not reduce postflop depth to manufacture
an execution gain. Evaluate strategy quality/per-street movement across seeds,
without retaining an automatic split-model fallback.

## 9. Rental protocol and retained references

1. Record GPU UUID/SKU/MIG, driver, Torch/CUDA/cuDNN, build architectures,
   code/config identity and clocks/power. Adjust SXM expectations for other SKUs.
   Run native BF16 and real fused-attention forward/backward preflight.
2. Run K=1/K=2 full-replay, failure and checkpoint gates. Fix defects in place;
   eager diagnostics are the same program, not a second trainer.
3. Run the complete K=16 pipeline (pre/post widths 32/8), then K=32 if admitted
   or K=8 if appropriate. Tune nearby widths, not a Cartesian menu of trainers.
4. Profile the largest roofline gap: launches, layouts, precision, fusion,
   memory or scheduling. Repair the responsible primitive/integration, rerun
   correctness and commit. Do not change queue/sampling/target mathematics.
5. Validate final settings over reference refreshes and independent seed sets,
   including affected branched/multi-way/distributed integrations. Report stable
   memory, measured full-pipeline rates, remaining bottlenecks and final commit.

Time synchronized wall boundaries and dependency-joined CUDA events; do not sum
overlapping events. Separate compile/capture/loading/profiling/checkpoint costs,
but include production schedule/packing work in learner totals. Compare batched
replay to sequential benchmark calls of the same K=1 shared model; report both
the measured ratio and $KT_1/T_K$. Cross-device comparisons distinguish matched
precision from each GPU's best valid configuration. Use repeated unprofiled
paired measurements, median/dispersion and confidence intervals where practical.

Use Nsight Systems for gaps, then short operator traces and permitted counters
for dominant kernels. Do not modify shared driver policy for counters. The
harness must save atomic progress/results and support resuming interrupted
benchmark orchestration. Keep successful evidence under `runs/experiments/` and
`configs/experiments/`; remove failed clutter without deleting unrelated data.

Earlier H100 audits were reviewed; they are evidence, not alternative build
instructions: [idea 1](h100_idea_1.md), [idea 2](h100_idea_2.md),
[idea 3](h100_idea_3.md), [consolidated plan](h100_plan.md),
[learner optimization](h100_learner_optimization_plan_20260803.md),
[execution changes](h100_normal_rb_execution_changes_20260802.md),
[Quest plan](quest_h100_rb_vs_normal_plan_20260724.md),
[launch blockers](queued_issues/quest_h100_launch_blockers_20260729.md), and
[replica parity](replica_learner_parity_and_rb_determinism.md).
Check the actual pinned runtime against PyTorch's
[autograd definitions](https://raw.githubusercontent.com/pytorch/pytorch/main/tools/autograd/derivatives.yaml),
[stacked-state ownership](https://docs.pytorch.org/docs/main/generated/torch.func.stack_module_state.html),
[compiled backward](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_backward.html)
and [CUDA Graph contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
Use NVIDIA's [GEMM guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
and [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
for kernel reasoning, not peak-FLOP promises.
