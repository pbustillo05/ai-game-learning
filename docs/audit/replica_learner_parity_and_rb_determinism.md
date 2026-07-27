# Replica learner parity and deterministic preflop RB

Status: requirements derived from the verified
[`runs/runpod-h100-handoff-20260726`](../../runs/runpod-h100-handoff-20260726/);
they do not override existing refactor contracts or authoritative theory.

## 1. Purpose

The queued rewrite must not preserve two defects exposed by the H100 replica work:

1. ordinary and replica training followed different forward/loss/backward graphs;
2. sequential RB preflop used a nondeterministic BF16 CUDA reduction from exact hands to model classes.

The result is one learner operation for ordinary, distributed, and replica execution; replicas add only ownership
and scheduling.

## 2. Verified evidence

The handoff manifest passes `sha256sum -c`. Its environment was:

- H100 80 GB HBM3, compute capability 9.0;
- PyTorch 2.11.0 and CUDA 12.8;
- learning repository HEAD `f952104600fb8cd0f7ff99c799c638527da71c63`;
- evaluator HEAD `9d35191b396cca5bf07518a07ba2976b11971a5a`.

The saved learning tree was already dirty. Classify its patch hunk by hunk; do not apply it wholesale.

Verified results after the retained prototype changes:

- focused CUDA tests: 11/11 passed;
- Normal replica-count-one one-step parity passed;
- maximum Normal parameter error: `2.3841858e-7`;
- maximum Normal gradient error: `4.5474735e-11`;
- sequential-RB replica-count-one one-step parity failed;
- maximum RB parameter error: `9.9711120e-5`;
- maximum RB gradient error: `1.9073486e-5`;
- complete 116-step parity, full CUDA gates, and lane-count `{1,2,4,8}` selection were not run.

## 3. Findings

### 3.1 Duplicate learner graphs

The replica implementation had accumulated a separate autograd path:

- `functional_call`, `grad_and_value`, and `vmap`;
- stacked loss/replay packs and replica-only adapters;
- padding rows and duplicate gathers;
- native outputs packed, stacked, and sliced back;
- a replica-only learner benchmark and CLI.

Forward agreement did not imply gradient agreement. AdamW amplified small graph-dependent gradient differences into
parameter errors near `1e-4`.

The prototype bound trainable zero-copy lane views and invoked the ordinary forward and loss directly, making the
Normal one-lane gate effectively exact.

### 3.2 Loss-control dtype drift

The replica entropy override inherited the BF16 dtype of a model output through `.to(natural_value)`, while the
ordinary coefficient remained FP32. This changed the actual objective and gradients under AMP.

Control scalars, posterior normalization, and objective-mass arithmetic need declared dtypes independent of model
output storage.

### 3.3 Nondeterministic preflop RB reduction

At full rank count, preflop maps 1,326 exact hands to 169 suit-equivalence classes. The old learner expanded
class-domain outputs with:

```python
value.index_select(-1, exact_to_model)
```

Its forward was correct. Backward accumulated posterior-weighted exact-hand gradients into repeated BF16 class
indices in scheduling-dependent order.

The diagnostic localized the first difference to preflop subspan 0 with 84 rows:

- storage, sampled rows, inputs, posterior weights, outputs, reference outputs, and loss components matched;
- the first unequal tensor was the model-output gradient;
- maximum value-output gradient error was `1.1920929e-7`;
- first parameter-gradient error was `1.5258789e-5`;
- repeated isolated backward executions varied by `1.9073486e-6` to `2.8610229e-6`;
- a fixed-incidence FP32 prototype was bit-identical across 12 executions.

Normal selects one realized class and avoids the reduction. Postflop's exact-to-model map is identity. The evidence
rules out sampling, RB normalization, clipping, reference-model behavior, and general CUDA nondeterminism.

### 3.4 Dense-gradient semantics

Replica optimization owns dense FP32 gradient rows. An unused parameter therefore receives an explicit zero
gradient. A conventional PyTorch oracle may leave it at `grad=None`, causing AdamW to skip moment creation and the
parameter transition. Tests that compare these two contracts are invalid.

## 4. Required end state

### 4.1 One learner operation

Every execution form must invoke the same semantic operation:

```text
sealed partition + sampled rows
  -> authoritative partition forward
  -> authoritative mode loss
  -> alpha / statistical_count scaling
  -> immediate backward into the flat FP32 gradient owner
```

Replica/distributed code may own only:

- stacked lane parameters, buffers, optimizer state, RNG, and counters;
- zero-copy lane binding and scheduling;
- inactive-lane handling, per-lane metrics, and checkpoint extraction;
- gradient synchronization required by the distributed contract.

It must not redefine model preparation, policy composition, loss arithmetic, partition mapping, reference handling,
gradient scaling, clipping, or AdamW semantics. There is no `ReplicaModeLoss`, stacked replay, alternate functional
learner, or replica-only policy adapter in the final production tree.

One and multiple lanes use this same operation. Lanes never share storage, sampled indices, RNG state, gradients,
moments, or step counters. Each row and unit of objective mass has one lane owner; inactive lanes draw no indices or
RNG and execute no forward.

### 4.2 Keep preflop RB learner work in class space

Follow the output partitions in [`storage_redesign.md`](storage_redesign.md):

```text
preflop RB learner domain   K = R^2 model classes
postflop RB learner domain  H = 2R(4R-1) exact hands
```

Evaluator completion may begin H-wide, but the sole layered target path must convert preflop mass and moment to K
with the declared `post -> pre` class sum before conditionalization and learner publication.

Consequently:

- sealed preflop heads, targets, posterior mass, advantages, and policy reductions are K-wide;
- the learner does not expand K-wide model outputs or targets to H;
- no learner-side repeated-index exact/class reduction exists;
- class membership comes from immutable partition/game geometry;
- the H-to-K target conversion uses a fixed within-class order and FP32 accumulation;
- identity conversions are allocation-free;
- conversion metadata and bounded scratch appear in target-phase admission.

If authoritative mathematics proves that a learner term must remain exact-hand-wide, amend the partition contract
explicitly. That exception must use one fixed-order FP32 differentiable reduction, cast once to the destination
dtype, avoid repeated-index atomics, declare exact memory, and replace every raw learner-side gather. Do not retain
both approaches in production.

### 4.3 Explicit dtype boundaries

- Model fields may use the configured inference/training dtype.
- Canonical policy arithmetic follows [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md) in FP32.
- Entropy, KL, value, clipping, alpha, and statistical-count controls use explicit contract dtypes.
- Posterior normalization and objective-mass accumulation are explicitly typed.
- AMP applies only at declared model/loss boundaries.
- No scalar inherits BF16 through `.to(output)` or an equivalent convenience cast.

### 4.4 Gradient and optimizer ownership

Follow the bounded learner lifecycle in [`storage_redesign.md`](storage_redesign.md):

- allocate one flat FP32 SUM owner in fixed parameter order per independent lane;
- bind parameter gradients directly to views of that owner;
- zero it once per optimizer step;
- gather one bounded physical subspan;
- run forward, loss, scaling, and immediate backward;
- retain no graph, replay pack, or full-gradient copy;
- release subspan staging before the next subspan;
- synchronize when required, then clip and transition AdamW once.

The optimizer contract is dense:

- every parameter has a gradient view;
- no contribution means an explicit zero;
- clipping covers the complete dense vector;
- AdamW state transitions follow that dense-vector contract;
- checkpoint import/export preserves parameters, moments, and step counters exactly.

### 4.5 Accounting and failure behavior

All resident mapping metadata, gradient owners, and bounded conversion/backward scratch must be included in the
appropriate phase ledger and allocated before admitted work. Autograd must not hide unbounded or post-hoc
workspace. Nonfinite gradients and rank-common failures follow the queued engine's transactional failure contract.

## 5. Refactor sequence

1. Register the saved H100 patch and artifacts in the migration ledger; classify every hunk as behavioral evidence,
   rewrite requirement, pre-existing, or delete.
2. Establish the sole ordinary, one-lane, and multi-lane learner operation and zero-copy lane binding.
3. Remove functional/replay/adaptor/benchmark compatibility paths in the same cutover.
4. Keep preflop RB heads and losses K-wide; implement deterministic FP32 H-to-K target conversion in the sole layered
   executor.
5. Make loss-control dtypes, dense gradients, clipping, AdamW, and checkpoint semantics explicit.
6. Add parity diagnostics and memory-ledger validation.
7. Run local gates, then target-H100 gates, before replica selection.

## 6. Validation

Use the RTX 2080 Ti development server for static checks, patch classification, supported CUDA tests, FP32
class-sum references, lane isolation, dense-gradient oracles, checkpoint round trips, and deletion audits. It is not
evidence for native H100 BF16 behavior.

On H100, run in order and stop at the first genuine failure:

1. repeated preflop H-to-K target conversion, requiring bit-identical results;
2. focused replica CUDA tests;
3. Normal one-lane one-step parity;
4. sequential-RB one-lane one-step parity;
5. complete 116-step Normal and RB parity;
6. full CUDA gate matrix;
7. lane-count `{1,2,4,8}` memory and throughput selection.

Parity artifacts record matched RNG state, sampled rows, objective mass, losses, first differing output/parameter,
maximum gradient and parameter errors, pre-clip norm, clip scale, Adam moments, environment, code revisions, config,
and commands. Diagnostics remain disabled outside gates.

Independent tests must cover:

- BF16 AMP with every loss-control override;
- all policy fields: `v`, `mu`, `s`, `omega`, `type_logits`, and `iota`;
- every rank count `R in {4,...,13}`;
- preflop class sums and postflop identity conversion;
- multi-lane parameter aliasing plus gradient, RNG, optimizer, and checkpoint isolation;
- inactive lanes and explicit-zero gradients for unused parameters;
- checkpoint/resume and distributed equivalence.

## 7. Acceptance

The refactor is complete only when:

- one learner math path serves ordinary, distributed, one-lane, and multi-lane execution;
- no learner-side preflop K-to-H expansion or repeated-index backward remains;
- deterministic conversion and gradient-owner bytes reconcile in memory reports;
- Normal and sequential-RB one-step and complete parity gates pass;
- repeated H100 executions are bit-identical at the preflop conversion boundary;
- dense-gradient AdamW matches an independent dense oracle;
- multiple lanes prove complete state isolation;
- full H100 CUDA gates pass;
- selection artifacts are reproducible;
- obsolete source, tests, CLI entries, and documentation are deleted.

## 8. Preserved references

- [Handoff summary](../../runs/runpod-h100-handoff-20260726/HANDOFF.md)
- [Test results](../../runs/runpod-h100-handoff-20260726/TEST_RESULTS.md)
- [Continuation commands](../../runs/runpod-h100-handoff-20260726/COMMANDS.md)
- [Saved working-tree patch](../../runs/runpod-h100-handoff-20260726/repositories/ai-game-learning/working_tree.patch)
- [RB diagnostic](../../runs/runpod-h100-handoff-20260726/results/parity_20260726T163407Z/rb_diagnostic_first_boundary.json)
- [Core mathematics](../theory/core_math.tex)
- [Distributed learner contract](../theory/pseudocode/distributed_update.tex)
