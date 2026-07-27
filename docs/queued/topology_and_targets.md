# Queued topology and layered targets

Status: Phase 3 executable contract, validated and audited at
`ec9ef6f725a8f986c4b2910b348b1c2a09b804ea`.

This document describes the tensor-fixture structural and mathematical path.
Poker rollout integration remains owned by later phases.

## 1. Sole owners

`Topology` is the compact CPU structural forest. `TopologyPacket` is its
transactional transfer unit, and `LayerPlan` is the immutable CPU execution and
movement plan. `LayeredTargetExecutor` is the only queued target executor.

The executor is generated from one `ModeSchema` and one `TargetAlgebra`. It
rejects any mismatch with the generated storage owners, partition contract,
physical widths, estimator, rollout family, or conversion vocabulary before a
CUDA layer launch.

There is no carry representation, counter, upload, kernel, or mode-specific
target builder in `pbg.queued`.

## 2. Topology contract

Every retained node stores its parent, child span, executed degree $N$, nominal
degree $\widetilde N$, protected target degree $N_{\rm tgt}$, owner slot, head
and target addresses, and output domain. Every child stores its logical label,
compressed path length, endpoint kind and row, and output domain.

Validation proves:

- node child spans partition the child table exactly once;
- every nonroot structural node has exactly one incoming topology edge;
- every leaf row is consumed exactly once per replica and leaf rows form one
  dense prefix across output domains;
- $1\leq N_{\rm tgt}\leq N\leq\beta_{\max}$ and
  $1\leq\widetilde N\leq\beta_{\max}$;
- sequential rows have
  $N=N_{\rm tgt}=\widetilde N=1$;
- trained and structural rows have disjoint closed address contracts;
- declared conversions cover every cross-domain child edge; and
- compressed unary paths preserve the retained parent label while adding path
  lengths and retaining the endpoint-domain cost multiplier.

For sequential trained rows, `FamilyWork`, `SigmaHead`, `SigmaTarget`,
`ActionHead`, and `ActionTarget` share one row identity after compaction.
`LayerPlan` rechecks its retained head/target identity, so reconstructed plans
cannot permute aligned learner rows.

Sigma destinations and action spans are unique. Their actual destination sets
must equal the declared dense extents exactly for every replica and partition;
no default-only target row can be published.

## 3. Movement and arenas

Completion payloads are closed CUDA FP32 contracts. Device, dtype,
contiguity, estimator type, sealed leaf-row extent, $P_{\rm train}$, output
width, and branched trace shape are checked before reservation. Sampled and RB
payloads cannot cross schemas. RB source mass and moment laws are checked
before post-to-pre conversion, preventing invalid components from cancelling.

The movement plan is a dense exact leaf permutation. It copies completion rows
into arena-zero work rows using an explicitly admitted CUDA index vector.
Movement destinations are dense within each replica/output group.

Two fixed SoA work arenas alternate by layer. A child row encodes both its
arena and local row. Short branches remain in their producing slot until their
last consumer; no physical copy exists. Source slots become reusable only
after every consumer CTA in the ordered layer has read them.

## 4. Work carriers and conversion

Sampled work carries `return_value`; branched sampled work also carries the
unbootstrapped `trace_vector`, `cost`, and `delta`.

RB work carries `mass` and `moment`, with

$$
\mathtt{moment}=\mathtt{mass}\times\mathtt{conditional\_value}.
$$

Branched RB work adds the same domain-independent trace, cost, and delta
fields. Sequential schemas physically omit them.

Identity conversion preserves component order. The declared post-to-pre
conversion deterministically accumulates FP32 source components in source
order:

$$
w_k=\sum_{h:c(h)=k}w_h,\qquad
m_k=\sum_{h:c(h)=k}m_h.
$$

Conditionalization occurs only after both reductions, and zero mass produces
an exact finite zero value.

## 5. One kernel per layer

The host passes the CPU `layer_ptr` once to the compiled CUDA executor. C++
enqueues exactly one `layered_target_kernel` per nonempty reverse layer on one
stream. One CTA owns one parent. There is no Python node/action numerical loop
and no host synchronization or device-to-host transfer between layer kernels.

For the protected child prefix, the kernel reduces in logical-child order:

$$
W_s=\frac{1}{N_{\rm tgt}}\sum_{a<N_{\rm tgt}}W_a.
$$

Sampled owner work applies lambda bootstrap after writing the unbootstrapped
sigma and action targets. RB work sums mass and moment first, writes posterior
mass, and conditionalizes values only at the target boundary.

Leave-one-out baselines use direct sibling summation in logical order.
Branched rows reduce `trace_vector`, `cost`, and `delta` only when those fields
exist. Headless best-response rows perform structural reductions but never
access heads, write targets, or apply owner bootstrap.

## 6. Failure and publication

The kernel validates every work channel for finiteness and, for RB work,
nonnegative mass plus zero moment at zero mass within tolerance. The minimum
failing logical node is recorded per replica. Later layers skip that replica.

Sigma and action targets are reserved and default-filled as one lane-local
transaction. They become learner-visible only after the final layer and
failure inspection succeed. Failure aborts every destination for that replica
without altering a successful replica.

All raw pointer addresses, spans, domain bindings, target extents, head and work
publication extents, and scratch requirements are proved before upload.

## 7. Accounting and selected alternatives

The plan records exact layer nodes, protected target-edge reads, RB source and
destination component traffic, target bytes, movement index bytes, and
executor scratch. Device-plan bytes and extension pointer, conversion,
failure, and capacity tables are admitted cumulatively before allocation.

The Phase 3 matched comparisons selected:

- last-consumer retention over physical copies: four copy layers would take
  `0.65830 ms` and move `173,801,472` bytes, while retention took
  `0.000252 ms` and allocated no extra payload;
- direct sibling summation over total-minus-child: direct took `0.02430 ms`
  and `4,073,472` incremental bytes versus `0.01436 ms` and `1,357,824`
  bytes, but direct returned the correct concentrated-row result `0` while
  total-minus-child returned `-1`.

The slower direct reduction is retained for correctness. Both losing
production implementations and every carry surface are deleted.

## 8. Verified boundary

The CUDA fixture suite covers all twelve mode rows, sequential and branched
multi-layer paths, signed sampled returns, zero-mass RB rows, trace/cost/delta,
internal post-to-pre conversion, protected refill prefixes, unary compression,
duplicate labels, unbalanced trees, maximum configured degree and depth,
headless rows, exact row bijections and identity, malformed plan rejection, and
multi-replica failure isolation.

Phase 3 does not own poker rollout queues, evaluator completion, model
execution, sampling laws, or learner execution. Those consumers must preserve
this closed contract when they land.
