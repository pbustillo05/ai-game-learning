# Refactor brainstorming

Status: resolved; decisions are merged into the refactor proposals.

## Resolved replica batching

Evidence:
[`replica_learner_parity_and_rb_determinism.md`](../../docs/audit/replica_learner_parity_and_rb_determinism.md).
The H100 run invalidated the replica-specific learner graph; it did not test multi-lane throughput.

1. **Replicas add ownership and scheduling only.**
   Per-replica queues, RNG, topology, targets, parameters, dense gradients, optimizer state, and counters remain
   independent.

2. **Keep one shared execution signature.**
   Replicas share game configuration, training regime, sequential/branched geometry, private estimator, model
   architecture, schemas, capacities, and training schedule. Weights, loss coefficients, and branching controller
   are replica-local. `branched.fixed` and `branched.KKT` share an execution signature.

3. **Use a leading owner dimension.**

   ```text
   Owner[input_type] <n_replica, cap[input_type], ...>
   QueueMetadata     <n_replica, input_type>
   ```

   `n_replica = 1` is the ordinary path, not an adapter or oracle.

4. **Active subsets are required.**
   The scheduler selects one `input_type`, an eligible replica set, and one common stable-prefix width. Inactive
   replicas draw no rows or RNG and execute no forward. Padding rows are forbidden.

5. **Withdraw replica-specific `functional_call`/`vmap` learning.**
   Do not stack replay packs, gather padded lane batches, sum lane losses into a new graph, or maintain a
   replica-only loss/policy adapter.

6. **Invoke the authoritative learner operation per active lane.**

   ```text
   sealed lane subspan
     -> ordinary partition forward
     -> ordinary mode loss with FP32 controls
     -> alpha / statistical_count scaling
     -> immediate backward into lane FP32 gradient owner
   ```

   Loss coefficients differ by lane; zero disables a term. No graph survives the subspan.

7. **Use zero-copy lane binding.**
   Bind each lane's parameters and gradient views, then call the ordinary model and loss directly. Initially schedule
   equal-shape lane operations on CUDA streams; do not introduce a second model implementation to force one batched
   autograd graph.

8. **Grouped non-autograd kernels remain valid.**
   Sampling, transition, segmented reservation, completion, target propagation, dense clipping/AdamW, and collectives
   may flatten `(replica_id, local_row)` into one launch. Each CTA retains a single lane owner and fixed reduction
   order.

9. **Make preflop RB deterministic before replica tuning.**
   The layered target path converts H-wide completion mass and moment to K-wide preflop rows in fixed class/member
   order with FP32 accumulation. The learner remains K-wide and performs no repeated-index K/H backward.

10. **Declare numerical ownership.**
    Loss controls, posterior normalization, and objective-mass arithmetic use explicit FP32 contracts. Every
    parameter owns an explicit FP32 gradient view; unused parameters receive zero, not `grad=None`.

11. **Batch target layers across lanes.**
    Layer launch $l$ concatenates layer $l$ from every replica that has one. Work rows and destinations remain
    lane-local; combining replicas must not change within-row reduction order.

12. **Recover only the failed lane.**
    It lowers root count or beta and retries with a disjoint RNG address. Completed replicas keep their stores.

13. **Keep replicas iteration-synchronous.**
    All lanes finish rollout, targets, and the same optimizer-step count before advancing model version.

14. **Support multi-rank composition, not joint tuning.**
    Every rank has the same `n_replica` and replica IDs. One ordered dense gradient buffer includes the replica axis.
    Correctness is required; performance tuning for multi-rank plus replicas is not.

15. **Gate native batched autograd as a future replacement.**
    A batched learner is admissible only if it becomes the sole ordinary/distributed/replica operation, keeps
    immediate bounded backward and dense gradients, and passes complete H100 Normal/RB parity. It replaces the
    lane-bound path; it never coexists with it.

16. **Revised performance hypothesis.**
    Replica gains initially come from concurrent ordinary model operations and grouped non-autograd kernels, not
    `vmap`. Measure lane counts `{1,2,4,8}` only after deterministic RB and complete one-lane parity pass.

17. **Execution map.**

    | Phase | Replica execution |
    |---|---|
    | family scheduler | one `input_type`, active lane cohort, common stable-prefix width |
    | rollout model forward | replica-batched inference; `vmap` is an admissible prototype because there is no backward |
    | sample / commit | one flattened CUDA launch with segmented lane ownership |
    | completion | concatenate compatible lane leaf rows |
    | topology flattening | independent lane plans; concatenate matching target layers |
    | layered targets | one flattened CUDA launch per combined layer |
    | learner forward/loss/backward | authoritative ordinary operation per lane; no replica `vmap` |
    | clipping / AdamW / all-reduce | dense lane axis may be vectorized |

    Every batched phase preserves lane-local RNG, tails, topology, reduction order, destinations, and failures.

18. **Use BF16-first mixed precision on H100.**
    BF16 is the default dtype for model compute, activations, raw heads, and other large model-facing floating
    storage. It is not used for indices, RNG counters, CDFs, policy arithmetic, posterior normalization, target
    reductions, loss controls, gradient owners, clipping, or AdamW state. Those use integer or FP32 contracts.
    Tensor-Core operations use BF16 inputs with FP32 accumulation. Benchmark BF16 storage for ranges, layer work,
    and published targets separately; adopt it only when complete-training accuracy and deterministic gates pass.

## Resolved base redesign

1. Storage uses fixed-capacity aligned SoA arenas. Family `model_input` prefixes are direct model batches.
2. Rows are append-only within an iteration. `[0,head)` remains live until `align-and-seal`.
3. The mode matrix is:

   ```text
   training_regime   self_play | best_response
   rollout_mode      sequential | branched.fixed | branched.KKT
   private_estimator sampled | rao_blackwell
   ```

4. One `model + sample` batch runs at once; its commit may overlap the next batch. Counts, scans, and reservations
   stay on-device.
5. Caps are fixed hyperparameters. Whole-attempt overflow reduces sequential root count or branched beta, records
   diagnostics, and retries with a disjoint RNG address.
6. Sampled branching intentionally uses one joint profile per tree. Branched RB reuses child zero and samples
   `N - 1` independent posterior profiles.
7. Action-conditioned sampler variants are prototypes. Select one by complete `sample -> commit` time and delete the
   rest. Poker needs at most two distinct action-conditioned actor ranges per frontier.
8. Topology packets are staged by transaction and become visible only after commit.
9. Target creation follows [`layered_target_creation.md`](layered_target_creation.md): one asynchronous CUDA kernel
   per layer, initially one CTA per parent or carry row.
10. Sampled target work keeps lambda-return and trace-value carriers separate.
11. A compressed edge contributes

    $$\mathtt{path\_length}
      +\mathtt{cost\_multiplier}\,C_{\rm child}.$$

12. Frozen best-response opponents own no learner heads or targets. Unary rows compress; branched rows remain
    headless structural reductions.
13. Learner indices are final when created. Self-play rows align directly; best response compacts only trained
    inputs.
14. Acceptance targets the audited routing, reservation, synchronization, sealing, and target-propagation costs
    while preventing sequential regression.
15. Experiments may select an implementation. Production retains one path and deletes every loser, adapter, legacy
    path, stale option, and compatibility surface.
