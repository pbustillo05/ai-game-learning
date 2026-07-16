# Core production algorithms

The repository has one rollout implementation and three estimator modes. The
mathematical source is `docs/theory/`; this document maps those laws to active
code.

| Mode | Start topology | Completion | Loss |
|---|---|---|---|
| `normal` | global independent starts, partitioned across ranks | sampled private profiles settle terminal trajectories | `ppo` |
| `rao_blackwell` | global independent starts, partitioned across ranks | sibling-evaluator posterior at each pre-terminal leaf | `rb_ppo` |
| `branched_rao_blackwell` | one distinguished root/tree per rank | sibling-evaluator posterior at every sampled leaf | `rb_ppo` |

## Family-scheduled rollout

`pbg/core/execution.py` owns the occurrence arena, byte accountant, bounded work
queues, stable family routing, reusable staging, and logical RNG primitives.
`pbg/core/rollout.py` implements sequential trajectories; `branching.py`
implements labeled randomized-rounded public branching. Both call the same
poker family interface.

Poker classifies model rows as `pre` or `post_A`, where $A$ is the active-player
count. Street stays within row metadata. A logical family block is dequeued in
insertion order and the poker backend may split it into safe actual-cost model
calls, scattering results back to the original logical rows. No alternative
executor or fixed-quantum packing path exists.

Every occurrence is allocated transactionally and follows
`NEW -> QUEUED -> EVALUATING -> DONE`. Admission charges persistent state,
reaches, queue storage, topology, policy retention, and execution workspace
before mutation. Node, queue, arena, or byte overflow fails loudly.

## Sequential modes

`rollout.sequential_starts` is global. `_rank_local_config` assigns contiguous
rank quotas and a global trajectory prefix, keeping logical addresses disjoint
even when quotas differ.

Each trajectory samples one collision-free private profile. Player events and
chance cards are drawn through purpose-separated logical keys. A normal rollout
continues to a sampled terminal reward and produces compact scalar decision
rows. A sequential-RB rollout stops at a pre-terminal leaf and retains its
single-child public path for posterior completion.

With `reach_storage: recompute`, only live trajectory reaches and retained
decision rows persist. Replay reconstructs requested decision reaches without
changing the sampled trajectory.

## Branched Rao--Blackwell mode

Each rank begins with exactly one root. At every non-pre-terminal node,
including the root, the backend supplies beta and the sampler applies the
ordinary randomized-rounding rule. Repeated event labels remain distinct
children. Empirical child frequency determines edge mass, and `path_weight`
stores the product of incoming edge masses.

The tree is one labeled CSR topology, never a forest of special root copies.
The configured node budget caps that tree. Offspring counts are validated
before topology or occurrence status changes.

## Logical RNG

Random draws use keys derived from the run seed/iteration, global trajectory or
rank tree identity, abstract path, child ordinal, and `DrawPurpose`. Physical
queue order, family batch size, completion batching, and device assignment are
not inputs. Child path keys preserve repeated-label identity by including the
child ordinal.

## Completion and value reduction

The sibling evaluator owns collision-conditioned joint profiles, runouts,
showdown shares, and folded-player partition summaries. The active completion
adapter returns bounded `BatchedCompletionStats` and immediately releases
diagnostic-only tensors after summarizing their quality.

`streamed_branched_value` is the sole RB value reducer. Completion batches feed
a contracted boundary frontier of posterior mass $P$ and weighted moment $M$.
Ready sibling sets reduce uniformly over stored abstract children. At a player
decision it:

1. captures parent and child posterior masses;
2. conditionalizes $M/P$ only where $P>0$;
3. forms leave-one-out sibling baselines for multi-child decisions;
4. uses the frozen old value for a single child; and
5. applies lambda bootstrap only to the acting player's propagated moment.

The reducer preflights its byte plan, validates that streamed batches uniquely
cover every leaf, and emits only the target fields consumed by production
losses. Completion batch boundaries do not change results.

## Normal targets

`build_normal_dataset_from_sampled` retains state references, compact decision
reaches, sampled actor types/events, frozen log probability/value, scalar
advantage, and value target. Settlement randomness uses logical keys. Sampling
and gathering keep rows aligned and do not rerun the behavior model.

## Loss and optimization

Normal PPO uses sampled scalar advantages. RB-PPO weights actor-type targets by
posterior mass and stored child masks. Both support policy clipping, MSE or
clipped value loss, exact legal entropy, and optional exact legal-action
`KL(new || reference)`. The reference model is refreshed only at configured
iterations and is otherwise frozen.

The first update after launch/resume verifies cached old log probabilities and
values against model recomputation. A sampled learner minibatch cannot silently
change old-policy identity.

## Distributed mass law

Normal and sequential decisions have weight one; branched decisions use
`path_weight`. Each rank/family stratum samples proportional to its weights and
scales by local stratum mass divided by global mass. Ranks then reduce one
zero-filled flat fp32 gradient buffer with `SUM`. There is no DDP mean and no
world-size division.

The asynchronous global-mass sum may overlap completion/target work. All ranks
make one common finite-gradient decision before clipping or stepping. Unequal
rank decision counts, masses, and family populations are valid.

## Checkpoints and reproducibility

Checkpoints atomically retain the model, optimizer, scaler, scientific config,
completed iteration, run identity, explicit per-rank RNG state, and optional
reference model. Resume requires the exact metrics prefix and scientific
identity. Strict reproducibility additionally selects deterministic Torch
controls; seeded and throughput profiles relax backend execution without
changing logical RNG addressing.
