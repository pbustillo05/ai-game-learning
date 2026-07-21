# Core production algorithms

The repository has one rollout implementation and three estimator modes. The
mathematical source is `docs/theory/`; this document maps those laws to active
code.

| Mode | Start topology | Completion | Loss |
|---|---|---|---|
| `normal` | common per-GPU independent roots | sampled private profiles settle terminal trajectories | `ppo` |
| `rao_blackwell` | common per-GPU independent roots | sibling-evaluator posterior at each pre-terminal leaf | `rb_ppo` |
| `branched_rao_blackwell` | one distinguished root/tree per rank | sibling-evaluator posterior at every sampled leaf | `rb_ppo` |

Branched training has an additional required `rollout.branching_mode` selector.
`fixed` uses the configured beta map for every iteration. `dynamic` uses that
same map during warmup, then enables public T/C-driven KKT control. Both modes
instantiate, target, optimize, and checkpoint the T/C heads from iteration zero.

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
before mutation. An inadmissible complete child wave stops cleanly before
random draws or topology mutation; it is never retried with a changed law.

## Sequential modes

`rollout.sequential_roots_per_gpu_initial` and its minimum/maximum bounds are
explicitly per GPU. Every rank freezes the same memory-feasible width before
the iteration's draws; the rank id keeps logical addresses disjoint.

Each trajectory samples one collision-free private profile. Player events and
chance cards are drawn through purpose-separated logical keys. A normal rollout
continues to a sampled terminal reward and produces compact scalar decision
rows. A sequential-RB rollout stops at a pre-terminal leaf and retains its
single-child public path for posterior completion.

Only one mutable reach row per live sequential trajectory exists. A player's
exact reach is committed to its decision row when the policy forward consumes
it, and the stopped-leaf reach is committed once for completion. No path or
policy replay reconstructs discarded reach state.

The data-plane contracts live in `partition.py`, `storage.py`, `topology.py`,
and `targets.py`. Chunked schema-owned arenas seal into exact contiguous
rollout columns; streaming target builders then freeze an immutable
`TrainingStorage`. Natural poker model rows are 169-wide preflop and 1326-wide
postflop for the full deck, while RB posterior/target rows remain in the exact
1326-hand domain.

## Branched Rao--Blackwell mode

Each rank begins with exactly one root. At every non-pre-terminal node,
including the root, the backend supplies beta and the sampler applies the
ordinary randomized-rounding rule. Repeated event labels remain distinct
children. Empirical child frequency determines edge mass, and `path_weight`
stores the product of incoming edge masses.

When `rollout.branching_min_nodes` is configured, a completed public tree below
that size receives one fresh root child at a time, with each new subtree
expanded by the ordinary downstream beta rules before posterior completion.
Final root edges are renormalized uniformly. The pre-refill root children form
a protected prefix used for every root target; refill branches contribute
learning rows only below the root. `nominal_child_count` retains its ordinary
randomized-rounded beta meaning.

The tree is one labeled CSR topology, never a forest of special root copies.
There is no configured node capacity. Paged topology, control, action, and leaf
owners grow only after the complete already-selected child wave is admitted by
the hard byte ledger. The scientific queue safeguard remains a distinct
nominal-versus-executed correction.

## Logical RNG

Random draws use keys derived from the run seed/iteration, global trajectory or
rank tree identity, abstract path, child ordinal, and `DrawPurpose`. Physical
queue order, family batch size, completion batching, and device assignment are
not inputs. Child path keys preserve repeated-label identity by including the
child ordinal.

## Completion and value reduction

The sibling evaluator owns collision-conditioned joint profiles, runouts,
showdown payoffs, and folded-player partition summaries. The active production
completion adapter returns only posterior mass, conditional value, and
spot-level effective sample size.

`build_sequential_rb_targets` and `build_branched_rb_targets` are the sole RB
target builders. The branched builder precomputes an exact reusable frontier
from sealed topology; completion batches feed posterior mass $P$ and weighted
moment $M$. Ready target-eligible sibling sets reduce uniformly. At a player
decision it:

1. captures parent and child posterior masses;
2. conditionalizes $M/P$ only where $P>0$;
3. forms leave-one-out sibling baselines for multi-child decisions;
4. uses the frozen old value for a single child; and
5. applies lambda bootstrap only to the acting player's propagated moment;
6. applies nominal/executed queue correction and protected-root targeting; and
7. writes $T/C$ targets and masks before discarding frozen head predictions.

The reducer preflights its byte plan, validates that streamed batches uniquely
cover every leaf, and emits only the target fields consumed by production
losses. Completion batch boundaries do not change results.

## Normal targets

`build_normal_targets` reads decision-only chronology from `RolloutStorage` and
performs the reverse segmented player-clock recurrence. It retains natural
inputs, selected actor type/event, frozen scalar log probability/value, and one
scalar lambda return. Advantage and clipped value arguments are derived during
the bounded learner gather; no state handle or decision reach survives.

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
