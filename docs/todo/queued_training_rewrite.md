# Unified dense queued training engine

Status: implementation proposal; not yet an authoritative production contract.

Date: 2026-07-24.

## Executive proposal

Replace the production rollout path in normal, sequential Rao--Blackwell, and
branched Rao--Blackwell modes with one fixed-address queued engine built around
flat, dense, per-family `FamilyWork` slabs.
Key word being **replace**: the legacy implementation gets completely overridden and would disappear from the codebase.

The central physical rule is:

> Assign every player occurrence its final family FIFO row when it is created.
> Every later hot consumer reads a direct contiguous FIFO prefix. Queue identity
> is semantic; poker street and padding width never create another queue.

`FamilyWork` is a schema-generated structure of arrays with batch-major
columns. It is flat and monotone within an admitted attempt: processed rows
become dead to rollout, but their retained subcolumns stay live and their
physical addresses are not recycled. This deliberately spends memory to
eliminate occurrence handles, separate active-reach owners, page lookup,
dequeue payload staging, status gathers/scatters, and hot scans whose cost
grows with accumulated tree size.

For one dequeued parent pack, the intended execution is:

1. run the family model forward;
2. reserve logical child, topology, and player-edge addresses;
3. produce lightweight sampled child descriptors;
4. compute stable destination offsets on device;
5. update the public-trajectory and range fields and write each child directly
   to its final family, leaf, or bounded chance-closure destination;
6. publish only complete destination prefixes.

Small packs may fuse steps 3--5 in one CTA. Larger packs should use a
descriptor-first pipeline so the large natural-input payload is written only
once, at its final destination.

Branched targets use a separate GPU layer engine. It flattens the completed
topology into root-depth order, carries all-node numerical state only in
adjacent-layer workspace, writes exact player-node outputs to `SigmaTarget`
and exact sampled-action outputs to `ActionTarget`, and executes the numerical
recurrence deepest-to-root with launch count proportional to tree depth rather
than node or edge count.

Rollout publication is closed before target creation, but sealing occurs only
after targets are complete. Seal deletes all hot columns, stable-deletes
frozen-opponent rows in best response, compacts the surviving learner fields,
reduces full topology to a minimal learner index, and atomically publishes the
immutable training store.

The same design targets two hardware profiles:

- **H100:** throughput-first tuning, with aggressive batch sizes, graph
  capture, and measured overlap where they improve complete-iteration time.
- **2080 Ti:** throughput under the tighter phase ceiling. Approximately
  1 GiB of rollout slab storage is acceptable when the complete schema-derived
  phase ledger fits without reducing the most valuable forward or backward
  width.

The architecture and mathematical semantics are identical on both profiles.
Only physical capacities, batch quanta, graph choices, and stream schedules may
differ.

## Authority and scope

`docs/theory/core_math.tex` remains authoritative for all mathematical and statistical design. The pseudocode in `docs/theory/pseudocode/`
remains authoritative for any algorithmic element not covered here.

### Intentional contract changes

The changes below require explicit amendments to the authoritative theory and
pseudocode before production acceptance.

#### Monotone total-work family slabs

The current storage pseudocode requires active mutable state and reach capacity
to follow live-frontier high water. This proposal instead preallocates a
configured maximum number of player rows per family and advances a monotone
tail without recycling the processed prefix during the attempt.

Processed rollout-only fields are logically dead after their children and
retained outputs commit. Their stale bytes may never be read by targets or the
learner. Nevertheless, the physical allocation is proportional to admitted
total family work, not only the live frontier.

The proposal also replaces the separate mutable public-state and active-reach
owners in the current pseudocode. Each committed player row owns exactly one
immutable natural model input. That input contains the public trajectory and
range representation needed by rollout and, when the row is trainable, by the
learner; transition writes the child's input directly rather than updating a
second state/reach copy. Best response is the row-lifetime exception:
frozen-player inputs occupy ordinary family rows through target creation but
are deleted rather than published to immutable learner storage.

This is an intentional implementation-contract change. V1 does not use a
circular slab, epoch compactor, or fallback owner. Reclamation should be
reconsidered only if admitted multiplayer or deep-stack measurements reject
the total-work capacity.

#### Class-native preflop RB targets and losses

The current pseudocode uses the exact-hand domain
$X_{\mathsf{pre}}=X_{\mathsf{post}_A}=H$. This proposal changes every preflop
posterior, moment, value, advantage, and loss row to the model-native quotient
$K=R^2$, while postflop remains on $H=\binom{4R}{2}$. To avoid redefining the
game's true private-state space, this document calls those proposed estimator
domains $\mathcal Z_{\mathsf{pre},\mathsf{RB}}$ and
$\mathcal Z_{\mathsf{post},\mathsf{RB}}$.

This changes the finite-sample preflop estimator and nonlinear clipped loss,
not merely storage. The change must be approved in `core_math.tex` and the
storage pseudocode before the target engine becomes production authority.

### Non-goals

This proposal does not:

- change branching intensity or queue-pressure laws;
- change the mode-specific true-profile and posterior-sampling laws;
- merge duplicate public-event children;
- change the completion estimator;
- retain a separate production legacy path;
- make the learner consume mutable rollout fields;
- introduce CPU rollout, loss, or target tensors;
- introduce CPU, automatic-device, or fallback execution;
- retain production $\widehat Q^\lambda$ rows for the deferred NeuRD work.

## Physical families and stable pack planning

Core schedules opaque game-declared semantic families. Poker declares exactly

$$
\mathcal M
=\{\mathsf{pre}\}\sqcup\{\mathsf{post}_A:2\le A\le P\}.
$$

Flop, turn, and river rows with the same live-player count $A$ share one
`post_A` FIFO.
For a selected
postflop prefix $Q$, the game-owned physical planner may use

$$
b_{\min}(Q)=\min_{q\in Q}|\operatorname{board}(q)|,
\qquad
H_Q=\binom{4R-b_{\min}(Q)}2
$$

to compact board-legal model rows to the smallest width covering that prefix.

## Data structures, row domains, and lifetimes

This section is the sole data-structure definition for the proposal. Later
sections describe algorithms over these owners and do not redefine their
fields.

Every allocation is generated from four independent schema axes:

```text
training_regime       self_play | best_response
rollout_geometry      sequential | branched
private_estimator     sampled | rao_blackwell
beta_mode             fixed_beta | KKT_beta       # branched only
```

Three domains must remain distinct:

- $\mathcal H$ is poker's exact private-state domain. A realized $x_p$ is
  always an exact two-card hand in $\mathcal H$, including preflop.
- $\mathcal Y_m$ is family $m$'s natural model-row domain. Poker uses
  $\mathcal Y_{\mathsf{pre}}=\mathcal K$ and
  $\mathcal Y_{\mathsf{post}_A}=\mathcal H$.
- $\mathcal Z_{m,e}$ is private estimator $e$'s target/loss domain. The
  private-state axis is physically absent in sampled mode. Under the proposed
  class-native RB contract,
  $\mathcal Z_{\mathsf{pre},\mathsf{RB}}=\mathcal K$ and
  $\mathcal Z_{\mathsf{post},\mathsf{RB}}=\mathcal H$.

Thus preflop sampled training retains the exact hand $x_p\in\mathcal H$ but
selects model row $c(x_p)\in\mathcal K$. Calling the hand itself a preflop
class would conflate the game state with the model symmetry quotient.

The four geometry/estimator combinations generate the following physical
contracts:

| Geometry | Private estimator | Row-lifetime true profile $x$ | `natural_input` contains $x_p$ | Target private axis | Rollout posterior-profile sampler |
|---|---|---:|---:|---:|---|
| sequential | sampled | yes | yes | absent | never |
| sequential | Rao--Blackwell | yes | no | $\mathcal Z_{m,\mathsf{RB}}$ | never |
| branched | sampled | yes | yes | absent | never |
| branched | Rao--Blackwell | no | no | $\mathcal Z_{m,\mathsf{RB}}$ | independently per logical child |

In both sequential modes, one $x\sim\mu$ is drawn at the root and carried
through the trajectory. In sampled branched geometry, the realized profile is
carried by the sampled tree according to that mode's sampling law. That mode
still requires an explicit authoritative theory contract: in particular,
sibling subtrees sharing $x$ are conditionally rather than marginally
independent. In branched RB, the posterior profile is ephemeral child
materialization state and is never committed to `FamilyWork`.

The trace-variance and cost heads are always trained in branched geometry.
`beta_mode` changes only whether their frozen outputs control branching
intensity; it does not change `SigmaTarget`, `ActionTarget`, or the learner
loss schema.

The currently authoritative best-response workflow is sequential. The
`best_response + branched` combinations below reserve the storage and
ownership architecture for the planned branched extension; its sampling and
loss law still require an explicit theory contract before production use.

The principal owners are:

1. monotone per-family `FamilyWork` slabs, divided only into hot and retained
   columns;
2. one compact sequential topology or a structural `FlattenedTargetTree` with
   player-node and trainable-decision sidecars;
3. pre/post-partitioned `SigmaTarget` and, in branched geometry only,
   `ActionTarget`;
4. pre/post-partitioned transient `LayerWork` in branched geometry;
5. a shared sequential-RB posterior pool and the final minimal learner index;
6. direct final leaf storage;
7. fixed failure records and bounded phase workspaces.

Topology owns identities and relationships. `FamilyWork.retained` owns
rollout inputs and frozen-model outputs that survive rollout. `SigmaTarget`
and `ActionTarget` own only outputs of target propagation. `LayerWork` owns
only transient numerical recurrence state. No field is duplicated across
these responsibilities.

### Numerical domains and physical families

The rollout families remain

$$
\mathcal M
=\{\mathsf{pre}\}\sqcup\{\mathsf{post}_A:2\le A\le P\}.
$$

RB targets and RB `LayerWork` use exactly two numerical partitions:

$$
\mathcal Z_{\mathsf{pre},\mathsf{RB}}=\mathcal K,\quad
|\mathcal K|=K=R^2,
\qquad
\mathcal Z_{\mathsf{post},\mathsf{RB}}=\mathcal H,\quad
|\mathcal H|=H=\binom{4R}{2}.
$$

Board-empty nodes, including the flop-dealing chance parent and preflop
completion leaves, use `pre`. Flop, turn, and river nodes use `post`,
independent of live-player count. These numerical partitions are not rollout
queues and never change semantic FIFO identity.

Sampled targets have no private-state dimension and need no numerical-domain
conversion. In RB, all direct children of a parent use the parent's recurrence
input domain. The only cross-domain recurrence is:

```text
post <- post    H -> H
pre  <- pre     K -> K
pre  <- post    H -> K
```

There is no `post <- pre` recurrence.

### `FamilyWork`: one FIFO with hot and retained columns

For family $m$, `FamilyWork[m]` is one monotone FIFO-addressed structure of
arrays. `hot` and `retained` are lifetime groups inside that one FIFO, not
separate queues or scheduling domains:

```text
FamilyWork[m]
  hot
    logical node/root identity
    logical RNG path key
    role/model selector                         # best_response
    true_profile [P]                            # except branched RB
    mode-specific controller/path fields
    reservation/publication metadata
    head
    reserved_tail
    ready_tail

  retained
    sigma-indexed                              # trainable player nodes
      natural_input
      owner
      old_v
      action                                    # sequential
      old_log_pi                                # sequential
      old_t                                     # branched
      old_c                                     # branched

ActionOccurence[pre|post]                       # branched, only separated pre|post for rao_blackwell
    action-indexed
      action
      old_log_pi
```

The generated schema omits inapplicable fields. Normal mode stores scalar
`old_v` and `old_log_pi` at the realized private type. RB modes store them on
the natural model domain $\mathcal Y_m$. Branched `old_t` and `old_c` are
scalars.

`natural_input` is the sole large player-row payload. It contains the public
state and history, family-native range fields, any immutable folded-summary
dependency, and, in sampled mode, the acting player's realized exact private
state $x_p$. There is no second public-state, history, or active-reach owner.
Poker supplies the following generated schemas:

`action-indexed` means that the data is written in the row corresponding to $\sigma a$ as opposed to $\sigma$ (where $\sigma$ is a processed player node, and $a$ a sampled action).

```text
NaturalInput[pre, sampled]
  preflop public/model fields
  range_weight[P, K]
  acting_private_hand                         # exact two-card hand in H

NaturalInput[pre, rao_blackwell]
  preflop public/model fields
  range_weight[P, K]

NaturalInput[post_A, sampled]
  postflop public/model fields
  board
  live_seat_map
  range_weight[A, H]
  folded_summary_ref                          # iff A < P
  acting_private_hand                         # exact two-card hand in H

NaturalInput[post_A, rao_blackwell]
  postflop public/model fields
  board
  live_seat_map
  range_weight[A, H]
  folded_summary_ref                          # iff A < P
```

The exact poker field encodings are owned by the immutable family
specification; core treats each schema as an opaque fixed-stride input row.
Bounded sampler or transition workspace derives any temporary
symmetry-preserving exact-hand preflop range view. A folded-summary reference
names an immutable game-owned dependency with an explicit last consumer and
seal remapping; it is not an unowned external payload.

Every FIFO row has exactly one `retained.natural_input` location, including a
frozen-opponent row in best-response mode. Thus every model pack reads one
direct contiguous family prefix with no tagged hot/retained input selection.
At post-target seal, self-play keeps every trainable player row. Best response
stable-deletes frozen-opponent rows and retains only exploitator inputs and
their learner fields. The word `retained` therefore means that the column can
survive rollout; it does not assert that every physical row survives seal.

Retention is field-specific rather than row-wide. In future branched best
response, a frozen-player natural input is consumed during rollout and its
row is deleted at seal, while that node's scalar `old_t` and `old_c` remain
available through target propagation. The selected player's `natural_input`
and `old_v` survive through optimization. Frozen-player nodes have no retained
`old_v`, action row, or learner target.

In sequential sampled, sequential RB, and sampled branched geometry,
`hot.true_profile` contains the realized joint profile $x$. Sequential modes
copy it unchanged along the trajectory. Sampled branched geometry follows its
declared true-profile tree law. The acting coordinate $x_p$ is additionally
present in `natural_input` only in sampled mode because it is part of the
model and learner input. In branched RB, a fresh posterior profile is sampled
independently for each child, used only to draw and materialize that child's
event, and discarded; no committed family row contains it. Root work and the
bounded chance frontier carry `true_profile` in the three applicable modes
before or between player-family rows.

In sequential geometry the one sampled action is implicit in its player-node
row, so `action` and `old_log_pi` are sigma-indexed. In branched geometry a
trainable player node reserves one exact contiguous retained action span
before draws:

```text
sampled
  action[A_m]
  old_log_pi[A_m]

rao_blackwell
  action[A_m]
  old_log_pi[A_m, Y_m]
```

The span order equals the target-eligible child prefix. Frozen parents in
branched best response create structural children but no retained action span.
The complete old policy is never retained merely to reconstruct selected log
probabilities.

Names follow `core_math.tex`, with `old` denoting frozen rollout-model output:

| Field | Mathematical quantity | Last consumer |
|---|---|---|
| `old_v` | $V^{\theta_k}(-,\sigma)$ | learner value clipping |
| `old_log_pi` | $\log\pi^{\theta_k}(a\mid-,\sigma)$ | learner policy loss |
| `old_t` | $T^{\theta_k}(\sigma)$ | branched target propagation |
| `old_c` | $C^{\theta_k}(\sigma)$ | branched target propagation |

`old_t` supplies $T^*$ when `target_degree == 1`; `old_c` supplies the
$\lambda$ bootstrap of $\widehat C^\lambda`. Both are released after target
propagation.

Each family satisfies

$$
0\le\mathtt{head}_m
\le\mathtt{readyTail}_m
\le\mathtt{reservedTail}_m
\le N_m^{\rm cap}.
$$

`[0, head)` is dead to rollout, though retained columns remain live.
`[head, ready_tail)` is committed FIFO work,
`[ready_tail, reserved_tail)` is reserved but incomplete, and the remaining
capacity is unused. Publication exposes only the longest consecutively
completed reservation prefix. The initial implementation uses ordered
routing and commit; a completion scoreboard requires an end-to-end measured
benefit.

After target creation, seal copies the surviving natural-input, sigma, and
action rows into compact immutable learner extents. It deletes every hot
column, every target-only column, unused rollout capacity, and, in best
response, every frozen-opponent row. A zero-copy retained view is allowed only
when its measured phase peak and optimization width beat exact sealing.

### Sequential topology

Sequential geometry never allocates `FlattenedTargetTree` or `ActionTarget`.
It stores one decision-only chronological structure:

```text
SequentialTargetTree
  trajectory_ptr[G + 1]
  family[D]
  family_sigma_row[D]
  sigma_target_partition[D]                   # pre | post
  sigma_target_row[D]
  posterior_row[D]                             # sequential RB only
  leaf_row[G]
```

There is exactly one sampled action per retained player node, and it is
implicit in that sigma row. Chance nodes and frozen-player decisions in best
response are absent. In self-play all player decisions are present; in best
response only exploitator decisions are present. Skipping chance and frozen
opponent nodes requires no propagation weight in a realized sequential
trajectory because neither advances the retained player's clock.

Sequential RB does not copy the same completion posterior into every
`SigmaTarget` row. `SequentialPosterior[pre|post]` stores it once per required
`(trajectory, player, target partition)` and `posterior_row` references that
shared row. A trajectory that has decisions on both sides of the pre/post
boundary may therefore own one class row and one exact-hand row for the same
player.

### Branched topology

Branched rollout first constructs append-only scalar topology with contiguous
logical child spans. Target preparation builds one exact root-depth-major
permutation and materializes:

```text
FlattenedTargetTree
  layer_ptr[L + 1]
  node                                        # N structural rows
    parent
    child_begin
    degree
    target_degree
    nominal_degree
    numerical_partition                       # pre | post; RB only
    node_kind                                  # player | chance | leaf
    leaf_row
    player_row                                 # -1 unless player

  player                                      # D_all player-node rows
    node
    owner
    family
    retained_control_row
    trainable_decision_row                     # -1 for frozen BR player

  trainable_decision                           # D_train learner rows
    player_row
    retained_sigma_row
    retained_action_begin
    sigma_target_partition
    sigma_target_row
    action_target_begin

  protected root metadata
```

The structural table does not carry owner, family, control, learner, or target
sentinels for chance and leaf nodes. The player sidecar does not carry learner
fields for frozen best-response nodes. Topology alone owns node identity,
parent/child relationships, target-eligible prefixes, source-to-retained
mappings, and source-to-target mappings.
`retained_action_begin` and `action_target_begin` each address exact
`target_degree` spans in identical child order. Refill-only root children lie
outside both root spans, while decisions below their subtrees remain ordinary
training rows.

Every direct child lies in the next root-depth layer, and stable sibling order
is preserved. Source-node permutations and inverse construction maps are
flattening scratch and are released after validation; they are not persistent
topology fields.

In self-play every player node has a retained sigma row and target row. In
branched best response only exploitator nodes do. Frozen-player and chance
nodes remain structurally present because their subtrees and public T/C
recurrences contribute to exploitator targets. Every branched player node,
including a frozen one, has a retained control row for `old_t` and `old_c`.

### Target outputs

`SigmaTarget` and `ActionTarget` contain only final numerical outputs written
by target propagation. They contain no natural input, owner, topology,
source-row mapping, action, or old model output.

Both use exact pre/post numerical partitions:

```text
SigmaTarget[pre]
SigmaTarget[post]
ActionTarget[pre]                              # branched only
ActionTarget[post]                             # branched only
```

A `SigmaTarget` row is indexed by one retained player node $\sigma$. Its
generated fields are:

```text
sequential sampled
  v_lambda_hat

sequential RB
  v_lambda_hat

branched sampled
  v_lambda_hat
  t_hat
  c_lambda_hat

branched RB
  p_hat
  v_lambda_hat
  t_hat
  c_lambda_hat
```

Sampled fields are scalars and have no private-state axis. RB posterior and
value rows use $\mathcal Z_{\mathsf{pre},\mathsf{RB}}$ or
$\mathcal Z_{\mathsf{post},\mathsf{RB}}$. In both sequential estimators, the
advantage for the unique implicit action is derived during the loss as
`v_lambda_hat - old_v`; it is not stored. Sequential RB reads posterior mass
through `SequentialTargetTree.posterior_row`, so its `SigmaTarget` also does
not duplicate that row. No sequential action-position target row is created.

An `ActionTarget` row exists only for a target-eligible sampled action
$\sigma a$ of a retained branched player node:

```text
ActionTarget[pre|post, sampled]
  a_lambda_hat

ActionTarget[pre|post, RB]
  p_hat
  a_lambda_hat
```

Sampled action fields are scalars and have no private-state axis. The RB
partition is the parent $\sigma$'s numerical domain. `ActionTarget` has no
`action`, `old_log_pi`, `child_node`, parent, owner, pointer, count, or
validity field. `FlattenedTargetTree` supplies the identity and mapping during
target creation; `FamilyWork.retained` supplies `action` and `old_log_pi`.

In branched RB, `SigmaTarget.p_hat` is the posterior for the sigma owner and
`ActionTarget.p_hat` is the child posterior for the parent owner. Sampled
targets contain neither. No all-node/all-player persistent posterior exists.
During RB self-play the all-player recurrence is transient in `LayerWork`;
during RB best response only the exploitator posterior moment need be carried
through the numerical recurrence. Production does not retain
$\widehat Q^\lambda$.

No trace-validity mask is stored. `SigmaTarget.t_hat` is valid exactly when
the corresponding source `FlattenedTargetTree.target_degree >= 2`;
`LearnerIndex` carries that structural gate after seal. Likewise, there is no
action-validity mask: `ActionTarget` rows exist exactly for the target-eligible
child prefix.

### Transient `LayerWork`

`LayerWork` is schema-generated adjacent-layer numerical workspace. Sampled
mode projects away the private-state dimension entirely:

```text
LayerWork[sampled]
  v_vector_hat[rows, P]
  c_lambda_hat_work[rows]
  delta[rows]

LayerWork[RB, pre]
  p_hat[rows, P, K]                            # self_play
  m_hat[rows, P, K]                            # self_play
  p_hat[rows, K]                               # best_response
  m_hat[rows, K]                               # best_response
  v_vector_hat[rows, P]
  c_lambda_hat_work[rows]
  delta[rows]

LayerWork[RB, post]
  p_hat[rows, P, H]                            # self_play
  m_hat[rows, P, H]                            # self_play
  p_hat[rows, H]                               # best_response
  m_hat[rows, H]                               # best_response
  v_vector_hat[rows, P]
  c_lambda_hat_work[rows]
  delta[rows]
```

Only the schema-selected estimator and self-play or best-response columns are
allocated. Sampled leaf rewards initialize `v_vector_hat` directly. In RB,

$$
\widehat M_p(-_p,\mathsf s)
=\widehat{\mathbb P}(-_p\mid\mathsf s)
 \widehat V_p(-_p,\mathsf s),
$$

so `p_hat` is probability mass and `m_hat` is posterior-weighted value moment.
The workspace is sized to the exact maximum simultaneous adjacent layers and
swaps source/destination roles after each launch. Leaf completion initializes
the applicable `p_hat`, `m_hat`, `v_vector_hat`, zero cost, and zero delta
rows. Child sums, leave-one-out sums, $\overline\Delta$,
$\widehat T^{\rm exec}$, conditionalized continuations, and baselines remain
kernel-local reductions.

### Leaves, failures, and bounded workspaces

Stopped children are written directly into their final leaf schema:

```text
RB leaf
  board and completion-public fields
  range weights [P,H]
  stratum/payoff fields
  logical completion key

sampled leaf
  terminal reward [P]
  sequential trajectory or branched topology association
```

A stopped child never enters a player family, generic occurrence owner,
intermediate reach pool, or temporary leaf representation.

Each admitted attempt owns:

```text
DeviceFailure
  status
  phase
  failure_code
  semantic_family
  logical_address
  requested_extent
  available_extent
  sampler_attempt_count
```

Concurrent failures resolve to the minimum logical failing address. Every
PBG-owned transient workspace is preallocated from the immutable schema:
descriptor columns, route scans, destination counts, chance-frontier buffers,
completion microbatches, flattening scratch, `LayerWork`, and learner ragged
scratch. Model/compiler/library workspaces are calibrated and charged even
though PBG does not allocate them directly.

### Capacity and phase lifetimes

Capacities are generated from the complete mode schema, configured $P,R,S$,
model layouts, fixed workspaces, and lagged per-family observations. A
calibration plan may begin near

$$
\sum_m N_m^{\rm cap}=25{,}000
$$

for HU 32 bb, but that count is not an admission rule. At $P=2$, $H=1326$,
the postflop `[P,H]` range component of 25,000 natural inputs costs about
252.9 MiB; at $P=6$ it costs about 758.7 MiB. The complete ledger also charges
all other retained fields, exact topology and target extents, leaves, seal
overlap, every bounded workspace, backend reservation, alignment, and the
unused guard.

Family capacities remain fixed within an admitted attempt and are divided
only among semantic families, never poker streets. Objects with disjoint
lifetimes are not added:

```text
rollout
  FamilyWork hot and retained capacities
  growing topology and leaves
  routing/chance/inference workspace

targets
  rollout FamilyWork source capacities
  topology and leaves
  exact SigmaTarget and optional ActionTarget
  sequential RB posterior pool
  completion workspace and LayerWork

seal
  rollout FamilyWork source capacities
  completed SigmaTarget and optional ActionTarget
  exact compact learner FamilyWork extents
  sequential RB posterior pool
  minimal LearnerIndex destination
  bounded copy/validation scratch

optimization
  compact immutable learner FamilyWork
  minimal LearnerIndex
  sequential RB posterior pool
  SigmaTarget and optional ActionTarget
  model/reference/optimizer and ragged learner workspace
```

Rollout first closes publication without copying or sealing the source slabs.
After target propagation, release leaves, completion outputs, `LayerWork`,
`old_t`, and `old_c`. Seal then deletes every hot column and unwanted
best-response row, compacts the surviving learner columns, reduces sequential
or branched topology to `LearnerIndex`, and releases the full source slabs and
topology. Exact target and posterior arrays are already final-address outputs
and need not be recopied merely to align their row numbers with compacted
inputs. After the final optimizer step, release compact `FamilyWork`,
`LearnerIndex`, the posterior pool, `SigmaTarget`, and `ActionTarget`.

## Rollout execution

### Mode behavior

Sequential geometry admits a fixed root set before current draws. Offspring
below the root is one. A capacity failure never drops, replaces, or rerolls a
trajectory.

Sampled sequential mode retains scalar sampled loss inputs and terminal
rewards.
Sequential RB additionally retains completion-ready leaves and decision-only
chronological topology; bounded completion outputs are consumed while writing
the shared posterior pool and exact pre/post `SigmaTarget` rows.

Branched geometry creates exactly one root. Every non-preterminal node,
including that root, uses the ordinary branching rule. In branched RB, every
logical child samples an independent collision-conditioned posterior profile.
In sampled branched geometry, the row-lifetime true profile follows that
mode's declared tree law and no posterior-profile sampler is invoked.
Duplicate public events remain distinct topology nodes with independent RNG
addresses, descendants, and completion keys. Lower-size refill records the
protected pre-refill root degree, rewrites final root transition/path masses
using the augmented root degree, and leaves ordinary descendant decisions
trainable.

The engine is a shared physical protocol, not a union schema. Sequential rows
do not carry branched-only path, controller, topology, or profile-sampler
fields.

Best response is a training-regime overlay on either geometry. Exploitator
rows use the current model and retain learner inputs. Opponent rows use the
frozen model. Their natural inputs occupy the same row-aligned retained column
during rollout and target creation but are deleted at post-target seal.
Stable consecutive role subspans select the current or frozen model without
reordering the family FIFO. Sequential best response retains no opponent
output. Branched best response retains only the frozen-player `old_t` and
`old_c` scalars needed by the public target recurrence.

### Parent transaction

For one stable player-family prefix:

1. reserve nominal/executed child counts, logical topology spans, RNG
   identities, and exact player-edge spans;
2. run the role-selected policy forward on the direct/compact family input;
3. write `old_v` and sampled-action inputs for trainable rows; in branched
   geometry also retain `old_t` and `old_c` for every player node; discard
   every other frozen-opponent output after sampling;
4. construct lightweight child descriptors;
5. count and scan semantic destinations;
6. reserve disjoint destination spans;
7. materialize large payloads directly at those final addresses;
8. atomically commit the complete multi-destination transaction.

The complete parent policy output is never expanded to child shape. Descriptor
rows carry a parent-position index and the selected action.

Logical reservations occur before random draws. Physical destination
reservations occur after classification reveals their exact family, leaf, or
chance destination. A failed transaction may leave provisional logical or
physical reservations and writes, but no ready tail advances. Sequential mode
stops cleanly; branched mode discards the whole attempt through the declared
recovery boundary.

### Stage 1: lightweight child descriptors

For each logical child, Stage 1 produces only bounded routing data:

```text
ChildDescriptor
  parent_position
  node_id
  logical_key
  ephemeral posterior profile ids              # branched RB only; omit if fused
  action or chance event
  old_log_pi source address
  compact state delta/classification fields
  destination
  destination-local ordinal
```

In the other three geometry/estimator combinations, Stage 1 reads the
row-lifetime true profile instead and the descriptor has no profile field.
The applicable profile source, public event draw, compact
transition/classification work, and chance-closure decision should be fused as
far as their shared device code and register pressure permit.

Crucially, Stage 1 does not materialize a large child range payload. Stable
destination addresses are computed from small descriptors before the
bandwidth-heavy natural-input write.

### Shared sampler device core

Branched-RB collision-conditioned profile sampling must have one canonical
implementation and retain the sibling evaluator's statistical, RNG-version,
legality, exhaustion, and row-local reproducibility contract.

The sampler's CUDA device core should be factored into one shared source
artifact that can be compiled into:

- the sibling evaluator's public `sample_joint_profiles` operation; and
- PBG's fused Stage 1/chance-closure kernel.

PBG owns the surrounding poker transition, descriptor, routing, and
chance-closure logic. The shared sampler core supplies accepted combo IDs in
registers or shared memory and reports exhaustion through the common device
failure record.

This rollout sampler is invoked only in branched RB. Sequential sampled,
sequential RB, and sampled branched execution use their carried true profile
and must perform zero rejection-sampler calls. Completion remains a separate
consumer of the sibling evaluator in the RB modes.

A literal copied implementation must not remain independently maintained in
both repositories. If ownership moves, both callers must compile or call the
one moved source and the authority documents must change with it.

This direct device integration is preferred over returning a newly allocated
`combo_ids[B,M,P]` tensor. A preallocated output-buffer API is an acceptable
development step, but it is not the intended final fused path.

### Stable destination routing

For a child pack small enough for one CTA:

1. materialize descriptors and classify one or more children per lane;
2. count destinations in shared memory;
3. compute stable within-destination offsets;
4. reserve one contiguous span per nonempty destination;
5. perform Stage 2 final writes;
6. commit all destinations together.

For larger packs, use a bounded device pipeline:

```text
descriptor generation
→ stable scan/partition of small descriptors
→ destination reservation
→ Stage 2 final materialization
→ transaction commit
```

The initial large-pack path should use ordinary fixed-workspace kernels.
Cooperative persistent execution or graph capture may replace launch
boundaries only after a paired end-to-end benchmark.

Cross-block atomics may count destinations, but they may not determine
insertion order from nondeterministic block scheduling. FIFO order follows
logical child order and the deterministic reservation sequence.

### Stage 2: final materialization

Once every destination base is known, Stage 2:

- reads the parent's natural input directly from its family row;
- applies the selected event to its public-trajectory and range fields;
- writes the complete child natural input once at a player destination;
- writes completion-ready fields at a stopped child;
- forms only the bounded descriptor/frontier state needed by a continuing
  chance child.

This two-stage pipeline is the default for large packs because it minimizes
large global-memory traffic. Kernel fusion is accepted when it removes a
transient tensor or global round trip, not merely because it reduces launch
count.

### Device-resident chance closure

Chance children remain in a bounded device workspace rather than a Python
queue. Successive chance generations repeat the descriptor, stable-route, and
final-write protocol until the chance frontier is empty.

The host observes only the final committed family/leaf extents and any failure
record at the declared control boundary. It does not loop over individual
chance generations or synchronize between sampler, transition, and routing
stages.

Poker chance closure continues to respect preterminal stopping. It does not
locally replace sibling-owned completion, posterior, equity, or showdown
evaluation.

### Natural-input range representation

The preferred natural-input representation is normalized or safely scaled
linear range weight on the family's native domain:

- the branched-RB sibling sampler and RB completion APIs consume finite
  nonnegative linear weights;
- each nonzero seat row is invariant to a finite positive scale;
- player transitions multiply the acting row by the selected policy
  likelihood and normalize it;
- stopped children write exact completion-ready weights directly into the leaf
  schema.

Preflop input stores class weights on $K$. Exact-hand sampling or a transition
out of preflop derives the symmetry-preserving $H$ view in bounded transient
workspace. Postflop input stores exact live-player weights on $H$ and retains
the natural-input reference to any folded-player summary. No mode retains a
second “active reach” beside these natural-input fields.

The representation is an empirical choice, not a dependency of the queued
architecture. If long-trajectory, short-deck, zero-support, distributional,
target, loss, or gradient tests reject linear fp32 range weights, the
natural-input schema uses log weights instead. There is one accepted
production representation, not two runtime fallback paths.

### Asynchrony and streams

Fixed addresses and explicit reservations permit overlap, but stream count is
not a design invariant.

The allocation-free eager implementation should begin with the simplest
deterministic issue and commit order. Model forward, descriptor work, final
materialization, and append/seal copies may use separate streams only where:

- source and destination lifetimes are protected by events or
  `record_stream`;
- commit visibility remains deterministic;
- queue-pressure decisions do not depend on completion timing;
- the overlap improves complete-iteration throughput;
- retained workspaces do not reduce feasible training width.

There is no undeclared device-wide synchronization in the ordinary rollout
loop.

## Flattened branched target engine

The current CPU ready-counter, per-parent, and per-action target loops are
replaced completely. After one topology-only flattening phase, every numerical
recurrence runs on GPU in deepest-to-root dependency-layer order.

Completion remains a separate bounded service. The host may advance the
root-depth layer loop, but it performs no numerical recurrence and constructs
no per-parent device tensors.

### Topology flattening

Build the exact `FlattenedTargetTree` defined in the data-structure section.
Within each root depth, preserve stable source-node and child-ordinal order.
Every direct child lies in the next root-depth layer, every sibling span
remains contiguous, and retained action inputs and `ActionTarget` rows remain
aligned with the target-eligible child prefix. The protected root uses its
pre-refill target degree; refill-only root children remain ordinary learning
data below the root but do not receive root action targets.

Flattening is a deliberate locality optimization. The scalar permutation and
topology rebuild are small compared with numerical target rows and make layer,
sibling, edge, and learner access predictable.

The first implementation may use one bulk CPU transfer of scalar topology,
perform a deterministic linear construction, and upload the result once. If
that becomes visible on H100, replace it with a device histogram/stable sort
without changing the flattened schema.

### Preflop class aggregation

Let $c:\mathcal H\to\mathcal K$ be the ordinary preflop class map. For exact
posterior probability mass and posterior-weighted value moment, define

$$
(Gz)_k=\sum_{h:c(h)=k}z_h,
$$

and aggregate before conditionalization:

$$
\widehat{\mathbb P}_{p,k}
=\sum_{h:c(h)=k}\widehat{\mathbb P}_{p,h},
\qquad
\widehat M_{p,k}
=\sum_{h:c(h)=k}\widehat M_{p,h}.
$$

Then

$$
\widehat V_{p,k}
=\operatorname{Cond}(\widehat M_{p,k},
                       \widehat{\mathbb P}_{p,k}).
$$

The same one-way aggregation seeds an RB preflop completion leaf and executes
the RB `pre <- post` boundary. The RB learner sums preflop losses over $K$
classes. Sampled target propagation has no private-state axis and performs no
$H\to K$ aggregation. These equations are the proposal's intentional
estimator change; the remaining recurrences are referenced from
`core_math.tex` rather than duplicated here.


### Layer kernel

For each non-leaf depth, one mixed-domain custom launch initially processes
every internal parent in that layer. A warp, CTA, or measured bounded group
owns one degree-ragged parent and:

1. reduces child `v_vector_hat`, `c_lambda_hat_work`, and `delta`, and in RB
   also reduces `p_hat` and `m_hat`;
2. writes the parent's transient propagated `LayerWork` row;
3. when `sigma_target_row >= 0`, writes scalar `v_lambda_hat`, `t_hat`, and
   `c_lambda_hat` in sampled mode, or additionally writes vector `p_hat` in
   RB;
4. when `action_target_begin >= 0`, writes each target child's scalar
   `a_lambda_hat` in sampled mode, or additionally writes vector `p_hat` in
   RB;
5. advances the acting player's scalar value recurrence in sampled mode; in
   RB, bootstraps the acting player's `m_hat` with
   `FamilyWork.retained.old_v`; best response does so only at exploitator
   nodes;
6. at every player node, bootstraps `c_lambda_hat_work` with
   `FamilyWork.retained.old_c`;
7. uses `FamilyWork.retained.old_t` as $T^*$ only when
   `target_degree == 1`;
8. applies the protected root prefix exactly.

The arithmetic is exactly `\eqref{eq:branched-backward-updates}`,
`\eqref{eq:branched-targets}`,
`\eqref{eq:branched-trace-variance-target}`, and
`\algref{alg:branched-value}`, with the approved $H\to K$ aggregation inserted
at the domain boundary.

Launch count is proportional to tree depth, never node or action count. A
small fixed number of pre/post or degree-specialized launches per layer is
allowed only if it wins end-to-end without changing storage or semantics.

After the root completes, target creation is complete. Release completion
output, `LayerWork`, `old_t`, and `old_c`; the post-target seal below then
reduces full topology to the minimal learner mapping.

### Post-target seal

Seal is executed only after all target rows and sequential-RB shared posterior
rows have been written and validated. It performs two deletions:

1. column deletion: discard every `FamilyWork.hot` column and every
   target-only retained column;
2. row deletion: in best response, stable-delete every frozen-opponent family,
   sigma, and action row.

It then copies the surviving `FamilyWork.retained` rows into exact immutable
extents. Self-play normally copies ordinary realized prefixes; best response
stable-compacts the exploitator subset. The default is an exact copy because
keeping full rollout capacities alive through repeated learner passes inflates
the optimization base. A zero-copy view is allowed only if measured peak
memory and throughput improve without reducing backward width.

Seal also constructs compact `LearnerIndex` rows containing only the
source/target row mappings, exact branched action spans, posterior references,
target degree, sampling mass, and structural validity needed by optimization.
It validates those mappings and then releases `SequentialTargetTree` or
`FlattenedTargetTree`, leaves, source construction maps, and seal scratch.
Family-input and target row numbers need not be aligned. If they happen to
align naturally, redundant mappings may be omitted, but alignment is not a
required optimization or a reason to complicate compaction.

Final partition-local decision masses are computed after any root refill.
Their log weights survive only through construction of the exact learner
sampler and its normalization; transient path/controller factors are then
released. The successful seal atomically publishes one complete immutable
training store.

### Exact ragged learner

For a selected retained sigma subspan $J$, the learner follows
`LearnerIndex` mappings and exact target-child spans. In one preallocated
workspace it
constructs:

```text
local_edge_ptr[|J| + 1]
child_row[E_J]
edge_to_local_decision[E_J]
```

where $E_J$ is the exact sum of selected `target_degree` values. The planner
admits both decision and edge counts and selects the largest feasible
consecutive decision subspan.

The current-policy forward remains sigma-major. Action policy work gathers
only real `FamilyWork.retained.action`,
`FamilyWork.retained.old_log_pi`, and `ActionTarget.a_lambda_hat` rows, plus
`ActionTarget.p_hat` in RB, then reduces contributions back to sigma rows with
segmented reductions.

RB value, entropy, and optional reference KL terms read
`SigmaTarget.p_hat`; sampled losses use the retained realized $x_p$ and have no
posterior row.
$T/C$ losses read `SigmaTarget.t_hat` and
`SigmaTarget.c_lambda_hat`; `t_hat` is gated by structural target degree.

There is no immutable or transient `[decision,max_actions,...]` tensor, no
invented safe action, no inactive event evaluation, and no
`max_actions_per_decision` production API.

## Memory admission and failure behavior

Admission is phase-specific and generated from allocated schemas rather than
handwritten payload estimates:

```text
rollout
  model/static
  complete FamilyWork hot and retained capacities
  growing topology and leaf capacities
  descriptor/routing/chance workspaces
  inference workspace

targets
  rollout FamilyWork source capacities
  sequential or flattened branched topology
  exact SigmaTarget and optional ActionTarget
  sequential RB posterior pool
  completion workspace
  maximum LayerWork

seal
  rollout FamilyWork source capacities
  completed SigmaTarget and optional ActionTarget
  sequential RB posterior pool
  exact compact FamilyWork and LearnerIndex destinations
  bounded copy/validation scratch

optimization
  compact immutable FamilyWork learner inputs
  LearnerIndex
  sequential RB posterior pool
  SigmaTarget and optional ActionTarget
  gradient/collective buffers
  one bounded ragged learner subspan
  model/reference/optimizer workspaces
```

Objects with disjoint lifetimes are not added. Allocator-reserved and NVML
peaks, graph-owned pools, compiler/library workspaces, alignment, and the
unused guard are included.

### Sequential failure

Sequential root width and logical keys are frozen before current draws. If a
family, leaf, seal, completion, or learner transaction unexpectedly cannot
fit, the run releases the current iteration, checkpoints the last completed
iteration, and stops. It does not drop, replace, retry, or reroll a trajectory.

### Branched failure

A family, topology, edge, leaf, descriptor, chance, seal, completion, or target
workspace overflow is a classified whole-attempt admission failure. Every rank
releases the entire attempt, agrees on the controller backoff and disjoint next
attempt address, and retries within the bounded configured limit.

Sampler exhaustion and non-memory correctness/evaluator failures remain
terminal unless their authoritative contract explicitly classifies otherwise.

### Transactional publication

One parent transaction may target several families, leaves, and chance
frontiers. It publishes none of them until:

- every exact destination extent fits;
- sampler and transition stages report success;
- every destination column has been written;
- required stream dependencies are complete.

Reserved tails may have advanced before failure; committed tails do not.
Unpublished bytes are not learner-visible and are discarded with the failed
sequential iteration or branched attempt.

## CUDA graphs and optional overlap

Graph capture is an optimization over the allocation-free eager path, never a
correctness dependency or storage authority.

Candidate capture units are:

- a common fixed-address family forward pack;
- the large-pack descriptor/scan/materialization pipeline;
- one bounded chance-generation step;
- one fixed-shape ragged learner microbatch.

Do not capture the complete rollout iteration: scheduler choices, realized
offspring, family mixtures, layer widths, completion counts, and distributed
control are genuinely dynamic.

A capture is retained separately per supported hardware profile only when a
paired end-to-end benchmark includes graph pools, warmup, recapture, tails,
and batch-width effects and still wins. On the 2080 Ti, reject a capture that
reduces the largest valuable forward or backward width. H100 may use more
aggressive capture and overlap, but it receives no semantic or storage variant.

Persistent kernels, additional streams, degree bucketing, and learner
elementwise fusion obey the same rule: one accepted production choice after
measurement, not a permanent fallback matrix.