# Queued training rewrite, in iteration order

Status: implementation proposal; not yet an authoritative production contract.

Date: 2026-07-24.

This is a chronological presentation of `docs/todo/queued_training_rewrite.md`. It proposes the same replacement:
normal, sequential Rao--Blackwell (RB), and branched RB training use one fixed-address queued engine, and the legacy
production path is deleted after all consumers migrate.

## 1. Fundamentals

### 1.1 One schema, four independent axes

Every allocation is generated from:

```text
training_regime       self_play | best_response
rollout_geometry      sequential | branched
private_estimator     sampled | rao_blackwell
beta_mode             fixed_beta | KKT_beta       # branched only
```

Inapplicable fields do not exist. The shared engine is a protocol, not a union of every mode's data. `beta_mode`
changes only whether frozen trace-variance and cost heads control branching intensity. Branched training always learns
those heads.

Throughout this document, a data-structure block is a catalog of every field that the structure can have in any mode.
Mode-dependent fields remain in that one block and carry an applicability comment such as `# branched only` or
`# except branched RB`. These are documentation supersets, not runtime union schemas: schema generation omits every
field whose applicability condition is false.

### 1.2 `input_partition` and `output_partition`

Fix a run and let $\Sigma$ be its set of possible player-node occurrences. The game and estimator declare two literal
partitions of the same $\Sigma$:

$$
\Pi_{\rm in}=\{I_i:i\in\mathcal I\},
\qquad
\Pi_{\rm out}=\{O_o:o\in\mathcal O\}.
$$

Here $\mathcal I$ is the set of `input_type` labels and $\mathcal O$ is the set of `output_type` labels. An
`input_type` or `output_type` below means its labeled partition block when no confusion can result.

The input partition must refine the output partition:

$$
\Pi_{\rm in}\preceq\Pi_{\rm out}
\quad\Longleftrightarrow\quad
\forall i\in\mathcal I\;\exists!\,o\in\mathcal O:\ I_i\subseteq O_o.
$$

This refinement induces the total map

$$
\operatorname{output}:\mathcal I\longrightarrow\mathcal O,
\qquad
\operatorname{output}(i)=o
\ \Longleftrightarrow\
I_i\subseteq O_o.
$$

A declaration is invalid if an input block crosses output blocks. Thus `input_partition` is always at least as fine
as `output_partition`; an `output_type` may combine several complete `input_type` blocks but may never split one.

Each `input_type` owns one model-work FIFO. Queue identity is therefore semantic and game-defined.

Street, observation width, history length, token width, padding width, and their cross-products never create extra
queues. Within the next stable FIFO prefix, the game may choose the minimum physical width covering that prefix or
split it into consecutive feasible subspans. It may never skip or reorder rows to improve padding.

Numerical target arrays are partitioned by `output_type`. Sampled mode has a singleton `output_partition`, whose rows
are scalar. Rao--Blackwell (RB) mode may have several `output_type` elements with different private-state domains.

The two partitions serve different purposes:

- `input_type` selects a model-input schema and schedules model execution.
- `output_type` selects a numerical target domain and target storage.
- A sigma row of `input_type` $i$ has `output_type` $\operatorname{output}(i)$. Storage may materialize both labels for
  routing, but they are not independent.
- Output partitioning never creates rollout queues.

For poker,

$$
\mathcal I
=\{\mathsf{pre}\}\sqcup\{\mathsf{post}_A:2\le A\le P\},
$$

while the estimator gives

```text
sampled
  output_partition     = {all}
  output(pre)          = all
  output(post_A)       = all             for every A

rao_blackwell
  output_partition     = {pre, post}
  output(pre)          = pre
  output(post_A)       = post            for every A
```

Flop, turn, and river rows with the same live-player count $A$ share the `post_A` `input_type`. All postflop RB targets
use the `post` `output_type`, independent of $A$. Thus `post_2` and `post_6` are distinct `input_type` elements but map
to the same `output_type`.

For a poker `post_A` prefix, the game-owned planner may use the minimum board width in that prefix to select the
smallest legal exact-hand width covering every row.

### 1.3 Input sizes, output sizes, and exact private states

Each type has one declared size, but input and output sizes measure different objects:

$$
\operatorname{size}_{\rm in}[i]
  :=\dim(\operatorname{input\_schema}(i)),
\quad i\in\mathcal I,
\qquad
\mathcal D_o
  :=\operatorname{output\_domain}(o),
\quad
\operatorname{size}_{\rm out}[o]
  :=|\mathcal D_o|,
\quad o\in\mathcal O.
$$

In typed schema notation these may be shortened to `size[input_type]` and `size[output_type]`; the subscripts are
retained in mathematics whenever labels such as `pre` occur in both partitions. Input size is the logical flattened
row width. Output size is the trailing numerical width of one row, not the number of rows in its slab. The byte/work
contract separately applies field dtypes, alignment, and pack-specific transient widths.

For poker, let

$$
B_i
=\max_{\sigma\in I_i}
  |\operatorname{public/history}(\sigma)|
$$

denote the fixed schema capacity of all non-range input fields, including public state, board/history fields, and
immutable dependency references. The logical input sizes are schematically

$$
\operatorname{size}_{\rm in}[\mathsf{pre}]
  =PK+B_{\mathsf{pre}},
\qquad
\operatorname{size}_{\rm in}[\mathsf{post}_A]
  =AH+B_{\mathsf{post}_A}.
$$

Thus, in particular, a `post_A` natural-input row has the requested $A H$ live-range coordinates plus the maximum
public/history extent for that input block. Here $K=R^2$ and $H=\binom{4R}{2}$.

The output sizes are

```text
sampled
  size_out[all]          = 1

rao_blackwell
  size_out[pre]          = K
  size_out[post]         = H
```

After the model forward and any estimator-specific selection, the retained value and selected-action log-probability
rows for `input_type` $i$ lie on $\mathcal D_{\operatorname{output}(i)}$. Targets of `output_type` $o$ use that same
$\mathcal D_o$. Consequently, combining several input blocks into one output block is sound precisely because their
retained numerical rows share one output domain. The model's transient unsliced head may be wider in sampled mode.

Keep the following three objects distinct:

- `private_domain`: the game's exact private-state domain.
- `input_schema(input_type)`: the natural model-input row, of width `size[input_type]`.
- `output_domain(output_type)`: the retained numerical and target/loss domain, of width
  `size[output_type]`.

For poker, `private_domain` is $\mathcal H$, and a realized $x_p$ is always an exact two-card hand in $\mathcal H$,
including preflop. Sampled targets have no private-state axis. Preflop sampled training therefore carries exact
$x_p\in\mathcal H$ while selecting the scalar policy/value entry associated with model class
$c(x_p)\in\mathcal K$. In RB, `output(pre)` uses $\mathcal K$ and `output(post_A)` uses $\mathcal H$.

Board-empty poker nodes, including the flop-dealing chance parent and preflop completion leaves, use the `pre`
`output_type`. Flop, turn, and river nodes use the `post` `output_type`.

The poker RB recurrence conversions are:

```text
post <- post    H -> H
pre  <- pre     K -> K
pre  <- post    H -> K
```

There is no `post <- pre`.

### 1.4 Sigma indexing and action indexing

Let $\sigma$ be a player-node occurrence and let $\sigma a$ be one sampled action occurrence at that player node.

- `FamilyWork` is **sigma-indexed**: every row represents one $\sigma$.
- `ActionOccurence` is **action-indexed** and exists only in branched
  geometry: every row represents one $\sigma a$.
- `SigmaTarget` is sigma-indexed over trainable $\sigma$ rows.
- `ActionTarget` is action-indexed over target-eligible $\sigma a$ rows.

Every sigma row that owns action rows points to one exact contiguous action span. No padded
`[decision,max_actions,...]` array exists.

Sequential geometry has exactly one retained sampled action per $\sigma$, so `action` and `old_log_pi` occupy the
corresponding `FamilyWork` sigma row. Branched geometry instead uses a ragged `ActionOccurence` span of exact
`target_degree`.

Below, `TreeTopology` is the append-only branched rollout structure, `SequentialTargetTopology` is the sequential
decision chronology, and `FlattenedTargetTopology` is the depth-major branched target structure. The last two were
called `SequentialTargetTree` and `FlattenedTargetTree` in the original document.

### 1.5 Authority and intentional contract changes

`docs/theory/core_math.tex` remains authoritative for mathematics and `docs/theory/pseudocode/` for implementation
philosophy. Production acceptance requires explicit amendments for two intentional changes:

1. `FamilyWork` slabs are sized by admitted total input work and advance
   monotonically. They replace separate mutable public-state and active-reach
   owners, rather than tracking only the live-frontier high water.
2. RB preflop posteriors, moments, values, advantages, and losses use the
   model-native class domain $\mathcal K$. This changes the finite-sample
   estimator and nonlinear clipped loss, not just storage.

Everything else keeps the original contract: branching and queue-pressure laws, sampling laws, completion, and
duplicate-child identity. There is no legacy path, production $\widehat Q^\lambda$, or CPU/automatic/fallback
execution.

## 2. Before rollout: derive capacities and allocate fixed owners

Admission derives exact schemas and phase ledgers from the run configuration, model/workspace calibration, lagged
per-`input_type` observations, alignment, allocator reservation, and an unused guard.

Allocate:

```text
FamilyWork[input_type]
ActionOccurence[output_type]                  # branched only
TreeTopology                              # branched only
Leaf[terminal output_type]
DependencyStore                           # game-declared immutable dependencies only
QueueBehaviorMetadata                     # queue-pressure safeguard only
DeviceFailure

descriptor workspace
routing workspace
chance-frontier workspace
inference workspace
completion workspace                      # RB only
```

Let

$$
\mathcal O_{\rm term}
=\{o\in\mathcal O:\text{a stopped child can have output type }o\}.
$$

The game/estimator terminal-boundary rule assigns each stopped child its output type; unlike
$\operatorname{output}(i)$, this assignment does not imply that the stopped child owns a model-input row.
There is one `FamilyWork` slab per $i\in\mathcal I$, in branched geometry one `ActionOccurence` slab per applicable
$o\in\mathcal O$, and one `Leaf` slab only per $o\in\mathcal O_{\rm term}$. The notation
`Leaf[terminal output_type]` means exactly this restriction; there is no leaf allocation for a nonterminal-only
output block. Every row gets its final address when created, and addresses are not recycled within an admitted
attempt.

For each `input_type`:

$$
0
\le\mathtt{head[input\_type]}
\le\mathtt{readyTail[input\_type]}
\le\mathtt{reservedTail[input\_type]}
\le N_{\mathtt{input\_type}}^{\rm cap}.
$$

`[head,ready_tail)` is committed FIFO work and `[ready_tail,reserved_tail)` is reserved but unpublished. Publication
exposes only the longest consecutively completed reservation prefix.

`FamilyWork[input_type]` is a batch-major, sigma-indexed structure of arrays:

```text
FamilyWork[input_type]
  hot
    logical node/root identity
    logical RNG path key
    role/model selector                    # best_response only
    true_profile[P]                        # except branched RB
    mode-specific controller/path fields
    reservation/publication metadata

  retained
    natural_input[size[input_type]]
    x_p                                   # sampled only
    owner
    old_v[size[output(input_type)]]
    action                                # sequential only
    old_log_pi[size[output(input_type)]]  # sequential only
    old_t                                 # branched only
    old_c                                 # branched only
```

The whole structure is sigma-indexed; `hot` and `retained` are column lifetimes, not row kinds or queues.
`natural_input` is an opaque fixed-stride schema selected only by `input_type`, containing the game-defined public
state, history, range representation, and immutable dependency references.

The realized $x_p$ is a separate retained column only for the sampled private-state estimator; it is not part of
`natural_input`.

In branched geometry, `ActionOccurence[output_type]` contains:

```text
ActionOccurence[output_type]
  action
  old_log_pi[size[output_type]]
```

Each trainable branched sigma row reserves one exact contiguous `ActionOccurence` span before draws. For a sigma row
in `FamilyWork[i]`, its action rows reside in `ActionOccurence[output(i)]`; refinement guarantees that their numerical
width is `size[output(i)]`.

In branched geometry, `TreeTopology` is append-only and node-indexed:

```text
TreeTopology                               # branched only
  node
    parent
    child_begin
    degree
    nominal_degree
    target_degree
    node_kind
    leaf_row
    player_row

  player
    node
    familywork_sigma_row
    action_occurrence_begin
```

Topology rows and child spans are assigned when created and filled as their `input_type` runs. Sequential geometry
reconstructs its chronology from `FamilyWork.hot` root/path identities and needs no structural tree.

Best response creates action rows only for the learner's decisions. Frozen branched parents keep structural children
and `old_t`/`old_c`, but no learner sigma or action rows.

`natural_input` is the sole large state/range payload; no second active reach or public-state owner exists. Poker
currently uses normalized linear range weights. If tests require log weights, they replace the one production schema
rather than become a fallback.

`DependencyStore` exists only when a game schema contains immutable dependency references. In poker it owns folded
summaries. A dependency receives a final source address when created; descendant `natural_input` rows retain only its
reference. Seal finds dependencies reachable from surviving learner rows, copies each reachable dependency once into
an exact immutable extent, remaps those references, and releases the source store and unreachable rows. The compact
dependency extent survives until the last model-input consumer.

`QueueBehaviorMetadata` is a separate fixed owner, not a `FamilyWork` column or target field. When the queue-pressure
safeguard is enabled, it maintains the existing frontier-cost forecast and controller counters. Enqueue adds the
required C-head contribution and dequeue removes that same contribution under the existing queue law; any cached
per-entry contribution is private queue-manager bookkeeping.

## 3. Rollout: assign and fill final rows

### 3.1 Mode laws

- Sequential modes freeze roots and logical keys before draws, sample one
  $x\sim\mu$ per root, and carry it unchanged. Only the root branches.
- Branched mode creates one labeled tree rooted at $\varnothing$; the root uses
  the ordinary randomized-rounding rule. Duplicate public events remain
  distinct nodes.
- Sampled branched mode carries its declared tree-law profile. Branched RB
  samples a fresh collision-conditioned posterior profile per logical child
  and discards it after materialization.
- Root refill protects the pre-refill target degree, then recomputes final root
  transition/path masses using the augmented realized degree. Refill-only root
  children receive no root action target, but their descendant decisions
  remain trainable.
- Best response selects current/frozen models through stable FIFO subspans.
  Frozen `FamilyWork` rows survive through targets and disappear at seal.

Sampled-branched and branched-best-response laws still need explicit theory contracts; currently only sequential best
response is authoritative.

### 3.2 One parent transaction

For the next stable `input_type` prefix:

1. Reserve child identities and, in branched geometry, logical child spans, topology rows, and exact action spans
   before random draws.
2. Run the role-selected model directly on the `input_type` prefix.
3. Write `old_v` for trainable sigma rows; in sequential geometry write the selected `action` and `old_log_pi` in
   those same rows; write `old_t` and `old_c` for every branched player row.
4. In branched geometry, sample actions and immediately fill their reserved `ActionOccurence` rows with `action` and
   `old_log_pi`.
5. Produce lightweight child descriptors and classify destinations.
6. Compute stable destination offsets and reserve exact `FamilyWork`, `Leaf`,
   or bounded chance-frontier spans.
7. Materialize each large child payload once at its final destination.
8. Publish every destination together only after the whole transaction is
   complete.

The full policy output is never expanded to child shape. Descriptors contain only routing/transition scalars and, when
unavoidable, ephemeral branched-RB combo IDs. Logical reservations precede draws; exact physical reservations follow
classification.

Before any ready tail advances, every destination extent must fit, sampling and transition must report success, every
required destination column must be written, and all required stream dependencies must complete. All destinations
then publish together. Failure may leave reserved, unpublished bytes, but never advances a committed tail; those bytes
are discarded with the failed iteration or attempt.

The partition abstraction remains the core/game boundary. Core removes a stable `input_type` prefix and invokes its
declared model and game hooks. The game may fuse action/event sampling, transition, chance closure, range
rescaling/normalization, destination classification, and final writes, but every resulting player row lands directly
in `FamilyWork[next_input_type]` and every stopped row in `Leaf[terminal output_type]`. Core reasons only about
declared schemas, sizes, mappings, and byte/work contracts.

For poker, this means that `pre` batches carry class-native ranges and `post_A` batches carry live-player exact-hand
ranges. A transition may derive a bounded symmetry-preserving exact-hand view, multiply an acting range by the
selected policy likelihood, safely normalize or rescale it, close intervening chance events, and write either the next
`pre`/`post_A` row or completion-ready leaf fields. Folded-player summaries remain immutable referenced dependencies
with an explicit last consumer and seal remapping; none of these poker details creates another core queue.

### 3.3 Stable routing and poker CUDA fusion

For a one-CTA pack, fuse descriptor generation, destination counting, stable offsets, reservations, final writes, and
commit when register/shared-memory pressure permits.

Larger packs use bounded fixed workspaces:

```text
descriptor generation
-> stable scan/routing
-> destination reservation
-> final materialization
-> transaction commit
```

Atomics may count destinations, but deterministic logical order fixes insertion order.

Branched-RB rejection sampling, transition, classification, and chance closure should fuse. Its collision-conditioned
sampler shares one CUDA device source with the sibling evaluator's `sample_joint_profiles`, preserving legality, RNG
version, exhaustion, and row-local reproducibility. Failures use `DeviceFailure`.

The other geometry/estimator combinations make zero rejection-sampler calls. RB completion remains a separate
sibling-evaluator consumer. A preallocated accepted-ID buffer is only an intermediate step toward the fused path.

### 3.4 Device-resident chance closure

Chance children remain in a bounded GPU frontier and repeat the same route and commit protocol until empty. The host
does not synchronize per generation.

Stopped children write directly to final leaf rows of their terminal output type:

```text
Leaf[terminal output_type]
  terminal_reward[P]                      # sampled only
  trajectory_or_topology_row              # sampled only
  completion_public_fields                # RB only
  range_weights                           # RB only
  stratum_fields                          # RB only
  payoff_fields                           # RB only
  logical_completion_key                  # RB only
```

A stopped child never visits a generic occurrence store or temporary leaf format.

When all `input_type` queues and chance frontiers are empty, close rollout publication. Do not seal yet: target
creation still consumes retained `FamilyWork` columns, topology, and leaves.

## 4. After rollout: create target-addressed storage

Allocate exact extents from realized work:

```text
SequentialTargetTopology                      # sequential only
FlattenedTargetTopology                       # branched only
SigmaTarget[output_type]
ActionTarget[output_type]                      # branched only
LayerWork[output_type]                         # branched only
SequentialPosterior[output_type]               # sequential RB only
```

Each array has one exact extent per applicable `output_type`.

`SigmaTarget` has one row per trainable $\sigma$; `ActionTarget` has one row per target-eligible branched $\sigma a$.
In best response, these are exclusively exploiter-owned rows; frozen-player nodes receive neither target. Target
address assignment already follows the stable order that seal will use. `SigmaTarget` and `ActionTarget` contain only
numerical target outputs. In branched geometry, `ActionOccurence` remains a separate input owner through optimization
rather than being copied into a target array.

Generated target contents are:

```text
SigmaTarget[output_type]
  v_lambda_hat[size[output_type]]
  t_hat                                     # branched only
  c_lambda_hat                              # branched only

ActionTarget[output_type]                    # branched only
  p_hat[size[output_type]]                  # RB only
  a_lambda_hat[size[output_type]]

SequentialPosterior[output_type]             # sequential RB only
  p_hat[size[output_type]]
```

Sampled targets are scalar. RB value/posterior fields have width `size[output_type]`, while the `old_v` row in
`FamilyWork[input_type]` has width `size[output(input_type)]`. Sequential RB stores posterior mass once for each
trajectory, player, and `output_type`, and derives advantage as `v_lambda_hat - old_v` within the common output
domain. Branched RB does not duplicate a posterior row in `SigmaTarget`: the sigma posterior used by value, entropy,
and reference-KL losses is the exact mean of its target-eligible `ActionTarget.p_hat` rows.

### 4.1 Sequential target topology

Build a decision-only chronological mapping:

```text
SequentialTargetTopology                  # sequential only
  trajectory_ptr
  input_type
  familywork_sigma_row
  output_type
  sigma_target_row
  posterior_row                          # RB only
  leaf_row
```

Chance and frozen-opponent decisions are absent because neither advances the retained player's clock in a realized
trajectory.

### 4.2 Branched flattening

`TreeTopology` owns append-only node identity, parent/child spans, node kind, leaf rows, target-eligible prefixes, and
mappings to input/player/action rows. Flatten it once into stable root-depth order:

```text
FlattenedTargetTopology                    # branched only
  layer_ptr[L+1]

  node
    TreeTopology.node columns
    output_type                              # present for every structural node

  player
    TreeTopology.player columns
    owner
    input_type
    retained_control_row
    trainable_decision_row

  trainable_decision
    player_row
    action_occurrence_begin
    sigma_target_row
    action_target_begin

  protected root metadata
```

Every child enters the next depth layer; sibling and target-action spans remain stable and contiguous. Frozen
best-response players keep structural/control rows but no learner mapping. Flattening permutations are scratch; a bulk
CPU scalar rebuild is acceptable until measurements justify a device stable sort.

The `output_type` on every structural node selects its numerical `LayerWork` block. Each parent recurrence therefore
knows both child and parent domains and applies the declared conversion (`post <- post`, `pre <- pre`, or
`pre <- post`) without requiring a target row on that node.

`LayerWork` holds only two adjacent depth layers:

```text
LayerWork[output_type]                     # branched only
  p_hat[P,Z]                              # RB only; omit P axis in best_response
  m_hat[P,Z]                              # RB only; omit P axis in best_response
  v_vector_hat[P]
  c_lambda_hat_work
  delta
```

Here $Z=\mathtt{size[output\_type]}$ in RB. Size the two alternating buffers to the exact largest adjacent-layer
requirement.

## 5. Completion: initialize leaves and direct target outputs

In RB, run sibling-evaluator completion in bounded microbatches.

Sampled leaves initialize `v_vector_hat` directly. RB completion initializes probability mass `p_hat`,
posterior-weighted moment

$$
\widehat M_p(-_p,\mathsf s)
=\widehat{\mathbb P}(-_p\mid\mathsf s)
 \widehat V_p(-_p,\mathsf s),
$$

plus `v_vector_hat`, zero cost, and zero delta. Sequential completion writes its exact `SigmaTarget` values and shared
`SequentialPosterior` rows directly.

For the RB `pre <- post` boundary, aggregate exact-hand mass and moment before conditionalization:

$$
(Gz)_k=\sum_{h:c(h)=k}z_h,\qquad
\widehat V_{p,k}
=\operatorname{Cond}\left(\sum_{h:c(h)=k}\widehat M_{p,h},
                           \sum_{h:c(h)=k}\widehat{\mathbb P}_{p,h}\right).
$$

The same aggregation seeds preflop completion leaves. Sampled mode performs no $H\to K$ aggregation.

## 6. Propagate deepest-to-root

Sequential targets batch equal reverse-time positions and use one CUDA launch per realized trajectory depth.

Branched targets use one mixed-domain CUDA launch per non-leaf depth. A warp, CTA, or measured bounded group owns one
ragged parent and:

1. reduces child value, cost, delta, and in RB mass/moment;
2. writes the parent's alternating `LayerWork` row;
3. writes sigma-indexed value, T, and C outputs;
4. writes each action-indexed advantage and optional RB posterior output;
5. bootstraps acting-player moment/value with `old_v`;
6. bootstraps cost with `old_c`;
7. uses `old_t` as $T^*$ only when `target_degree == 1`;
8. applies the protected root target prefix.

Arithmetic remains exactly the authoritative branched backward, target, trace-variance, and value recurrences, with
only the approved $H\to K$ aggregation inserted.

Launch count is proportional to tree/trajectory depth, never node or edge count. A small fixed number of domain- or
degree-specialized launches per depth is acceptable only when end-to-end measurements win.

After root completion, release leaves, completion outputs, `LayerWork`, `old_t`, and `old_c`.

## 7. Seal and atomically publish the learner store

Sealing begins only after every target and shared posterior row validates. Target addresses were assigned in this same
stable order, so alignment is construction-time bookkeeping rather than a target reshuffle.

1. Delete all `FamilyWork.hot` columns.
2. In best response, stable-delete frozen-opponent `FamilyWork` and sigma rows.
3. Stable-compact surviving `FamilyWork.retained` rows and seal
   branched `ActionOccurence` into exact immutable extents.
4. Within each `output_type`, use one deterministic concatenation of its
   contributing `input_type` subspans. A base offset then aligns each sealed
   `FamilyWork[input_type]` row with its `SigmaTarget` row without a per-row
   mapping. Align branched `ActionOccurence` rows one-for-one with
   `ActionTarget`; sequential `action` and `old_log_pi` are already columns of
   the aligned sigma row.
5. Preserve, for every sigma row, the pointer and exact count of its
   contiguous $\sigma a$ rows in every action-indexed array.
6. After any root refill, compute final partition-local decision masses.
   Retain their log weights through exact learner-sampler normalization, then
   release transient path/controller factors.
7. Reduce sequential or flattened topology to `LearnerIndex`.
8. Validate every alignment and action span.
9. Compact and remap reachable game dependencies, then release source slabs, source dependencies, full topology,
   construction maps, and seal scratch.
10. Atomically publish the complete immutable training store.

`LearnerIndex` retains only posterior references, exact action spans, target degree, sampling mass, and structural
gates that optimization needs.

No node, parent, child, or layer topology survives sealing.

`t_hat` is valid exactly when `target_degree >= 2`; no validity mask is stored. Action rows exist exactly for
target-eligible children.

Exact copying is the default because rollout capacity should not remain live through repeated learner passes. A
zero-copy view is allowed only if measured phase peak and optimization width improve.

## 8. Learner

The current-policy forward remains sigma-major. For each largest feasible consecutive sigma subspan $J$, construct in
one preallocated workspace:

```text
local_edge_ptr[|J|+1]
child_row[E_J]
edge_to_local_decision[E_J]
```

$E_J$ is the exact sum of action-span lengths. Admission constrains both sigma and edge counts.

Sequential work reads the unique `action` and `old_log_pi` from its sealed `FamilyWork` sigma row and derives
advantage from `SigmaTarget.v_lambda_hat - FamilyWork.old_v`. Branched action work additionally gathers advantage and
optional RB posterior rows from `ActionTarget`, then segment-reduces to sigma rows. Branched RB derives sigma
posterior mass as the exact mean over each action span; sequential RB reads `SequentialPosterior`. RB value, entropy,
and optional reference-KL terms use that sigma posterior mass. Sampled losses use the retained exact $x_p$. T/C losses
use `t_hat` and `c_lambda_hat`, structurally gating T by target degree.

There is no padding, invented safe action, inactive-event evaluation, or `max_actions_per_decision` production API.

After the final optimizer step, release compact `FamilyWork`, sealed branched `ActionOccurence`, compact game
dependencies, `LearnerIndex`, shared posterior rows, `SigmaTarget`, and `ActionTarget`.

## 9. Failure, memory, and execution policy

`DeviceFailure` is a fixed device record:

```text
status
phase
failure_code
input_type
logical_address
requested_extent
available_extent
sampler_attempt_count
```

Concurrent failures resolve to the minimum logical failing address.

Sequential overflow releases the iteration, checkpoints the last completed iteration, and stops. It never drops,
replaces, retries, or rerolls a trajectory.

Branched capacity overflow is a whole-attempt admission failure. All ranks release it, agree on controller backoff and
a disjoint next attempt address, then retry within the configured bound. Sampler exhaustion and evaluator or
correctness failures remain terminal unless their authoritative contract says otherwise.

For a structure-of-arrays owner $X$ with $N_X$ rows and fields $f$, define

$$
\operatorname{bytes}(X,N_X)
=\sum_{f\in X}
  \operatorname{align}\!\left(N_X\,b_f,a_f\right)
+M_X^{\rm metadata},
$$

where $b_f$ is the byte width of one field element including its fixed trailing shape and $a_f$ is that column's
required alignment. Alignment is applied to each allocation, not to a fictitious packed row.

For phase $\phi$, let $\mathcal L_\phi$ be its simultaneously live owners. Admission computes

$$
M_{\rm required}
=\max_\phi\left[
  \sum_{X\in\mathcal L_\phi}
    \operatorname{bytes}(X,N_X)
  +W_\phi^{\rm bounded}
  +W_\phi^{\rm external}
\right]
+M^{\rm guard}
\le M^{\rm budget}.
$$

Exact realized arrays use their realized row counts; capacity owners use admitted capacities. The ledger takes the
maximum of complete phases rather than adding objects with disjoint lifetimes:

```text
rollout
  model/static
  FamilyWork and applicable ActionOccurence capacities
  source DependencyStore and QueueBehaviorMetadata when applicable
  topology and leaf capacities
  descriptor/routing/chance/inference workspaces

targets
  rollout sources, topology, and leaves
  exact SigmaTarget, ActionTarget, and SequentialPosterior
  completion workspace and maximum LayerWork

seal
  rollout sources and completed targets
  exact compact FamilyWork and applicable ActionOccurence destinations
  source and exact compact dependency extents
  LearnerIndex destination and copy/validation scratch

optimization
  compact FamilyWork, sealed applicable ActionOccurence, targets, and LearnerIndex
  compact game dependencies
  model/reference/optimizer and gradient/collective buffers
  bounded ragged learner workspace
```

`W_\phi^{\rm external}` includes calibrated compiler/library workspaces, backend and allocator reservation, and
graph-owned pools even when PBG does not allocate them directly. Calibration reconciles allocator-reserved and
NVML-observed peaks. Alignment and the unused guard are mandatory charges.

Start with deterministic allocation-free eager execution. Add streams only when event-protected lifetimes,
deterministic publication, queue-pressure decisions, complete-iteration throughput, and feasible learner width all
remain sound. Ordinary rollout has no undeclared device-wide synchronization.

CUDA graphs are optimizations, never correctness or storage authorities. Candidate capture units are a fixed-address
`input_type` forward, the large-pack route pipeline, one bounded chance-generation step, and one fixed-shape learner
microbatch. Do not capture the whole dynamic iteration.

H100 may favor larger batches, more capture, and measured overlap. The 2080 Ti must reject any graph pool or overlap
schedule that reduces the most valuable forward/backward width; roughly 1 GiB of rollout slabs is acceptable when the
full phase ceiling still fits. Hardware profiles may change capacity, batch quantum, graph selection, and stream
schedule, but never semantics or storage layout.

Persistent kernels, degree specialization, extra streams, and elementwise fusion follow the same rule: keep one
measured production choice, not a permanent fallback matrix.
