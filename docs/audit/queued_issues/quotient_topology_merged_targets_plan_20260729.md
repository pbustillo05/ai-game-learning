# Quotient topology and merged-target implementation plan

Date: 2026-07-29

Status: design plan; no production implementation is included in this
document.

## 1. Decision and objective

The queued rollout should retain two related trees:

- `SampledTopology` is the duplicate-preserving stochastic occurrence tree.
  It contains every sampled copy and remains the authority for sampling,
  degrees, sibling sets, path masses, target propagation, and leaf completion.
- `Topology` is the image of the public-label map. It contains one row per
  distinct complete public game-tree label and one edge per distinct public
  event from that label.

The learner sees only `Topology` rows and `Topology` action edges. Targets from
all sampled occurrences with the same label are merged before the learner.
The configured loss, including any nonlinear term, is evaluated once against
the merged target. Reproducing the loss obtained by evaluating every
occurrence separately is not a goal.

The purpose is to increase sampled-tree size without making model inference
and learning scale with duplicate public labels. Model forward and backward
work should scale primarily with:

```text
number of distinct trained Topology rows
+ number of distinct trained Topology action edges
```

rather than:

```text
number of trained SampledTopology occurrences
+ number of target-eligible sampled action occurrences.
```

`Leaf` and numerical propagation work remain sampled-size. Persistent model
inputs, heads, final targets, learner indices, and learner batches become
labelled-size.

The unique labelled structure is named `Topology` throughout.

## 2. Mathematical objects

For one replica, let:

```text
S                 SampledTopology node set
E_S               SampledTopology edge set
T                 Topology node set
E_T               Topology edge set
l_node : S -> T   node label map
l_edge : E_S -> E_T
```

`l_node(s)` is the complete public game-tree label of occurrence `s`.
`l_edge(e)` is the unique labelled edge from `l_node(parent(e))` carrying the
same public event as `e`.

`Topology` is the image of `l_node`, not a state abstraction. Equality must
mean equality of the complete model-relevant public history. For poker this
includes, directly or by deterministic derivation:

- the complete betting and chance history;
- board cards;
- stacks, commitments, player status, street, and actor;
- input partition and numerical output domain;
- live/folded-seat structure;
- exact public reach state and folded summaries;
- best-response role/model authority; and
- every other retained model-input field.

Private `true_profile`, sampled actor hand, logical RNG address, randomized
degree, path mass, and completion sample are not part of the label. They are
properties of a sampled occurrence.

Merging is replica-local. Identical labels from different replicas cannot
share `Topology` rows because replicas can have different weights, controller
state, targets, and gradients.

For the exact public-history definition used here, every nonroot `Topology`
node has one unique labelled parent. Thus `Topology` is a tree or forest, not a
DAG. If a future model intentionally identifies histories with different
parents, that is a different abstraction and is outside this plan.

## 3. Required ownership split

The present `FamilyWork` row owns both model payload and occurrence-local
sampling payload. That is incompatible with a labelled-size `FamilyWork`.
Those responsibilities must be separated.

### 3.1 `SampledTopology`: occurrence authority

During rollout, every sampled node needs enough device-resident state to be
expanded without reconstructing an occurrence from a canonical row:

```text
SampledTopologyNode <sampled_topology_cap>
  parent                         # sampled node, -1 at a sampled root
  incoming_edge                  # sampled edge, -1 at a sampled root
  topology_row                   # l_node(s)
  logical_rng_key                # root and occurrence path identity
  true_profile <P>
  path_mass_or_log_mass
  log_player_ancestor_sum        # KKT, if required
  log_ancestor_product           # KKT, if required
  branch_beta
  nominal_degree
  executed_degree
  target_degree
  child_begin                    # into SampledTopologyEdge after expansion
  expansion_state
  refill/protected-prefix metadata
```

```text
SampledTopologyEdge <sampled_topology_edge_cap>
  parent                         # sampled parent occurrence
  child_ordinal                  # exact sibling identity and order
  event_label
  topology_edge                  # l_edge(e)
  target_kind                    # sampled node or Leaf
  target_row
  target_eligible
  path_length                    # sealed compact form
  endpoint_output_type           # sealed compact form
```

The hot form may retain additional transition state needed by poker. The
essential rule is that no occurrence-local profile, RNG identity, degree, or
sibling identity may move to `Topology`.

`SampledTopology` remains duplicate-preserving even when:

- two siblings draw the same action;
- two siblings draw the same chance event;
- different sampled copies of one label draw the same continuation;
- separate logical roots reach the same public history; or
- refill adds a continuation already present in `Topology`.

### 3.2 `Topology`: labelled authority

`Topology` owns one immutable public payload and one model row per distinct
label:

```text
TopologyNode <topology_cap>
  parent                         # unique labelled parent
  incoming_edge
  child_begin                    # distinct labelled children
  child_count
  label_identity
  input_type                     # -1 when no player model is evaluated
  output_type
  owner
  role/model authority
  familywork_row                 # player nodes only
  sigma_head_row                 # trained player nodes only
  sigma_target_row               # trained player nodes only
  action_begin                   # trained distinct action edges only
  action_count
  preimage_fragment_begin/count  # hot append metadata
  expanded_preimage_tail
  model_state                    # absent/reserved/published
```

```text
TopologyEdge <topology_edge_cap>
  parent
  child
  event_label
  action_head_row                # trained player edge only
  action_target_row              # trained player edge only
  preimage_fragment_begin/count
```

`Topology` does not store a single `N`, `N_tilde`, or `N_tgt`. Those quantities
are not functions of the label because randomized rounding, queue pressure,
and protected root refill are occurrence-specific. `child_count` is the
number of distinct observed labelled children and must never be interpreted
as a branching degree or as an averaging denominator for target propagation.

`FamilyWork`, `SigmaHead`, and model outputs are indexed from `Topology`.
Chance and other headless labels remain in `Topology` for structural identity
but own no `FamilyWork` row.

### 3.3 `TopologyMap`: explicit bidirectional relation

The map must be explicit in both directions. Reconstructing preimages by
searching labels after rollout is not sufficient for scheduling or target
reduction.

The sealed node map is:

```text
TopologyMap.node_to_topology <sampled_node_count>
TopologyMap.topology_to_node_ptr <topology_node_count + 1>
TopologyMap.topology_to_node <sampled_node_count>
```

with:

```text
node_to_topology[s] == t
iff
s occurs in topology_to_node[
    topology_to_node_ptr[t] : topology_to_node_ptr[t + 1]
]
```

The sealed edge map is analogous:

```text
TopologyMap.edge_to_topology <sampled_edge_count>
TopologyMap.topology_to_edge_ptr <topology_edge_count + 1>
TopologyMap.topology_to_edge <sampled_edge_count>
```

Additional maps are required at the target boundary:

```text
sampled_trained_node -> topology sigma target row
sampled target edge  -> topology action target row
leaf_to_topology <sampled_leaf_count>
topology_to_leaf_ptr <topology_node_count + 1>
topology_to_leaf <sampled_leaf_count>
```

The leaf CSR is the physical preimage for labelled preterminal/terminal nodes
whose sampled endpoints live in `Leaf` rather than in the internal sampled
node table.

The node and edge inverse arrays must be exact permutations of dense sampled
IDs. Validation proves:

- every sampled node appears in exactly one node preimage;
- every sampled edge appears in exactly one edge preimage;
- every inverse entry agrees with its forward map;
- a sampled edge's canonical parent is the canonical image of its sampled
  parent;
- its event label agrees with the `TopologyEdge`;
- its sampled child maps to the `TopologyEdge` child when the endpoint is a
  sampled node;
- every leaf row appears in exactly one leaf preimage and agrees with the
  canonical endpoint of its sampled incoming edge;
- every sampled root maps to a root-compatible `Topology` row; and
- no map crosses replica identity.

Raw pointers must not be stored in either direction. IDs remain valid through
append and sealing; compaction emits new explicit ID maps.

## 4. Hot map construction

### 4.1 Root creation

All logical roots are sampled first. Group them by complete public label
within each replica:

1. allocate every sampled root occurrence;
2. sort or classify roots by label;
3. allocate one `Topology` root per distinct label;
4. set every sampled root's `node_to_topology`; and
5. append the sampled roots to the corresponding canonical preimage.

Different root `true_profile` values do not prevent merging. Different
intrinsic public/model conditions do.

### 4.2 Expanding a canonical family row

A scheduled family item is a `Topology` player node plus a nonempty suffix of
its sampled-node preimage:

```text
preimage[
  expanded_preimage_tail : current_preimage_tail
]
```

The model is evaluated only if the canonical model output has not already
been published. The same output is then addressed by every sampled occurrence
in the suffix.

For sampled occurrence `s` in that suffix:

1. compute its independent randomized `N(s)` and related degrees;
2. allocate its exact logical child ordinals;
3. draw occurrence-local profiles/actions using `s.true_profile` and RNG key;
4. retain the ragged parent assignment;
5. chance-close each sampled child without losing occurrence identity; and
6. append all resulting sampled nodes, sampled edges, or leaves.

The physical sampler may flatten the ragged envelope, but every flattened
child carries:

```text
sampled_parent
child_ordinal
topology_parent
event_label
true_profile
logical_rng_key
```

Sampling merely `sum_s N(s)` unordered actions is invalid. The
`sampled_parent` and ordinal are required to reconstruct sibling sets,
leave-one-out baselines, protected prefixes, and path mass.

### 4.3 Grouping children into `Topology`

For each canonical parent and event label:

- look up or create exactly one `TopologyEdge`;
- look up or create its labelled child;
- set `edge_to_topology` for every sampled edge in the group;
- set `node_to_topology` for every nonleaf sampled child; and
- append sampled edges/nodes to the inverse-map fragments.

Player action and chance labels must be grouped at their actual game-tree
levels. Chance closure is a physical fusion and must not collapse the
semantic intermediate labels. If a sealed sampled edge later compresses
several unary nodes, the hot uncompressed map still retains every intermediate
canonical edge.

The labelled child must be validated against its canonical payload on every
collision. At minimum, input/output partition, owner/model authority, public
state, model input, and folded summaries must agree bitwise or by a declared
exact semantic comparison. A hash match alone is never sufficient.

### 4.4 Appendable preimages are necessary

A canonical preimage is not necessarily final after its first scheduling:

- a preimage can exceed one physical pack;
- later chunks can generate a child label seen in an earlier chunk;
- root refill can append new occurrences after the initial tree completed;
- a refilled branch can reach an already-created descendant label.

Therefore the hot inverse map cannot be a one-shot CSR. It needs appendable
fragments or chunk lists per canonical node and edge. Each canonical node owns
`expanded_preimage_tail`, so newly appended occurrences can schedule another
expansion suffix without repeating work for old occurrences.

The canonical model output is cached after its first publication. Reopening a
canonical node for a new preimage suffix reuses that output; it does not
perform another model forward.

At final seal, append fragments are flattened into the exact CSR arrays
defined above.

### 4.5 Canonical rollout-model cache

`SigmaHead` and `ActionHead` are not a sufficient cache for reopening a
canonical node. `SigmaHead` retains values and public heads, while
`ActionHead` retains selected log policies only for action labels already
observed. A newly appended occurrence still needs the raw factorized policy
output to draw an action that might not have appeared earlier.

If the contract is one rollout model forward per `Topology` label and model
version, introduce a labelled-size cache:

```text
TopologyModelCache[input_type] <labelled player rows>
  raw factorized policy heads
  value head or reference to SigmaHead
  trace/cost heads or reference to SigmaHead
  validity/model-version marker
```

It is populated atomically with the first canonical model evaluation and
remains live until no refill or newly appended preimage can reopen the row.
The conservative lifetime is through rollout seal.

The preferred contract defers final saved-head materialization until seal:

- `SigmaHead` gathers the cached value and public heads once final trained
  sigma rows are known; and
- `ActionHead.old_log_policy` is evaluated from the cached raw policy output
  once final distinct canonical action edges and their order are known.

This avoids trying to reserve a contiguous action span before refill and later
preimage chunks have finished discovering action labels. An implementation
may stage hot heads earlier, but sealing must still produce the exact final
labelled permutation and may not expose append-order spans to the learner.

This cache is potentially large, especially for natural private-domain policy
heads, and must be included explicitly in admission and benchmarks. If a
future implementation elects to recompute reopened rows instead, it weakens
the one-forward contract and must be specified and measured as a different
design; it cannot silently omit both cache and recomputation.

## 5. Rollout batch flattening

There are two different flattening operations and they must not be conflated.

### 5.1 Canonical model flattening

Model inference flattens:

```text
(replica, selected Topology row)
```

exactly once per distinct label and model version. Replicas remain separate.
The stable canonical family prefix remains a direct model batch.

### 5.2 Occurrence sampling flattening

Sampling flattens the selected canonical rows' unexpanded preimage suffixes:

```text
canonical_ptr <canonical_batch + 1>
sampled_node <occurrence_count>
degree <occurrence_count>
child_ptr <occurrence_count + 1>
flattened_child <sum degree>
```

Model outputs are gathered through:

```text
sampled node -> Topology row -> model batch row
```

and child operations through:

```text
flattened child -> sampled parent -> Topology/model row.
```

The implementation should use indices/views rather than materializing one
copy of the wide model output per occurrence or per child.

Queue cost accounting must count occurrences, not canonical rows. A canonical
queue entry therefore contributes the cost-head estimate multiplied by its
currently unexpanded occurrence count. Appending a new occurrence adds one
such contribution; consuming an occurrence suffix removes the same amount.

The safe-tree controller is scheduling-sensitive. The quotient scheduler is
the new declared scheduling law: it processes canonical rows and their
occurrence suffixes. Tests must freeze its frontier accounting and cutoff
behavior rather than claiming equivalence to the previous occurrence FIFO.

## 6. Sealing and structural flattening

Sealing produces two different immutable products.

### 6.1 Sealed `Topology`

Flatten canonical nodes in deterministic parent-before-child order. Within a
parent, order distinct edges by a stable semantic order:

1. event kind;
2. canonical action/chance label; and
3. canonical child ID as a final tie-breaker.

Hot IDs and flat IDs are separate. Sealing emits:

```text
hot_topology_node_to_flat
flat_topology_node_to_hot
hot_topology_edge_to_flat
flat_topology_edge_to_hot
```

and rewrites every `SampledTopology` and `TopologyMap` canonical reference
through these permutations. It also emits partition-local row maps:

```text
flat_topology_node -> FamilyWork row
flat trained node  -> sealed SigmaHead/SigmaTarget row
flat trained edge  -> sealed ActionHead/ActionTarget row
```

The final action-edge permutation groups trained action edges by canonical
sigma row, making each `action_begin/action_count` span contiguous even when
the labels were discovered in different preimage chunks or during refill.
`SigmaHead` and `ActionHead` are materialized in these sealed orders from
`TopologyModelCache`.

Allocate dense labelled rows:

```text
FamilyWork              all canonical player nodes
SigmaHead/Target        canonical trained player nodes
ActionHead/Target       distinct observed trained action edges
LearnerIndex            canonical trained player nodes
```

The canonical `action_count` is the number of distinct target-contributing
action labels, not any occurrence's `N_tgt`. There is no final learner
`target_degree`.

Frozen best-response canonical player nodes retain rollout model inputs but
have no learner head or target rows.

### 6.2 Sealed compact `SampledTopology`

`SampledTopology` is independently flattened for reverse target propagation.
Its parent/child relation and degrees remain occurrence-specific.

A sampled node is retained when any of the following holds:

- it is a learner-owned player occurrence and contributes a sigma target;
- it has more than one executed child;
- its nominal degree is greater than one;
- it owns a protected/refilled root boundary; or
- omitting it would lose a required recurrence or domain conversion.

Ordinary headless unary sampled nodes may still be compressed. Compression
preserves:

- summed `path_length`;
- endpoint output type;
- the child cost multiplier required by the recurrence; and
- the first canonical trained action-edge identity when the retained parent
  owns an action target.

One compact sampled edge cannot, in general, map one-to-one to a canonical
edge because it can summarize several unary public edges. The sealed map must
therefore distinguish:

```text
compact_sampled_edge.endpoint_topology_row
compact_sampled_edge.first_topology_edge
compact_sampled_edge.topology_path_begin/count   # diagnostic/validation map
```

`first_topology_edge` is used only when the retained sampled parent is a
trained player and this edge contributes to its canonical action target.
`endpoint_topology_row` validates the compact endpoint. The full
`topology_path` preserves bidirectional auditability for omitted hot edges.

Compaction must not rewrite the hot map and pretend omitted edges never
existed. The final map includes both:

- the full uncompressed sampled-to-labelled relation; and
- the compact propagation relation used by `LayerPlan`.

### 6.3 Final path masses

Decision path masses are computed from sealed `SampledTopology`, using each
occurrence's executed degree. Root descendants use the final augmented root
degree. Protected root action targets use only the original target prefix,
while refill descendants remain ordinary data below the root.

Only after this recomputation are quotient target weights final.

## 7. Target creation: sampled propagation, labelled accumulation

The current direct write:

```text
one sampled node -> one SigmaTarget row
one sampled edge -> one ActionTarget row
```

must be replaced. Many sampled nodes and edges now contribute to the same
labelled destination.

The target path becomes:

```text
Leaf completion
  -> SampledTopology LayerPlan
  -> occurrence-local reverse recurrence
  -> per-layer quotient contribution reduction
  -> labelled target finalization
  -> atomic SigmaTarget/ActionTarget publication
```

### 7.1 What remains occurrence-local

For every sampled retained node, the existing recurrence still uses:

- its own protected child prefix;
- its own `N`, `N_tgt`, and `N_tilde`;
- its own child work values;
- its own leave-one-out sibling set;
- its own trace sample covariance;
- its own cost and delta correction;
- its own old canonical value/trace/cost head; and
- its own sampled path mass.

No child work is averaged by canonical label before these quantities are
computed.

`LayerPlan` therefore remains derived from compact `SampledTopology`. Its node
descriptor replaces unique destination target rows with:

```text
topology_sigma_row
sampled_decision_weight
sampled_private_index            # sampled estimator
```

and each target child descriptor includes:

```text
topology_action_row
sampled_action_weight
target_eligible
```

The work destination is still sampled propagation work. The topology target
destination is an accumulator key, not an exclusive row write.

### 7.2 Target weights

Let `w_s` be the final decision path mass of a trained sampled occurrence
`s`. For an eligible sampled action edge `e` under `s`, define:

$$
w_e=\frac{w_s}{N_{\rm tgt}(s)}.
$$

For canonical trained node `t` and action edge `a`:

$$
W_t=\sum_{s:l_{\rm node}(s)=t}w_s,
$$

$$
W_{t,a}
=
\sum_{\substack{e:l_{\rm edge}(e)=a\\
                 \operatorname{parent}(e)\in l_{\rm node}^{-1}(t)}}
w_e.
$$

All accumulators store numerator and mass separately. Finalization divides
only where mass is positive.

The equations above use unnormalized masses for clarity. Production must not
exponentiate deep-tree `log(w_s)` directly. Before target execution, compute:

$$
\log W_t
=
\operatorname{logsumexp}_{s:l(s)=t}\log w_s,
\qquad
q_s=\exp(\log w_s-\log W_t),
$$

and use:

$$
q_e=\frac{q_s}{N_{\rm tgt}(s)}
$$

in the physical accumulators. Thus private value mass sums to one for a
nonempty canonical sigma row, total action/private mass across all its
canonical action rows sums to one, and trace mass is a validity fraction in
`[0,1]`. `log W_t`, preferably retained in FP64 where the current alias path
uses FP64, supplies the between-sigma sampling law. The normalized quotient
masses and target numerators remain FP32.

### 7.3 Sampled-estimator merging

The sampled learner can no longer retain one scalar sampled hand on a
canonical sigma row. It must bucket occurrences by the acting private index.
Let `h_s` be the sampled actor hand/class for occurrence `s`.

For value targets:

$$
M^V_{t,h}
=
\sum_{\substack{s:l(s)=t\\h_s=h}}w_s,
\qquad
Y^V_{t,h}
=
\sum_{\substack{s:l(s)=t\\h_s=h}}w_s\widehat V_s.
$$

For action advantages:

$$
M^A_{a,h}
=
\sum_{\substack{e:l(e)=a\\h_{\operatorname{parent}(e)}=h}}w_e,
\qquad
Y^A_{a,h}
=
\sum_{\substack{e:l(e)=a\\h_{\operatorname{parent}(e)}=h}}
w_e\widehat A_e.
$$

Final targets are:

$$
\overline V_{t,h}=\operatorname{Cond}(Y^V_{t,h},M^V_{t,h}),
\qquad
\overline A_{a,h}=\operatorname{Cond}(Y^A_{a,h},M^A_{a,h}).
$$

This makes sampled-mode learner targets natural-private-domain tensors with
explicit empirical mass. Propagation can remain scalar (`all` domain); the
learner domain is separately `pre` class width or `post` hand width. Zero-mass
private entries are masked.

### 7.4 Rao--Blackwell merging

For RB occurrence `s`, let `p_s(h)` be its sigma posterior used by the value
loss. For sampled action occurrence `e`, let `p_e(h)` be its child posterior.

Accumulate:

$$
M^V_{t,h}
=
\sum_{s:l(s)=t}w_s p_s(h),
\qquad
Y^V_{t,h}
=
\sum_{s:l(s)=t}w_s p_s(h)\widehat V_s(h),
$$

$$
M^A_{a,h}
=
\sum_{e:l(e)=a}w_e p_e(h),
\qquad
Y^A_{a,h}
=
\sum_{e:l(e)=a}w_e p_e(h)\widehat A_e(h).
$$

The final value and advantage targets are the corresponding conditional
means. `M^V` and `M^A` replace a separately normalized occurrence posterior:
they are both posterior mass and learner weight.

### 7.5 Public trace and cost merging

For each canonical trained node:

$$
M^C_t=\sum_{s:l(s)=t}w_s,
\qquad
Y^C_t=\sum_{s:l(s)=t}w_s\widehat C_s.
$$

For trace:

$$
M^T_t
=
\sum_{s:l(s)=t}w_s
\mathbf 1_{\{N_{\rm tgt}(s)\geq2\}},
$$

$$
Y^T_t
=
\sum_{s:l(s)=t}w_s
\mathbf 1_{\{N_{\rm tgt}(s)\geq2\}}\widehat T_s.
$$

Final targets are `Y/M`. The learner uses `M^T_t/W_t` as the trace-loss
validity weight. A distinct-child count in `Topology` is never a substitute
for this weight.

### 7.6 Per-layer deterministic reduction

Multiple sampled CTAs can target one labelled accumulator, including sampled
copies at different reverse depths. Unordered FP32 atomic addition should not
define the result.

At seal, build a `QuotientReductionPlan` for every reverse layer:

```text
sigma_contribution_source
sigma_destination
sigma_group_ptr

action_contribution_source
action_destination
action_group_ptr
```

Sources are ordered by:

```text
(canonical destination, sampled logical node/edge ID)
```

For each layer:

1. the propagation kernel computes occurrence-local targets into bounded
   layer contribution scratch;
2. a segmented reduction kernel accumulates ordered numerators and masses
   into labelled FP32 accumulators; and
3. contribution scratch may be reused after that layer completes.

This scratch is transient sampled-layer workspace, not a persistent learner
owner. Persistent `SigmaTarget` and `ActionTarget` remain labelled-size.

Contributions reaching the same canonical destination from different reverse
layers are accumulated in fixed increasing-layer order. This defines
deterministic arithmetic for a fixed sampled tree.

An alternative fused deterministic kernel is allowed only if it implements
the same declared ordering and does not reintroduce sampled-size persistent
targets.

### 7.7 Finalization and publication

After the final reverse layer:

1. validate accumulator finiteness and nonnegative masses;
2. conditionalize every positive-mass target;
3. set zero-mass target payloads to finite zero and retain a zero mask;
4. derive `W_t`, private value weights, action/private weights, trace-valid
   weight, and cost weight;
5. validate exact dense labelled destination extents; and
6. atomically publish all target and learner-index owners for the replica.

Failure aborts the complete labelled target transaction. No partially merged
target becomes learner-visible.

## 8. Final labelled target schemas

The precise physical grouping can be schema-generated, but the semantic
fields are:

```text
SigmaTarget[input_type] <trained Topology rows>
  value_target <learner_private_width>
  value_mass   <learner_private_width>
  trace_target                              # branched
  trace_mass                                # branched validity fraction
  cost_target                               # branched
  sigma_log_mass = log W_t
```

```text
ActionTarget[action_type] <distinct trained Topology action edges>
  advantage_target <learner_private_width>
  action_mass       <learner_private_width>
```

RB does not require a separate posterior field: normalized `action_mass`
plays that role and also preserves the action's decision mass. A separate
field may be retained only if it has a distinct consumer.

```text
SigmaHead[input_type]
  old_value <learner_private_width>
  old_trace
  old_cost
```

```text
ActionHead[action_type]
  action
  old_log_policy <learner_private_width>
```

For sampled mode, this changes saved learner heads from one selected scalar
private row to the natural private domain. The rollout still selects scalar
hands/actions per occurrence; only labelled learner storage is widened.

`LearnerIndex` contains:

```text
topology_row
action_begin
action_count                  # distinct observed action labels
log_sampling_weight = log W_t
```

It does not contain `N_tgt`, an occurrence profile, or occurrence action
count.

## 9. Merged quotient learner

The branched sampler samples canonical trained `Topology` row `t` in
proportion to `W_t`. A physical batch contains each selected canonical model
input once.

Within a selected row:

- value loss is reduced over private entries using
  normalized `value_mass`;
- policy loss is reduced over its distinct canonical action rows and private
  entries using normalized `action_mass`;
- entropy, reference KL, and policy diagnostics use the same merged action
  mass;
- trace loss is multiplied by the normalized `trace_mass`;
- cost loss uses the merged cost target; and
- the configured nonlinear loss is applied to the finalized merged target.

The action rows are not averaged uniformly. `action_count` is only a ragged
span length. `action_mass` supplies the statistical measure.

The learner does not receive sampled occurrence IDs, occurrence targets,
sampled `N_tgt`, or a sampled preimage. It cannot reconstruct or evaluate the
old per-occurrence loss, by design.

Alias sampling can draw the same canonical row more than once in one
statistical batch. Before the model call, coalesce equal drawn topology IDs,
retain their integer presentation counts/statistical weights, and evaluate
each distinct model input once. Reducing those weighted merged losses
preserves the quotient sampler while avoiding accidental duplicate model rows
inside a physical batch.

Sequential mode is the same quotient learner with occurrence degree one.
Multiple sequential roots or paths with identical labels can still merge.

## 10. Capacity and lifetime changes

The capacity vocabulary must separate sampled and labelled bounds:

```text
sampled_topology_cap
sampled_topology_edge_cap
topology_cap
topology_edge_cap
leaf_cap

labelled_player_cap[input_type]
labelled_trained_player_cap[input_type]
labelled_action_cap[action_type]

node_map_index_cap = sampled_topology_cap
edge_map_index_cap = sampled_topology_edge_cap
quotient_reduction_layer_cap
topology_model_cache_bytes
```

The existing `topology_cap` currently bounds the sampled structural forest.
Under this design it should bound `Topology`; the sampled bound receives the
explicit `sampled_topology_*` names. Configuration conversion must be
offline/explicit rather than silently reinterpreting an old value.

Lifetimes become:

| Owner | First producer | Last consumer |
|---|---|---|
| hot `SampledTopology` occurrence state | roots/child commit | occurrence expansion or seal |
| hot `Topology` and append map | root grouping | seal |
| canonical `FamilyWork` retained input | labelled child commit | final backward |
| `TopologyModelCache` | first canonical model evaluation | rollout seal |
| `Leaf` | sampled terminal commit | completion |
| sealed compact `SampledTopology` | seal | final target layer |
| sealed `TopologyMap` | seal | target finalization/audit |
| `LayerWork` | completion | final target layer |
| quotient accumulators | target setup | target finalization |
| labelled heads/targets | model/target publication | final backward |

`LayerWork` remains bounded by sampled completion/frontier needs. Exact
reverse-live-row admission must be recomputed from compact
`SampledTopology`; it must not use labelled `Topology` size.

## 11. Transaction, recovery, and distributed rules

One child commit atomically publishes:

- sampled nodes and edges;
- leaves;
- new or existing canonical child references;
- canonical FamilyWork rows when first created;
- forward and inverse map fragments; and
- canonical queue/preimage-tail updates.

A failed transaction publishes none of them. Retrying uses the existing
disjoint logical RNG epoch and cannot observe half-created canonical labels.

Concurrent creation of the same `(Topology parent, event label)` must resolve
to one canonical edge. The implementation may serialize canonical insertion
per parent, use a deterministic sort/unique stage, or use a checked device
table followed by deterministic compaction. It may not allow duplicate
canonical rows and repair them only after model evaluation.

Replica lanes have independent maps. Distributed ranks must agree on
capacity/failure decisions but do not share topology IDs or target
accumulators across ranks. Gradients continue to combine through the declared
distributed optimizer path.

## 12. Current contracts that this design replaces

This is not an additive side table beside the current topology. The following
one-to-one assumptions must be replaced together.

### `pbg/core/contracts.py`

- Split sampled structural capacities/owners from labelled topology owners.
- Move occurrence profile, RNG, degree, and controller fields out of
  `FamilyWork`.
- Make saved sampled learner heads and targets natural-private-domain
  labelled fields.
- Add quotient masses and remove learner `target_degree`.

### `pbg/core/capacity.py`

- Stop deriving FamilyWork, trained, and action capacity from sampled
  occurrence counts.
- Admit sampled topology/map indices independently from labelled rows.
- Admit `TopologyModelCache`, quotient accumulators, and maximum per-layer
  contribution scratch.
- Size `LayerWork` from sampled reverse liveness, never labelled topology.

### `pbg/core/rollout.py`

- Schedule canonical family rows with unexpanded preimage suffixes.
- Separate canonical model flattening from occurrence sampling flattening.
- Make queue depth/frontier cost count unexpanded occurrences.
- Publish map fragments atomically with both sampled and canonical children.

### `pbg/games/poker/queued/branching.py`

- Replace the current occurrence-key topology recorder with dual hot builders.
- Move `true_profile` propagation to `SampledTopology`.
- Preserve sampled-parent and child-ordinal identity through every action and
  chance grouping operation.
- Reopen canonical rows for appended preimages/refill while reusing the
  canonical output cache.
- Group full semantic chance levels even when physical chance closure is
  fused.

### `pbg/core/topology.py`

- The current duplicate-preserving structural type becomes
  `SampledTopology`.
- Add the unique labelled `Topology` and the hot/sealed `TopologyMap`.
- Split canonical flatten permutations from sampled reverse-layer planning.
- Remove the requirement that a sampled trained node owns a unique sigma
  target destination.
- Remove the requirement that sampled action target spans do not overlap;
  they intentionally map many-to-one before quotient reduction.
- Retain exact dense uniqueness only for final labelled target rows.

### `pbg/core/targets.py` and `pbg/core/csrc/layered_targets.cu`

- Stop writing `SigmaTarget` and `ActionTarget` directly from one sampled CTA.
- Read canonical old heads but propagate over sampled work.
- Emit occurrence-local target contributions and reduce them through the
  per-layer quotient plan.
- Finalize numerator/mass pairs only after every sampled layer.

### `pbg/core/learner.py`

- Replace `action_count == target_degree` with distinct canonical action
  spans plus explicit action mass.
- Remove sampled occurrence profiles from learner sigma rows.
- Use natural-private-domain sampled buckets.
- Sample canonical sigma rows by `log W_t`, coalesce repeated draws, and
  reduce every loss by quotient masses.
- Use trace-valid mass rather than `N_tgt >= 2`.

These changes must cut over atomically at the schema/plan boundary. A mixed
state in which rollout is canonical but target destinations or learner
denominators retain sampled one-to-one semantics is invalid.

## 13. Implementation sequence

### Stage 1: measurement and identity proof

- Reconstruct full public-label hashes from existing sampled topologies.
- Measure sampled versus unique node/action counts by street, depth, mode,
  beta, and root count.
- Verify colliding rows have identical canonical model inputs, owners,
  partitions, folded summaries, and old public heads.
- Freeze the exact poker public-label encoding and collision check.

### Stage 2: dual hot trees and appendable map

- Introduce `SampledTopology`, `Topology`, and `TopologyMap` capacities.
- Move profile/RNG/degree state from occurrence `FamilyWork` to
  `SampledTopology`.
- Group roots and child commits into canonical rows.
- Support appendable preimages, suffix expansion, and cached canonical model
  output.
- Keep the old target/learner presentation temporarily only as a validation
  oracle; it is deleted before final cutover.

### Stage 3: canonical rollout inference

- Change family scheduling to canonical rows plus preimage suffixes.
- Add canonical-model and occurrence-sampling flatten plans.
- Reuse one model result across every occurrence in the suffix.
- Make chance closure preserve and group both sampled and canonical identity.
- Freeze queue-cost and safe-tree behavior under canonical scheduling.

### Stage 4: dual sealing and map validation

- Emit deterministic sealed `Topology`.
- Emit compact sealed `SampledTopology`.
- Flatten append fragments into exact node/edge CSR maps.
- Add compact-edge endpoint/first-edge/path summaries.
- Compute final sampled path masses after refill.

### Stage 5: quotient target creation

- Remove unique sampled target destinations from `LayerPlan`.
- Add topology destination IDs and finalized occurrence weights.
- Produce per-layer target contributions.
- Add deterministic segmented reductions and labelled accumulators.
- Finalize sampled private buckets, RB posterior-weighted targets, T/C
  targets, and all masses.
- Publish only dense labelled target extents.

### Stage 6: quotient learner

- Widen sampled learner heads/targets to the natural private domain.
- Sample canonical sigma rows by `W_t`.
- Coalesce repeated canonical alias draws before each model call.
- Replace uniform action averaging with `action_mass`.
- Apply every configured loss to the merged targets only.
- Remove occurrence profiles, `N_tgt`, and sampled action spans from learner
  inputs.

### Stage 7: deletion and performance closure

- Delete the one-occurrence/one-target-row contracts and adapters.
- Delete any temporary occurrence-loss oracle from production.
- Recompute exact capacity admission and lifetimes.
- Benchmark model forward rows, backward rows, target-reduction overhead,
  memory, and end-to-end sampled nodes per second.

## 14. Required tests

### Mapping and identity

- Duplicate actions under one sampled parent map to one `TopologyEdge`.
- Equal labels under different sampled copies map to one `TopologyNode`.
- Different full histories with equal physical poker state do not merge.
- Different replicas never merge.
- Root duplicates, multi-pack preimages, and refill append correctly.
- Hot fragment maps flatten to exact forward/inverse CSR permutations.
- Hash collisions fail unless the semantic payload also agrees.

### Flattening

- Canonical model rows appear once per label.
- Occurrence sampling retains exact parent and child ordinal.
- Compact sampled unary paths retain endpoint, first action edge, full
  canonical path, path length, and cost multiplier.
- Every retained trained sampled node and target edge has one labelled
  destination.
- Frozen unary compaction does not delete canonical rollout model rows.

### Targets

- Leave-one-out targets are computed before quotient merging.
- Duplicate action labels merge only after their occurrence baselines exist.
- Trace covariance is occurrence-local; trace validity mass is exact.
- Root refill excludes refill-only root edges but includes their descendant
  decisions.
- Sampled hand buckets and RB posterior-weighted numerators match a direct
  reference reducer.
- Layer-by-layer segmented results are deterministic.
- Zero masses are finite and masked.
- Failed target creation publishes no labelled target.

### Learner

- Alias mass equals the sum of sampled decision masses in each canonical
  preimage.
- Normalized action masses sum to one per canonical sigma row, within declared
  numerical tolerance.
- Sampled and RB learners consume no occurrence rows.
- Nonlinear losses receive the merged target, not individual targets.
- Distinct-action span length is never used as the statistical denominator.
- Trace loss uses trace-valid mass rather than distinct child count.

### Capacity and execution

- Independent sampled/labelled caps fail at the correct boundary.
- Large duplicate-heavy trees increase leaves and propagation work without
  proportionally increasing model rows.
- CUDA tests cover all twelve modes, root counts, replica execution, refill,
  safe-tree cutoff, best-response roles, and distributed failure agreement.
- GPU memory and processes return to their pre-test baseline.

## 15. Acceptance criteria

The design is complete only when all of the following are true:

1. `SampledTopology` alone determines the stochastic tree and every reverse
   recurrence.
2. `Topology` alone determines model rows, saved heads, final targets, and
   learner rows.
3. `TopologyMap` is exact and bidirectional for nodes and edges.
4. Model inference never duplicates a published canonical label within a
   replica/model version.
5. Target merging happens after occurrence-local target construction.
6. Persistent `SigmaTarget`, `ActionTarget`, and learner inputs are
   labelled-size.
7. The learner evaluates configured losses only on merged targets with
   explicit value/action/trace/cost masses.
8. Sampled-private collisions are merged by private-state bucket rather than
   discarding private identity.
9. Refill, multi-pack append, chance closure, compaction, and safe-tree
   scheduling have explicit tested semantics.
10. The measured model-row reduction justifies the map and segmented-reducer
    overhead on representative NLHE workloads.
