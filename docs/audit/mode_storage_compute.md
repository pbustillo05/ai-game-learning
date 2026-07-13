# Required compute and storage by training mode

This document defines the efficient execution contract implied by the requested three modes. “Store” means retain past rollout long enough to construct targets and run all gradient steps for an iteration. Transient frontier tensors are not rollout-database storage.

## Shared principles

All modes need the frozen rollout policy, public navigation, Bayesian range updates during rollout, legal actions, and model inputs. They should not share a maximal data structure merely for modularity. A shared interface may dispatch to specialized layouts.

The lifetime phases should be explicit:

1. rollout frontier;
2. leaf reward/completion;
3. backward target construction;
4. compact decision dataset;
5. gradient steps.

Data whose final consumer is in an earlier phase must be released before the next phase. In particular, leaf reaches, completion diagnostics, whole-tree working moments, the frozen rollout model, and unused public states should not coexist through all gradient steps.

For poker, a decision example must reproduce the model input under the frozen public belief. That can be done by storing a compact input snapshot or by replaying from checkpoints. Replaying requires complete public events, including chance payloads, and must be benchmarked against snapshot storage.

## Mode 1: ordinary terminal-trajectory training

### Required rollout computation

- Sample one collision-free private profile per root.
- Carry it through all player actions and public chance events until a genuinely terminal state, including a complete poker board and deterministic showdown settlement for that sampled profile.
- Maintain belief/range state only because the policy model consumes it; the gradient target itself is sampled-private-state.
- Cache the selected actor type/combo/class, selected public action, rollout log probability for that selected type, and rollout value for that selected type.
- Backward-propagate scalar player-clock lambda returns from the terminal reward vector.

No pre-terminal multi-sample completion, private posterior tensor, posterior-weighted mass, or sibling baseline is needed.

### Minimal persistent data

Per trajectory:

- terminal reward vector `[P]` until scalar targets are built;
- compact decision offsets/counts.

Per decision:

- public/model-input snapshot or replay key;
- owner and sampled actor type;
- sampled action/event;
- scalar old log probability and scalar old value;
- scalar advantage and value target;
- legal-action metadata needed by the policy adapter.

Not required after rollout:

- `[P,1326]` leaf posterior ranges;
- all-type old log-probability/value rows;
- parent/child posterior targets;
- MC completion output;
- branch topology or path weights (flat trajectory offsets suffice);
- folded full ranges after their last rollout/model-summary consumer.

### Folded-player lifecycle

During rollout, a folded range may be needed to update folded-card summaries on later streets. Once the terminal trajectory and every stored decision input have been materialized, normal training needs only the folded summaries consumed by the model. It does not need folded ranges for reward computation because the sampled private profile is already known.

### Current implementation

Implemented through `training.mode: normal`. It shares the one-path frontier
sampler, settles one sampled terminal reward, and compacts rollout output into
scalar decision rows before minibatch optimization. It does not construct or
invoke a completion backend.

## Mode 2: sequential Rao--Blackwell training

### Required rollout computation

- Sample one private profile per root and carry it to a pre-terminal public leaf.
- Maintain all dealt-seat reach rows, including frozen folded ranges, through leaf completion.
- At each leaf, run collision-conditioned completion with the configured MC budget to estimate actor-type posterior mass and conditional reward.
- Propagate player-clock lambda returns along the one path.
- Integrate every actor type in policy/value/entropy losses.

### Minimal persistent data

Per leaf, transient until backward propagation:

- posterior mass `[P,H]`;
- conditional value `[P,H]` or directly posterior-weighted value mass `[P,H]`.

Per decision, after target construction:

- public/model-input snapshot or replay key;
- selected public action;
- all-actor-type old selected-action log probability `[H_actor]`;
- all-actor-type old value `[H_actor]`;
- one actor posterior vector `[H_actor]` (with one child, parent and child posterior are identical);
- advantage `[H_actor]` and value target `[H_actor]`;
- legal/type masks.

Not required after target construction:

- leaf reach rows and full completion objects;
- separate parent and child posterior copies;
- generic CSR branch arrays (a flat trajectory layout is sufficient);
- public moment/equity/standard-error diagnostics unless an enabled loss or adaptive sampler consumes them.

### Folded-player lifecycle

Folded full ranges are required through leaf completion because folded cards participate in the collision-conditioned posterior. They can be released immediately after completion/target construction if compact model inputs have been stored, or recomputed from valid checkpoints if replay is selected.

### Current implementation

Algorithmically present through `training.mode: rao_blackwell`, but it uses the
same maximal tree/target DTO as branching, duplicates the one-child posterior
as parent and child tensors, retains generic completion fields, and has no
byte-aware streaming.

## Mode 3: fixed branched Rao--Blackwell training

### Required rollout computation

- At each public parent, draw randomized-rounded offspring count from fixed public beta.
- Evaluate the policy/value model once per unique player parent.
- Draw a fresh posterior event-generating private sample per abstract child (full profile at chance nodes; only an actor marginal is theoretically needed at player nodes).
- Preserve duplicate labels as distinct edges and store uniform edge mass `1/K`.
- Complete every pre-terminal leaf under its own reach state.
- Propagate posterior mass and posterior-weighted moments backward.
- Emit leave-one-branch advantages and path-weighted decision examples.

### Minimal persistent data

Until backward propagation:

- labeled topology: parent/child relation, duplicate-preserving event label, edge mass, owner, tree id;
- leaf reach or a valid replay route;
- frozen value row per player parent;
- selected-action old log probability per abstract child and actor type.

Per decision after target construction:

- compact model input or replay key;
- path sampling weight;
- parent posterior `[H_actor]`;
- for each abstract child: label, child posterior `[H_actor]`, advantage `[H_actor]`, old selected-action log probability `[H_actor]`;
- old value and value target `[H_actor]`;
- child masks/counts and legal/type masks.

Transient only:

- leaf completion statistics;
- per-node/per-player `P` and `M` working tensors;
- sibling totals after advantages are emitted;
- full nondecision public states once compact decision/leaf consumers are satisfied.

### Folded-player lifecycle

Every branch owns independent reach evolution. Folded full ranges remain necessary within a branch through completion and through any later-street folded-summary refresh. After that branch's completion and compact decision-input materialization, they should be released. Shared immutable public payloads may be deduplicated, but branch random state, reaches, descendants, and completion draws may not be merged.

### Current implementation

The probability law and backward target semantics are mostly aligned. The implementation nevertheless repeats identical model forwards for sibling children, preallocates capacity-sized reach/state tensors, completes all leaves before streaming the backward pass, and retains a full target table for all decisions.

## Comparative storage matrix

| Data | Normal | Sequential RB | Branched RB |
|---|---:|---:|---:|
| Sampled private profile | one per trajectory | one per rollout spine | temporary per abstract child |
| Full public terminal runout | required | completion-owned after pre-terminal cut | completion-owned after each leaf cut |
| Multi-sample completion | no | yes | yes, independently per leaf |
| Branch topology/path mass | no | no (flat offsets suffice) | yes |
| Leaf all-player posterior `[P,H]` | no | transient | transient |
| Stored actor posterior target | no | one `[H]` per decision | parent `[H]` plus child `[E,H]` |
| Stored advantage/value target | scalar | `[H]` | `[E,H]` policy and `[H]` value |
| Stored old log probability | scalar | `[H]` | `[E,H]` |
| Stored old value | scalar | `[H]` | `[H]` |
| Folded full ranges after target build | no | no | no |
| Folded compact model summaries | as needed per decision | as needed per decision | as needed per decision |
| `vvec/u/equity/stddev` | no | only if an enabled consumer exists | only if an enabled consumer exists |

## Required memory-planning contract

Every production configuration should be rejected or downscaled before allocation if the estimated peak exceeds a configured fraction of free device memory. The estimator must account for:

- tree/state capacity actually scheduled in the current chunk, not `n_trees * per_tree_max` for the whole iteration;
- reach/frontier tensors;
- frozen and trainable model parameters;
- optimizer/scaler state;
- rollout policy activations;
- completion inputs and outputs;
- backward `P/M` workspace;
- compact decision targets;
- training activations and attention workspace.

Tree generation, completion, backward propagation, and target compaction should operate as a pipeline over whole-tree chunks. Chunking only the backward `P/M` workspace while retaining all leaves, all capacity-sized reaches, all targets, and the frozen model does not bound peak memory.
