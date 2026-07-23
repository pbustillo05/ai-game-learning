# Throughput-first storage design

Status: pre-optimization hand-off, 2026-07-23.

The immediate priority is the physical data path, not logical byte trimming.
The current branched rollout repeatedly routes hot mutable rows through growing
256-row page/chunk lists. That violates the intended dense-live/append-only-cold
split, produces superlinear routing work as a tree grows, fragments CUDA
allocation, and makes the Section 15 branched baseline roughly 4.6 times slower
per player decision than sequential RB. Removing that defect is ROI item 1.

Storage is not an objective by itself. The objective is the highest end-to-end
rate of fresh, correctly trained player decisions that fits the hard device
memory envelope:

$$
\max_{\mathcal W,\mathcal P,\mathcal C}
\frac{D_{\mathrm{fresh}}(\mathcal W)}
{t_{\mathrm{snapshot}}+t_{\mathrm{rollout}}+t_{\mathrm{targets}}
+t_{\mathrm{seal}}+t_{\mathrm{optimization}}}
$$

subject to the unchanged sampling law, targets, losses, and learner
presentation ratio, and to every physical phase fitting the device. Here
$\mathcal W$ is the admitted statistical workload, $\mathcal P$ is its
physical packing plan, and $\mathcal C$ is the selected set of compute caches.
For branched mode, $\mathcal W$ is one complete labeled tree rooted at
$\varnothing$ per GPU; a tiny-tree floor may remove the near-empty mode but
must not prescribe the desired tree size.

The concrete target is useful throughput at the RTX 2080 Ti ceiling, not the
smallest logical tensor payload. Under the current 10.5 GiB usable-device
contract, the initial guard is 0.75 GiB and the hard admitted ceiling is
9.75 GiB. Inefficient allocation can make a logically small workload reserve
most of the card, force narrow model/backward packs, cause drains, or lose an
iteration to OOM. Conversely, a larger compute cache can be the right use of
memory when it makes repeated learner presentations faster without reducing
safe admitted work.

A representation is useful when it does at least one of the following:

1. reduces work or synchronization on a hot path;
2. admits materially more statistical work per iteration;
3. reduces a phase peak that currently constrains execution;
4. removes duplicated ownership or conversion without moving work into a more
   frequently executed consumer.

The smallest representation from which a later consumer *could* reconstruct
its input is not automatically the best representation. In particular,
optimization reads a retained decision many times. A forward-ready immutable
model input can be a valuable compute cache even when some of its scalar fields
are algebraically derivable.

The governing physical distinction is:

| Domain | Physical rule |
|---|---|
| Hot mutable rollout state | Dense recyclable slots bounded by the live frontier; direct tensor indexing |
| Growing retained rollout data | Schema-derived, append-only rows with stable logical addresses |
| Sealed target/training data | Exact realized extents; forward-ready inputs; no discarded rollout owner |
| Transient work | Bounded by the physical pack or completion/target microbatch, independently of logical statistical work |

This note does not amend the sampling law, estimator, target, loss, or
last-consumer rules in `docs/theory/`. It proposes physical representations and
implementation experiments under those contracts.

All byte arithmetic below must be read at one of three explicitly reported
levels:

1. **Logical payload:** unique realized tensor bytes, useful for finding
   algebraic duplication.
2. **Allocated capacity:** immutable-schema bytes at allocated row capacities,
   including alignment, pages, slack, explicit phase overlap, and bounded
   workspaces.
3. **Physical high-water mark:** measured CUDA peak allocated and reserved,
   plus non-Torch/device-context occupancy.

Only the latter two can support admission decisions. A raw field sum is never
an OOM proof or evidence that a family width, tree, or backward subspan fits.

## Performance evidence from repository history

The storage design must internalize what has already been measured:

- Commit `a399e30` put generic paged-column access on hot poker state and
  regressed production-scale sequential-RB rollout from approximately 259 to
  90 decisions/s.
- Commit `d95b7d3` restored dense recyclable live-state access. Subsequent
  direct contiguous append/gather changes (`5ecfeb4`, `cf6bc34`, `48ce2a3`)
  restored the current approximately 263 decisions/s baseline.
- The accepted lesson is not “never page retained storage.” It is “do not put
  generic page routing between a hot kernel and a dense or already-known
  contiguous span.”
- Commit `9d90fb2` deferred a linked-history ownership rewrite because it is a
  large shared-ancestor lifetime change and can fragment model packs.
- Current measurements show optimization can dominate the iteration. Moving
  reconstruction from one rollout capture into every learner reuse is therefore
  presumptively a throughput regression.
- The dominant page regression was not fixed by changing page size. Commit
  `d95b7d3` restored a dense recyclable compute slab for sequential mutable
  state and reduced the controlled rollout from 87.45--90.27 seconds to the
  mid-30-second range. Its governing rule was: paging may own allocation and
  lifetime, but generic per-field page routing must not sit in hot poker
  arithmetic.
- Commits `5ecfeb4` and `cf6bc34` were small follow-ups, not the dominant fix.
  Direct host-proven contiguous appends were about 1.1% faster, and removing
  one redundant first-page gather merge was about 0.35% faster.
- Commit `48ce2a3` specialized immediate unsplit sequential child commits, but
  deliberately left branched subspan commits on arbitrary-row routing. The
  current branched implementation still pays that generic path even though
  each reserved physical child subspan has a host-known contiguous logical
  extent.
- Section 15 falsifies redundant policy forwards as the main branched
  slowdown. After excluding reference-cache construction, branched rollout
  has essentially one policy-forward row per retained player decision and
  fills 84-row packs well. The unresolved excess is data movement, routing,
  state/history/reach access, child materialization, private-profile sampling,
  and synchronization.

## Immediate hand-off: repair the branched physical data path

The first implementation milestone is:

> Make the cost of one bounded branched dequeue-to-enqueue pack depend on the
> realized pack and the pages it actually intersects, never on every page or
> chunk accumulated by the tree.

This is a storage project. It includes physical layout, address translation,
bounded recyclable ownership, append/commit strategy, queue organization, and
pack-local materialization. It does not authorize changing the sampled object,
branching law, targets, losses, or estimator.

### Replacement-only rule

Every production change in this project **replaces** an inefficient
representation or path. It does not add a second implementation beside it.
Introducing a new data structure is permitted only as the means of replacing
an existing one; it is never accepted as an additional layer around the old
owner. The resulting production code should have fewer ownership boundaries,
fewer conversions, fewer configuration choices, and a smaller discoverable
API. If an optimization leaves the old representation underneath or beside
the new one, the migration is incomplete.

Once the replacement passes its correctness and performance gates:

- migrate every training, evaluation, benchmark, variance, and inference
  consumer that is in production scope;
- delete the superseded classes, functions, schemas, allocators, queues,
  adapters, configuration fields, compatibility branches, and tests;
- delete conversions whose only purpose was to cross between the old and new
  representations;
- delete stale Python owners even when their tensors are empty, because their
  existence obscures lifetime, keeps dead APIs discoverable, and invites later
  code to wire itself to the wrong path;
- update documentation and examples so they name only the surviving
  implementation.

Do not retain a legacy path “for safety,” behind a feature flag, under an
optional fallback, or as an apparently reusable abstraction. A time-bounded
migration with two paths requires explicit user approval and is not the
default for this project. Rejected experimental code is removed immediately.

Within production `pbg/` modules, every Python object, schema, owner, and
adapter must be reachable from the sole current training or supported
inference path and must have a real lifetime consumer. Diagnostic scripts may
live under `tools/` and historical code may live under `archive/`, but neither
may be imported by production. An unused but importable production object is a
defect, not harmless scaffolding.

Before implementing a replacement, trace the current production entry point
through allocation, writes, reads, last consumer, and release. After migration,
repeat that reachability trace and prove:

1. the optimized implementation is the one actually timed by the production
   benchmark;
2. no production entry point can instantiate the obsolete owner;
3. no compatibility conversion or duplicate allocation remains;
4. repository search finds no obsolete symbol, config field, or test fixture
   outside explicitly historical records;
5. deleting the old path did not weaken semantic, failure, teardown, or CUDA
   acceptance coverage.

Performance evidence from an unwired helper, monkeypatch-only path, or
benchmark-only implementation is not an accepted result.

### Confirmed defects

1. **Branched mutable state lost the dense fast path.**
   `pbg/games/poker/bundle.py` gives branched state a 256-row live slab.
   Section 15 measured 6,212 simultaneously live state rows and 14,073 live
   shared-history rows. The state/history columns use direct tensor indexing
   only while the arena owns one page; they therefore fell back to generic
   multi-page routing across approximately 25 and 55 pages, respectively.
   The accepted ROI 1 state/history slice below replaces that implementation
   with grouped, dense recyclable slabs; active reach remains separate.
2. **Active reach has the same physical mistake.**
   `pbg/core/branching.py` constructs the recyclable
   `[F,P,H]` reach owner in 256-row pages. Each gather/write checks arbitrary
   logical handles against a growing page list even though reach is hot
   recyclable state.
3. **Hot topology/controller access scales with accumulated tree pages.**
   The median 23,162-node tree owns about 91 topology/controller pages at 256
   rows/page. `PagedTensorColumn` gathers loop over every page. Parent
   metadata, controller factors, and updates are read repeatedly during
   expansion, so later packs cost more than earlier packs.
4. **Known-contiguous branched commits use arbitrary-row writes.**
   Branched rollout reserves child topology in stable contiguous subspans,
   then calls `commit_children` without the host cursors required by
   `write_contiguous`. Generic writes build page-match work for all admitted
   pages. The sequential immediate-child specialization already demonstrates
   the intended fast boundary.
5. **Occurrence chunk routing repeats the page-scan pattern.**
   `ChunkedOccurrenceArena` gathers/scatters by building selectors and merging
   candidates for every accumulated chunk. A realistic tree reaches roughly
   the same chunk count as topology page count.
6. **Current working-tree execution scheduling removed exact-street queues.**
   Postflop rows now share a natural `post_2` queue and a pack discovers its
   minimum street at dequeue time. A mixed flop/turn/river prefix pads every
   row to its earliest street and adds a device scalar boundary. Commit
   `83aab85` had already established exact-street physical execution as about
   a 4.9% HU rollout improvement. Stable training partitions need not dictate
   physical rollout queues.
7. **Integrity checks contain unreported CUDA synchronizations.**
   Repeated `torch.equal` checks in child reservation/commit synchronize the
   host but are absent from the declared scheduler-synchronization counters.

These are implementation defects, not proposals inferred only from payload
arithmetic. They are visible in the current code and in the page/live-row
metrics already recorded by Section 15.

### Required implementation sequence

Work in isolated, paired steps. Do not combine all of these into one rewrite:

1. Instrument generic/direct page and chunk access counts, pages/chunks
   visited, rows moved, transient routing bytes, and CUDA launch/synchronization
   time. Reuse the existing Section 15 artifacts rather than first spending
   several minutes reproducing an uninstrumented known result.
2. Give branched mutable public state, shared history, and active reach a
   dense or explicitly segmented recyclable compute interface bounded by the
   admitted live frontier. A large page sized to total possible tree work is
   not an acceptable fix. State, history, and reach may need different
   capacities and lifetime accounting. The state/history slice is accepted;
   active reach remains pending after its isolated experiment missed the
   throughput gate.
3. Pass host-known contiguous cursors through branched child reservation and
   commit. Preserve the two-pass rule that fixes every child/RNG address before
   the first draw; optimize the physical commit without changing logical
   order.
4. Remove all-chunk scans from hot occurrence/topology/controller access.
   Keep stable logical addresses and paged append-only ownership, but resolve a
   bounded pack to only its intersecting physical chunks. Splitting mutable
   controller/live metadata from cold append-only topology is allowed when
   their lifetimes differ.
5. Restore exact-street physical rollout execution, or demonstrate with paired
   padded-pair and wall-time evidence that another bounded packer is better.
   Natural sealed/training partitions remain unchanged.
6. Only after the hot path is corrected, proceed to the smaller retained-row,
   canonical-capture, seal-overlap, and learner-cache items later in this
   document.

Every step must preserve page/slab byte admission, last-consumer release,
logical RNG addresses, stable queue order, topology, retained inputs, targets,
and gradients. The replacement-only rule is a hard acceptance gate: migrate
all consumers and delete the superseded path in the same accepted change.
Trace every ownership change to `docs/theory/pseudocode/` before
implementation; do not amend theory without explicit user approval.
Production state, reach, sampling, transition, model, and target work remains
on CUDA with no CPU staging or fallback.

### First benchmark gate

Use the Section 15 branched workload:

- HU, 32 bb, 13 ranks, fixed $\beta=3.0$;
- one distinguished-root tree per GPU;
- tiny-tree refill floor 1,000 total nodes;
- diagnostic upper trigger 10,000 total nodes, followed by complete
  multiplicity-one drainage rather than truncation;
- 10k--20k realized player decisions;
- 84-row physical model/child packs;
- 256-row pages for append-only retained owners;
- checkpoint wrapper and logical seeds 104,830, 209,559, and 314,288 from
  Section 15;
- one idle RTX 2080 Ti under the approximately 10 GB physical ceiling.

Start with seed 104,830 because it naturally produced 14,565 player decisions
without refill, making it a clean hot-path comparison. After a material paired
improvement, run all three seeds, including the refill case, then the full
rollout/targets/learner gate. Report time per player decision and per total
node, because branch leaf/decision composition varies.

The existing branch artifact is the time-to-beat implementation baseline:
249.792 seconds rollout, 14,565 player decisions, 23,162 total nodes,
4.388 GiB peak allocated, and 8.812 GiB peak reserved for the median sample.
It is not an acceptable estimate of intrinsic branching efficiency.

Every proposal below is judged against this history.

# 1. Notation and row domains

- $P$: players, $2\leq P\leq6$.
- $R$: active ranks and $4R$ active cards.
- $H=\binom{4R}{2}$: exact private-hand count; $H=1326$ for $R=13$.
- $K=R^2$: preflop model width; $K=169$ for $R=13$.
- $G$: sequential trajectories. Sequential modes have $L=G$ leaves.
- $N$: all generated public occurrences.
- $D=\sum_mD_m$: retained player decisions.
- $A=\sum_mA_m$: retained target-eligible sampled player actions in branched
  mode. This is not necessarily the number of public-history tokens.
- $L$: stopped leaves.
- $C$: maximum simultaneously active sequential trajectories.
- $F$: maximum simultaneously live branched occurrences.
- $B$: current player/model pack.
- $B_c$: current branched child-materialization pack.
- $T$: configured lossless public-history width used by the model.
- $E=3+n_{\text{ticks}}$: ordered public action domain.
- $X_m=H$: exact loss/target width for poker.
- $Y_m$: model width: $K$ preflop and $H$ postflop.
- $A_{\text{live}}$: live-player count in a postflop partition.
- $S_m$: distinct folded-summary dependency anchors in partition $m$.
- $Q$: folded-summary spectral rank.
- $M_{\text{device}}$: usable device capacity; currently 10.5 GiB on the
  target RTX 2080 Ti.
- $M_{\text{guard}}$: intentionally unused reserve; initially 0.75 GiB.
- $M_{\text{hard}}=M_{\text{device}}-M_{\text{guard}}$: hard admitted ceiling,
  initially 9.75 GiB.
- $\operatorname{cap}(e)$ and $b(e)$: allocated row capacity and
  schema-derived aligned bytes/row for owner $e$.
- $W_\phi$: calibrated transient and backend workspace bound for phase $\phi$.

The lossless history bound is

$$
T_{\min}=SP+3P-S+4.
$$

For $P=6,S=128$, this is $T_{\min}=662$, not 3,123. The latter is a
separate conservative rollout-state estimate and must not be used as the model
history width in retained-input arithmetic.

## 1.1 Physical admission model

Logical row formulas rank duplication, but the scheduler admits physical
owners. For a phase $\phi$ define

$$
M_\phi^{\mathrm{cap}}
=M_{\mathrm{static},\phi}
+\sum_{e\in\mathcal O_\phi}\operatorname{cap}(e)b(e)
+M_{\mathrm{overlap},\phi}
+W_\phi,
$$

where $\mathcal O_\phi$ is the exact owner set live in that phase,
$M_{\mathrm{overlap},\phi}$ includes transactional source/destination overlap,
and $W_\phi$ includes such work as model activations, attention and evaluator
workspaces, autograd-saved tensors, flat gradients, optimizer state, reference
cache construction, and physical gather/reduction arenas. Then

$$
M_{\mathrm{peak}}^{\mathrm{cap}}
=\max_{\phi\in\{\mathrm{rollout},\mathrm{seal},\mathrm{targets},
\mathrm{optimization}\}}M_\phi^{\mathrm{cap}}
\leq M_{\mathrm{hard}}.
$$

Admission must additionally cover allocator bin rounding, fragmentation,
non-releasable segments, branched live-tail overshoot, rank-to-rank maxima, and
non-Torch occupancy. In practice this means validating the schema prediction
against a conservative quantile of measured reserved/NVML peaks across ranks
and representative iterations:

$$
Q_{1-\alpha}\!\left(M_{\mathrm{peak}}^{\mathrm{physical}}\right)
\leq M_{\mathrm{hard}}.
$$

The purpose of a storage change is therefore to improve the optimum admitted
$(\mathcal W,\mathcal P,\mathcal C)$, not merely to lower one logical byte
sum.

# 2. What survives each phase?

The following is the first-principles ownership inventory. It determines
lifetime legality, not physical fit by itself. Each implementation must derive
the corresponding allocated capacities from the immutable schemas and charge
the last live source owner, the first destination owner, and bounded transient
work together wherever their lifetimes overlap.

## 2.1 Rollout: live work

Sequential live work requires, for at most $C$ admitted trajectories:

$$
\begin{aligned}
&\text{mutable public state},\\
&\text{active log reach}[C,P,H],\\
&\text{sampled private profile}[C,P],\\
&\text{root id, path/step ordinal, queue state}[C].
\end{aligned}
$$

Branched live work requires, for at most $F$ occurrences:

$$
\begin{aligned}
&\text{mutable public state},\\
&\text{active log reach}[F,P,H],\\
&\text{logical node/path address and controller factors}[F].
\end{aligned}
$$

Private profiles for branched children need only be transient
$[B_c,P]$ values while sampling their action/chance event.

## 2.2 Rollout: append-only retained rows

At a player decision, write:

- one common immutable, forward-ready model input;
- the minimum mode-specific old policy/value information;
- stable topology/address metadata;
- in branched mode, target-eligible sampled-action rows.

At a stopped leaf, write:

- normal mode: exact sampled reward;
- either RB mode: evaluator-ready range weights, board/stratum data, fixed
  payoff data, and logical completion key.

No later phase should consult mutable public state or replay a discarded path.

## 2.3 Target construction

Target construction owns:

- compact sequential or branched topology;
- immutable decision inputs and old outputs;
- completion-ready leaves;
- exact target destinations;
- one bounded completion/recurrence workspace.

After targets finish, topology, leaves, completion outputs, ready counters, and
controller scratch are released.

## 2.4 Optimization

Optimization owns only:

- forward-ready immutable partition inputs;
- natural-width old outputs;
- exact targets/posteriors and their references;
- action CSR in branched mode;
- sampling metadata;
- optional reference caches selected by measured time saved per byte.

This phase is a repeated consumer. Avoid giving it compact encodings that
require repeated irregular reconstruction unless an end-to-end benchmark shows
a net win.

# 3. Hot mutable poker state

The current live-state payload in
`pbg/games/poker/tensor_state.py` is:

| Information | Shape | Payload bytes/row | Consumer |
|---|---:|---:|---|
| stacks, committed this street, committed total | three $[P]$ fp32 | $12P$ | transition and legality |
| status, acted, reopen permission | three $[P]$ bytes | $3P$ | transition and legality |
| actor, street, button, raises, phase | five int32 | 20 | routing and transition |
| board | $[5]$ int16 | 10 | chance, model, completion |
| last full raise, current bet | two fp32 | 8 | legality |
| history reference and length | two int32 | 8 | history packing |
| fold payoff | $[P]$ fp32 | $4P$ | terminal capture |
| board at fold | $[P,5]$ uint8 | $5P$ | folded summary |
| folded line metadata | $[P,8]$ fp32 | $32P$ | folded summary |
| folded-summary dependency key | int32 | 4 | shared summary lookup |

The raw payload is therefore

$$
S_{\text{state}}=56P+50
$$

bytes per resident state before independent field alignment.

The current public-history row contains

$$
S_{\text{history}}=35+4J,\qquad J=\operatorname{bitlength}(T),
$$

bytes: one refcount, $J$ binary ancestors, one event id, three byte fields, and
six fp32 arithmetic fields.

These are logical lower bounds. The live-state admission charge is the
schema-derived aligned slab capacity plus history-page capacity, queue and
successor overlap, active reaches/profiles, the inference workspace, allocator
tail, and any branched frontier overshoot. In particular, multiplying the two
raw row sizes by an expected live count is not a safe node guard.

## Throughput-first target

Keep hot arithmetic fields in their currently convenient dtypes unless a
controlled CUDA test shows a speedup from changing them. Saving tens of bytes
per live state is not worth repeated casts in every transition.

The owned-payload bounds should be:

### Sequential

$$
M_{\text{seq-live}}
\lesssim
(C+B)S_{\text{state}}
+F_hS_{\text{history}}
+4CPH
+4CP
+O(C+B),
$$

where $F_h$ is the admitted live/shared history-row high-water mark. The $B$
term permits one successor pack to overlap its parents.

This expression describes owned tensor payload. The admission predictor must
replace every realized row count by allocated capacity and add
$W_{\mathrm{infer}}$, allocator headroom, and the measured reserved-byte
residual from Section 1.1.

The current implementation creates all $G$ roots before admission and allocates
root/working reach and profile tensors with leading dimension $G$. Removing
that mismatch is worthwhile, but it is a dedicated sequential-engine refactor,
not a dtype cleanup.

### Branched

$$
M_{\text{branch-live}}
\lesssim
FS_{\text{state}}
+F_hS_{\text{history}}
+4FPH
+O(F)
+W_{\text{child}}(B_c).
$$

At the instrumentation baseline, recyclable ownership did **not** preserve the
required dense compute boundary. Branched construction set both mutable
state/history and active-reach allocation quanta to 256 rows. Once live work
exceeded one page, ordinary `PagedTensorColumn` and
`RecyclableTensorPool` access fell back to generic all-page gather/write
routing. The accepted ROI 1 state/history slice now uses grouped dense slabs
bounded by admitted live high water; active reach still uses the paged owner.

This distinction is central:

- recyclable logical handles are correct;
- stable page-backed capacity accounting is useful;
- scanning every accumulated page for each bounded hot pack is not acceptable.

Section 15 measured state/history live-row high waters of 6,212/14,073 for the
median tree, or approximately 25/55 pages. These live owners need a direct
dense or bounded-segment compute interface whose capacity follows an admitted
frontier high-water bound. They must not be sized to total generated nodes, and
discarded state must not be recomputed.

## Fields worth removing from hot state

`fold_payoff[P]` is written only when a terminal fold is recognized. It can be
computed directly into the leaf/reward row instead of being copied through
every successor. This saves $4P$ bytes per live state, which is negligible
beside the $4PH$ exact-reach row. Its plausible benefit is reduced
copy/update traffic, not capacity. Keep it as an isolated hot-transition
experiment rather than an early memory priority.

# 4. Common forward-ready decision input

All three modes should use one `DecisionInput` concept and the same canonical
field names. Mode changes affect targets and old-output retention, not the
meaning of the public/model input.

The current public payload per decision is, excluding field alignment,

$$
I_{\text{public}}
=70+14P+E+4n_{\text{ticks}}+28T.
$$

The range payload is

$$
I_{\text{range},m}
=
\begin{cases}
4PK,&m=\mathsf{pre},\\
4A_{\text{live}}H,&m=\mathsf{post}_{A_{\text{live}}},
\end{cases}
$$

plus one int32 folded-summary reference where $A_{\text{live}}<P$.

Thus

$$
I_m=I_{\text{public}}+I_{\text{range},m}
+4\mathbf 1_{\{m\text{ uses a folded summary}\}}.
$$

These $I_m$ values are unaligned per-row payloads. A capacity report must use
the generated partition schema, its independently aligned columns, and its
allocated row capacity. The examples below explain field scale; they do not
predict how many decisions fit on a card.

Examples:

### HU, 32 bb, $R=13$

Here $P=2$, $T=42$, $n_{\text{ticks}}=64$, and $E=67$:

$$
I_{\text{public}}=1{,}597\text{ bytes}.
$$

The preflop range is $4PK=1{,}352$ bytes and the postflop range is
$4PH=10{,}608$ bytes.

### Six players, 128 bb, $R=13$

Here $T=662$, $n_{\text{ticks}}=256$, and $E=259$:

$$
I_{\text{public}}=19{,}973\text{ bytes}.
$$

The no-fold postflop range is $4PH=31{,}824$ bytes. The history contribution is
$28T=18{,}536$ bytes: material, but not larger than the six-player exact range.

## Minimal information versus best physical representation

| Field family | Semantically derivable? | Used when? | Throughput-first decision |
|---|---|---|---|
| pot and call amount | yes, from commitments/current bet | every model forward | retain; cheap cache |
| legality summary and raise bounds | yes, from mask/tick grid | every model forward | retain; cheap cache |
| `n_ticks`, tick values, seat mask | run/partition constants | every model forward | do not own per row; supply broadcast views |
| exact legal-action mask | no for the learner without replaying legality state | rollout and every loss/model use | retain once in `DecisionInput` |
| legal event ids | fixed `arange(E)` | policy adapter/loss | run constant, never a $[D,E]$ owner |
| padded public history | derivable from a linked path | every repeated model forward | retain forward-ready by default |
| natural range | no after active reach is released | every model forward | retain once |
| folded summary | expensive shared dependency | postflop forwards with folds | retain one anchor row and one int32 reference |

The per-row constants cost

$$
D(4+4n_{\text{ticks}}+P)
$$

bytes for `n_ticks`, tick values, and the all-ones seat mask. They can be
provided as expanded immutable run constants without algebraic reconstruction.
This is a worthwhile schema cleanup because it reduces storage without adding
state-dependent learner work.

## History policy

Keep dense forward-ready $[D_m,T]$ history fields as the default. Their payload
is

$$
M_{\text{history}}=28DT.
$$

This is intentionally a compute cache. A shared linked history that requires
ancestor traversal or materialization during each optimization reuse is
removed from the target design.

If history storage or padded attention becomes a measured capacity limiter, a
later experiment may materialize each history once at seal into a small set of
dense history-width buckets. This preserves forward-ready learner inputs while
potentially reducing both retained padding and attention work. The experiment
must include any seal-source representation, one-time materialization,
source/destination overlap, and the cost of splitting learner and rollout
packs. It is accepted only on end-to-end throughput and safe admitted work,
not byte savings alone.

# 5. Mode-specific retained decision and action data

One canonical vocabulary should be used from rollout capture through
`TrainingStorage`:

- `owner`
- `event`
- `realized_type`
- `old_value`
- `old_log_prob`
- `log_weight`
- `nominal_child_count`
- `executed_child_count`
- `frozen_trace_variance`
- `frozen_cost`
- target fields with their final canonical names

Do not maintain source aliases such as `old_v`, `oldV`, and `old_value` for the
same tensor.

Canonical vocabulary does not imply one union allocation. Normal,
sequential-RB, and branched-RB rows have different widths, consumers, and
lifetimes. Generate separate mode-and-partition physical schemas and omit
unconsumed columns entirely; do not introduce null columns or a common
capacity merely to share names.

| Concept | Modes | Last consumer | Minimal representation |
|---|---|---|---|
| `owner` | all | target routing and learner grouping | one int32 per decision |
| `event` | normal, sequential RB; branched action rows | policy loss | one int32; narrower storage is not useful while adapters consume int32 |
| `realized_type` | normal | exact-row learner projection | one int32 per decision |
| `old_value` | all | learner reference/value loss | scalar in normal; natural model width in both RB modes |
| `old_log_prob` | normal, sequential RB; branched action rows | learner policy ratio | scalar in normal; natural model width in sequential RB and per sampled action in branched RB |
| `log_weight` | branched RB | backward target recurrence | one fp64 per decision under the current numerical contract |
| child counts and frozen controller inputs | branched RB | backward target recurrence and controller diagnostics | one scalar of the consumed dtype; omit a field entirely when that controller head is disabled |
| final target/posterior fields | mode-specific | optimizer | scalar only in sampled normal mode; exact $H$ width wherever private hands are integrated out |

The width distinction is semantic, not a reason to invent mode-specific names:
for example, `old_value` is the same concept whether its row shape is scalar or
$[Y_m]$.

The authoritative decision contract includes `owner`. In poker it equals the
already-retained `DecisionInput.to_act` at a player decision. A poker-specific
implementation may test a view/reference rather than a second allocation if
all generic target consumers, schema validation, alignment, and lifetimes
remain valid. Until that is demonstrated, the equations below count the
independently owned `owner` column. This is not grounds to delete `owner` from
the generic schema.

## 5.1 Normal sequential

At capture time, the acting private hand is still available. Select the scalar
model rows immediately:

$$
\begin{aligned}
&\text{owner}[D],\quad
\text{event}[D],\quad
\text{realized\_type}[D] && 3D\text{ int32},\\
&\text{old\_value}[D],\quad
\text{old\_log\_prob}[D] && 2D\text{ fp32}.
\end{aligned}
$$

The pre-target retained policy payload under the current independent-owner
schema is

$$
M_{\text{normal-policy}}=20D.
$$

After targets, add scalar `lambda_return[D]`, for a total of

$$
M_{\text{normal-train-metadata}}=24D
$$

bytes, excluding common inputs and sampling metadata.

The current source stores two $[D_m,Y_m]$ rows and projects them later. Direct
capture saves

$$
8\sum_mD_m(Y_m-1)
$$

bytes in the rollout/seal source and avoids wide writes.

## 5.2 Sequential Rao--Blackwell

The exact private hand is integrated out, so natural model rows are required:

$$
\begin{aligned}
&\text{owner},\text{event},\text{posterior\_ref}
&&3D\text{ int32},\\
&\text{old\_value}_m,\text{old\_log\_prob}_m
&&2[D_m,Y_m]\text{ fp32},\\
&\text{lambda\_return}_m
&&[D_m,H]\text{ fp32}.
\end{aligned}
$$

The retained decision/target payload under the current independent-owner
schema is

$$
M_{\text{seqRB-decision}}
=12D+8\sum_mD_mY_m+4DH.
$$

Let $\mathcal U$ be the required trajectory/player pairs that own at least one
decision. The current shared posterior pool retains the fp32 posterior plus
one int32 trajectory and one int32 player key per row:

$$
M_{\text{seqRB-posterior}}=(4H+8)|\mathcal U|,
\qquad |\mathcal U|\leq GP.
$$

This is already the correct deduplication boundary: one completion posterior
per required trajectory/player, referenced by all of that player's decisions.
The $4H|\mathcal U|$ matrix term is dominant, but its lookup metadata should
not be hidden in exact accounting. It may be removed only if a deterministic
indexing rule replaces it without adding a more expensive map or
synchronization.

## 5.3 Branched Rao--Blackwell

Before targets, retain per decision:

$$
\begin{aligned}
&\text{owner} &&4D,\\
&\text{old\_value}_m &&4\sum_mD_mY_m,\\
&\text{log\_weight} &&8D,\\
&\text{nominal\_child\_count},\text{executed\_child\_count} &&8D,\\
&\text{frozen\_trace\_variance},\text{frozen\_cost} &&8D
\quad\text{when consumed by dynamic targets}.
\end{aligned}
$$

Per target-eligible sampled player action:

$$
\text{event}[A]\text{ int32},
\qquad
\text{old\_log\_prob}_m[A_m,Y_m]\text{ fp32}.
$$

Use one action pointer:

$$
\text{action\_ptr}_m[D_m+1]\text{ int32}.
$$

After target construction, retain:

$$
\begin{aligned}
&\text{value\_hat}_m[D_m,H],\\
&\text{trace\_variance\_hat}[D]\text{ and validity mask},\\
&\text{cost\_hat}[D],\\
&\text{child\_posterior}_m[A_m,H],\\
&\text{advantage\_hat}_m[A_m,H].
\end{aligned}
$$

The exact posterior and advantage rows dominate branched training storage.
Saving a handful of int32 topology columns is not valuable if it makes these
rows harder to schedule or reduce.

The continuous branching intensity vector is used only for diagnostics after
rollout. Accumulate count, sum, extrema, and clipping counts and retain those
scalars rather than `branching_beta[D]`.

# 6. Topology

## 6.1 Sequential

The final decision-only topology is:

$$
\text{trajectory\_ptr}[G+1],\quad
\text{decision\_partition}[D],\quad
\text{decision\_row}[D],\quad
\text{leaf\_row}[G].
$$

The current topology payload is

$$
M_{\text{seq-topology}}=4(G+1)+8D+4G=8D+8G+4.
$$

If leaf capture is changed and validated to write every leaf directly at its
root index, `leaf_row[G]` then becomes unnecessary and the payload falls to
$8D+4G+4$. Do not remove the map before that stronger invariant exists.

No final sequential tensor should scale with all public occurrences $N$.
Generated occurrence count and per-trajectory length are diagnostics that can
be accumulated as scalar/$[G]$ counters.

The current rollout still builds all-occurrence node/edge topology and a second
occurrence owner. Replacing them with dense reusable trajectory slots is a
high-value but large refactor. It should be implemented only in the sequential
engine and must preserve:

- direct dense state/reach access;
- logical RNG addresses;
- queue order and physical pack behavior;
- append-only decision capture;
- exact chronological CSR sealing.

## 6.2 Branched

Backward targets need efficient node-to-child, node-to-decision, and
node-to-leaf access. Keep the throughput-oriented sealed representation:

$$
\begin{aligned}
&\text{parent}[N],\
\text{child\_begin}[N],\
\text{degree}[N],\\
&\text{decision\_partition}[N],\
\text{decision\_row}[N],\
\text{leaf\_row}[N],
\end{aligned}
$$

plus `nominal_degree[N]` where required and two protected-root scalars.

Its payload is

$$
M_{\text{branch-topology}}
=24N+4N\mathbf 1_{\{\text{nominal degree retained}\}}+8.
$$

Sparse decision/leaf maps can reduce this toward
$12N+4D+4L$, but they would require indirect search, merged sparse traversal,
or transient inverse maps during target construction. The wide
$[D,H]$/$[A,H]$ target rows dominate memory, so the sparse form is not the
default target.

Sealing currently reorders nodes because protected root refill appends new root
children after the initial tree. Those children are not contiguous in
generation order. Eliminating that reorder requires a different root topology
contract or a measured two-span special case; it is not a free storage cleanup.

# 7. Leaves

Under the authoritative storage-lifecycle contract, each RB leaf owns the
evaluator-ready contiguous fields:

- `board_cards uint8[L,5]`;
- `range_weights float32[L,P,H]`;
- `stratum_level int32[L,P]`;
- `payoff_baseline float32[L,P]`;
- `stratum_amount float32[L,P]` for the no-side-pot poker contract;
- `seed int64[L]`.

This is a physical retention contract, not a claim that the evaluator ABI
proves every field is semantically irreducible. The evaluator declares
`stratum_amount` as `[B,K_{\mathrm{pot}}]`, where $K_{\mathrm{pot}}$ is the
pot-stratum count. The current equal-stack, no-side-pot schema uses a
player-width padded row, so the formulas below describe the present schema.
Likewise, leaf `owner` is not an evaluator argument but remains required by
target construction to distinguish terminal fold from showdown; it may be
released only after that consumer.

A compact leaf representation expanded into a bounded completion workspace
would change the governing leaf-ownership contract. It is therefore outside
this recommendation set unless the theory contract is explicitly amended and
the expansion cost, overlap peak, and end-to-end throughput are then measured.

## 7.1 Normal sequential

Write reward directly at root index:

$$
\text{reward}[G,P]\text{ fp32},
\qquad
M_{\text{normal-leaf}}=4GP.
$$

Do not store node id, root id, path key, or log reach.

## 7.2 Sequential RB

Write completion-ready fields directly at root index. The raw payload per leaf
is

$$
S_{\text{RB-leaf}}=4PH+16P+17
$$

bytes before independent field alignment:

- range weights: $4PH$;
- fold payoff, stratum level, payoff baseline, stratum amount: $16P$;
- owner, board, completion key: $4+5+8=17$.

Thus

$$
M_{\text{seqRB-leaf}}=G(4PH+16P+17)
$$

before alignment.

Compute `range_weights` from active log reach and compute the logical completion
key when the leaf is created. Then release active reach, state, path key, and
private profile. This moves normalization and key construction onto leaf
capture; it may add rollout kernels or extend parent/live-owner lifetimes until
those kernels finish. Accept it only when the measured phase peak falls and
neither rollout nor end-to-end throughput regresses materially.

## 7.3 Branched RB

Use the same completion-ready leaf schema with leading dimension $L$. The
sealed topology's `leaf_row[N]` supplies the node association; no node id belongs
inside the leaf payload:

$$
M_{\text{branchRB-leaf}}=L(4PH+16P+17)
$$

before alignment.

Direct completion-ready capture avoids simultaneous `log_reach[L,P,H]` and
`range_weights[L,P,H]` owners during a later conversion, saving an overlap of
approximately

$$
4LPH
$$

logical bytes. The physical claim must be verified at the transactional
capture/seal overlap and include any newly retained live tail.

# 8. Folded-summary anchors

The shared-anchor design should remain. Its common payload per anchor is
exactly, before independent field alignment,

$$
4H+4Q+16RQ+4
$$

bytes in the sealed learner pool for hand log ratio, spectral factors, and the
tight-cluster scalar. The mutable rollout cache additionally owns the int64
logical anchor, so its raw common payload is $4H+4Q+16RQ+12$ bytes per row.

For $F_m=P-A_{\text{live}}$ folded players, folder-dependent payload is

$$
F_m(45+16R)
$$

bytes with the current fp32/int64/mask fields:

- log mass: $4F_m$;
- card mass: $16RF_m$;
- mask: $F_m$;
- line metadata: $32F_m$;
- seat: $8F_m$.

Using exact $F_m$ per partition rather than the global $P-2$ is structurally
clean and can save

$$
(P-2-F_m)(45+16R)S_m
$$

bytes. It is worth implementing only after the more direct capture/leaf
changes, because common spectral/hand payload usually dominates each anchor.

Do not compact `folder_seat` merely to save a few bytes if that introduces a
cast in every model use.

# 9. Work queues and routing

## 9.1 Queues

The current queue payload per partition is approximately

$$
8(Q_{\max}+B)+8
$$

bytes for handles, dequeue staging, and count. Even six 4,096-row queues are
only about 0.2 MiB before allocator overhead.

Keep independent per-partition queues. A shared queue allocator is not worth
extra hot-path coordination for this byte scale. Handle dtype may change only
as part of a broader occurrence-id migration that removes conversions
end-to-end.

Stable sealed/training partitions do not require identical physical rollout
queues. In particular, exact flop/turn/river execution queues avoid padding a
mixed postflop prefix to its earliest street. The current working-tree natural
`post_2` rollout queue removed this accepted physical distinction and must be
restored or beaten with paired evidence.

## 9.2 Routing

Current routing to $M$ destinations constructs approximately

$$
F_bM\text{ membership bytes}
+8F_bM\text{ candidate-handle bytes}
+F_bM\text{ validity bytes},
$$

or roughly

$$
10F_bM+O(M)
$$

bytes, and performs one stable sort per destination.

A one-sort representation uses:

$$
\text{order}[F_b],\quad
\text{sorted\_handles}[F_b],\quad
\text{counts}[M],\quad
\text{offsets}[M+1],
$$

for approximately

$$
16F_b+O(M)
$$

bytes with int64 order/handles before queue-interface staging. It replaces $M$
sorts with one stable sort by destination, but this is not yet an
apples-to-apples physical bound: the current path emits fixed-shape candidates
and validity tensors without dynamic host slicing. The experiment must include
the fixed-window materialization or device-side offsets needed by the queue
interface, as well as synchronization and kernel-launch costs. Treat it as an
attractive isolated CUDA experiment, not as an assumed improvement.

Routing analysis must also include address resolution inside each owner.
`ChunkedOccurrenceArena` currently constructs selectors and merges candidates
for every accumulated chunk on each gather/scatter. `PagedTensorColumn`
performs analogous all-page merging for topology/controller data. Replacing
the destination sorts while retaining those page scans would optimize the
wrong layer.

# 10. Physical model and child workspaces

For one player pack $B$, legitimate transient owners include:

$$
\begin{aligned}
&\text{state/reach/profile gathers},\\
&\text{forward-ready public/range inputs},\\
&\text{logical RNG keys and legal action rows},\\
&\text{model outputs on }[B,Y_m,\ldots],\\
&\text{selected log probabilities},\\
&\text{successor state/reach/event work}.
\end{aligned}
$$

The hard workspace bound must also include model activations, padded-attention
workspace, backend/JIT allocations, output staging, and allocator rounding for
the realized physical shape. Optimization separately charges autograd-saved
tensors, AMP temporaries, flat gradients, collective workspace, optimizer
state, reference-cache build/output, and exact-loss gathers and reductions.
These calibrated bounds, rather than the retained-row formulas, commonly
determine the maximum safe family and backward widths on a 2080 Ti.

For a branched child pack $B_c$, use:

$$
\text{parent\_position}[B_c],\
\text{child\_ordinal}[B_c],\
\text{profile}[B_c,P],\
\text{child\_reach}[B_c,P,H],\
\text{successor\_state}[B_c,\ldots].
$$

Indexing the complete parent `PolicyOutput` into a child-sized copy is a
possible branched throughput target. A better sampling/transition interface
could consume parent outputs through `parent_position`. This is worth testing
only in branched mode and must include kernel-launch and gather costs; it is not
a storage-only mechanical rewrite.

Use operation-specific bounded workspaces. Do not force every phase into one
universal arena if doing so prevents overlap, graph capture, or efficient
shape-specific allocation.

# 11. Canonical vocabulary, mode-specific schemas

Reuse concepts across modes without forcing fields into modes that do not
consume them or forcing unlike lifetimes into one allocator. Each object below
generates a separate schema for the selected mode and partition.

## Common objects

1. `DecisionInput[m]`
   - identical public/range meaning in every mode;
   - one legal mask;
   - one folded-summary reference where applicable;
   - forward-ready dense history;
   - broadcast run constants are not row-owned.
2. `DecisionAddress`
   - sequential source: root id and decision ordinal until CSR seal;
   - branched source: node id until tree seal.
3. `OldDecisionData[m]`
   - canonical `owner`, `old_value`, and optional event/type/old-log-probability
     or branching metadata selected by mode.
4. `ActionData[m]`
   - exists only for branched target-eligible actions;
   - canonical event and old log probability.
5. `LeafData`
   - normal reward variant;
   - shared completion-ready RB variant.
6. `TargetData[m]`
   - mode-selected final targets under canonical names.

Schemas should be generated from these concepts and mode field subsets. They
should not rename tensors between source, rollout, and training phases, nor
allocate a union of every mode's optional fields.

## Seal boundary

The current scheduler seals paged captures into `SealedRolloutSample`, after
which training translates policy and leaf fields into `RolloutStorage`.
Immutable input blocks are generally reused rather than copied wholesale, so
this is not literally two copies of every input. The avoidable work is:

- wide normal policy sources projected after sealing;
- log reach converted to range weights after sealing;
- renamed source/destination fields;
- duplicated validation and ownership boundaries;
- overlapping source and translated destination columns.

First make normal policy and leaf capture canonical. Then test a direct
`RolloutStorage` seal. The direct seal is accepted only if it does not replace
contiguous copies with per-row/page gathers and does not regress end-to-end
throughput or seal peak.

# 12. ROI-ordered implementation and experiment table

Priority is expected throughput return per implementation/validation cost,
conditioned on the measured binding phase. Byte effects are logical payload
unless explicitly labeled as allocated capacity.

Section 15 is the pre-optimization baseline. Do not first rerun its
uninstrumented 250-second branched sample. Add the page/chunk-path
instrumentation described in the hand-off, then use the same checkpoint,
geometry, and seed for the first paired candidate.

| ROI | Change | Expected byte effect | Throughput expectation | Scope and acceptance gate |
|---:|---|---:|---|---|
| 1 | Remove generic all-page scans from branched mutable state, shared history, and active reach | keep capacity proportional to admitted live high water; remove per-256-row tensor fragmentation and routing transients | strongly positive; this is the sequential paged-state regression class still present in branching | **State/history accepted 2026-07-23; reach pending.** Strict seed 104,830 reduced selected-owner generic accesses from 81,616 to zero, aggregate candidate rows by 88.5%, rollout time by 40.3%, and total time by 18.1%. Reserved peak was unchanged and allocated peak rose 0.02%. Exact discrete work and zero undeclared synchronizations were preserved. A reach-only slice removed 767 scans and 55.4GB of routing temporaries but improved rollout only 0.22%; it was rejected and removed. |
| 2 | Commit host-proven contiguous branched child subspans directly | capacity-neutral; removes page-match matrices and candidate copies | positive and low-risk | **Instrumented 2026-07-23; implementation pending.** Topology append already used 319 node and 318 edge contiguous writes, but hot updates still included 970 generic node writes and 312 generic edge writes, materializing 28.2M candidate rows. Pass physical subspan cursors through the remaining commit boundary; preserve two-pass address publication and stable child order; exact children/edges/events; paired seed-104830 branch gate |
| 3 | Make occurrence/topology/controller access pack-local instead of all-chunk/all-page | retain paged append-only capacity; remove $O(B\,n_{\text{pages}})$ hot routing work | potentially strongly positive on large trees | **Instrumented 2026-07-23; implementation pending.** Occurrence, node/edge topology, and controller owners made 10,640 generic accesses, visited 560,060 chunks/pages, and materialized 85.3M candidate rows; each reached 91 accumulated segments. Medium core refactor. Stable logical addresses and last-consumer ownership; report early/late-pack timing and scaling versus realized page count |
| 4 | Restore exact-street physical rollout queues or an equally efficient bounded packer | no required retained-byte change; reduces mixed-street padded work | positive; previously about 4.9% in HU | Separate scheduling experiment. Natural training partitions unchanged; identical keyed RNG/topology; padded pairs and wall time must improve |
| 5 | Replace per-destination routing sorts with one stable partition only after owner-local page scans are gone | candidate core is roughly $10F_bM\to16F_b$, excluding queue staging | unknown until fixed-window/device-offset costs are included | Localized experiment. Identical stable queue order and RNG; no new host slicing/sync; realistic sequential and branched CUDA timing |
| 6 | Capture scalar normal old value/log probability and realized type immediately | save $8\sum_mD_m(Y_m-1)$ source bytes | positive: fewer wide writes and later gathers | Small–medium. Exact normal targets/losses; paired CUDA rollout and end-to-end gate |
| 7 | Capture canonical completion-ready leaves directly | normal removes roughly $4GPH$ unused log-reach bytes; RB removes about $4LPH$ logical seal overlap | uncertain: fewer owners/copies, but normalization and key work move into capture | Medium. Evaluator-ready equality, completion-key invariance, lower measured phase peak, unchanged live-tail bound, and paired rollout/end-to-end gate |
| 8 | Remove per-row run constants and legality duplication | save $D(4+4n_{\text{ticks}}+P)$ owned input bytes; remove logical $4DE$ event-id schema | neutral-to-positive if only broadcast views are added | Medium schema/learner change. Identical logits/losses, lower unique/capacity bytes, and repeated-learner throughput gate |
| 9 | Compute terminal fold payoff directly into the leaf/reward row | save only $4P$ per live state plus possible successor copy traffic | capacity-neutral; hot-path effect may be positive or negative | Small poker-only experiment, separate from leaf canonicalization. Exact normal rewards and RB fold payoff; paired transition/rollout gate |
| 10 | Use canonical field names and test a direct rollout-storage seal | removes translated policy/leaf overlap and compatibility code | probably neutral; seal is currently a small phase | Apply vocabulary with the first owner it simplifies, then isolate the direct seal. Byte-for-byte canonical output, no peak increase, paired timing with dispersion |
| 11 | Aggregate branching-intensity diagnostics online | save $4D$ retained diagnostic bytes | neutral-to-positive | Small. Exact diagnostic equality within fp reduction tolerance |
| 12 | Sequential-only live-slot/topology rewrite | removes all $N$ node/edge/occurrence source tensors; active reach/profile $G\to C$; removes root reach | potentially strongly positive, but large and high-risk | Preserve dense access and sampled object; paired multi-repeat throughput/headroom gate |
| 13 | Exact folded-summary folder geometry per partition | save $(P-2-F_m)(45+16R)S_m$ | neutral if no new casts/indirection | Medium poker-only. Benchmark multiplayer folded workloads |
| 14 | Avoid child-sized copies of complete parent policy output | save mode/output-dependent $O(B_cY_m)$ transient bytes | uncertain but promising in branching | Medium–large interface change. Require meaningful branched end-to-end improvement |
| 15 | One-time seal materialization into dense history-width buckets, only if history/attention is measured as binding | workload-dependent reduction from $28DT$ and padded attention | uncertain; materialization and pack fragmentation can lose | Experimental. Forward-ready learner rows; charge seal overlap; multiplayer packing study and end-to-end win required |

Do not bundle multiple ROI items into one benchmark. Establish byte and timing
deltas independently before combining them. If attribution shows that a
different phase or owner is binding, re-order the remaining items rather than
following the table mechanically. Isolation during experimentation does not
authorize permanent parallel implementations: an accepted item ends with one
production path and deletion of the implementation it supersedes.

The 2026-07-23 instrumentation run is
`runs/experiments/nlhe_hu_32bb_branched_fixed_storage_access_20260723/benchmark.json`.
Its aggregate totals were 82,654 generic accesses, 2.47M segments visited,
768.0M candidate rows, and 81.9GB of cumulative routing temporaries issued
(allocation/traffic sum, not a simultaneous peak). It preserved 58/58 applied
optimizer steps, 14,848 learner presentations, zero undeclared host
synchronizations, and the 8.960GiB reserved rollout peak. The fresh tree had
23,267 nodes versus 23,162 in the historical artifact after an independently
executed throughput-mode warmup, so its 244.790-second rollout is diagnostic
only, not a strict paired timing acceptance result.

The strict cold comparison required for ROI 1 is
`runs/experiments/nlhe_hu_32bb_branched_fixed_storage_roi1_baseline_20260723/benchmark.json`
against
`runs/experiments/nlhe_hu_32bb_branched_fixed_storage_state_history_candidate_20260723/benchmark.json`.
Both used `--warmup 0 --repeats 1 --seed 101`, so logical seed 104,830 began
from byte-identical checkpoint weights. Nodes, children, events, queue work,
RNG-derived work, retained-row counts, model/evaluator work, learner
presentations, and optimizer attempts/outcomes were identical. Loss and
gradient summaries remained within the dispersion observed in a same-code
throughput-mode rerun. Rollout time was 376.289 to 224.568 seconds and total
time was 827.087 to 677.514 seconds. Rollout peak reserved memory remained
7,807,696,896 bytes; peak allocated memory increased from 4,622,451,200 to
4,623,313,408 bytes. The selected owners use one direct slab segment per
access, with capacities following their 6,532/15,996-row live high waters
rather than total generated nodes.

Independent ordinary-CLI runs for logical seeds 104,830, 209,559, and 314,288
are stored under
`runs/experiments/nlhe_hu_32bb_branched_fixed_storage_state_history_seed{104830,209559,314288}_20260723/`.
All three recorded zero selected-owner generic scans, zero undeclared
synchronizations, unchanged 14,848 learner presentations and 58 optimizer
attempts, and clean teardown. Their concurrent timings are robustness evidence,
not substitutes for the paired acceptance timing.

# 13. Acceptance and reporting

Every accepted storage change must report:

- the production entry points and consumers migrated to the replacement;
- the obsolete symbols, owners, schemas, config fields, conversions, tests,
  and fallback branches deleted, plus a repository-search audit showing that
  none remain outside explicit historical records;
- exact logical schema payload and allocated-capacity deltas by phase,
  including alignment, page/slack capacity, and transactional overlap;
- unique retained tensor bytes, accounting correctly for aliases/views;
- rollout, seal, targets/completion, learner, and end-to-end wall time;
- fresh decisions and learner presentations so throughput is not improved by
  doing less statistical work;
- model calls, rows/call, padded tokens/pairs, kernel launches, and device/host
  synchronizations where relevant;
- direct versus generic page/chunk access counts, pages/chunks visited per hot
  pack, logical rows requested versus candidate rows materialized, and
  early-tree versus late-tree pack time for branched changes;
- forced drains, backward physical-subspan sizes, completion calls, and
  optimizer steps attempted/applied;
- peak allocated, peak reserved, non-Torch occupancy, and teardown state;
- per-rank and per-iteration peak distributions or conservative quantiles,
  including the branched live-tail overshoot;
- paired fixed logical RNG seeds and equality of topology, retained inputs,
  targets, and gradients within their declared arithmetic tolerance;
- realistic HU sequential, sequential-RB, branched-RB, and multiplayer
  coverage appropriate to the changed owner.

Acceptance requires executing the ordinary production CLI/runtime path.
Calling a new helper directly is useful unit evidence but does not prove that
training uses it. Tests must cover the surviving implementation through its
production entry point and must not preserve obsolete fixtures merely to keep
the deleted abstraction testable.

Use paired fixed-seed repeats and report dispersion. Correctness, the
$M_{\text{hard}}$ ceiling, guard/headroom, OOM, and clean-teardown checks are
hard gates. Timing gates must use a predeclared practical regression threshold
or confidence interval supported by the repeat count; fixed 98% or 99%
per-case cutoffs are not justified without that statistical design.

Minor storage inefficiency is acceptable when it improves throughput. A memory
reduction is valuable when it removes a binding phase peak or admits enough
additional useful work to improve end-to-end decisions/s. Neither line count
nor byte count overrides measured training performance. Decisions/s also does
not override unchanged work: the accepted plan must preserve the sampled
object, learner presentation ratio, minimum useful family width, physical
backward width, and optimizer/completion work.

# 15. Efficiency-to-beat baseline: 2026-07-23

## 15.1 Scope and provenance

This is a single-rank, one-iteration CUDA benchmark on an NVIDIA GeForce RTX
2080 Ti (compute capability 7.5), PyTorch 2.12.1+cu130, CUDA 13.0, and the
`local_3x2080ti` hardware profile. It is a storage/phase regression target, not
a replacement for the sustained multi-rank training gate in
`docs/todo/optimization.md`.

The latest source checkpoint is
`runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt`:

- source SHA-256:
  `bced881945512f2a4152d3acd88a0a06395a00130cdf68a61aa2652bc1b5aebb`;
- current-schema sequential wrapper:
  `runs/experiments/nlhe_hu_32bb_rb_storage_baseline_20260723/benchmark_init_ckpt_0001110_v8.pt`,
  SHA-256
  `854c449b3b55a158d0fe97533b75cd1ca3d98262ee31cafd95a657b6abfaaa28`;
- current-schema branching wrapper:
  `runs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723/benchmark_init_ckpt_0001110_dummy_heads_seed91301_v8.pt`,
  SHA-256
  `7ddaad84f691ee31336f4e852ca44e566596275ba05fcdf56cb527087009aa21`.

All 366 source model tensors are byte-identical in both wrappers. The
sequential wrapper adds no tensor. The branching wrapper adds exactly 32
randomly initialized T/C-head tensors with seed 91,301 and changes no source
tensor.

Every mode used one unprofiled warmup iteration followed by three profiled
repeats. Each repeat resets model, optimizer, scaler, and cost-safeguard state
to the identical post-warmup snapshot. Training seed is 101; repeat logical
seeds are 104,830, 209,559, and 314,288; the profiled logical iteration is one.

Shared physical settings are:

| Setting | Value |
|---|---:|
| Family/model pack maximum | 84 rows |
| Maximum active sequential trajectories | 64 |
| Storage page | 256 rows |
| Statistical minibatch | 256 rows |
| Gradient steps attempted | 58 |
| Learner presentations | 14,848 rows/rank |
| RB completion samples | 1,024/leaf |

The statistical minibatch is not the physical backward width. Current
admission splits the regular learner work into 84-row preflop and 12-row
postflop subspans. That distinction is a central storage optimization target.

## 15.2 Mode geometry

| Mode | Roots/rank | Fixed $\beta$ | Tiny-tree floor | Diagnostic upper trigger | Median $N$ | Median player $D$ | Median actions $A$ | Median leaves $L$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 3,072 | — | — | — | 18,112 | 11,931 | 0 | 3,072 |
| Sequential RB | 3,072 | — | — | — | 17,875 | 11,773 | 0 | 3,072 |
| Branched RB | 1 | 3.0 | 1,000 | 10,000 | 23,162 | 14,565 | 17,503 | 6,567 |

The sequential roots were chosen because they realize about 12k player
decisions/rank. Branched mode samples exactly one distinguished-root tree per
GPU. Its fixed-beta scale was exactly one: the required
`branching_ideal_nodes=20000` config field did not drive this benchmark.

The branched player-decision range was 11,258--14,665, entirely inside the
requested 10k--20k band. The total-node range was 18,933--23,272.

The tiny-tree floor was used only as the intended bimodality guard. Two repeats
needed no refill. In the other, the ordinary $\beta=3.0$ tree had only seven
nodes; two additional complete root-child subtrees added 18,926 nodes and
produced one 18,933-node distinguished-root tree. The floor was 1,000, not
18,933: the overshoot is the indivisible complete refill subtree. No forest or
minimum-18k workload rule was used.

The diagnostic upper trigger first activated at 9,828--9,865 nodes (median
9,850). It then used multiplicity one for already-live/future frontier work so
the single tree could finish; it was not a hard truncation. This explains why
final $N$ exceeds 10,000. This guarded result is valid as a paired throughput
baseline under the same diagnostic rule. It is not an unguarded fixed-$\beta$
sampling baseline and must not be used to assess estimator quality.

## 15.3 Wall time and throughput

Values are per-metric medians of the three profiled repeats.

**Warning:** the branched timing values are far from a plausible efficient
implementation, but they are the primary reproducible baseline for ROI items
1--5. The gap is not a separate problem: it is evidence that the physical
storage/access path is defective. Use paired same-seed branched timing to rank
those changes. Do not interpret the current cross-mode ratio as an intrinsic
cost of the branching estimator.

| Mode | Snapshot | Rollout | Targets/backward | Seal | Optimization | Total | Rollout decisions/s | End-to-end decisions/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 0.0007 s | 45.793 s | 0.0238 s | 0.0013 s | 85.359 s | 131.177 s | 260.5 | 91.0 |
| Sequential RB | 0.0006 s | 43.737 s | 0.8987 s | 0.0026 s | 80.667 s | 125.306 s | 269.2 | 94.0 |
| Branched RB | 0.0007 s | 249.792 s | 15.891 s | 0.0043 s | 92.461 s | 358.274 s | 58.3 | 40.7 |

Repeat ranges expose the measurement and tree-size variance:

| Mode | Player decisions | Total wall time | Rollout wall time |
|---|---:|---:|---:|
| Normal | 11,412--12,065 | 125.250--133.114 s | 43.194--46.687 s |
| Sequential RB | 11,242--11,934 | 119.017--127.525 s | 41.046--44.743 s |
| Branched RB | 11,258--14,665 | 241.815--360.548 s | 177.103--252.007 s |

The normal raw-rollout median is within about 1.1% of the historical 263.3
decisions/s family-84 result in `docs/todo/optimization.md`, which is useful
cross-check evidence. End-to-end time is learner-dominated in both sequential
modes.

The 249.792-second branched rollout is not explained by completion or backward
targets: those are separately measured at 15.891 seconds. Sampled-action
capture adds real traffic, and branching samples a collision-conditioned
private profile for every child rather than once per sequential trajectory,
but the current data path also makes each later bounded pack route through
more accumulated pages/chunks. Section 15 therefore treats the rollout gap as
a physical data-handling defect until controlled attribution proves otherwise.

## 15.4 CUDA phase high-water marks

`Allocated` and `reserved` are PyTorch CUDA high-water marks reset at each
phase boundary. They include persistent model/reference/optimizer owners.
Resetting peaks does not empty the caching allocator, so rollout reservation
remains visible in targets and seal. Equal reserved values do **not** mean
those phases each own that much incremental storage.

| Mode | Phase | Peak allocated | Peak reserved |
|---|---|---:|---:|
| Normal | Snapshot | 0.228 GiB | 2.992 GiB |
| Normal | Rollout | 4.389 GiB | 7.785 GiB |
| Normal | Targets/backward | 0.325 GiB | 7.785 GiB |
| Normal | Seal | 0.435 GiB | 7.785 GiB |
| Normal | Optimization | 2.762 GiB | 3.016 GiB |
| Sequential RB | Snapshot | 0.229 GiB | 3.158 GiB |
| Sequential RB | Rollout | 4.390 GiB | 7.877 GiB |
| Sequential RB | Targets/backward | 0.568 GiB | 7.877 GiB |
| Sequential RB | Seal | 0.511 GiB | 7.877 GiB |
| Sequential RB | Optimization | 2.845 GiB | 3.158 GiB |
| Branched RB | Snapshot | 0.250 GiB | 3.621 GiB |
| Branched RB | Rollout | 4.388 GiB | 8.812 GiB |
| Branched RB | Targets/backward | 0.976 GiB | 8.812 GiB |
| Branched RB | Seal | 0.806 GiB | 8.812 GiB |
| Branched RB | Optimization | 3.355 GiB | 3.621 GiB |

The median branch sample also exposes the hot-owner page geometry:

| Owner | Realized/live high water | Approximate 256-row pages/chunks |
|---|---:|---:|
| Mutable poker state | 6,212 | 25 |
| Shared public history | 14,073 | 55 |
| Topology/controller/occurrence extent | 23,162 | 91 |

Direct `PagedTensorColumn` state/history access is available only while the
owner has one physical page. Generic gather/write and occurrence access loop
over the growing page/chunk list. This creates both increasing work per pack
and many field-by-page CUDA allocations. A smaller earlier branch artifact
with 9,398 nodes had only 973 live states and 1,779 history rows and completed
in 45.326 seconds for 3,761 decisions. The Section 15 median has 3.87 times as
many decisions but takes 5.51 times as long, consistent with a page-count
dependent cost.

Unique retained target storage was 97.2 MiB normal, 255.9 MiB sequential RB,
and 584.1 MiB branched RB. This is the key quantitative warning for this note:
logical retained bytes are far below the 4.39 GiB allocated and
7.79--8.81 GiB reserved rollout high-water marks.

Before the final branch run, a conservative 49 kB/decision retained-capacity
estimate put 20k decisions below about 0.94 GiB. The measured complete-tree
run then peaked at 4.39/8.81 GiB allocated/reserved, below the approximately
10.0 GiB per-card profile ceiling. The three branch samples' rollout peaks
ranged over 4.316--4.400 GiB allocated and 8.207--8.960 GiB reserved. This
empirical allocation check, not the logical payload alone, justified the 10k
trigger.

## 15.5 Other work and saturation metrics

| Mode | Policy model calls / rows | Max model pack | Physical learner subspans | Evaluator calls / rows | Completion draws | Optimizer applied / attempted |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 843 / 33,885 | 84 | 522 | 146 / 3,685 | — | 58 / 58 |
| Sequential RB | 841 / 33,712 | 84 | 522 | 60 / 6,144 | 3,145,728 | 58 / 58 |
| Branched RB | 840 / 31,288 | 84 | 580 | 344 / 29,728 | 6,724,608 | 58 / 58 |

The policy-call column covers the whole iteration, including learner and
reference-cache forwards; it is not a rollout-forward count. In the median
branch sample, inference owned 16,440 rows, reference-cache construction owned
1,871 rows, and small feasibility probes account for the residual. The
remaining rollout rows match the 14,565 player decisions to within those
declared probes. There is no evidence of repeated ordinary policy forwards.

The evaluator rows expose a real sampling-law cost: 23,161 non-root edges each
sample one collision-conditioned private profile, and 6,567 leaves are later
completed, totaling 29,728 rows. Optimize the sampler interface and batching
only after page-routing attribution; do not remove these draws or carry one
root profile through a branched tree.

All modes used 14,848 learner presentations. Regular physical backward spans
were 84 rows preflop and 12 rows postflop. No profiled optimizer step was
skipped as nonfinite.

Branched OOM retry count was zero. The dummy C-head produced median cost loss
60.295 and Gamma-deviance calibration 2.638 versus the activation threshold
0.1. Calibration failed in every repeat, its streak remained zero, and both
`safe_tree_active` and `safe_tree_active_next_iteration` were zero even though
`safe_tree_size=30000` was configured. Median T-head trace-variance loss was
74.533. Thus the C-head safeguard was automatically inactive while independent
tiny-tree prevention remained enabled.

Completion quality remained near the 1,024-sample ceiling:

- sequential RB mean effective sample size 1,023.84, minimum 987.51;
- branched RB mean effective sample size 1,023.80, minimum 981.03.

## 15.6 Regression use

The raw artifacts are:

- `runs/experiments/nlhe_hu_32bb_normal_storage_baseline_20260723/benchmark.json`;
- `runs/experiments/nlhe_hu_32bb_rb_storage_baseline_20260723/benchmark.json`;
- `runs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723/benchmark.json`.

For a storage-only change, use the same checkpoint wrappers, logical seeds,
profile iteration, tree guard, family width, statistical rows, and completion
count. Require:

1. identical per-seed topology/work counts unless the change intentionally and
   correctly changes scheduling without changing the sampled object;
2. identical retained inputs, targets, losses, and gradients within declared
   arithmetic tolerance;
3. all 58 optimizer steps applied and all 14,848 rows presented;
4. no worse admitted family/backward physical widths;
5. no material paired wall-time regression and no increase in a binding memory
   peak without a compensating measured throughput win;
6. zero OOM retries, clean teardown, and the same safeguard/refill state.

Report each repeat as well as medians, especially in branched mode: the one
tiny-tree refill sample demonstrates why a median alone can conceal a
meaningfully different realized workload.
