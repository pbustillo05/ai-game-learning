# Queued sampled branched execution

Status: Phase 7 executable contract, validated and audited at
`0d035afc6c300c382d22f336f7ea3be52b138e70`.

## Closed mode surface

One sampled-branched composition implements:

| Training regime | Branch controller |
|---|---|
| self play | fixed |
| best response | fixed |
| self play | KKT |
| best response | KKT |

Fixed and KKT controllers share the same decision, rollout, topology, target,
sampler, learner, optimizer, recovery, and persistence path. `beta_max` is a
positive integer and must equal the storage manifest. Checkpoint root count
must equal the manifest root count exactly.

## Branch identity and degrees

Every active player or chance node uses ordinary keyed randomized rounding.
The controller returns nominal, executed, and target degrees through one
interface. Logical child identity combines the root key, parent path, and
one-based child ordinal, so duplicate poker action labels remain different
tree children.

A tree owns one sampled private profile. Every sibling and descendant carries
that same profile; branching changes public action/chance history, not the
private draw. Inactive chance rows perform no ROUND draw, do not deal cards,
and do not advance their logical path.

The protected root prefix is retained even when a root is an untrained unary
node. A deficient tree is refilled by copying its immutable root family row and
adding children to the same structural root. Final executed root degree is
used for descendant decision mass; protected target degree remains the
learner action span.

## Best-response roles

`role_selector` is a CUDA family field. Learned versus frozen ownership is
derived from `owner == role_selector` at each model call. One shared
`compact_best_response_evaluation` operation:

- compacts learned and frozen rows separately;
- invokes each model only for its rows under `no_grad` and configured
  autocast;
- restores exact FIFO order; and
- is used by both sequential and branched execution.

No host role queue or pinned role mirror owns rollout state. Root refills copy
both owner and selector from the original root.

## Multiplayer folded state

Nonterminal folds reserve and atomically publish `FoldedRanges.pre` or
`FoldedRanges.post`. Preflop ranges use the $K=R^2$ class domain; postflop
ranges use the exact $H=\binom{4R}{2}$ hand domain. Each stored range includes
its raw log mass.

The destination family row receives the encoded folded-range reference.
Folded summaries refresh after a nonterminal fold and on street change, and
otherwise copy the already resolved suffix. Chance descendants of one fold
reuse the same folded-range row. Heads-up terminal folds consume no folded
range.

## Topology, targets, and sampling

Rollout publishes bounded compact topology packets. Ordinary unary nodes may
be compressed, while trained, branched, nominally branched, and protected
roots remain explicit. The sealed topology preserves:

- exact parent-before-child order;
- nominal, executed, and target degrees;
- distinct duplicate-label edges;
- family, sigma-head, and ragged action bindings; and
- terminal leaf bindings.

Layered target execution is the sole target authority. It propagates sampled
values plus trace and cost targets over the executed topology while learner
spans use protected target degrees.

Decision log mass is propagated additively by one linear CUDA operation, so
valid deep trees cannot underflow before the logarithm. Each stratum builds
one deterministic linear-time CUDA Walker alias table. Sampling gathers exact
ragged action spans with no padding or maximum-action axis.

Cost-aware statistical counts are accepted by the branched sampler. Reference
outputs are optional: a bounded first chunk proves `resident + build +
payload` admission before full construction, every later forward is capped by
`learner_sigma_pack_cap`, and cache gathers use exact sealed sigma indices.
Rejected caching retains the ordinary bounded reference forward.

## Admission and lifetimes

Rollout admission simultaneously covers:

- family model/transition workspace;
- a second family bound for best-response merge lifetime;
- chance-expanded state and routing;
- expanded folded-statistics scratch;
- 95-byte compact node/edge packet rows plus the dummy row; and
- library workspace.

Align-and-seal admits CUDA parent/degree inputs and additive log mass. Learner
admission covers sigma/action packs, the 44-byte-per-decision alias peak,
reference cache resident/build overlap, and library workspace.

Topology packets, failed-attempt owners, completion state, sealed stores,
reference caches, alias tables, log normalizers, and stratum/store references
are explicitly released at their final consumer.

## Recovery and agreement

Only declared capacity overflow is recoverable. A failed attempt is discarded
in full, the controller performs bounded backoff, and the retry uses a
disjoint retry epoch and logical address range. Rank-common agreement selects
the same failure on every distributed rank. Any other failure terminates.

Checkpoint state contains the exact fixed/KKT controller and optional
cost-safeguard state. Retry never changes the sampled law except through the
declared controller scale and disjoint logical RNG address.

## Verified boundary

The exact candidate passed:

- 32 focused Phase 7 CUDA tests;
- 21 selected adversarial CUDA tests;
- 184 cumulative Phase 1–7 CUDA tests;
- explicit bounded-cache and alias-release probes; and
- independent exact-SHA validation and read-only audit.

Coverage includes all four modes, four-player best-response root selection,
three-player nonterminal folds, inactive chance identity, duplicate labels,
shared profiles, refill, family-pack and KKT batching invariance, 700-decision
finite log mass, 65,536-row alias construction, exact root/controller
compatibility, allocation/cache wiring, recovery, and GPU cleanup.

Phase 8 may add branched Rao--Blackwell sampling only by extending this engine;
it must not add a second branching, topology, target, role-selection, or
learner path.
