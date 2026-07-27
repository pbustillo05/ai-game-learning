# Storage-redesign migration ledger

Status: opened in Phase 1. A row closes only when the old definition and all
callers, tests, configuration, and stale documentation are deleted.

Phase 1 contract/inventory rows closed against candidate
`7e371b8daf9f978ad2e479edfda6ae87fdaadb6b`: twelve-mode vocabulary,
partitions/conversions, owner fields, replica rules, capacity vocabulary,
opaque hooks, old-module/root/config/persistence/artifact inventories, and
static firewall gates. All replacement and deletion rows for later phases
remain open.

Phase 2 replacement evidence passed against candidate
`5bd7d0ebb900ec4dddc6a951646d6e1e62622a3b`: immutable aligned schemas,
fixed monotone slabs, exact blocks, lane-local failure/retry, atomic
multi-destination publication, generated owners, immutable capacity admission,
exact byte reports, and last-consumer release. The corresponding old-definition
rows remain open until their legacy callers and definitions are deleted at
cutover; isolated replacement does not weaken that closure rule.

Phase 3 replacement evidence passed against candidate
`ec9ef6f725a8f986c4b2910b348b1c2a09b804ea`: compact CPU topology and
packets, one immutable exact movement/layer plan, two last-consumer work
arenas, one schema-generated compiled CUDA layer executor, sampled and RB
carriers, deterministic post-to-pre conversion, branched trace/cost/delta,
headless structural rows, and atomic target publication. Physical carries and
total-minus-child are deleted. Legacy topology and target definitions remain
open until their callers and owners are deleted at cutover.

Disposition meanings:

- `move`: representation-independent behavior relocates once to its final
  owner and the old definition is deleted in the same change;
- `rewrite`: required behavior is implemented from its contract under a new
  sole owner;
- `delete`: stale, duplicated, forbidden, or test-only production behavior and
  all retention evidence disappear.

No row means reuse in place, adaptation, or indefinite compatibility.

## Legacy generic core

| Old module / symbol chain | Disposition | New sole owner / phase | Authority and gate |
|---|---|---|---|
| `core.config`: all dataclasses, parser, validation, seed construction | rewrite | closed queued config + logical RNG, Phases 4–5 | Phase 1 inventory; dotted paths, twelve rows, determinism |
| `core.devices` | move | neutral accelerator validation, Phase 10 | accelerator-only repository contract |
| `core.game`: `EventBatch`, `RolloutBackend`, `GameBundle` | rewrite | opaque hooks + top-level composition, Phases 4–5 | mode contracts; firewall |
| `core.partition`: protocols, packs, byte work, partition specs/set | rewrite | `PartitionContract`, game hook, manifest, Phases 1–2/5 | exact partitions/conversions; no game labels in core |
| `core.execution`: accountant, admission, occurrence arena, work queue, stable partition | rewrite | monotone slabs + semantic FIFOs, Phases 2/4 | atomic publication, stable FIFO, exact bytes |
| `core.execution`: draw purposes and logical RNG functions | rewrite | schedule-independent queued RNG, Phase 4 | subdivision/stream/lane invariance |
| `core.storage`: tensor schemas, paged/recyclable arenas and columns | delete | `SoASchema` + `MonotoneSlab`, Phase 2 | no paging/recycling/wrapping |
| `core.storage`: seal DTOs, rollout/training partitions/stores | rewrite | `ExactBlock` and generated owners, Phase 2 | one owner/lifetime; exact release |
| `core.storage`: occurrence/topology/leaf/target schema builders | rewrite | `ModeSchema`-generated owners, Phase 2 | twelve exact allocations |
| `core.tree.PagedRolloutTopology` | rewrite | compact CPU `Topology`, Phase 3 | no parallel topology owner |
| `core.topology`: sequential CSR and branched spans | rewrite | `Topology`/`LayerPlan`, Phase 3 | one structural representation |
| `core.rollout`: sequential sampling and public dispatch | rewrite | semantic FIFOs and one state machine, Phases 4–5 | one generated rollout path |
| `core.branching`: tree sampler and helpers | rewrite | controller + common sample/commit, Phase 7 | root/refill/duplicate/RNG laws |
| `core.branching`: diagnostic cutoff objects | delete | experiment evidence only, Phase 10 | no diagnostic production mode |
| `core.adaptive_branching`: formulas and fixed/KKT/safeguard controllers | rewrite | one controller interface, Phase 7 | math references; recovery gates |
| `core.completion`: completion DTOs, reach view, protocol | rewrite | exact leaf/work views + poker hook, Phases 3/5/6 | evaluator v2 boundary |
| `core.targets`: all three target builders and frontier planner | rewrite | one layered executor, Phase 3 | twelve fixture rows; no Python node loop |
| `core.targets.finalize_training_storage` | delete | atomic target publication, Phase 3 | no post-hoc compatibility seal |
| `core.learner`: validation, physical planner/subspans/index buffers | rewrite | aligned/ragged sampler and one learner operation, Phase 5 | exact statistical presentations |
| `core.learner`: loss plumbing and distributed update | rewrite | one forward/loss/immediate-backward lane op, Phases 5/9 | gradient/reference parity |
| `core.losses` | rewrite | schema-driven loss algebra, Phase 5 | preserve configured loss surface |
| `core.optimization` | rewrite | dense FP32 optimizer transition, Phase 5 | Adam/AdamW/scaler/clipping oracle |
| `core.memory_control`: ledgers/models/forecasters/controllers | rewrite | immutable manifest/admission/failure, Phase 2 | no runtime capacity growth |
| `core.observability.ExecutionMetrics` | rewrite | generated diagnostics schema, Phases 4–10 | required global/rank/lane metrics |
| `core.distributed_failure` | rewrite | rank/lane transactional agreement, Phases 2/9 | minimum-address and common retry |
| `core.preemption` | rewrite | iteration owner, Phase 5 | signal handler notice only |
| `core.checkpointing` | rewrite | strict queued persistence + offline converter, Phase 5 | atomic/resume/drift/source-hash |
| `core.training`: orchestration, memory/seal/target/iteration helpers | rewrite | one queued iteration composer, Phases 4–10 | one path; last-consumer audit |
| `core.br`: probe seat and paired-stat formulas | move | response consumer/neutral statistics, Phase 6 | `core_math.tex` probe contract |
| `core.utility` | move | neutral target/converter utility contract, Phase 5 | raw/stack units |
| `core.replica_learner`, `core.replica_optimizer`, `core.replica_update`, `core.replica_replay` | rewrite | lane-local learner + dense optimizer, Phase 9 | no vmap learner/replay adapter |
| `core.replica_training`, `core.replica_checkpoint` | rewrite | ordinary ownership/scheduling + sole persistence, Phase 9 | ordinary/replica identity |

## Poker implementation

| Old module / symbol chain | Disposition | New sole owner / phase | Gate |
|---|---|---|---|
| `poker.rules`: game/action/strata rules | rewrite | queued poker hooks, Phase 5 | every legal `(P,S,R)` boundary |
| `poker.cards`: geometry/category and test-oriented card helpers | rewrite/delete | minimal hook geometry; reference-only helpers to tests, Phase 5 | evaluator combo/deck identity |
| `poker.tensor_state`, `poker.model_inputs` | rewrite | direct hot/retained fields + model input hook, Phase 5 | history and lifetime |
| `poker.partitions`, `poker.policy_planning` | rewrite | immutable partition contract + stable prefix plan, Phases 1/5 | no street family |
| `poker.rollout_backend` and capture stores | rewrite | pre/post hooks and sample/transition, Phases 5–8 | no legacy representation crosses |
| `poker.folded`, `poker.partition_cache` | rewrite | local `FoldedRanges` and folded-summary hook, Phase 6 | refresh/release/folded math |
| `poker.mc`: completion/showdown facades | rewrite | exact evaluator-v2 leaf hook, Phases 5–6 | version/shapes/device support |
| `poker.mc.PokerGPURejectionSampler` | rewrite | local queued posterior sampler, Phase 8 | conditioned law/exhaustion |
| `poker.adapter`, `poker.bundle` | rewrite | canonical policy operator + composition, Phase 5 | same operator rollout/learner |
| `poker.replica` | rewrite | replica-vmapped inference hook only, Phase 9 | lane parity; no learner vmap |
| `poker.model.preflop`, `postflop`, `primitives`, `incidence`, `heads`, `capacity` | rewrite | authoritative two-network model operation, Phase 5 | forward/backward/model authority |
| model diagnostic/ablation-only paths | delete | tests or archive at consumer migration | no production object retained by tests |

## Products and composition roots

| Old root | Disposition | New owner / phase | Gate |
|---|---|---|---|
| `pbg.config`, `pbg.runtime`, training CLIs | rewrite | queued composition, Phase 10 | no dual-engine reachability |
| `pbg.benchmark` and benchmark CLIs | rewrite | matched-work queued gates, Phase 10 | frozen tolerances/identity |
| `pbg.eval.br_probe`, BR CLI | rewrite | ordinary response consumer, Phase 6 | balanced paired lower-bound report |
| `pbg.inference`, artifact/export/benchmark CLIs | rewrite | queued model-only inference, Phase 10 | CUDA/MPS contract |
| RunPod/replica benchmark CLIs | rewrite | Phase 9 experiment consumers | H100 hunk classification/parity |
| package `__init__` modules | rewrite/delete exports at cutover | final public surface, Phase 10 | reachability and import audit |

The accepted root list is frozen in
`queued_phase1_inventory_20260726.md`. Static tests fail if a production root
reaches both engines.

## Test-only retention evidence

Objects identified by the reachability/compactification audits as test-only
have `delete`, not preservation, disposition: legacy alias target builders,
diagnostic tree/alias helpers, test-only categorical implementations,
deprecated checkpoint migrations, unused completion row DTOs, compatibility
policy forwards, and CPU product scaffolding outside supported inference.
Useful representation-independent equations move into direct-reference tests
only when their consumer phase lands.

Dynamic reachability that must be preserved includes poker transition and
probe-fold callbacks, checkpoint/preemption paths, evaluator methods,
distributed failure agreement, and console entry points. Broad protocol
reachability does not authorize retaining an old owner.

## Preservation ledger

Every row of `features_to_preserve.md` maps to a new owner and regression family
in `queued_phase1_inventory_20260726.md`. Those rows remain open until their
consumer phase passes. The default is rewrite/preserve; deletion or semantic
change still requires explicit user approval.

## Artifact/converter ledger

| Artifact family | Disposition | Consumer phase | Closure evidence |
|---|---|---:|---|
| selected current sampled sequential checkpoints | offline convert | 5 | source/result hashes, exact resume |
| selected current sequential-RB checkpoints | offline convert | 6 | source/result hashes, exact state |
| selected current branched artifacts | offline convert | 8 | mode/controller conversion |
| selected ordinary/distributed/replica lanes | offline convert | 9 | lane/rank identity |
| inference artifacts | offline convert | 10 | model/config/content identity |
| tracked range-database pairs | versioned dataset conversion | first consumer | source hash and semantic validation |
| configs/run/benchmark JSON/YAML | evidence only | cumulative gates | read-only identity |
| unknown/unselected legacy artifacts | delete | 10 | no runtime compatibility loader |

## Saved H100 branch

The branch is not merged or cherry-picked. Replica batching/selection is a
Phase 9 rewrite requirement; range files and hardware profiles are calibration
evidence; fail-fast parity and progress hunks are observability evidence; the
pre-refactor snapshot is baseline evidence. Every later-used hunk receives its
own row before implementation.

## Validation columns

Rows close only with exact candidate SHA, CUDA commands/results, failure
injection, determinism, memory/high-water accounting, matched-work timing, old
definition/caller deletion, and audit result. Phase 1 opens the rows and proves
the closed schema/firewall; it does not prematurely close later behavior.

Phase 2 candidate `5bd7d0ebb900ec4dddc6a951646d6e1e62622a3b`
passed 25 queued cumulative CUDA tests and 66 legacy storage/memory regression
tests on explicitly selected GPU 0. Evidence covers all twelve generated owner
sets, exact analytical/allocated/observed bytes, allocation and publication
failure injection, minimum-address determinism, whole-lane retry isolation,
immutable-manifest rejection, and release exactly at every last consumer. The
read-only audit found no parallel queued allocator, capacity authority, failure
owner, compatibility path, forbidden import, paging, recycling, growth, or
wraparound. See `queued_phase2_report_20260726.md` and
`../queued/storage_and_lifetimes.md`.

Phase 3 candidate `ec9ef6f725a8f986c4b2910b348b1c2a09b804ea`
passed 45 focused CUDA tests, 70 cumulative queued tests, 66 legacy
topology/target/memory regressions, and four targeted static gates on
explicitly selected GPU 1. Evidence covers all twelve mode rows, one kernel per
layer without interlayer synchronization, exact topology/child/leaf/target
bijections and sequential row identity, maximum degree/depth, deterministic
H-to-K, sampled/RB/trace/cost/delta recurrences, exact memory and traffic,
schema/bounds/failure injection, and replica-local atomic publication. Matched
comparisons selected direct sibling summation for correctness and
last-consumer retention for time and memory; both losing implementations and
all carry surfaces are absent. The independent read-only audit found one
topology owner, one plan, one target executor, closed completion and conversion
contracts, and no parallel path or forbidden import. See
`queued_phase3_report_20260726.md` and
`../queued/topology_and_targets.md`.
