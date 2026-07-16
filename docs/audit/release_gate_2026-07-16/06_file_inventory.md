# File-by-file inventory

## Root Markdown

| File | Role | Audit result |
|---|---|---|
| `AGENTS.md` | repository scope, tree convention, game domain, GPU hygiene | authoritative operational constraints followed; its max-history formula exposes B-02; advertised R domain conflicts with B-01 |
| `CONTRACTS.md` | active implementation contract | tree, estimator, ownership, checkpoint, and mode contracts mostly match; config strictness, deck domain, finite-step, profile, and sync claims do not |
| `HARDWARE.md` | local/Quest execution procedure and profiles | profile separation and GPU procedure match; Python API can bypass required CUDA profile |
| `IMPLEMENTATION_NOTES.md` | historical implementation ledger | clearly labeled historical; not used as current acceptance; no active code imported from recorded retired paths |
| `NOTATION.md` | implementation-facing estimator/loss notation | fixed estimator and `rb_ppo` mapping are accurate; moment-head and reference-KL status has stale wording |
| `README.md` | production overview and commands | architecture and commands are accurate for full deck; 4--13 ranks and strict-production claims overstate implementation |

## Project documentation

| File/group | Audit result |
|---|---|
| `docs/README.md` | accurate index; should point readers to this newer no-go audit |
| `docs/architecture.md` | accurately describes one scheduler, P/M reducer, global mass, and package ownership; finite-update and no-undeclared-sync claims are not met |
| `docs/core_algorithms.md` | accurate description of the active three modes and fixed estimator subset |
| `docs/file_reference.md` | package/file ownership map matches reachable source; no active retired game omitted |
| `docs/poker.md` | rule/evaluator/folder/model description is accurate for R=13; compact-deck production support is false; history limit is undocumented |
| `docs/training_and_evaluation.md` | CLI, modes, checkpoints, benchmark, and arena descriptions are substantially accurate; relies on incomplete strict validation and does not disclose domain blockers |
| `docs/audit/README.md` and older audit files | useful cleanup/performance history, but previous production-accepted conclusion is superseded by this report |
| `docs/todo/BR_probe.md` | correctly identifies missing approximate-BR acceptance |
| `docs/todo/control_branching_size.md` | correctly identifies adaptive branch-size control as research, not shipped behavior |

## Theory documentation

| File/group | Audit result |
|---|---|
| `docs/theory/core_math.tex` | fixed public-game/RB/branching construction maps well to code; named-method/convergence limits remain as recorded in its audit; poker section fixes a 52-card 6-max example and contains stale reference-KL status |
| `docs/theory/core_math_audit.md` | candidly and correctly bounds the mathematical confidence; its moderate open findings remain unresolved |
| `docs/theory/MHA_model.md` | substantially implemented for the full 52-card deck, including folder statistics; optional heads are not backed by losses; product-level reduced-deck generalization is absent |
| `docs/theory/folder_impact.md` | mathematical background and evaluator ownership are respected; local code consumes supplied fields instead of reimplementing private partition laws |
| `docs/theory/pseudocode/family_scheduled_rollout.tex` | scheduler, family, logical RNG, actual-cost, and one-root semantics trace to code |
| `docs/theory/pseudocode/distributed_update.tex` | mass scaling and gradient SUM trace to code; explicit finite-decision step is violated by B-03 |
| `docs/theory/preamble/` | document build support only; no runtime contract issue |
| `docs/references/` | 138 vendored third-party provenance files; not active implementation requirements |

## `pbg` package

### Composition, CLI, benchmark, and evaluation

| Module | Audit result |
|---|---|
| `pbg/config.py` | thin public export; correct |
| `pbg/runtime.py` | direct NLHE composition confirms sole active game; library CUDA path does not require a profile |
| `pbg/cli/train.py` | correct CLI and default local profile; existing-output protection works |
| `pbg/cli/benchmark.py` | production benchmark entry point; mode constraints match core |
| `pbg/benchmark.py` | benchmarks the same implementation and supports checkpoint identity/performance comparison; not a strategic evaluator |
| `pbg/eval/arena.py` | read-only duplicate-deal CPU public navigation plus sibling showdown; explicitly not approximate BR |

### Generic core

| Module | Audit result |
|---|---|
| `pbg/core/game.py` | clean protocols/owner constants; no poker import |
| `pbg/core/config.py` | strict recursive schema and many cross-field checks; incomplete numeric/domain validation is B-05 |
| `pbg/core/tree.py` | duplicate-preserving host/device tree, path weights, replay; topology invariants well tested |
| `pbg/core/execution.py` | bounded arena/queues, logical RNG, transactional planning; complex but coherent |
| `pbg/core/rollout.py` | shared sequential family scheduler, reach/profile/event logic; conforms to sequential laws |
| `pbg/core/branching.py` | one-root randomized-rounded duplicate-preserving branch sampler; conforms to fixed beta math |
| `pbg/core/normal.py` | sampled settlement and player-clock lambda return; correct if lambda is valid |
| `pbg/core/completion.py` | typed fp32 posterior/moment validation and streaming contracts; correct |
| `pbg/core/values.py` | streamed P/M frontier, conditionalization, sibling baselines, owner bootstrap; faithful and well tested |
| `pbg/core/losses.py` | only `ppo`/`rb_ppo`; reductions match notation; entropy schedule bug B-04 |
| `pbg/core/minibatch.py` | aligned normal/RB logical gather and masks; correct |
| `pbg/core/distributed.py` | weighted SUM helpers; correct, though main training owns the flat reduction |
| `pbg/core/training.py` | integrates all modes, memory, optimizer, reference, checkpoint, DDP; finite gate B-03 and oversized responsibility M-04 |
| `pbg/core/observability.py` | useful complete counters; cannot detect uninstrumented host syncs in M-01 |
| `pbg/core/logging_util.py` | atomic/simple logging support; no issue found |

### Poker rules, evaluator, and runtime

| Module | Audit result |
|---|---|
| `pbg/games/poker/cards.py` | correct compact `DeckGeometry`, class/combo maps, and full-deck compatibility constants |
| `pbg/games/poker/engine.py` | independent public NLHE oracle and config validation; rule arithmetic tested against tensor backend |
| `pbg/games/poker/tensor_state.py` | preallocated public state/history structure; capacity larger than model history cap |
| `pbg/games/poker/ranges.py` | CPU/reference likelihood range state and compatibility; active support only |
| `pbg/games/poker/adapter.py` | legal factored policy probability, entropy, KL, event mapping, model bundle; no local private evaluator fallback |
| `pbg/games/poker/mc.py` | strict sibling sampling/completion/showdown facade; validates API/device/shape/numerics |
| `pbg/games/poker/folded.py` | typed sibling partition/folder DTO and validation; ownership boundary is respected |
| `pbg/games/poker/bundle.py` | model/evaluator/backend construction and memory preflight; does not reject deck-width/history mismatches early |
| `pbg/games/poker/rollout_backend.py` | full public tensor navigation and model routing; explicit R=13 blocker, hot-path history/folder host work, oversized module |

### Poker model

| Module | Audit result |
|---|---|
| `model/primitives.py` | LN, FFN, reach-biased temperature MHA with flex/chunked-dense execution; focused CUDA test passes |
| `model/incidence.py` | exact compact maps and incidence residual; numerical identities tested |
| `model/heads.py` | type, DLogisticTick mixture/all-in, value, optional moments; legal normalization owned by adapter |
| `model/preflop.py` | class-native pre-LN model over all seated ranges; fixed 169/52 geometry and 256 history default |
| `model/postflop.py` | compact live-combo model, evaluator folder features, folded-seat readout, fixed 1326/52 geometry and same history cap |

## Organization conclusion

The high-level organization is good and should be preserved: generic math/core,
poker implementation, evaluator boundary, evaluation, and composition are cleanly
separated. The most important organization improvement is targeted extraction
from `training.py` and `rollout_backend.py` after the release fixes, not a broad
rewrite. There is no obvious dead active game, alternate executor, or local hand
evaluator to remove.
