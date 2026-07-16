# File-by-file inventory

## Root and active documentation

| File | Final audit result |
|---|---|
| `AGENTS.md` | authoritative game/tree/GPU constraints; all repaired code follows them |
| `CONTRACTS.md` | game domain, finite update, hardware profile, residency, checkpoint, and validation claims now match production |
| `HARDWARE.md` | local/Quest procedure and conservative GPU hygiene remain valid |
| `IMPLEMENTATION_NOTES.md` | historical ledger only; current readiness lives here |
| `NOTATION.md` | dynamic private width, reference KL, and structural-only moment-head status are accurate |
| `README.md` | production surface is accurate and explicitly excludes strategy-quality claims |
| `docs/README.md` | accurate active-document index and declared mixed-row control boundary |
| `docs/architecture.md` | scheduler, P/M reducer, global mass, binary-lift history, cache, and device movement match code |
| `docs/core_algorithms.md` | accurate three-mode fixed estimator subset |
| `docs/file_reference.md` | package ownership remains accurate |
| `docs/poker.md` | dynamic geometry, exact history, folded cache, model, and evaluator boundaries match code |
| `docs/training_and_evaluation.md` | CLI, profiles, checkpoint/resume, benchmark, and arena behavior match production |
| older readiness audit | explicitly superseded evidence snapshot |
| `docs/todo/BR_probe.md` | correctly records excluded M-05 work |

## Theory

| File/group | Final audit result |
|---|---|
| `docs/theory/core_math.tex` | configurable P/S/R poker example; implemented reference KL and deferred public-moment loss/control are distinguished |
| `docs/theory/core_math_audit.md` | research-proof qualifications remain applicable |
| `docs/theory/MHA_model.md` | parameterized by $D=4R$, $H=\binom{D}{2}$, and $K=R^2$; optional heads accurately bounded |
| `docs/theory/folder_impact.md` | evaluator-owned background mathematics; local code consumes supplied fields only |
| rollout/distributed pseudocode | one-root scheduler, mass scaling, SUM reduction, and finite gate trace to code |
| `docs/references/` | vendored provenance, not active implementation requirements |

## Composition, CLI, benchmark, and evaluation

| Module | Final audit result |
|---|---|
| `pbg/config.py`, `pbg/runtime.py` | strict NLHE composition; CUDA library launch requires a profile |
| `pbg/cli/train.py` | profile-aware production entry point and existing-output protection |
| `pbg/cli/benchmark.py`, `pbg/benchmark.py` | benchmark the sole implementation; no strategy-quality implication |
| `pbg/eval/arena.py` | read-only duplicate-deal comparison, explicitly not approximate BR |

## Generic core

| Module | Final audit result |
|---|---|
| `pbg/core/game.py` | clean generic protocols and owner codes |
| `pbg/core/config.py` | recursive type/value/cross-field validation with dotted errors |
| `pbg/core/tree.py` | duplicate-preserving topology, path weights, exact reach replay, and decision caches |
| `pbg/core/execution.py` | bounded arena/queues, logical RNG, and transactional admission |
| `pbg/core/rollout.py` | shared sequential family scheduler and sampled-profile law |
| `pbg/core/branching.py` | one-root randomized-rounded duplicate-preserving branching |
| `pbg/core/normal.py` | sampled settlement and player-clock lambda return |
| `pbg/core/completion.py` | typed fp32 posterior/moment streaming contract |
| `pbg/core/values.py` | streamed P/M frontier and owner-clock bootstrap |
| `pbg/core/losses.py` | `ppo`/`rb_ppo`, corrected entropy horizon, values, entropy, and reference KL |
| `pbg/core/minibatch.py` | aligned logical normal/RB gather and masks |
| `pbg/core/distributed.py` | weighted SUM helpers |
| `pbg/core/optimization.py` | common gradient reduction, finite decision, clip/step/scaler lifecycle |
| `pbg/core/checkpointing.py` | atomic payload, per-rank RNG, identity, and metrics-prefix lifecycle |
| `pbg/core/training.py` | orchestration; rollout and learner precision now agree; extracted lifecycles reduce responsibility |
| `pbg/core/observability.py` | complete transfer/sync/evaluator/model counters |
| `pbg/core/logging_util.py` | atomic/simple logging support |

## Poker rules, evaluator, runtime, and model

| Module | Final audit result |
|---|---|
| `cards.py` | variable `DeckGeometry`, suffix-deck mapping, combo/class/category tables |
| `engine.py` | independent public NLHE oracle and strict game validation |
| `tensor_state.py` | preallocated public state plus binary-lifting history ancestors |
| `ranges.py` | CPU/reference likelihood support |
| `adapter.py` | legal factored policy probability, entropy, KL, and bundle dispatch |
| `mc.py` | strict read-only sibling sampling/completion/showdown facade for every R |
| `folded.py` | typed masked sibling partition/folder boundary |
| `bundle.py` | dynamic construction, lossless history, hardware/model preflight, memory plan |
| `model_inputs.py` | bounded batched device history materialization |
| `partition_cache.py` | fixed memory-planned device-indexed folded-summary cache |
| `policy_planning.py` | host-declared bounded physical policy packing and one counted control block |
| `rollout_backend.py` | dynamic public tensor navigation and family routing; no compact-deck blocker or Python cache |
| `model/primitives.py` | LN/FFN/reach-biased MHA with bounded attention execution |
| `model/incidence.py` | geometry-derived exact incidence residual |
| `model/heads.py` | factored type/raise/value and optional structural moment outputs |
| `model/preflop.py` | $R^2$ class-native trunk with lossless history width |
| `model/postflop.py` | geometry-derived compact live combos and evaluator folder features |

## Organization conclusion

The active package boundary is clean: generic core does not import poker,
poker does not duplicate evaluator math, and composition is explicit. M-04's
targeted extraction is complete; a broad rewrite is neither required nor
desirable. No active retired game, alternate executor, local hand evaluator, or
unreachable production selector remains.
