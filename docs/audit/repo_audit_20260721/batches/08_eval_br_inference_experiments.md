# Evaluation, BR, inference, benchmark, and experiment audit

## Scope and conclusion

This batch reviewed the evaluation and deployment-adjacent surfaces requested by
the coordinator:

- `pbg/eval/*.py`;
- `pbg/inference.py` and `pbg/inference_artifact.py`;
- `pbg/benchmark.py`;
- `pbg/cli/br_eval.py`, `export_inference.py`,
  `benchmark_inference.py`, `rao_blackwell_variance.py`, and
  `branched_rao_blackwell_variance.py`;
- their directly corresponding tests, tools, configs, package entry points, and
  evaluation documentation.

The approximate BR probe is a genuine reachable fifth *semantic* workflow: one
trainable model controls one selected seat per sequential-RB trajectory, all
other seats use a hash-bound frozen checkpoint, only selected-seat decisions
are retained for learning, and a selected-seat fold stops the trajectory because
that player's payoff is then public and fixed. It is not a separately enumerated
training mode, however, and it cannot be certified as a best-response algorithm
against `core_math.tex`: `MATH-BR-001` records that the authority defines no BR
sampling law, optimization objective, approximation class, stopping rule, or
exploitability estimator. The implementation and its documentation correctly
use the weaker terms "approximate best-response probe" and "lower bound" in
several places, but the requested five-mode claim needs a new authoritative BR
contract before it can be certified.

The other major conclusion is deletion/archival rather than repair. The
independent arena is CPU-only and full-deck-specific; the 938-line inference
session is expressly CPU/Apple-MPS-only and maintains a second state/storage
implementation; and both variance packages are checkpoint-specific research
experiments that manually reconstruct targets outside the sole production
storage path. None is part of the five NVIDIA production workflows. The training
benchmark is legitimate production support, but currently covers only three
labels and its comparison gate can silently omit metrics or semantic modes.

## Findings

### EVAL-001 — BR is an approximate shared-policy probe with no authoritative BR contract

- Severity: blocker
- Confidence: confirmed
- Contract: MATH-BR-001; MATH-BR-002; five-mode audit requirement
- Modes affected: BR
- Evidence:
  - `docs/audit/repo_audit_20260721/batches/00_math_and_modes_contract.md:304-317`
    establishes that `core_math.tex` contains no BR definition, loss, sampling
    law, stopping rule, or exploitability estimator.
  - `pbg/core/config.py:122-128`, `ProbeConfig`, calls the workflow a shared
    approximate best response; `pbg/core/config.py:282-305` represents it as
    `training.mode == "rao_blackwell"` plus a non-null `probe`, rather than an
    independently enumerated fifth mode.
  - `pbg/core/training.py:2600-2635` hash-binds the opponent checkpoint and
    checks identical `(P,S,R)`; `:2686-2707` constructs a frozen opponent and
    optionally initializes the probe from it.
  - `pbg/core/rollout.py:369-389` selects one probe seat per trajectory;
    `:780-809` sends that seat through the trainable model and other seats
    through `opponent_model`; `:953-957` emits `ProbeRollout`.
  - `pbg/core/training.py:1510-1613` seals the probe through the ordinary
    `sequential_rb` representation. `pbg/core/targets.py:248-312` then builds
    posterior/return rows only for owners which have retained decisions.
  - `pbg/games/poker/rollout_backend.py:384-393` detects when the selected probe
    seat folds; `pbg/core/rollout.py:860-867` changes that stopped leaf to
    `PROBE_FOLD`; and `pbg/games/poker/rollout_backend.py:2590-2610` uses the
    selected player's now-public committed-chip payoff.
  - There is no convergence or exploitability stopping condition. The run ends
    after the generic configured iteration budget, and the checkpoint records
    only logical work (`pbg/core/training.py:2742-2747`).
- Current behavior: the implementation optimizes one shared parameterization
  across randomly selected seats against a frozen opponent, using sequential-RB
  PPO and an externally chosen training budget. That is a defensible approximate
  BR lower-bound probe, but it is not evidence of exact BR and its approximation
  target cannot be checked against the designated authority.
- Required behavior: the fifth production mode needs an authoritative statement
  of whether it is exact BR, approximate BR, or a diagnostic probe; the selected
  seat law; whether one shared seat-conditioned policy or independent per-player
  responses define the approximation class; the loss; fold stopping semantics;
  training/stopping budget; and the evaluation/NashConv estimator.
- Consequence: configuration reachability proves that a fifth workflow exists,
  but neither mathematical fidelity nor the meaning of its reported
  "approximate_nashconv" can be certified.
- Recommended disposition: add a BR section to an authoritative theory document
  before calling this one of five certified modes. Preserve the current
  "approximate probe/lower bound" language unless the stronger algorithm is
  actually specified and implemented.
- Verification needed: equation-level BR tests for seat sampling, frozen
  opponents, selected-owner-only loss rows, fold stopping, response-class
  semantics, and the aggregate per-seat estimator.

### EVAL-002 — BR has no NVIDIA acceptance test and silently permits CPU execution

- Severity: high
- Confidence: confirmed
- Contract: MATH-BR-001; repository NVIDIA-only requirement; MODE-001; MODE-007
- Modes affected: BR
- Evidence:
  - `pbg/eval/br_probe.py:129-140` defaults to CUDA only when available and
    otherwise silently selects CPU.
  - `pbg/cli/br_eval.py:11-34` exposes no explicit device argument or CUDA-only
    guard.
  - `tests/test_br_probe.py:37-73` explicitly configures the CPU completion
    backend. Its rollout/backward test (`:92-132`), paired evaluation test
    (`:135-169`), completion test (`:172-213`), and train/resume test
    (`:216-257`) all execute on CPU.
  - `tests/test_refactor_gpu_acceptance.py:63-111` covers normal, sequential RB,
    fixed branched RB, and KKT branched RB, but not a probe.
- Current behavior: BR training/evaluation is reachable and the CPU integration
  suite passes, but no test establishes that opponent-model routing, selected
  completion, storage sealing, backward, checkpoint/resume, or paired evaluation
  works on an NVIDIA GPU.
- Required behavior: the sole production BR workflow must reject non-CUDA
  execution and have a tiny CUDA train/evaluate/resume gate.
- Consequence: the fifth mode has weaker evidence than the other four and may
  carry device-specific regressions unnoticed.
- Recommended disposition: make BR production entry points CUDA-explicit; retain
  only genuinely tiny CPU unit tests for pure helpers, not an alternate
  end-to-end implementation.
- Verification needed: one idle-GPU smoke with a frozen target, at least one
  optimizer step, a hash-bound resume, and paired evaluation, followed by an
  allocator/process-idle check.

### EVAL-003 — BR sealing computes and stages all-player completion although only the selected owner is consumed

- Severity: high
- Confidence: confirmed
- Contract: STORE-014; STORE-018; STORE-019; STORE-021; production minimal-payload requirement
- Modes affected: BR
- Evidence:
  - Probe rollout retains only selected-seat decision rows
    (`pbg/core/rollout.py:780-809`, `:953-957`; asserted by
    `tests/test_br_probe.py:92-110`).
  - `_seal_sequential_rollout_storage` notices `ProbeRollout` only to unpack it
    (`pbg/core/training.py:1510-1518`); it does not record selected seats in the
    final leaf schema or completion request.
  - The closure at `pbg/core/training.py:1692-1722` always calls
    `complete_sealed` and declares dense `[leaf,P,H]` outputs/bytes.
  - `pbg/games/poker/bundle.py:198-292`, `complete_sealed`, allocates dense
    `p_hat` and `v_hat` for every player and asks the evaluator for all-player
    completion at pre-terminal leaves.
  - `pbg/core/targets.py:248-312` proves the downstream consumer only copies the
    player owners that occur in the trajectory's retained decisions; for a
    probe, that is the one selected seat.
  - The older `complete_selected_keyed` method at
    `pbg/games/poker/bundle.py:294-365` is no longer called by production and is
    itself not a compact implementation: it allocates dense `[B,P,H]` zeros and
    calls `_complete_impl` for all players before copying one selected row.
- Current behavior: BR avoids retaining opponent decisions, but completion
  compute, transient output, and declared output bytes still scale with all
  players. A test-only pre-seal helper gives the appearance of selected
  completion without reducing the expensive ordinary completion.
- Required behavior: immutable BR leaf schema should carry the selected owner,
  the sibling evaluator boundary should project only the required posterior/value
  rows if its API supports this, and the final posterior pool should be exactly
  one player row per trajectory. If the evaluator necessarily computes all
  players internally, at minimum project immediately and declare that workspace
  honestly.
- Consequence: multiway BR wastes completion memory/bandwidth and contradicts
  last-consumer/minimal-storage discipline. `complete_selected_keyed` is stale
  test-only code, not evidence that production is optimized.
- Recommended disposition: design one sealed selected-player completion path,
  migrate BR and its tests to it, then delete `complete_selected_keyed` and its
  mutable-tree interface.
- Verification needed: exact output-shape/byte tests, equality to the current
  selected row on a tiny game, and GPU peak/throughput measurements for
  $P=2$ and $P=6$.

### EVAL-004 — BR report labels work and pairing more strongly than it records

- Severity: medium
- Confidence: strong
- Contract: MATH-BR-001; audit evidence integrity
- Modes affected: BR evaluation
- Evidence:
  - `pbg/eval/br_probe.py:142-169` uses logical-address RNG with the same
    `(seed, iteration=0, trajectory_start_index)` for mixed and baseline
    rollouts. This does pair private profiles and matching public-history draws;
    after policies choose different histories the path addresses intentionally
    diverge.
  - `pbg/eval/br_probe.py:172-185` computes the sample mean/Student interval and
    reports `approximate_nashconv_bb_per_hand = players * mean`. The configured
    seat rows are balanced by `:121-128` and `:145-151`.
  - `pbg/eval/br_probe.py:186-195` nevertheless reports
    `evaluation_completion_samples = 2 * n_hands` unconditionally.
    `capture_probe_leaf` at `pbg/games/poker/rollout_backend.py:2601-2609` does
    not perform a showdown completion for a selected-seat cutoff and ordinary
    fold-terminal payoffs also need no showdown sample.
  - `BRProbeReport` records no probe hyperparameters, response iteration, model
    architecture/response class, completion backend, CUDA device, logical RNG
    version, or per-seat confidence interval (`pbg/eval/br_probe.py:29-49`).
- Current behavior: the common-random-number comparison is meaningful, but the
  completion counter is a nominal replay count rather than actual completion
  work, and the report lacks enough scientific identity to compare approximate
  responses or reproduce the claimed lower bound.
- Required behavior: distinguish physical replays, fold settlements, showdown
  completion draws, and configured evaluation samples; bind the report to the
  response configuration/work budget and runtime identity; define the aggregate
  NashConv approximation in the BR authority.
- Consequence: run-cost comparisons and approximate-exploitability comparisons
  can be misleading even when the reward estimate itself is correct.
- Recommended disposition: retain the paired logical-RNG design, rename or
  measure the work counters accurately, and expand report provenance after the
  BR contract is approved.
- Verification needed: a forced-fold case proving zero showdown completion work,
  a diverging-policy common-random-number test, and per-seat estimator tests for
  $P=2\ldots6$.

### EVAL-005 — The independent arena is a stale CPU-only, full-deck-specific duplicate evaluator

- Severity: high
- Confidence: confirmed
- Contract: repository NVIDIA-only requirement; NLHE $(P,S,R)$ contract; sole-production-path requirement
- Modes affected: offline checkpoint evaluation
- Evidence:
  - The module describes itself as CPU-driven (`pbg/eval/arena.py:1`), builds
    CPU models (`:52-63`), requires a CPU generator (`:264-272`), instantiates
    CPU sampling/evaluation (`:66-85`, `:150-174`, `:177-229`), and loops one
    replay at a time in Python (`:279-299`).
  - It hard-codes global full-deck `N_HANDS` in reach arrays
    (`pbg/eval/arena.py:18`, `:72`, `:197`) and uses global 52-card helpers in
    `_linear_ranges`/postflop compact size (`:88-118`), even though `_poker_config`
    accepts every configured `rank_count` (`:44-49`).
  - The only integration test forces `rank_count = 13`
    (`tests/test_p12_eval.py:87-102`); there is no compact-deck evaluation test.
  - `np_gen` is validated at `pbg/eval/arena.py:271-272` but never consumed.
  - Static reachability finds `arena` used only by `tests/test_p12_eval.py` and
    documentation; `br_probe.py` imports only its `_student_t_critical` helper.
- Current behavior: a second model/reach/action/showdown replay algorithm exists
  outside the current GPU rollout path, is prohibitively serial, and is not
  correct for supported compact decks $R<13$.
- Required behavior: retained production evaluation must run through the sole
  NVIDIA rollout/evaluator implementation for all legal $(P,S,R)$ settings.
- Consequence: this module violates both the platform restriction and the
  sole-production-path rule; it can disagree with the production state/reach
  implementation and silently evaluate the wrong hand domain.
- Recommended disposition: archive `pbg/eval/arena.py`,
  `tests/test_p12_eval.py`, and the corresponding duplicate-arena documentation.
  Move `_student_t_critical` to a small generic statistics helper if BR still
  needs it.
- Verification needed: replace any desired checkpoint-versus-checkpoint arena
  with a batched CUDA workflow and compact-deck coverage before calling it
  production.

### EVAL-006 — The inference package is an active CPU/MPS-only second implementation with unbounded retained state

- Severity: high
- Confidence: confirmed
- Contract: repository NVIDIA-only requirement; STORE-006; STORE-021; STORE-022; production-only requirement; MODE-002
- Modes affected: deployed policy inference, not one of the five production workflows
- Evidence:
  - `pbg/inference.py:1` explicitly limits the 938-line session to CPU or Apple
    MPS. `_resolve_device` selects only MPS/CPU and rejects CUDA
    (`pbg/inference.py:256-273`); fp16 is MPS-only (`:180-185`).
  - `pbg/cli/benchmark_inference.py:1`, `:44-50` exposes only `auto/cpu/mps`, and
    `pyproject.toml:18-21` installs both inference commands plus a macOS extra.
  - The session owns a second host `NLHEEngine`, per-hand dictionaries and
    log-reach tensors (`pbg/inference.py:189-235`, `:275-299`), reimplements
    action resolution and state advance (`:326-401`, `:811-928`), and rebuilds
    folded summaries through a separate cache (`:468-576`).
  - `_hands` only grows (`pbg/inference.py:221-222`, `:275-289`); there is no
    close/release API. `_partition_cache` retains cloned summaries for every
    `(hand, board, fold_version)` (`:223-225`, `:476-576`) and likewise has no
    eviction or last-consumer release.
  - Every uncached postflop summary transfers all hand-width reach weights back
    to CPU (`pbg/inference.py:491-514`) before sending the summary back to the
    model device (`:643-680`).
  - `Decision.seat_values_bb` and `FullStrategy.seat_values_bb`
    (`pbg/inference.py:63-86`) are dead output fields: `_evaluate_full_device`
    initializes `seat_values = None` at `:589-594` and never assigns another
    value before returning at `:692-698`.
- Current behavior: this is a separate non-NVIDIA serving product with its own
  state lifetime, scheduling modes, batching heuristics, CPU summary path, and
  tests. Long-lived sessions leak logical hand and summary state until the
  entire session is discarded.
- Required behavior: code retained in this repository should serve the five
  core NVIDIA workflows. A deployment API is out of scope unless explicitly
  approved, CUDA-native, and given bounded ownership/lifetime contracts.
- Consequence: more than one thousand production/package lines preserve CPU/MPS
  support, duplicate game-state semantics, and add unbounded memory behavior.
- Recommended disposition: archive `pbg/inference.py`,
  `pbg/inference_artifact.py`, both inference CLIs, their installed entry points,
  macOS extra, and `tests/test_inference_api.py`. If CUDA deployment is later
  required, design it on top of the production tensor-native state/model input
  path rather than migrating this second implementation wholesale.
- Verification needed: before removal, confirm no external deployment consumer
  was intentionally authorized. If retained, require explicit hand/cache release,
  bounded cache accounting, CUDA acceptance, and parity with production inputs.

### EVAL-007 — “Model-only” inference artifacts retain unused training configuration and legacy migration coupling

- Severity: medium
- Confidence: confirmed
- Contract: STORE-014; sole-production-path/no-legacy requirement
- Modes affected: inference export
- Evidence:
  - `export_inference_artifact` describes a model/scientific-only artifact at
    `pbg/inference_artifact.py:105-108` but copies the complete checkpoint
    `config` at `:114-123`.
  - A real scientific checkpoint configuration contains rollout, completion,
    loss, training, and possibly probe identity in addition to the only sections
    consumed by inference. `PokerInferenceSession.__init__` reads only `game`
    and the model builder inputs (`pbg/inference.py:189-219`).
  - `load_inference_source` uses private checkpoint migration machinery
    (`pbg/inference_artifact.py:13-16`, `:91-102`, `:110-113`), keeping inference
    coupled to retired training checkpoint schemas.
  - `_validate_artifact` checks only digest string length, not hexadecimal
    content (`pbg/inference_artifact.py:58-60`).
  - `tests/test_inference_api.py:20-36` uses an artificial config containing only
    game/model, so `:57-90` does not detect the extra retained sections from a
    production checkpoint.
- Current behavior: exported artifacts are optimizer-free but not minimal
  inference artifacts and preserve compatibility/migration surface that the
  inference consumer does not need.
- Required behavior: if inference remains, export an exact immutable schema of
  only consumed game/model/version fields and perform migration once at export,
  not every deployment load.
- Consequence: stale config keys and probe/training metadata remain deployment
  compatibility commitments and enlarge the deletion surface.
- Recommended disposition: archive with the inference subsystem. If inference
  is explicitly retained, tighten the schema and tests rather than preserving
  full training configuration.
- Verification needed: export from a real current checkpoint, assert exact keys,
  and reject malformed/nonhexadecimal source hashes.

### EVAL-008 — Two checkpoint-specific variance experiments duplicate obsolete rollout, completion, and target algorithms

- Severity: high
- Confidence: confirmed
- Contract: STORE-001; STORE-007; STORE-014 through STORE-022; production-only/no-stale-experiment requirement; MODE-010
- Modes affected: none; sequential-RB and fixed-branched-RB research experiments
- Evidence:
  - `pbg/eval/rao_blackwell_variance.py:1-7` takes its mathematics from the TODO
    `docs/todo/rao_blackwell_variance_experiment.md`, not `core_math.tex`.
    `run_variance_job` is hard-coded to HU, 32 bb, full deck
    (`:377-400`) and manually retains raw tree/reach/policy, reconstructs leaf
    inputs, JVP rows, dense functional coefficients, and completion moments
    (`:442-617`) instead of consuming sealed production training storage.
  - It calls the sibling evaluator through private
    `bundle.completion_backend._bound_evaluator()` at
    `pbg/eval/rao_blackwell_variance.py:502-506`, bypassing the declared opaque
    completion service contract.
  - `pbg/eval/branched_rao_blackwell_variance.py:1-6`, `:158-179`, and
    `:437-524` lock the experiment to checkpoint 1110, beta values
    `[2,2.25,2.5]`, legacy optional branching heads, and private evaluator access.
  - Its runner calls raw `sample_branched_tree`, materializes whole topology on
    CPU, manually reduces node moments, reconstructs leave-one-out target
    derivatives, and propagates dense leaf coefficients
    (`pbg/eval/branched_rao_blackwell_variance.py:259-430`, `:533-733`). This is
    a parallel target implementation, not the sole `RolloutStorage`/target path.
  - The branched experiment intentionally supplies evaluation-only global
    `BranchingDiagnosticLimits` (`:480-483`, `:546-567`) that change geometry and
    are not the authoritative queue-pressure safeguard.
  - Active custom configs are tied to historical run directories and checkpoint
    1110 (`configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml:1-55` and
    `nlhe_hu_32bb_branched_rao_blackwell_variance.yaml:1-60`).
- Current behavior: these modules are reproducibility machinery for specific
  July 2026 experiments. They preserve independent formulas and old mutable
  representations that can drift from production after every refactor.
- Required behavior: only the five production implementations remain active;
  historical experiments and their custom mathematics belong with archived run
  evidence.
- Consequence: roughly 2,300 Python lines plus configs/launchers look like active
  implementations, violate sole-path auditability, and can produce findings
  about a non-production estimator.
- Recommended disposition: move both `pbg/eval/*variance.py` modules, both
  variance CLIs, their three experiment YAMLs, tests, and tool launchers into one
  dated archive alongside their `runs/` evidence. Future performance/variance
  probes should consume sealed production outputs and avoid reimplementing
  targets.
- Verification needed: preserve manifest/run reproducibility paths in the
  archive and prove no installed production entry point imports these modules.

### EVAL-009 — Active variance CLIs include CPU fallback, advertised dead phases, and an unused loader

- Severity: medium
- Confidence: confirmed
- Contract: repository NVIDIA-only requirement; production-only/no-unused-object requirement
- Modes affected: none; experiment control plane
- Evidence:
  - Both experiment device resolvers fall back to CPU
    (`pbg/cli/rao_blackwell_variance.py:35-50` and
    `pbg/cli/branched_rao_blackwell_variance.py:30-37`).
  - Sequential variance advertises seven phase choices at
    `pbg/cli/rao_blackwell_variance.py:307-317`, but only `smoke`, `variance`, and
    `analyze` execute; all other choices reach `_unimplemented_phase` at
    `:178-182` and `:332-347`.
  - `_load_geometry` at
    `pbg/cli/branched_rao_blackwell_variance.py:51-60` has no repository caller.
  - `tools/run-rao-blackwell-variance` and
    `tools/run-branched-rao-blackwell-variance` are the only launcher surfaces;
    neither experiment is one of the installed five-mode production commands.
- Current behavior: incomplete and unused research control flow remains under
  the active package and explicitly supports CPU.
- Required behavior: active CLI choices should be implemented production
  operations; historical partial workflows should be archived, not preserved
  as callable stubs.
- Consequence: users can select advertised phases that deterministically fail,
  and static reachability is polluted by dead/experimental objects.
- Recommended disposition: archive the complete experiment bundles rather than
  individually repairing these helpers.
- Verification needed: after archival, package import and console-script tests
  should find no variance experiment surface under `pbg`.

### EVAL-010 — The performance gate covers only three labels and can silently omit modes or metrics

- Severity: high
- Confidence: confirmed
- Contract: MATH-FIXED-001; MATH-KKT-001 through MATH-KKT-005; MATH-BR-001; HIST-PERF-01; TEST-STORE-008; MODE-003; MODE-004
- Modes affected: all five semantics, especially fixed/KKT distinction and BR
- Evidence:
  - `pbg/benchmark.py:1`, `:28` and `pbg/cli/benchmark.py:1`, `:28` expose only
    `normal`, `rao_blackwell`, and one `branched_rao_blackwell` label.
  - `_mode_config` always forces that branched label to fixed mode and deletes
    KKT (`pbg/benchmark.py:68-100`). BR is absent entirely.
  - `_median_metrics` takes the intersection of metric keys across repeats
    (`pbg/benchmark.py:123-129`). A metric missing from one malformed repeat is
    silently dropped instead of failing the artifact.
  - `compare_benchmark` iterates only current modes and only metrics present in
    both artifacts (`pbg/benchmark.py:430-461`). It does not require identical
    semantic-mode sets, so a current artifact containing only one requested mode
    can pass against a broader baseline; dropped metrics are likewise skipped.
  - The CLI permits CPU/automatic execution and CPU completion
    (`pbg/cli/benchmark.py:23-28`, `:49-51`), contrary to the intended NVIDIA
    regression surface.
  - `tests/test_aud32_performance_gates.py:13-87` uses CPU-shaped synthetic
    artifacts and tests individual mismatch/tolerance checks, but not missing
    modes, missing metrics, KKT, or BR.
- Current behavior: the benchmark is valuable because it invokes
  `_one_iteration`, the current production pipeline, but it cannot gate all five
  semantics and can report success with incomplete evidence.
- Required behavior: benchmark identities must distinguish normal,
  sequential-RB, fixed branched-RB, KKT branched-RB, and BR; a baseline gate must
  require the exact expected semantic-mode set and required metric set, and run
  only on an explicitly selected NVIDIA device.
- Consequence: KKT/BR regressions are wholly invisible, and failures that cause a
  metric to disappear can make the gate weaker instead of failing closed.
- Recommended disposition: retain and refactor this benchmark around a canonical
  five-semantic-mode matrix and strict artifact schema. Diagnostic cutoffs should
  be a separately labeled experiment, never a production baseline.
- Verification needed: negative tests for missing/extra modes and metrics, five
  tiny CUDA workloads, and comparable GPU baselines tied to exact config,
  checkpoint, software, and hardware identity.

### EVAL-011 — Benchmark checkpoint compatibility preserves randomly initialized legacy branching heads

- Severity: medium
- Confidence: confirmed
- Contract: MATH-BRANCH-008; sole-production-path/no-legacy requirement
- Modes affected: fixed branched RB benchmark
- Evidence:
  - `pbg/benchmark.py:202-225` loads checkpoints with `strict=False` and permits
    every branching-head parameter to be missing in branched mode.
  - Those missing T/C heads remain randomly initialized, are reported at
    `pbg/benchmark.py:293-308`, and are trained by the benchmark because
    `_mode_config` forces positive `c_t/c_c` (`:87-99`).
  - `tests/test_aud32_performance_gates.py:171-227` explicitly entrenches this
    compatibility behavior.
  - The current math requires branched T/C heads from the first update
    (`MATH-BRANCH-008`, `MATH-FIXED-002`); the repository policy requires obsolete
    compatibility paths to be removed after migration.
- Current behavior: a legacy sequential checkpoint can be presented as a
  branched performance baseline while an unbound random T/C subsystem changes
  loss work and gradients.
- Required behavior: production performance baselines use current, fully
  compatible checkpoint schemas, or explicitly declare a one-time conversion
  artifact before timing.
- Consequence: cross-commit results can mix checkpoint migration cost/semantics
  with implementation performance and preserve indefinite legacy loading.
- Recommended disposition: remove permissive benchmark loading after historical
  baselines are archived; require strict current checkpoints for regression
  gates.
- Verification needed: strict-load rejection test and one explicitly versioned
  offline conversion test if old run comparison remains necessary.

## Reachability and disposition inventory

The table classifies every top-level object in the reviewed non-benchmark
modules. "Experiment-only" means reachable only through a custom experiment
launcher/config, not through any of the five core workflows. Private nested
helpers/closures are covered by their owning top-level object.

| File | Objects | Classification and disposition |
|---|---|---|
| `pbg/eval/arena.py` | `ArenaReport`, `_poker_config`, `_build_model`, `_load_model`, `_profile_runouts`, `_linear_ranges`, `_policy_probabilities`, `_uniform_for`, `_draw_event`, `_settle_showdown`, `_replay`, `arena` | CPU-only duplicate evaluation; archive all. `_student_t_critical` alone is shared by BR and should move to a generic helper. |
| `pbg/eval/br_probe.py` | `BRProbeReport`, `_sha256`, `_selected_reward`, `evaluate_br_probe` | BR production evaluation support, pending BR authority and CUDA gate; retain/refactor. Imports `DrawPurpose`, `PROBE_FOLD`, and `sequential_leaf_logical_rng_keys` are unused. |
| `pbg/inference.py` | `HandHandle`, `PokerAction`, `Decision`, `FullStrategy`, `_FoldRecord`, `_HandState`, `parse_card`, `_move_public`, `_move_summary`, `PokerInferenceSession` | CPU/MPS serving subsystem, outside five NVIDIA modes; archive all. `seat_values_bb` fields are never populated. |
| `pbg/inference_artifact.py` | `INFERENCE_ARTIFACT_SCHEMA_VERSION`, `sha256_file`, `_validate_artifact`, `load_inference_source`, `export_inference_artifact` | Used only by the inference subsystem/export CLI; archive all with it. |
| `pbg/eval/rao_blackwell_variance.py` | `SCHEMA_VERSION`, `logical_seed`, `checkpoint_sha256`, `canonical_jobs`, `assigned_jobs`, `consecutive_blocks`, `weighted_ratio_moments`, `rademacher_direction`, `parameter_jvp`, `SufficientStatistics`, `theoretical_variance`, `variance_cube`, `cpu_oracle`, `_model_directional_rows`, `run_variance_job`, `load_experiment_config`, `manifest_payload`, `atomic_json`, `ensure_manifest` | Checkpoint-specific experiment/analyzer; archive all. Several generic names duplicate helpers elsewhere and should not remain active merely for reuse. |
| `pbg/eval/branched_rao_blackwell_variance.py` | `SCHEMA_VERSION`, `_conditional`, `projected_loss_and_child_partials`, `reconstruct_variance`, `load_config`, `manifest_payload`, `atomic_json`, `ensure_manifest`, `beta_key`, `_quantile`, `geometry_summary`, `_tree_layout`, `_node_moments`, `_decision_edges`, `_model_directional_rows`, `_propagate_functional_coefficients`, `_slice_rows`, `BranchedVarianceRunner`, `component_summary`, `variance_table` | Checkpoint-1110 experiment and parallel target implementation; archive all. |
| `pbg/cli/rao_blackwell_variance.py` | `_distributed_identity`, `_device`, `_linear_fixture`, `_jvp_smoke`, `run_smoke`, `_unimplemented_phase`, `_save_job`, `run_variance`, `run_analysis`, `main` | Experiment-only; archive all. `_unimplemented_phase` represents advertised dead functionality. |
| `pbg/cli/branched_rao_blackwell_variance.py` | `_device`, `_save_pt`, `_job_path`, `_load_geometry`, `_load_moments`, `run_smoke`, `_run_geometry_jobs`, `_run_moment_jobs`, `_objective`, `_refinement_candidates`, `_timing_gate`, `run_experiment`, `analyze`, `main` | Experiment-only; archive all. `_load_geometry` has no repository caller. |
| `pbg/cli/benchmark_inference.py` | `_prepare_hands`, `main` | CPU/MPS inference-only; archive all. |
| `pbg/cli/export_inference.py` | `main` | Inference-only; archive with artifact subsystem. |
| `pbg/cli/br_eval.py` | `main` | BR production support, pending BR authority/CUDA-only enforcement; retain/refactor. |

The benchmark objects `MODES`, `LOWER_IS_BETTER`, `HIGHER_IS_BETTER`,
`MODEL_INITIALIZATION_PROTOCOL`, `CHECKPOINT_INITIALIZATION_PROTOCOL`,
`_benchmark_checkpoint`, `_mode_config`, `_environment`, `_median_metrics`,
`_cpu_snapshot`, `benchmark_modes`, and `compare_benchmark` are all reachable
through the installed `pbg-benchmark` command. They are production support rather
than training-mode objects, but need the strict five-mode and fail-closed changes
in EVAL-010/EVAL-011. `pbg/cli/benchmark.py:main` is likewise reachable and
should remain only after that refactor.

Outside the assigned files, `pbg/games/poker/bundle.py:294-365`,
`PokerCompletionService.complete_selected_keyed`, is test-only legacy code: the
only repository call is `tests/test_br_probe.py:197-205`. Production BR sealing
uses `complete_sealed`. It should be deleted when EVAL-003 is resolved.

## Tests and checks performed

- `32` relevant tests passed and `3` MPS tests were skipped:

  ```text
  /home/pablob/.venvs/poker/bin/python -m pytest -q \
    tests/test_br_probe.py tests/test_inference_api.py tests/test_p12_eval.py \
    tests/test_rao_blackwell_variance.py \
    tests/test_rao_blackwell_variance_cli.py \
    tests/test_branched_rao_blackwell_variance.py \
    tests/test_aud32_performance_gates.py
  ```

- No CUDA work was launched for this read-only batch. Existing tests provide no
  BR CUDA acceptance, so GPU use here would have been a new test design rather
  than verification of an existing gate.
- Static symbol/repository searches were cross-checked with
  `tools/audit_python_objects.py --all --scope pbg`. Its lexical handling of
  common names such as `main`, `load_config`, and `SCHEMA_VERSION` overcounts
  references, so disposition above is based on import/call-site tracing rather
  than raw name counts.

## Files fully reviewed

- `pbg/eval/__init__.py`, `arena.py`, `br_probe.py`,
  `rao_blackwell_variance.py`, `branched_rao_blackwell_variance.py`
- `pbg/inference.py`, `pbg/inference_artifact.py`, `pbg/benchmark.py`
- all five assigned CLIs plus `pbg/cli/benchmark.py`
- `tests/test_br_probe.py`, `test_inference_api.py`, `test_p12_eval.py`,
  `test_rao_blackwell_variance.py`, `test_rao_blackwell_variance_cli.py`,
  `test_branched_rao_blackwell_variance.py`,
  `test_aud32_performance_gates.py`
- directly relevant sections of `pbg/core/config.py`, `rollout.py`,
  `training.py`, `targets.py`, `checkpointing.py`, `pbg/core/game.py`,
  `pbg/games/poker/bundle.py`, and `rollout_backend.py`
- `pyproject.toml`, `docs/training_and_evaluation.md`, the two variance tool
  launchers, and the corresponding variance experiment YAMLs

## Open authority decisions

1. Is the fifth mode formally an approximate shared-policy response probe, or
   must it approximate independently optimized per-player best responses?
2. What exact training budget/stopping rule makes two probe checkpoints
   comparable, and what quantity is the reported aggregate lower bound intended
   to estimate?
3. Is any deployment inference API actually in repository scope? Current policy
   says NVIDIA-only five-mode production, which makes archival the default.
4. Should checkpoint-versus-checkpoint evaluation remain a production feature?
   If yes, it needs a new CUDA implementation on the sole rollout path; the CPU
   arena is not a viable retained implementation.
