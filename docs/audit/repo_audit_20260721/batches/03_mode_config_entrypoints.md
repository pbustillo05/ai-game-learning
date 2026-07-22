# Mode, configuration, and entry-point audit

## Scope and conclusion

The production training schema has **three training labels**, not five literal
labels.  With the orthogonal `rollout.branching_mode` and optional `probe`
section, those labels encode the five requested semantic modes:

| Requested semantic mode | Current public configuration | Production entry point |
|---|---|---|
| normal | `training.mode: normal`, no `probe` | `pbg-train` -> `pbg.runtime.train` |
| sequential RB | `training.mode: rao_blackwell`, no `probe` | `pbg-train` -> `pbg.runtime.train` |
| fixed-beta branched RB | `training.mode: branched_rao_blackwell`, `rollout.branching_mode: fixed` | `pbg-train` -> `pbg.runtime.train` |
| KKT branched RB | `training.mode: branched_rao_blackwell`, `rollout.branching_mode: dynamic`, `rollout.adaptive_kkt` present | `pbg-train` -> `pbg.runtime.train` |
| approximate BR | `training.mode: rao_blackwell`, `probe` present; paired evaluation via `pbg-br-eval` | `pbg-train` and `pbg-br-eval` |

The schema rejects other `training.mode` and `branching_mode` values.  Thus no
sixth estimator mode was found on this surface.  However, the repository does
not present this five-mode mapping as one canonical public contract, has no
committed normal/KKT/BR training preset, does not benchmark all five semantics,
and still exposes substantial CPU/MPS and historical experiment surfaces.

BR fidelity cannot be certified against `core_math.tex`: contract
`MATH-BR-001` explicitly records that the authority contains no BR algorithm.

## Findings

### MODE-001 — Production training remains fully reachable on CPU

- Severity: high
- Confidence: confirmed
- Contract: A-07; repository NVIDIA-only requirement
- Modes affected: normal, sequential RB, BR, fixed branched RB, KKT branched RB
- Evidence:
  - `pbg/core/config.py:278`, `_validate_config`, accepts `completion.backend` values `auto`, `cpu`, and `cuda`.
  - `pbg/core/training.py:2546`, `run_training`, deliberately selects a CPU device for `backend == "cpu"` and falls back to CPU for `backend == "auto"` when CUDA is unavailable.
  - `pbg/cli/benchmark.py:23`, `main`, documents `cpu` as an explicit benchmark device; `pbg/benchmark.py:173`, `benchmark_modes`, also falls back to CPU.
  - `tests/test_br_probe.py:37-73` builds a production-valid CPU BR config, and `tests/test_br_probe.py:216-246` calls `train(cfg)` on it.
  - `tests/test_aud32_performance_gates.py:90-119` and `:124-216` exercise the production benchmark on CPU.
- Current behavior: CPU is a first-class validated training and benchmarking backend, not a tiny incidental test shim. `auto` silently changes the execution platform.
- Required behavior: the production surface is NVIDIA GPU only. CPU utilities may remain only where they add negligible code and must not form a production training path.
- Consequence: runs can execute an unsupported implementation with materially different scheduling, memory, synchronization, and performance behavior while still passing production config validation.
- Recommended disposition: make production config require `completion.backend: cuda`, make `run_training` fail without CUDA, remove automatic CPU fallback, and isolate any small CPU fixtures behind test-only helpers.
- Verification needed: loader tests rejecting `auto`/`cpu`; one CUDA-unavailable launch rejection; GPU smoke for each of the five semantics.

### MODE-002 — Installed inference surface is CPU/MPS-only and packaging advertises macOS support

- Severity: high
- Confidence: confirmed
- Contract: repository NVIDIA-only requirement; A-07
- Modes affected: deployed policy inference, hence BR/evaluation consumers
- Evidence:
  - `pbg/cli/benchmark_inference.py:1` describes a CPU/Apple-MPS benchmark and `:44-47` offers only `auto`, `cpu`, and `mps` devices.
  - `pyproject.toml:18` installs `pbg-inference-benchmark` as a public command.
  - `pyproject.toml:20-21` publishes a `mac` optional dependency for Apple arm64.
- Current behavior: a sizeable non-NVIDIA inference implementation is an installed production command and explicit package feature.
- Required behavior: production code should target NVIDIA GPUs; non-NVIDIA implementations should remain only if they barely add code.
- Consequence: maintenance, tests, and API choices preserve a separate deployment platform outside the project goal.
- Recommended disposition: delete or archive the CPU/MPS inference benchmark and macOS optional dependency unless the user explicitly designates model-only CPU/MPS inference as a supported exception; otherwise replace its device contract with CUDA.
- Verification needed: inventory `pbg.inference` consumers before deletion and add a CUDA inference smoke if inference export remains production.

### MODE-003 — The performance benchmark omits KKT branched RB and BR

- Severity: high
- Confidence: confirmed
- Contract: MATH-FIXED-001; MATH-KKT-001 through MATH-KKT-005; MATH-BR-001; five-mode audit requirement
- Modes affected: KKT branched RB, BR
- Evidence:
  - `pbg/benchmark.py:28`, `MODES`, contains only `normal`, `rao_blackwell`, and `branched_rao_blackwell`.
  - `pbg/benchmark.py:87-99`, `_mode_config`, unconditionally converts the branched label to `branching_mode = "fixed"` and removes `adaptive_kkt`.
  - `pbg/cli/benchmark.py:1` calls itself a “three-mode” benchmark and `:28` limits `--mode` to that tuple.
  - `tests/test_aud32_performance_gates.py` contains no dynamic/KKT or probe/BR case.
- Current behavior: the only installed regression benchmark collapses the two branched semantics and cannot run KKT or BR at all.
- Required behavior: efficiency regression gates must cover all five production semantics, with fixed and KKT recorded as distinct workloads.
- Consequence: KKT controller overhead, dynamic geometry, BR opponent-model overhead, and their memory behavior can regress without any benchmark signal.
- Recommended disposition: define a canonical five-semantic-mode enum/matrix for audit and benchmarking, or expose explicit `branched_fixed` and `branched_kkt` benchmark cases plus a BR case; never infer fixed mode from the generic branched label.
- Verification needed: comparable GPU baselines for all five semantics and gate tests that reject a baseline/config semantic-mode mismatch.

### MODE-004 — Benchmark configuration mutation bypasses the strict schema and can create invalid comparisons

- Severity: high
- Confidence: confirmed
- Contract: MATH-FIXED-001; MATH-KKT-003; production configuration strictness
- Modes affected: normal, sequential RB, fixed branched RB
- Evidence:
  - `pbg/cli/benchmark.py:59` validates only the original input, then `:60-86` mutates roots, beta, queue widths, completion, minibatch, and optimizer-step settings without revalidation.
  - `pbg/benchmark.py:68-100`, `_mode_config`, mutates a deep-copied dataclass without passing it through `load_config_mapping` or poker validation.
  - `pbg/benchmark.py:92` changes only the mutable runtime `roots_per_gpu` view for branched mode, leaving `sequential_roots_per_gpu_initial/minimum/maximum` potentially inconsistent with the production invariant checked at `pbg/core/config.py:321-325` and `:389-409`.
  - `pbg/cli/benchmark.py:71-84` does not reject nonpositive row counts, completion samples, minibatch sizes, or GD steps.
- Current behavior: benchmark-only configurations need not be legal production configurations and may retain scientific/model fields from the source mode.
- Required behavior: every compared workload must be produced from the same immutable strict schema and pass full game/model validation after overrides.
- Consequence: benchmark results may measure an impossible production setup and cannot reliably support regression claims.
- Recommended disposition: build a mapping for each semantic mode, apply typed overrides, and run `load_config_mapping` after every transformation. Remove mutation of the runtime-only `roots_per_gpu` field from the CLI layer.
- Verification needed: negative override tests; conversion starting from both sequential and branched presets; assertion that every benchmark workload round-trips through the strict loader.

### MODE-005 — Non-branched configs may retain unused branching heads and losses

- Severity: medium
- Confidence: confirmed
- Contract: MATH-NORMAL-004; MATH-SRB-005; MATH-BRANCH-008; production-only/minimal-payload requirement
- Modes affected: normal, sequential RB, BR
- Evidence:
  - `pbg/core/config.py:380-388`, `_validate_config`, requires `use_branching_heads` and positive `c_t/c_c` in branched mode but has no converse rejection for non-branched modes.
  - `pbg/benchmark.py:73-86`, `_mode_config`, removes branching geometry for normal/sequential RB but does not turn off `model.use_branching_heads` or zero `loss.c_t/c_c` when the source config is branched.
  - `pbg/games/poker/bundle.py:679` includes branching-head state in memory planning whenever the model flag remains true.
  - `tests/test_refactor_gpu_acceptance.py:26-38` manually zeros `c_t/c_c`, and `:93-103` manually sets the model flag, demonstrating that the production converter does not enforce this itself.
- Current behavior: valid sequential configs can instantiate production-unused T/C heads, and cross-mode benchmarks can retain their parameters/forward cost.
- Required behavior: only branched modes own and train T/C heads; normal, sequential RB, and BR should reject or canonicalize all branching-only fields.
- Consequence: dead parameters and allocations remain reachable, and cross-mode performance comparisons are biased.
- Recommended disposition: add symmetric validation (`use_branching_heads == false`, `c_t == c_c == 0`) outside branched modes and make mode conversion explicit.
- Verification needed: strict loader rejection tests and parameter/allocation parity tests for normal/sequential configurations derived from a branched preset.

### MODE-006 — No committed preset represents normal, KKT, or BR production

- Severity: medium
- Confidence: confirmed
- Contract: MATH-NORMAL-001 through MATH-NORMAL-004; MATH-KKT-001 through MATH-KKT-005; MATH-BR-001; five-mode audit requirement
- Modes affected: normal, KKT branched RB, BR
- Evidence:
  - `configs/train/nlhe6.yaml:7-30` and `configs/train/nlhe_6max_10bb_branched.yaml:7-33` are fixed branched presets.
  - `configs/train/nlhe_hu_32bb_rb.yaml:58` is sequential RB.
  - Repository searches find no `training.mode: normal`, no `branching_mode: dynamic`, and no `probe:` section in any YAML under `configs/`.
  - `tests/test_refactor_gpu_acceptance.py:17-60` constructs normal and KKT configurations by mutating another mode after validation rather than loading committed production presets.
- Current behavior: code-level construction can reach all four training estimators and BR, but three requested semantics have no checked-in launchable configuration.
- Required behavior: each supported production semantic mode should have a canonical, fully validated GPU preset or a documented typed transformation that is itself tested.
- Consequence: users must hand-edit multiple coupled fields, increasing the chance of unvalidated or stale combinations; BR has no discoverable training launch despite `pbg-br-eval` being installed.
- Recommended disposition: add minimal canonical NVIDIA presets for normal, KKT, and BR after the semantic contracts are approved; alternatively replace duplicated YAML with one typed generator whose five outputs pass the strict loader.
- Verification needed: load all five presets and perform tiny CUDA launch gates.

### MODE-007 — BR lacks NVIDIA acceptance coverage and an authoritative algorithm

- Severity: high
- Confidence: confirmed
- Contract: MATH-BR-001; MATH-BR-002; repository NVIDIA-only requirement
- Modes affected: BR
- Evidence:
  - `tests/test_refactor_gpu_acceptance.py:63-111` covers normal, sequential RB, fixed branched RB, and dynamic branched RB, but not a probe config.
  - `tests/test_br_probe.py:37-73` explicitly selects `backend: cpu`; its rollout, training, checkpoint, and paired-evaluation integration tests use CPU (`:92-236`).
  - `pbg/core/config.py:288-305` validates only the BR probe's parent mode, target path/hash, and initialization choice.
  - Contract `MATH-BR-001` states that `core_math.tex` contains no best-response sampling, objective, or stopping contract.
- Current behavior: BR is production-reachable but its end-to-end evidence is CPU-only and its mathematical semantics cannot be audited against the designated authority.
- Required behavior: a fifth NVIDIA production mode needs both an explicit algorithmic authority and a CUDA acceptance test.
- Consequence: calling the probe “BR” overstates what can currently be certified, and GPU-specific storage/queue/device failures can escape tests.
- Recommended disposition: specify whether this is exact BR, approximate BR, or a diagnostic probe; add its law/objective/stopping/evaluation contract to an authoritative document, then add a tiny GPU train-and-evaluate acceptance case.
- Verification needed: target-checkpoint binding, frozen-opponent behavior, selected-seat sampling, optimizer isolation, and paired evaluation on CUDA.

### MODE-008 — Removed `use_moment_heads` remains an accepted compatibility key in active configs

- Severity: medium
- Confidence: confirmed
- Contract: MATH-MODEL-002; MATH-LOSS-002; sole-production-path/no-legacy requirement
- Modes affected: sequential RB experiment presets
- Evidence:
  - `pbg/games/poker/model/preflop.py:95-111`, `PokerModelConfig.from_dict`, silently removes `use_moment_heads: false` and also translates legacy `p_max` to `max_players`.
  - 24 of the 30 YAML files in `configs/experiments/` contain `use_moment_heads: false`; for example `configs/experiments/nlhe_hu_32bb_optimizer_baseline_20step.yaml:37`.
  - `tests/test_adaptive_kkt_branching.py:97-99` requires only `use_moment_heads: true` to fail, preserving false as a compatibility no-op.
- Current behavior: an obsolete V/U-head key remains part of accepted production model configuration, and current experiment presets depend on that exception.
- Required behavior: superseded configuration fields and compatibility translations should be deleted once all production/evaluation consumers migrate.
- Consequence: stale configs appear valid, obscure the sole current T/C head contract, and keep legacy parsing code and tests alive.
- Recommended disposition: remove `use_moment_heads` from all retained configs, reject it regardless of value, remove the compatibility pop, and separately decide whether the `p_max` alias has any live checkpoint requirement; archive rather than preserve indefinite compatibility.
- Verification needed: search checkpoint migration code before removing `p_max`; strict unknown-key tests for both names after migration.

### MODE-009 — Hardware scheduler fields are validated and persisted but never control execution

- Severity: medium
- Confidence: confirmed
- Contract: production-only/no-unused-object requirement
- Modes affected: all training modes
- Evidence:
  - `pbg/core/config.py:132-152`, `HardwareProfileConfig`, declares `scheduler`, `worker_count`, and `slurm`.
  - `pbg/core/config.py:518-568` only validates those fields; no production runtime reads `profile.scheduler`, `profile.worker_count`, or `profile.slurm`.
  - `configs/hardware/quest_slurm.yaml:1-27` supplies a detailed Slurm section that `pbg-train` does not submit or consume.
  - `tests/test_p0_config.py:318-342` tests `worker_count` validation, thereby entrenching a no-op field.
- Current behavior: configuration claims scheduler behavior that is actually provided externally, while `worker_count` and `slurm` have no runtime effect.
- Required behavior: every production configuration object should affect a live production consumer; external-launch documentation should not masquerade as runtime config.
- Consequence: operators may expect the profile to request resources or set worker count when it does neither.
- Recommended disposition: delete `scheduler`, `worker_count`, and `slurm` from the runtime schema and profiles, or implement the explicit launcher as a separately authorized production component. Given current external `torchrun`/Slurm ownership, deletion and documentation are preferable.
- Verification needed: confirm no run-manifest comparison relies on these fields before removing them.

### MODE-010 — Historical experiment workflows and presets remain in the active production package/config tree

- Severity: medium
- Confidence: strong
- Contract: production-only/no-stale-experiment requirement
- Modes affected: none of the five production semantics directly; experimental variance/optimizer workflows
- Evidence:
  - `pbg/cli/rao_blackwell_variance.py:1-25` and `pbg/cli/branched_rao_blackwell_variance.py:1-27` are staged research launchers with custom schemas, manifests, phases, and analyzers.
  - `pbg/cli/rao_blackwell_variance.py:307-347` exposes phases outside the five production semantics; several choices intentionally reach `_unimplemented_phase`.
  - `pbg/cli/branched_rao_blackwell_variance.py:374-395` is independently runnable as a module.
  - `configs/experiments/` contains 30 active YAMLs, including three custom variance schemas and 27 one-off optimizer/tuning/probe presets; e.g. `configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml:1-55` and `configs/experiments/nlhe_hu_32bb_branched_rao_blackwell_variance.yaml:1-60`.
- Current behavior: completed calibration/tuning machinery and one-off presets remain importable under `pbg` and discoverable under active `configs`, even though they are not core production modes.
- Required behavior: keep only code used by the five production NVIDIA modes; preserve historical evidence under `archive/` or `runs/`.
- Consequence: audit reachability is polluted by alternate schemas and experimental control flows, and stale presets continue to preserve legacy keys.
- Recommended disposition: determine which experiment outputs are historical evidence, then move their launchers/eval modules/configs to `archive/` together; retain only a small current benchmark that covers the five production semantics.
- Verification needed: trace any current `tools/` launcher and `runs/` reproducibility reference before moving files, then ensure no installed script or production import depends on them.

### MODE-011 — Duplicate and probe-specific presets remain in active config locations

- Severity: low
- Confidence: confirmed
- Contract: sole-production-path/no-stale-config requirement
- Modes affected: fixed branched RB; hardware profiles
- Evidence:
  - `configs/nlhe6_h100.yaml:1-52` and `configs/train/nlhe6.yaml:1-52` describe the same fixed-branched run/out directory but disagree on `training.minibatch_size` (`6` versus `128`).
  - Six of eight active hardware profiles are named optimizer/SDPA probes, for example `configs/hardware/local_2x2080ti_optimizer_probe.yaml` and `local_2x2080ti_sdpa_row{3,6,11}_probe.yaml`.
- Current behavior: it is unclear which duplicate H100 config is authoritative, and historical tuning profiles sit beside actual launch profiles.
- Required behavior: one production preset per intended workload; measured historical probes should live with their run evidence or in `archive/`.
- Consequence: users and tests can choose inconsistent scientific geometry under the same apparent workload name.
- Recommended disposition: select one authoritative fixed-branched config, archive the duplicate and probe hardware profiles with their benchmark evidence, and update tests to consume the authoritative path.
- Verification needed: inspect run manifests to identify which minibatch value produced existing baselines.

### MODE-012 — Runtime dependency and package description preserve generic/test-only scope

- Severity: low
- Confidence: confirmed
- Contract: NLHE-only and production-only repository requirements
- Modes affected: packaging
- Evidence:
  - `pyproject.toml:8` describes generic “public-action Bayesian games,” although NLHE is the sole active game.
  - `pyproject.toml:11` installs `pytest` as a production dependency.
- Current behavior: package metadata advertises broader game scope and carries a test runner in every production environment.
- Required behavior: metadata and runtime dependencies should describe the sole NLHE NVIDIA production package.
- Consequence: minor dependency/scope drift and unnecessary deployment surface.
- Recommended disposition: move `pytest` to a development/test extra and update the description after confirming packaging consumers.
- Verification needed: clean-venv install and import/CLI smoke without the test extra.

## Validation gaps summary

- The strict loader does test fixed/dynamic coupling in
  `tests/test_adaptive_kkt_branching.py:102-170`, and the four training
  estimators have tiny CUDA coverage in
  `tests/test_refactor_gpu_acceptance.py:63-111`.
- BR alone lacks CUDA acceptance and an authoritative mathematical contract.
- The installed benchmark covers only three labels and forces fixed branching.
- No test requires production config/loading/training to reject CPU or automatic
  CPU fallback; several tests explicitly require those paths to work.
- No loader test enforces absence of T/C heads and coefficients in normal,
  sequential RB, or BR.
- There is no “load every production preset and classify it as exactly one of
  five semantics” test.

## Files and objects reviewed

Fully reviewed for this batch:

- `pyproject.toml`
- `pbg/config.py`: `load_config`, `load_config_mapping`
- `pbg/runtime.py`: `build_game_bundle`, `train`
- `pbg/core/config.py`: all dataclasses, strict parsing, validation, YAML/profile loading, seeding
- `pbg/benchmark.py`: mode transformation, device selection, workload identity, benchmark setup
- `pbg/cli/train.py`, `benchmark.py`, `br_eval.py`, `export_inference.py`, `benchmark_inference.py`
- `pbg/cli/rao_blackwell_variance.py`, `branched_rao_blackwell_variance.py`
- all YAML under `configs/train/`, `configs/hardware/`, and `configs/experiments/`, plus `configs/nlhe6_h100.yaml`
- directly relevant sections of `tests/test_p0_config.py`,
  `test_adaptive_kkt_branching.py`, `test_refactor_gpu_acceptance.py`,
  `test_br_probe.py`, `test_aud32_performance_gates.py`,
  `test_rao_blackwell_variance_cli.py`, and
  `test_branched_rao_blackwell_variance.py`

Objects strongly indicated as stale or outside the five-mode NVIDIA production
surface:

- `HardwareProfileConfig.scheduler`, `.worker_count`, `.slurm`
- the `use_moment_heads: false` compatibility path and likely `p_max` alias in
  `PokerModelConfig.from_dict`
- CPU/automatic completion backend options in production `CompletionConfig`
- CPU/MPS inference benchmark command and macOS packaging extra
- staged RB/branched-RB variance CLI modules and their custom configs
- one-off optimizer/tuning configs and probe hardware profiles
- duplicate `configs/nlhe6_h100.yaml`

## Checks performed

- Every root-shaped YAML under `configs/` was passed through `pbg.load_config`;
  all 27 such experiment/train presets loaded successfully. The three variance
  experiment YAMLs use separate custom schemas.
- `57` targeted configuration and benchmark-gate tests passed:
  `/home/pablob/.venvs/poker/bin/python -m pytest -q tests/test_p0_config.py tests/test_aud32_performance_gates.py`.
- Static repository searches established that no committed YAML represents
  normal, dynamic/KKT, or BR and that 24 experiment YAMLs retain
  `use_moment_heads: false`.
