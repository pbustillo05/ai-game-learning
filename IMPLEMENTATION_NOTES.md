# Implementation notes

## P00 — repository skeleton and configuration

- Added the complete package, config, test, and evaluation directory skeleton.
- Kept all future-prompt modules to ownership docstrings only.
- Implemented strict recursive YAML-to-dataclass loading with dotted-path errors.
- Kept synthetic and poker sub-configurations as game-owned mappings.
- Poker root configs require the `poker_gpu` completion boundary.
- Converted YAML optimizer beta lists to the frozen tuple representation.
- `seed_all` returns explicit Torch and NumPy generators; NumPy globals are untouched.
- Metrics are append-only JSONL; wandb remains an import-guarded optional sink.
- Added permanent AST-based import-direction test T12.

## P01 — synthetic family and exact tools

- Added colex private types, collision-free deals, and a deduplicated SoA state table.
- Implemented capped limit betting, public chance events, fold/showdown chip rewards.
- Final chance prefixes are pre-terminal; fold endings remain terminal only.
- Added collision-conditioned reach updates and exact/rejection posterior sampling.
- Exact tools enumerate public trees and profiles in float64 with the 2M guard.
- Best responses share actions across opponent profiles at each information set.
- Closed-form Kuhn equilibria at alpha 0, 1/6, and 1/3 evaluate at numerical zero.
- Added the synthetic embedding/MLP, categorical adapter, and table exporter.
- CONTRACT-QUESTION: Standard specified Kuhn rules give 9 total public states and 5 terminal histories, not the prompt's 9 non-root decisions and 6 terminals.
- CONTRACT-QUESTION: P00 required core/policy_adapter.py to remain empty, so P01 cannot use the frozen core PolicyOutput without violating file ownership; a local identical DTO is used.

## P02 — public NLHE engine and card representations

- Added the frozen lexicographic combo map and the standard AA-to-22 13x13 class map.
- Complement draws use random-key permutations, giving uniform ordered samples without replacement on CPU or CUDA.
- Added an immutable public NLHE state table with blind/action order, affine raise-to ticks, full-raise reopening, short-all-in handling, street chance transitions, and fold terminals.
- Showdowns remain pre-terminal; public contribution levels produce nested affine pot strata and no local private-hand settlement logic.
- The engine accepts caller-supplied chance cards and canonicalizes the flop as one unordered event.

## P03 — ranges and Bayesian bookkeeping

- Stored one fp32 log-likelihood row per rollout slot and player, initialized uniformly.
- Action updates gather only the observed event and stabilize row gauges below -30.
- Board deals apply compatibility masks without modifying surviving hypotheses.
- Fold snapshots clone linear weights, the fold-time board, and an 8-float public metadata row.
- Metadata is caller-populated because ranges have no dependency on the public engine.
- Branch clones append independent reach, live, board, metadata, and stability rows.
- Class aggregation normalizes combo rows on demand and scatter-sums the standard 169 map.

## P04 — evaluator partition-summary boundary

- Added typed batches for fold snapshots and opaque evaluator summary fields.
- The adapter lazily imports and version-checks only the stable sibling facade.
- Local -1 board padding is mapped to facade-owned uint8/255 padding on-device.
- Live and enabled folded likelihood rows are concatenated without normalization.
- Evaluator result tensors are shape/device/dtype checked and returned unchanged.
- No partition, blocker, collision, removal-set, or private-assignment math is local.

## P05 — poker-private evaluator adapters

- Added game-neutral single/batched completion DTOs, an opaque reach view, and the completion protocol.
- Completion maps public engine boards, dealt/contender status, stabilized unnormalized reach rows, affine pot strata, and explicit-generator seeds to the sibling facade.
- Evaluator completion statistics and diagnostics are validated and retained without local reconstruction.
- Exact profile sampling makes one batched facade call, preserves folded blocker seats, and raises a typed diagnostic error on any partial failure.
- Fixed showdown validation checks only shape, eligibility, tie shares, and mask consistency; poker winner selection remains entirely evaluator-owned.
- All three operations lazily import and require the stable `pbg-v2` public facade, with no local evaluator or CUDA fallback.

## Evaluator 0.3.0 contract reconciliation

- Reconciled the consumer boundary with the shipped
  `poker_gpu_evaluator.pbg` API documented in the sibling repository.
- Removed the ignored-and-now-deleted `query_combo_ids` argument from the
  partition adapter; it now calls the deterministic 0.3.0+ signature exactly.
- Validate the sibling's defensive uint8 `[1326,2]` `COMBO_CARDS` export against
  the frozen local lexicographic representation before any evaluator operation.
- Updated the binding IO document, corrected CONTRACTS P4 signature, and
  carried the shipped API constraints into the unimplemented P12/P13 prompts.

## P07 — postflop model and poker public-tree adapter

- Added generic padded compact-row maps with differentiable gather/scatter and representation-row permutation helpers.
- Added the combo-level postflop transformer with supplied compatibility maps, fixed-width output scatter, opaque evaluator-summary projection, optional fold-line features, autocast support, and public moment heads.
- Added consistent suit relabeling for boards and combo-indexed tensors; it performs only representation transforms.
- Added exact composition of contextual passive/call/raise-tick laws over engine-supplied legal events, including full-row probabilities, stipulated-profile row selection, sampling, entropy, and KL.
- Added mixed-street model routing with preflop class-to-combo lifting at the bundle boundary.
- Added the public poker facade for exact-profile delegation, complement chance draws, public encoding/reach updates, owned range slots, and evaluator-mediated fixed-showdown settlement.
- P07 focused tests: 9 passed (`tests/test_p7_postflop.py`, `tests/test_p7_adapter.py`, seed values 70/73 where stochastic).
- Full suite after P07: 60 passed, 3 skipped in 10.20s on CPU; skipped tests are optional sibling-boundary/device cases.

## P09 — branched rollout and variance-guided intensities

- Preserved the sequential sampler path and its random-number consumption; non-sequential modes dispatch to the P09 implementation.
- Added spine-Poisson child counts, one fresh joint posterior profile per event draw, stable event deduplication, and empirical multiplicity weights.
- Branched policy caches store one all-type log-probability row per unique child; reach updates remain per-edge and copy-on-branch.
- Node-budget pressure permanently clamps the affected tree frontier to spine-only continuation and records `budget_clamped` rollout metrics.
- Added a bucketed warmup/Neyman controller with padded-seat masking, chance-node fixed intensities, iteration updates, and checkpoint state round-tripping.
- P09 focused tests include the 50k fixed-beta-one law and 20k beta-three ν-induced terminal-law checks, multiplicity statistics, budget degradation, and controller formulas.

## P10 — backward value propagation

- Added the capture-before-mix branched lambda-return pass with owner-only type-value bootstrapping and optional all-player moment bootstrapping.
- Completion posteriors remain unmixed; invalid private-type and padded-seat entries are explicitly zero-masked from game dimensions.
- Complete trees are greedily streamed under the resident-row cap without splitting dependency chains.
- Added an independent recursive fp64 oracle plus focused branching, sequential-boundary, lambda-endpoint, moment-gating, padding, and chunk-consistency tests.

## Corrected implementation plan (C00/C08–C11, P12, P13A/P13B)

- Froze randomized-rounding offspring and duplicate-preserving labeled sampled trees; historical P09 quotient-tree behavior is superseded by C09.
- Corrected the mathematical contract to propagate unbiased bucket mass and posterior-weighted target mass, treating conditional values as self-normalized finite-sample ratios.
- Added correction prompts for the device-resident rollout boundary, fixed branching, posterior-moment backpropagation, and the minimal RB-PPO objective.
- Deferred Neyman allocation, public moment heads/losses, stored Q targets by default, and reference-policy KL.
- Rewrote P12 against the typed bundle and split poker work into P13A GPU rollout plus P13B integration/presets.
- Made every active prompt independently launchable in a cleared context: each now names its workspace bootstrap, binding interfaces, staged compatibility obligations, owned test migrations, exact validation commands, and handoff requirements. P08–P11 are visibly marked historical.

## C08 — device-resident rollout foundation

- Added the typed `GameBundle`, `RolloutBackend`, and opaque `EventBatch` boundary.
- Added a preallocated device tree buffer with contiguous parent-major appends and CSR finalization.
- The new sequential sampler runs a fixed number of frontier waves and keeps topology, reach, policy caches, and RNG draws device-local.
- Player and chance frontier partitions use backend-owned policy routing, event sampling, stepping, classification, and reach updates.
- Synthetic rollout remains the CPU oracle and now applies the completion cut only after final betting; folds remain terminal.
- Buffer exhaustion raises requested/capacity diagnostics and cannot return a truncated tree.
- The positional P08 sampler remains behaviorally intact and warns only under debug assertions.
- Explicit Torch generators can now be constructed directly on the rank device.

## C09 — fixed-intensity labeled branching

- Added the typed `sample_branched_labeled` entry point on the C08 device path.
- Randomized rounding produces `floor(beta)` plus one Bernoulli fractional child.
- Abstract children remain parent-major and duplicate public labels remain distinct.
- Profiles, events, steps, reach updates, classification, and appends are batched.
- Every edge stores exactly reciprocal offspring mass and propagates path occupancy.
- Fixed intensities are backend bucket lookups validated as finite fp32 values at least one.
- Capacity exhaustion raises from the preallocated buffer without clamping or partial output.
- Isolated the P09 quotient sampler under `_legacy_sample_branched_trees` for migration only.

## C10 — posterior-moment backward propagation

- Added the corrected engine alongside the untouched historical P10 value module.
- Propagated posterior mass and posterior-weighted return mass with uniform abstract-child averages.
- Conditionalized only at emission, with explicit zero-mass handling and fp32 working storage.
- Added O(K) total-minus-child sibling baselines and capture-before-bootstrap advantages.
- Kept Q allocation opt-in and emitted explicit child masks, counts, labels, and posterior rows.
- Streamed whole trees in fixed configured-budget chunks and rejected unsafe row-cap configurations.
- Added generic dense auxiliary channels with optional capture and every-player-node bootstrapping.
- Verified focused CPU behavior, historical P10 compatibility, and small CPU/CUDA parity.

## C11 — minimal RB-PPO training correction

- Made corrected posterior targets the sole public value API and removed default Q allocation.
- Unified sequential and fixed labeled rollout behind `sample_trees(bundle, cfg, gen)`.
- Aligned alias outcomes, targets, frozen policy caches, reach rows, and minibatches by decision row.
- Normalized synthetic and poker policy adapters over complete legal-event rows with duplicate gathers.
- Reduced the active composite to clipped direct-advantage policy, actor-value, and entropy terms.
- Added loss target-field declarations and fail-fast target availability checks before optimization.
- Replaced controller/reference/moment checkpoint state with explicit model, optimizer, scaler, config, and RNG state.
- Migrated the strict config and Kuhn smoke preset away from deferred Neyman, KL, and moment fields.

## P12 — read-only checkpoint evaluation

- Added exact synthetic checkpoint evaluation through the C11 bundle and P01 policy-table tools.
- Curves are sorted by stored iteration and return rows identical to their JSONL representation.
- Added CPU-engine poker arenas with exact sibling joint profiles and paired complement runouts.
- Duplicate replays rotate the singled-out button-relative seat and share seat/history uniforms.
- Pair-level aggregate A payoff drives bb/100 means, Student-t intervals, and per-seat reports.
- Fold rewards remain local; every fixed showdown crosses the stable evaluator facade.
- Arena reconstruction uses the P07 poker model bundle and policy adapter directly.

## P13A — GPU-native poker rollout backend

- Added a preallocated device SoA state store with tensor-cursor prefix allocation and full action histories.
- Implemented one Torch transition path for CPU and CUDA, including blinds, button rotation, affine raises, short all-ins, reopening, folds, chance deals and completion cuts.
- Kept fold settlement and public pot-stratum construction local while exposing device tensors for sibling completion.
- Added deterministic colex chance labels with street-disjoint offsets and padded dealt-card payloads.
- Routed one preflop queue and shared postflop queues by live-player count, lifting preflop classes to combo rows at the boundary.
- Partition and exact-profile inputs stay on the owning device and use the evaluator 0.3.0+ public adapters without removed arguments.
- Removed the policy adapter's device-scalar tick-width synchronization from the rollout event path.
- Random CPU trajectories match the independent `NLHEEngine`; focused CUDA residency passed on an idle RTX 2080 Ti.
- Batch histograms are retained as `preflop` and `postflop_<n_live>` dispatch records without CI throughput thresholds.

## P13B — corrected integration and first-run presets

- Registered synthetic and poker factories through the explicit importlib boundary.
- Added the poker-owned model/adapter/rollout/completion bundle with one owning model.
- Tensor-native completion calls the sibling facade directly for showdowns and uses
  sibling exact-profile sampling for collision-conditioned fold posteriors.
- Added strict Kuhn, Leduc, synthetic-dial, and six-player H100 presets with only
  fixed-intensity RB-PPO settings; DDP retains tree and batch values per rank.
- Poker `cpu`, `cuda`, and `auto` select matching rollout/completion residency.
- CPU sibling integration passed three train iterations and checkpoint roundtrip.
- CUDA tensor-residency passed on an idle RTX 2080 Ti; the real sibling CUDA smoke
  cannot JIT on this host because no CUDA toolkit/nvcc is installed (`CUDA_HOME` absent).
- H100 profiling uses the backend `preflop`/`postflop_<n_live>` histograms alongside
  `nvidia-smi dmon`; no fixed throughput threshold is imposed by CI.
