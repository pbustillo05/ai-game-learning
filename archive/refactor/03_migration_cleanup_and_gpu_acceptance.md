# Prompt 3 — finish migration, remove superseded memory paths, and earn the GPU green light

You are working in `/home/pablob/ai-game-learning`. Treat this as one large hardening and optimization task sized for roughly a 250k-token context window. Do not merely review the preceding refactor: make every necessary fix, run conservative single- and multi-GPU acceptance tests, document evidence, and git commit the final production-ready state.

## Objective

Finish the migration to the new pseudocode design, remove superseded production paths and compatibility allocations, validate checkpoint/config behavior, and tune measured bottlenecks so the normal and sequential-RB modes are safe for stable long runs and process more sampled player nodes per second with materially cleaner memory ownership. Branched RB remains a correct fixed-intensity experimental mode; do not attempt the general adaptive-beta research problem.

This is phase 3 of three. In a fresh context, first inspect `git log`, `git status`, and the code produced by phases 1 and 2. Verify that the new sealed storage/targets/root controller and bounded learner exist. If an earlier requirement is incomplete, fix it here rather than preserving a broken compatibility path.

## Authority and read order

1. Read `AGENTS.md` completely and obey its shared-GPU and sibling-repository rules.
2. Read all five implementation-design documents in `docs/theory/pseudocode/`:
   - `family_scheduled_rollout.tex`, especially Algorithms `alg:partitioned-queue-rollout`, `alg:seal-rollout-storage`, `alg:reserve-logical-children`, `alg:materialize-reserved-children`, and `alg:chance-close` at lines 41–319;
   - `sequential_memory_control.tex`, Algorithms `alg:sequential-memory-root-width` and `alg:observe-sequential-memory` at lines 34–127;
   - `backward_targets.tex`, all four algorithms at lines 36–187;
   - `distributed_update.tex`, Algorithms `alg:distributed-fixed-rollout-update` and `alg:mode-loss-map` at lines 38–249;
   - `storage_lifecycle.tex`, especially exact/natural domains at lines 34–167, ownership/lifetimes at lines 169–368, byte/control/failure formulas at lines 370–534, the HU 32bb estimate at lines 536–597, and acceptance gates at lines 599–642.
3. Use `docs/theory/core_math.tex` as estimator authority:
   - primitive v1 losses in `eq:primitive-losses`, lines 264–330;
   - normal decision law and player-clock return, lines 345–412;
   - sequential RB law/completion/return, lines 553–710;
   - branched law and independence warning, lines 748–949;
   - branched bounded-moment equations and fixed-v1 no-Q statement, lines 1200–1329;
   - production-v1 boundary, lines 1437–1446.
4. Read `docs/theory/MHA_model.md`, `docs/theory/folder_impact.md`, and the sibling read-only `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md` before changing poker integration.
5. Read current readiness/history in `docs/audit/`, especially the sequential-RB OOM/crash audit and any benchmark evidence. Treat historical numbers as comparison context, not an excuse to skip same-revision measurements.

## Required work

### 1. End-to-end ownership and dead-path audit

Trace one iteration in each mode from root selection through store release. Use code and measured tensor-owner inventories, not only names.

- There must be exactly one owner for every retained tensor field. Delete zero-copy compatibility wrappers only if they are unnecessary; delete any compatibility layer that clones or keeps old trees/reaches/targets alive.
- Remove production reach replay, root/all-node reach caches, recursive `recompute`, full-capacity `[N,P,H]` decision reach buffers, padded maximum-action target axes, duplicated parent posterior rows, separate retained V/A/return algebraic duplicates, and mandatory rollout-model clones.
- Remove retry/reroll-on-memory-pressure loops and retry-dependent RNG keys. Failures must happen through pre-allocation transactions and clean checkpoint/stop behavior.
- Remove dead Q-target/qHat, W/S target/loss, NeuRD, RNaD, adaptive-beta, and player-node-root-target configuration paths from production v1. Research prose may remain; runtime allocations/dispatch may not.
- Keep useful current structures: immutable public state/summary sharing, completion-quality diagnostics, identity gates, logical RNG, queue metrics, finite-gradient gates, reference scheduling, and checkpoint integrity.
- Search for hidden `torch.cat`/clone/list accumulation in rollout, sealing, completion, target, cache build, minibatch, and gradient paths. Replace payload-proportional growth with exact destinations or bounded reusable staging. Do not mechanically ban small metadata lists.
- Verify every source release with alias-aware storage accounting and allocator/NVML observations. A Python `del` without last-reference release is not evidence.

### 2. Poker integration and strict shape audit

Run full-deck `R=13` shape assertions through rollout, old-policy storage, targets, reference cache/recompute, current learner forward, and loss:

- dense preflop model input/output/cache width is always 169;
- dense postflop model input/output/cache width is always 1326, with board-blocked zeros allowed;
- RB posterior/lambda-return/advantage exact domain is canonical 1326 for both;
- normal retained old/value/log-probability slices may be scalar but never become a fake 1326 preflop row;
- no street-specific postflop partition or structural-zero packing has slipped in;
- folded summaries are immutable dependency anchors, refreshed on folds/street changes, and absent for `A == P`;
- completion inputs match the sibling `api_pbg.md` dtypes/layouts exactly and all posterior/equity simulation still goes through that sibling backend.

Test multiple active-player counts and at least `R=4` plus `R=13`; verify the canonical quotient and impossible-hand zero convention do not depend on one deck size.

### 3. Config and checkpoint migration

Make configuration and checkpoint behavior explicit and boring.

- Use unambiguous per-GPU initial/min/max sequential-root fields. Remove or reject obsolete global `sequential_starts`, target-decision root controls, retry settings, and reach-storage modes unless an explicit one-time migration maps them with a logged version change. Never silently reinterpret old YAML.
- Update all repository profiles/examples/tests and user-facing training documentation.
- Version storage schema, controller state, and checkpoint metadata. Support loading a recent legacy checkpoint only through a deterministic migration that does not resurrect legacy runtime stores. Reject incompatible partial state with a useful error before GPU allocation.
- Verify controller state, optimizer/reference schedule, RNG iteration/address state, and loss scale round-trip. A resumed run must choose the same next width and sample the same logical objects as an uninterrupted run.
- Do not commit generated run checkpoints or large benchmark artifacts.

### 4. Memory-controller and OOM-safety soak

Validate the exact design in `sequential_memory_control.tex` rather than just checking that a number changes.

- Synthetic tests must force rollout-, seal-, target-, and optimization-limited cases; cold start; no-fit pre-draw stop; cross-rank minimum; immediate downward step; delayed and rate-capped upward step; positive residual widening; delayed envelope tightening; checkpoint resume; external-memory ceiling contraction; cache exclusion; and singleton failure.
- On real GPUs, run a bounded multi-iteration normal soak and a sequential-RB soak with enough iterations to observe steady allocator behavior and at least controller observation/checkpoint transitions. Start tiny and increase only after the measured four-phase ledger proves headroom.
- Log per iteration: selected/safe width, limiting phase/field, predicted phase peaks, logical counts and allocated capacities, source/destination seal bytes, allocated/reserved/NVML high water, guard, reference-cache decision, physical shapes/tails, and spots/player-decisions per second.
- Reserved/active memory must not show unexplained monotone growth after warm-up. Controller width and forecasts must remain finite and deterministic across ranks. The run must not rely on `empty_cache()` as ownership management.
- Do not intentionally OOM. Use synthetic ceilings to test rejection. Any real CUDA OOM is a test failure requiring a smaller staged workload and root-cause fix, not repeated probing.

### 5. Throughput profiling and focused optimization

The redesign's purpose includes more sampled player nodes per second. Measure end-to-end, then optimize only evidence-backed hot paths.

- Establish a fixed-seed, fixed-model, fixed game/config benchmark for normal and sequential RB. Report rollout decisions/s, completion rows/s, target decisions/s, learner sampled decisions/s, total spots/s, model rows, launch counts, physical-tail fraction, cache build amortization, and phase wall times.
- Where a reproducible legacy baseline exists in audits or recorded prior-commit metrics, compare equal hardware, sampled-object semantics, precision, completion budget, and loss. Clearly distinguish measured same-machine comparisons from historical inference.
- Profile for remaining host synchronizations, excessive tiny launches, repeated gathers, allocator churn, over-conservative chunk sizes, poor queue fill, repeated reference forwards, and model padding. Fix the largest end-to-end bottleneck while preserving RNG/topology and estimator equality.
- Tune only physical execution knobs: queue thresholds, active frontier, chunk size, warmed subspan shapes, completion microbatch, and reference-cache choice. Do not change statistical `B_m`, root sampling law, lambda target, posterior domain, or loss just to improve a benchmark.
- A throughput gain is not accepted if it reduces sampled-object equality, completion quality, safe root capacity, or memory guard. Conversely, do not chase kernel occupancy that lowers end-to-end spots/s.
- Require stable repeated measurements (warm-up plus multiple timed repetitions). Investigate a material regression before green-lighting; do not hide it behind aggregate averages.

### 6. Distributed correctness and failure behavior

- Run unequal-local-mass tests and verify log-mass normalization, arbitrary integer stratum counts, SUM gradient semantics, and identical parameters/optimizer/loss scale on all ranks.
- Exercise empty local partitions, one-row physical tails, repeated sampled indices, cache versus recompute, AMP finite and injected-nonfinite gates, and checkpoint/clean stop at collective-safe boundaries.
- Ensure every rank takes collectives in the same order even when local strata/caches differ. No rank may wait forever after another rank detects a clean memory failure; implement an explicit common stop decision at safe boundaries.
- Branched mode must still produce exactly one tree per rank, root row zero with ordinary multiplicity, contiguous child/action spans, fixed beta, exact weighted decision law, and no Q target. Keep its GPU smoke tiny.

## Test procedure

Use the repository virtual environment and stage risk carefully.

1. Run focused CPU/unit tests, then the full feasible CPU suite.
2. Run `nvidia-smi` before CUDA work. If sandbox access fails, request escalation. Select only idle GPUs and make `CUDA_VISIBLE_DEVICES` explicit. Never touch another user's process.
3. Use `/home/pablob/.venvs/poker` or `./tools/poker-python`; do not install system packages and do not modify the sibling evaluator.
4. Precompute the structural byte plan and begin with tiny one-GPU normal smoke, then sequential RB, then fixed-intensity branched. Inspect GPU state and memory between modes.
5. Run one-GPU multi-iteration normal and sequential-RB soaks with checkpoints/resume. Increase workload conservatively only while predicted and measured headroom remain healthy.
6. If at least two GPUs are idle, run a two-GPU normal and sequential-RB distributed test plus a tiny branched smoke. Include unequal local stratum/partition coverage.
7. After all GPU work, synchronize/exit cleanly, run `nvidia-smi`, and verify your processes and allocations are gone.

Compile any changed TeX to `/tmp`, never into the source directory. Do not add `.aux`, `.log`, or generated PDFs to the commit.

## Final acceptance report and commit

Write a concise current audit under `docs/audit/` containing:

- exact commit-under-test and hardware/GPU ids;
- commands and configurations;
- correctness results for all three modes;
- four-phase predicted versus allocated/reserved/NVML peaks and guards;
- retained-store bytes by schema/partition, including 169/1326 assertions;
- controller soak trace summary and resume equality;
- reference cache build/training choices;
- end-to-end phase timing and sampled-player-nodes/second comparisons;
- any remaining limitation. The only acceptable major deferred item is the general adaptive branching-beta research problem.

Update `docs/training_and_evaluation.md`, file references, config examples, and current readiness notes to describe the implemented state rather than the old architecture. Do not rewrite historical audits.

Run final searches proving no superseded production path remains, inspect the complete diff, and commit only your changes with a message such as `refactor(core): complete memory-safe training pipeline`. The commit must include code, tests, migrations, and the audit evidence; do not leave work staged or uncommitted.

End your response with:

- the commit hash;
- an explicit green light or a specific blocker;
- CPU and GPU test commands/results;
- GPU ids and final idle-state confirmation;
- peak memory/headroom for all four phases;
- normal and sequential-RB sampled player nodes/s before/after where comparable;
- confirmation of preflop 169, postflop 1326, no reach replay, no reroll, and no stored Q target.
