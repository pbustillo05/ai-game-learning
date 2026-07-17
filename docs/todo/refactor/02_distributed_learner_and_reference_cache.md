# Prompt 2 — implement the bounded distributed learner and transactional reference caching

You are working in `/home/pablob/ai-game-learning`. Treat this as one large implementation task sized for roughly a 250k-token context window. Work through code, tests, conservative GPU validation, and a git commit; do not stop at analysis or a plan.

## Objective

Complete the new implementation by making the immutable partitioned `TrainingStorage` from phase 1 the only learner input. Implement the stratified distributed update, production-v1 storage-to-loss map, index-only statistical sampling, bounded plan–gather–forward/backward–release execution, hard reference-cache build admission, and direct flat SUM gradient storage. Reuse the current loss, distributed, optimizer, observability, and model code where it already has the right semantics. Do not rewrite all of `pbg/core/`.

This is phase 2 of a three-phase refactor. Before editing, verify that phase 1's schema-owned sealed stores, streaming targets, and memory-only root controller are present. If the repository lacks them, identify the missing prerequisite and implement only the smallest prerequisite needed; do not build this learner on legacy reach replay or duplicate dense stores.

## Required reading and authority

1. Read `AGENTS.md`, inspect `git status`, and preserve unrelated changes.
2. Read `docs/theory/pseudocode/distributed_update.tex` completely. The exact implementation targets are:
   - production-v1 scope and exact distributed decision law, Algorithm `alg:distributed-fixed-rollout-update`, lines 38–111;
   - asynchronous mass exchange and transactional reference-cache selection/build, lines 113–147;
   - arbitrary statistical counts, index-only draws, physical subspans, direct gradient views, and SUM reduction, lines 149–213;
   - exact storage-to-math map, Algorithm `alg:mode-loss-map`, lines 215–249.
3. Read the relevant storage authority in `docs/theory/pseudocode/storage_lifecycle.tex`:
   - v1 scope and exact/model-row domain map, lines 34–70;
   - poker 169/1326 and canonical exact-hand contracts, lines 99–167;
   - retained fields and sampling metadata, lines 211–276;
   - phase lifetimes and forbidden old patterns, lines 308–368;
   - compact-store, cache-build, training-transient, four-phase byte formulas, and common failure agreement, lines 370–534;
   - learner acceptance tests at lines 619–642.
4. Read `docs/theory/pseudocode/backward_targets.tex`, especially final `TrainingStorage` postconditions at lines 170–187. Read `docs/theory/pseudocode/sequential_memory_control.tex` lines 49–64: root feasibility assumes the uncached learner and optional caches cannot invalidate it.
5. Mathematical authority is `docs/theory/core_math.tex`:
   - no model Q-head and synchronous-iterate assumptions, lines 250–262;
   - production primitive PPO/value/value-clip/entropy/reference-KL formulas in `eq:primitive-losses`, lines 264–330. NeuRD and RNaD shown nearby are explicitly out of scope;
   - normal player-clock target at `eq:sequential-lambda-return`, lines 400–412;
   - exact Rao–Blackwell loss identity/domain and target at `eq:rao-blackwell-loss-identity` and `eq:rao-blackwell-lambda-return`, lines 628–658;
   - branched conditional decision law/targets, lines 809–949, and backward moments at `eq:branched-backward-updates`, lines 1208–1322;
   - fixed-intensity production-v1 loss boundary, lines 1321–1329 and 1437–1446.
6. Inspect and reuse the useful current implementation:
   - `pbg/core/losses.py` PPO/RB PPO, value clipping, entropy, legal-action KL, and metrics;
   - `pbg/core/distributed.py` mass/objective helpers;
   - `pbg/core/optimization.py` finite gate and SUM collective semantics, while replacing copy-flatten behavior with direct views;
   - `pbg/core/training.py` orchestration, alias sampling, async collectives, diagnostics, checkpoint/reference scheduling, and any correct adaptive optimizer-work logic;
   - `pbg/core/minibatch.py` only where its gather behavior can become bounded physical-subspan gathering;
   - poker adapter/model outputs and natural sufficient reference outputs.

## Required implementation

### 1. One immutable learner input and exact strata

- `DistributedFixedRolloutUpdate` accepts only GPU-local immutable `TrainingStorage` plus opaque `PartitionSpec`s. Assert at the boundary that no topology, leaf reach/completion payload, rollout queue/state, mutable occurrence, replay object, or target scratch survives.
- Each nonempty local partition has fp64 `log Z_{g,m}` and either an implicit sequential uniform sampler or a branched Walker alias sampler. Keep all normalization in log space until computing `alpha_{g,m}`.
- Begin the common-order asynchronous all-gather/reduction of local log masses as soon as a rank's target builder finishes. Do not add an unnecessary preceding barrier. Handle empty local strata without changing collective order.
- The estimator must equal lines 203–209 of `distributed_update.tex`: sample decisions from the normalized local stratum law, multiply each stratum mean by exact global `alpha`, and SUM gradients across ranks. Never multiply a sampled loss by the path weight again and never introduce DDP equal-rank averaging.
- Statistical `B_{g,m}` counts are arbitrary positive integers determined by measured time/cheap lagged variance allocation. Do not round them to 32 or any physical shape. Physical shapes are a separate planner concern.

### 2. Exact production-v1 ModeLoss only

Implement one explicit storage-to-math map corresponding to `alg:mode-loss-map`.

- Supported losses are clipped PPO policy, actor value regression with optional value clipping, entropy, and exact legal-action KL to an optional frozen reference.
- Remove production configuration, dispatch, retained fields, and allocations for stored Q targets/Q-losses, NeuRD, RNaD, W/S auxiliary losses/heads as training consumers, and adaptive branching intensity. Structural model compatibility may load unused legacy head weights only through an explicit checkpoint migration, never allocate targets for them.
- Normal uses scalar `lambdaReturn`, scalar old value/log probability at the realized type/event, and derives value/advantage at loss time.
- Sequential RB reads one shared exact-hand posterior through `posteriorRef`, exact-hand `lambdaReturn`, and natural old model rows through `phi_m`; posterior weighting stays inside the nonlinear loss.
- Branched RB derives the node posterior as the mean of child posterior rows, evaluates node value/entropy/KL terms once, and averages sampled-edge policy losses using `childPosterior * advantageHat`. There is no `qHat` argument.
- Current, old, and reference outputs are read directly at `phi_m(x)`. Do not store an exact-domain duplicate of a natural model output. Preflop dense forwards/caches remain width `K=169` for `R=13`; postflop remain `H=1326`; exact RB posterior/target loops remain `H=1326` in both.
- Preserve exact legality masks and the direction `KL(current || reference)` from `core_math.tex`. Add cache/recompute equality tests for all coefficients and value-clipping choices.

### 3. Hard-bounded physical execution

Separate statistical sampling from physical execution.

- Allocate one reusable int32 index buffer sized by configured time-budget caps. Draw only indices for one stratum. Do not gather all `B_{g,m}` rows before planning.
- Implement/calibrate `PhysicalPlan_{g,m}(I, A; C)` over a small finite warmed set of legal physical shapes plus an exact tail. It preserves index-vector order and never sorts by poker street, board, owner, or hidden shape.
- Its hard predictor includes only simultaneously live input/target/posterior/action/cache gathers, uncached reference output, current activations, exact loss reductions, AMP temporaries, and backend workspace. Use allocator-reserved high-water increments, not only tensor `numel`.
- Fail cleanly before allocation if even one decision cannot fit. Such failure checkpoints/stops and supplies conservative controller evidence; it does not retry/reroll the rollout.
- For each subspan: gather only aligned rows, compute optional no-grad reference, run current forward and exact loss, immediately backward the correctly normalized contribution, update only cheap scalar diagnostics, and release graph/gathers/reference/reductions/staging before the next subspan.
- No retained graph crosses subspans. Do not normalize each physical subspan independently. For a fixed statistical index vector, every legal split including a one-row tail must match the unsplit objective/gradient within the declared precision tolerance.
- Activation checkpointing is off by default. Admit it only as a separately measured physical-shape variant if it improves end-to-end spots/second or is required for a singleton, and include recomputation in measured time.

### 4. Reference cache is an optional hard transaction

Implement lines 120–143 of `distributed_update.tex` literally enough to make both construction and training safe.

- The no-cache singleton path is the required fallback and must be proven feasible first.
- For every partition, compute exact retained natural-output bytes and measured total learner time for candidate cache sets, including cache build and any physical-plan shrinkage. The poker partition count is small enough for a finite exact subset search; otherwise use a deterministic finite solver with the same result.
- Candidate feasibility charges the full final resident cache destination plus the maximum of calibrated cache-build and training subspan transients. Define and measure a separate `W_refbuild`; do not assume inference build is free because outputs will be retained.
- Always include the empty set. Minimize predicted total iteration learner time, not simply saved reference forward time or bytes/time ratio.
- Transactionally reserve all selected destination blocks before the first build forward. Build one bounded subspan at a time, scatter natural rows, and release staging immediately.
- If reservation or a calibrated build contract fails before any learner randomness, release every partial cache and select the empty set without changing targets. Do not use exception-driven CUDA OOM as the admission mechanism.
- Cache only natural sufficient reference outputs. For full-deck poker, preflop cache rows are 169 wide and postflop rows are 1326 wide. Add hard assertions preventing preflop 1326 expansion.
- Cache selection and observed build/train peaks must be visible in the phase ledger and long-run metrics.

### 5. Direct flat SUM gradient storage and finite gate

Refactor the useful `pbg/core/optimization.py` machinery so there is one fp32 flat SUM buffer per rank in fixed parameter order and every parameter's `.grad` is a view into it.

- Allocate/bind once for the update, zero once per optimizer step, and accumulate every subspan directly into those views. Remove per-step flatten/copy buffers and avoid DDP gradient buckets.
- All-reduce SUM exactly once in common order, with no division by world size. Derive the common nonfinite decision from the reduced buffer without a redundant collective, unscale once, clip globally once, and make the identical optimizer/loss-scale transition on all ranks.
- Preserve zero views for parameters unused by a given subspan/rank. Check aliasing and layout, optimizer compatibility, AMP scaling, nonfinite skip/backoff, and checkpoint reload.
- Ensure there is no extra frozen clone of `theta_k` during synchronous rollout/target/optimization. Parameters remain read-only until targets are fixed. The optional reference model is semantically separate and scheduled by optimizer step as before.

### 6. Orchestration, observability, and compatibility cleanup

- Replace `_IterationData`/minibatch paths that depend on trees, `ReachCache`, `DecisionReachCache`, padded global event axes, or full-minibatch gathers. Consumers should use partition/address views into `TrainingStorage`.
- Keep useful reference scheduling, cheap variance proxies, completion quality metrics, identity gates, checkpointing, and async collective structure.
- Keep optimizer-step/reuse control independent of root-width control. Decision counts may choose how often immutable data is reused, but may not alter the next root width; that is memory-only phase-1 state.
- Emit per-partition statistical counts, physical subspan histograms/tails, model/reference rows, cache build rows/time/bytes, activation variant, exact objective mass, flat-buffer bytes, and allocated/reserved phase peaks.
- Any temporary compatibility layer from phase 1 must be removed once the new learner passes. There must be one tensor owner for each retained field.

## Testing and proof obligations

Add direct tests for:

- local/global stratum mass, empty strata, log-domain stability, alias distribution, and conditional estimator equality;
- normal, sequential RB, and fixed-intensity branched ModeLoss against direct `core_math.tex` formulas, including `lambda=0,1`, zero posterior rows, value clipping, entropy, reference KL, and arbitrary legal masks;
- no Q/W/S/RNaD/NeuRD target allocation or learner dispatch;
- preflop 169/postflop 1326 dense outputs and caches with 1326 exact RB rows;
- cache set feasibility including build workspace, rollback to empty cache, and cache/recompute equality;
- arbitrary statistical counts, one-row tails, repeated indices, multiple physical splits, and split/unsplit objective plus gradient equality;
- direct `.grad` aliasing into one flat fp32 buffer, SUM semantics across unequal rank masses, AMP finite/nonfinite transitions, and optimizer checkpoint round trips;
- no full statistical gather, no host sync in hot loops, no duplicate training store, and correct release after every subspan/update.

Run focused CPU tests, then the full feasible suite. GPU validation is mandatory and must follow `AGENTS.md`:

1. Check `nvidia-smi`; request escalation if sandboxed. Use only idle GPUs, make `CUDA_VISIBLE_DEVICES` explicit, and never disturb other users.
2. Use `/home/pablob/.venvs/poker` or `./tools/poker-python`, with the sibling evaluator read-only.
3. Precompute the schema/ledger bound and begin with tiny one-GPU normal and sequential-RB updates, cache disabled. Then test cache build/recompute equality and a tiny branched update. If two GPUs are idle, run an unequal-local-stratum two-GPU test that proves exact SUM/global-mass behavior.
4. Never induce an OOM. Exercise singleton/cache failure with synthetic ceilings checked before allocation.
5. Capture reserved/allocated peaks for cache build and every training subspan, model/reference row counts, flat-gradient bytes, and spots/second. Verify GPU state returns to idle after the tests.

## Commit requirements

Before committing, inspect the diff and search the production path for `qHat`, Q-target flags, full-minibatch gathers, copy-flatten gradients, DDP averaging, reach replay, and preflop `H`-wide dense caches. Remove or explicitly quarantine dead research-only code so it cannot allocate in v1.

Update current configuration/examples and implementation/audit documentation. Record exact test commands, GPU ids, peak bytes, and any benchmark comparison. Commit only this phase's changes with a message such as `refactor(core): add bounded stratified distributed learner`. End with the commit hash, concise implementation summary, CPU/GPU results, memory/throughput measurements, and concrete remaining phase-3 items.
