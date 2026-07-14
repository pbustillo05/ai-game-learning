# Capability and validation matrix

## Current capability matrix

| Capability | Current status | Evidence / blocker |
|---|---|---|
| Normal sampled-private terminal training | Present; CPU and tiny CUDA poker smokes pass | Resolved `AUD-001` |
| Sequential Rao--Blackwell training | Present for small runs | `beta_mode: sequential` + completion + `rb_ppo`; completion/full reaches compact before optimization |
| Fixed branched Rao--Blackwell training | Core estimator present | One tree/rank, duplicate children, unique-parent forwards, path law, streamed completion and `P/M` frontier |
| Variable/learned beta | Deliberately deferred | Allowed by user; no action required for v1 |
| Kuhn rules/exact analysis | Present | Exact tree/exploitability/completion and three-mode gradient gates |
| Leduc rules/exact analysis | Present as synthetic limit game | Exact completion and all three production training modes pass |
| Synthetic two-axis stress family | Present with collision-free public history | Card/deck/private/board support and action width/depth are the only dials; exact Kuhn/Leduc anchors (`AUD-017`) |
| Full 6-max 100bb NLHE rules | Present at fixed 0.5bb denomination | Both engines expose all 200 raise-to amounts and the factored mixture samples the exact legal subset (`AUD-011`) |
| Full MHA model | Binding paths, evaluator-owned folder DTO, event cache, and exact bounded-memory attention present | `AUD-006`–`AUD-010` resolved; H100 ceilings remain `AUD-005` |
| Poker CPU completion/evaluator boundary | Small smoke passes | Real `pbg-v2` CPU tests pass |
| Poker reach recomputation | Correct for small CPU rollouts | Exact `full`/`recompute` parity through action, fold, flop, turn, river, and duplicate branched children (`AUD-003`) |
| Shipped H100 training | Exact bounded-memory attention, effective batching, and byte-budgeted completion/frontier reduction implemented; production H100 scale unverified | Quest 40/80 GiB ceilings remain to be measured (`AUD-005`); `AUD-013` is resolved |
| GPU validation in shared venv | Verified | Shared Python 3.10 venv has CUDA-enabled PyTorch; targeted CUDA parity and rollout tests pass on GPU 1 (`AUD-030`) |
| Checkpoint save/load | Present | Unit tests pass |
| End-to-end resume | Present; strict interrupted/uninterrupted parity passes | Resolved `AUD-028` |

## Validation performed during this audit

### Repository test suite

Command:

```text
/home/pablob/.venvs/poker/bin/python -m pytest -q
```

Result:

```text
188 passed, 7 skipped in 85.02s
```

This establishes current unit/contract behavior only; it does not override the capability gaps above.

### Recompute verification

A deterministic 2-player CPU rollout was evaluated in both sequential and
fixed-branching modes under `reach_storage: full` and `recompute`. The forced
check/call line reaches flop, turn, and river; every stored node's reach is
bitwise equal between the two storage modes. A separate forced-fold line has
the same parity. The fixed-branching case retains duplicate public child
labels. This verifies `AUD-003`; the separate allocation work is recorded as
resolved in `AUD-004`.

### Static memory sizing

The old `256 * 8192` configuration allocated 62.16 GiB for its recompute
reach arena and 19.15 GiB for copied action histories. The current preset uses
`8 * 512 = 4,096` nodes. Its conservative rollout plan is:

```text
state/tree/cache bounds  = included in plan
two reach frontiers      = included at full 4,096-node capacity
estimated rollout peak   = 305,265,732 bytes (about 291 MiB)
```

CUDA construction checks this plan against half of currently free memory
before backend allocation. Model, completion, optimizer, and attention
workspaces remain outside this rollout estimate.

For a lower-bound flop token count of 7,063, `B=128`, and six attention heads,
one explicit fp32 attention matrix would be approximately 142.7 GiB. This is
the historical failure mode removed by fused/chunked execution; effective 128
is accumulated from hardware-budgeted physical batches.

### GPU hygiene

`nvidia-smi` was checked before CUDA work. GPU 1 was idle (9 MiB used, no
compute process) and was selected explicitly with `CUDA_VISIBLE_DEVICES=1`.
The AUD-026 synchronization-debug gate and three targeted CUDA
preflop/postflop/rollout tests completed successfully.
A final GPU check confirmed no audit process or additional memory remained.

## Required correctness gates

### Cross-mode small-game gates

1. Run normal, sequential RB, and branched RB on identical frozen Kuhn/Leduc policies. **Passed with enumerative completion.**
2. Compare empirical policy-gradient means against exact enumerated gradients. **Passed for all three modes against the exhaustive Kuhn root oracle.**
3. Verify normal and RB agree in expectation while RB has reduced private-state variance.
4. Verify fixed beta one matches sequential RB in public-path law and target expectation. **Passed via terminal-law and exact-gradient gates.**
5. Verify duplicate-labeled branches remain independent and are weighted as separate children. **Passed.**
6. Test lambda returns with zero/one/multiple downstream same-player decisions.
7. Test every mode at lambda 0, intermediate lambda, and 1.

### Storage/lifetime gates

1. Assert normal targets contain no posterior/completion/tree tensors.
2. Assert sequential RB stores one posterior vector rather than duplicate parent/child copies.
3. Assert leaf reaches/completion results and the frozen model are released before gradient activations. **Passed with weak-reference lifetime checks.**
4. Assert folded full ranges are unavailable after their documented last consumer. **Passed for poker decision caches in all three modes.**
5. Measure peak bytes for each phase and reject configurations before exceeding budget.
6. Verify chunked and unchunked small runs are numerically identical.

### Poker public-engine gates

1. Differential-fuzz list-based and tensor engines over random legal histories for 2 through 9 players. **Passed.**
2. Cover heads-up and multiway blind/action order, every street, folds, checks, calls, full raises, short all-ins, cumulative reopening policy, side pots, uncalled excess, ties, and button rotation.
3. Decide/enforce integer chip denomination or explicitly test the chosen action abstraction.
4. Verify two histories with different public action sequences produce distinguishable model inputs.
5. Verify normal mode samples final boards and settles sampled terminal profiles without completion MC.

### Reach/replay gates

1. Compare `full`, `decisions_only`, and repaired replay reaches at every decision and leaf.
2. Include flop/turn/river chance payloads and duplicate branch labels.
3. Verify exact board-conflict masks after replay.
4. Verify fold-time ranges stay frozen except later public-card compatibility.
5. Run the shipped storage mode in the real sibling CPU smoke; do not substitute `full` in that gate.

### MHA conformance gates

1. Lossless public-history tokenization. **Passed.**
2. Reach prior added to key logits untempered and without per-layer host sync.
3. Exact incidence residual against brute-force small tensors, including identical-combo channel. **Passed.**
4. Preflop orbit geometry and expected overlap features. **Passed.**
5. No-folder folded correction is exactly neutral. **Passed.**
6. Unary `Lambda_B` and signed-beta features match evaluator-returned finite differences. **Passed.**
7. Fold/street cache invalidation occurs exactly when dependencies change. **Passed.**
8. Fold line/card-pressure inputs change outputs and are suit-equivariant. **Input/output path passed; the optional suit-consistency loss is explicitly deferred and is not an accepted training control.**
9. Compact/fixed row outputs and gradients agree.
10. Attention implementation has a measured feasible batch ceiling and no explicit quadratic fp32 matrix at production sizes.

### Evaluator gates

1. Real `pbg-v2` CPU and CUDA completion parity within estimator tolerance.
2. Per-leaf `n_eff`, validity, and standard-error policy is logged/enforced. **Passed for active training: invalid posterior mass is rejected and scalar quality summaries are logged.**
3. Minimal-return mode does not retain unused equity tensors. **Passed via immediate learning-only collapse.**
4. Actor-marginal event sampling, if added, matches joint-profile action marginals.
5. Folder-only partition calls match small exact finite-difference oracles supplied by the sibling repository.

## Required performance gates

Implemented by `pbg-benchmark` and the device-local baseline comparison in
`benchmarks/`. Profiling is opt-in, uses explicit CUDA phase boundaries, and
ordinary training does not incur synchronization overhead. The committed
workload matrix defines Kuhn, Leduc, the third dial point, tiny/medium local
NLHE, and the readiness-gated production H100 chunk.

For every mode/game/device combination, record:

- realized trajectories (sequential) or the one tree (branched), nodes, leaves,
  decisions, and offspring by stage;
- rollout/model/evaluator/completion/backward/minibatch/optimizer wall time;
- peak allocated and reserved device memory by phase;
- bytes retained per decision and per leaf;
- decisions/second and completed trajectories/second;
- completion effective samples/second and error estimates;
- host-device synchronization count in hot loops;
- GPU utilization and clean pre/post task state.

At minimum, benchmark:

- Kuhn and Leduc on CPU as correctness baselines;
- three or more defined dial/spectrum points;
- heads-up short-stack NLHE as a boundary smoke;
- 6-max 100bb at tiny, medium, and intended production chunks after memory/model redesign.

## Recommended implementation order

1. Correct the TeX lambda exponent and freeze explicit three-mode/config contracts.
2. Implement normal terminal training with a scalar flat dataset and exact Kuhn/Leduc tests.
3. Specialize sequential RB storage and add cross-mode exact-gradient tests.
4. Repair poker chance/reach replay or replace it with checkpointed compact snapshots.
5. Retain byte-budgeted completion/frontier reduction and the single-tree
   rollout preflight; reject abstract row caps and branched root forests. **Completed.**
6. Remove redundant branch model forwards and dead completion/partition outputs. **Completed.**
7. Resolve full-NLHE action-space semantics.
8. Extend the evaluator folder-summary boundary and implement MHA conformance features.
9. Measure Quest H100 40/80 GiB physical-batch ceilings for the exact fused operator.
10. Define the synthetic/Hold'em spectrum and shared model policy.
11. Stage tiny-to-production GPU validation with memory guards (the shared CUDA environment is now verified; `AUD-030` is resolved).
12. Add continuous correctness/performance regression gates. **Completed for
    repeatable same-device baselines; end-to-end strict resume is also complete.**
