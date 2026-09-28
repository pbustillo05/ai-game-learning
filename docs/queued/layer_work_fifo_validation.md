# LayerWork FIFO identity repair

The planner previously assigned arena-zero leaf slots in compacted topology-edge
order, while production completion appended each output partition in Leaf FIFO
order. No payload permutation reconciled the two. Decisions consequently read
another trajectory's returns or posterior/moment tensors.

`build_layer_plan` now assigns initial slots in increasing Leaf-row order within
each replica and output partition. The movement columns retain the same schema
and describe these direct completion addresses. Last-consumer slot reuse,
ping-pong arenas, estimator arithmetic, and CUDA kernels are unchanged. There is
no additional GPU launch, payload copy, or completion scratch allocation. The
existing CUDA profiler regression still requires one propagation kernel per depth.

## Permanent CUDA regressions

- The layer fixture now writes completions in independently sorted FIFO order;
  it no longer silently repairs incorrect planner order.
- Twelve mode cases cover sampled/RB, sequential/fixed/KKT branching, and
  self-play/best-response. Each compares every published target column exactly
  under different leaf arrival orders, with two replicas, uneven trees, multiple
  roots, and mixed RB output partitions.
- Four real poker iteration cases drain eight leaves in packs of three and check
  the learner's gathered terminal values, advantages, and RB posteriors against
  the corresponding original completion payloads before optimizer updates.
- A negative control loaded only the original `build_layer_plan` from the parent
  revision into the test process. All 16 cases failed; all pass with the repair.

## Historical checkpoint replays (2026-09-28)

These used CUDA GPU 0 (RTX 2080 Ti), the shared poker venv and CUDA 13 toolkit,
and the existing diagnosis scripts with their `OUT` redirected to
`runs/experiments/nlhe_hu_32bb_rb_layer_identity_fix_20260928/`.
No experiment-only alignment patch was enabled and historical evidence files
were preserved.

| Replay | Result with production repair |
|---|---|
| Normal checkpoint 50, seed 472918, 128 roots | 0 slot mismatches; all 128 terminal learner values exact, maximum error 0 |
| RB checkpoint 1,100, seed 918273, 512 roots | 0/512 slot mismatches; decision 1,017 reads leaf 493 at its actual postflop slot 78 |
| Same RB decision | Entire 169-class posterior equals its original completion and independent evaluator replay exactly; class 55 mass is 0 |
| Four actual RB learner updates, same checkpoint and seed | 4/4 optimizer steps applied; finite gradients; approximate KL 0.00003008186 |
| Same four-update replay | Zero posterior mass on exactly-zero or below-floor old action probabilities; zero mass on ratios above $10^6$ |

Raw JSON files in that directory are `sampled_leaf_alignment_baseline.json`,
`posterior_leaf_audit_rb_i1100_seed918273.json`, and
`ratio_audit_rb_i1100_seed918273.json`. The word `baseline` is the existing script's
name for its unpatched production path, which now includes the production fix.
These checks establish correct trajectory identity and learner signal; they do
not establish long-run policy strength or optimal training hyperparameters.

## Broader validation

On GPU 0 with `CUDA_VISIBLE_DEVICES=0`, the following suite completed with
**231 passed, 1 skipped** in 115.14 seconds:

```bash
python -m pytest \
  tests/test_layered_targets.py \
  tests/test_sequential_modes.py \
  tests/test_sequential_sampled_training.py \
  tests/test_branched_sampled_modes.py \
  tests/test_branched_rb_modes.py \
  tests/test_iteration_architecture.py \
  tests/test_rollout_transactions.py \
  tests/test_replica_identity.py -q
```

The skipped case requires multiple visible CUDA devices (distributed NCCL).
Lint and `git diff --check` passed. All three physical GPUs returned to their
original idle display-only memory footprints of 9/9/19 MiB after testing.
