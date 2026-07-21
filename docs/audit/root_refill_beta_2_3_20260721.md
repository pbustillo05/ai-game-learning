# Protected root-refill validation — 2026-07-21

The protected-cohort lower-size safeguard was validated against read-only
early, middle, and final checkpoints from
`runs/nlhe_hu_32bb_full_sequential_rb` on idle RTX 2080 Ti GPUs. Runs used fixed $\beta=2.3$,
`branching_min_nodes: 1000`, `node_budget: 12000`, and the existing
`branching_upper_cutoff_nodes: 4000` pressure latch.

| Checkpoint | Pre-refill nodes | Refills | Final nodes | Root degree | Peak allocated | Root-weight max error |
|---:|---:|---:|---:|---:|---:|---:|
| 37 | 3 | 2 | 2,992 | $2\to4$ | 1.08 GiB | 0 |
| 555 | 4 | 4 | 4,840 | $3\to7$ | 1.11 GiB | $6.39\times10^{-9}$ |
| 1,110 | 3 | 1 | 5,435 | $2\to3$ | 1.12 GiB | $9.94\times10^{-9}$ |
| 1,110 (ordinary large seed) | 4,279 | 0 | 4,279 | $3\to3$ | 3.50 GiB | $9.94\times10^{-9}$ |

The tiny-mode seeds exercise one, two, and four loop iterations. Every finalized
tree passed `TreeBatch.validate()`, exceeded the 1,000-node floor, retained its
ordinary root-child prefix, and assigned uniform weight to every final root
edge. The ordinary large-mode seed exercises the no-op path.

Checkpoint 37 was additionally run end to end through CUDA completion and the
streamed target reducer. Its refilled tree contained 1,208 decision nodes; the
root target exposed exactly the two protected children (both in `child_count`
and `child_mask`) and retained a valid root trace target. Peak allocation for
rollout, completion, and reduction together was 1.08 GiB.

Focused verification:

- 81 passed across the branching, adaptive KKT, streamed-value,
  posterior-moment, and GPU-rollout modules with CUDA available.
- 74 additional config, topology, loss, memory-control, and diagnostics tests
  passed. One unrelated existing assertion failed because the checked-in
  sequential-RB config uses 2,560 roots while that test expects 2,048.
- The real-poker checkpoint completion/reduction path passed separately.

All three GPUs were checked after the runs and returned to display-only idle
usage with no compute process.
