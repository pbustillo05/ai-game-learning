# NLHE HU 32 bb beta-2.4 execution-capacity audit, 2026-07-28

## Answer

The guarded fixed-branched beta-2.4 workload is stable in the measured
32-seed sample with the execution manifest now written to
`configs/profiles/nlhe_hu_32bb_branching_profile_v1.yaml`.

The recommended settings are:

```yaml
execution:
  manifest:
    player_node_cap: {pre: 6804, post_2: 20000}
    trained_player_cap: {pre: 84, post_2: 20000}
    action_cap: {pre: 252, post: 35999}
    folded_range_cap: {pre: 252, post: 252}
    leaf_cap: 20000
    topology_cap: 36000
    topology_edge_cap: 35999
    family_pack_cap: 84
    chance_frontier_cap: 8192
    completion_pack_cap: 8192
    learner_sigma_pack_cap: 20
    learner_action_pack_cap: 60
    topology_packet_cap: 10080
    target_arena_cap: 56000
```

Keep `safe_tree_size: 25000.0` and ensure the execution path supplies a
non-null `CostSafeguardController`. The threshold is a soft frontier-cost
guard, not a 25,000-node topology cap. The largest guarded tree had 30,742
nodes.

Results:

- 32/32 guarded discovery seeds completed;
- all three final-manifest tail validations completed;
- maximum topology was 30,742 of 36,000 rows;
- maximum leaves were 14,719 of 20,000 rows;
- maximum postflop decisions were 14,582 of 20,000 rows;
- maximum postflop actions were 27,261 of 35,999 rows;
- family pack 84 had the lowest measured model-forward and wall time;
- completion pack 8,192 had the lowest measured completion CUDA
  milliseconds per player decision.

This is a recommendation for the guarded scientific workload. It is not an
unsuppressed beta-2.4 tree bound.

There is also a must-fix capacity defect in the current execution allocator:
both `LayerWork` output partitions are physically allocated to the complete
static `target_arena_cap`, even though the exact per-output arena extents are
known before the completion phase allocates them. The current 36k guarded
configuration fits despite that waste. Larger tree caps must not be selected
from the present static `LayerWork` accounting.

## Scope and method

The measured workload was:

- heads-up NLHE, 32 bb, all 13 ranks;
- self-play, `branched.fixed`, Rao--Blackwell private estimator;
- player and chance beta 2.4, with `beta_max: 3`;
- one root and one replica;
- FP16 model execution with SDPA;
- 1,024 completion samples per leaf;
- `minimum_tree_size: 1000`;
- a genuinely active `safe_tree_size: 25000.0`;
- rollout, completion, alignment, and target construction only.

The benchmark did not execute learner forward/backward or an optimizer step.
It did enter the declared learner and final-backward ownership phases so their
slab allocations were reconciled.

The frozen initialization checkpoint was:

```text
runs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723/
benchmark_init_ckpt_0001110_dummy_heads_seed91301_v8.pt
```

Its SHA-256 was:

```text
7ddaad84f691ee31336f4e852ca44e566596275ba05fcdf56cb527087009aa21
```

The 32 seeds were generated with:

```python
random.Random(20260728).randrange(1, 2**31)
```

repeated 32 times. They were assigned round-robin to physical GPUs 0, 1, and
2. Each process stayed on one selected GPU and reused its loaded model.

The three cards are RTX 2080 Ti devices. GPUs 0 and 1 report 11,346,575,360
CUDA-visible bytes, while GPU 2 reports 11,344,740,352 bytes. The manifest now
declares the lower value so the same configuration is valid on every card.

## Corrections found during preflight

Three old-audit assumptions were wrong and failed before useful benchmarking.

### `FamilyWork.pre` cannot be reduced to observed high-water

A 256-row `FamilyWork.pre` slab failed immediately:

```text
ValueError: FamilyWork.pre device pack exceeds destination capacity
```

The retained high-water is small, but a device publication stages the complete
bounded chance-closure envelope. With family pack 84 and `beta_max: 3`, that
envelope is:

$$
84 \times 3^4 = 6{,}804.
$$

Therefore `player_node_cap.pre: 6804` is a real execution contract under the
current implementation, not arbitrary overprovisioning.

### `folded_range_cap: 1` is invalid

A one-row folded-range cap also failed immediately:

```text
ValueError: FoldedRanges.pre device pack exceeds destination capacity
```

HU retained folded-range high-water remained zero, but the destination must
still cover its staged device pack. The valid pack envelope is:

$$
84 \times 3 = 252.
$$

Both folded-range partitions therefore remain at 252.

### A profile threshold alone does not activate the guard

`FixedBranchController` suppresses player branching only when it receives a
non-null `frontier_cost`. The queued poker attempt supplies that value only
when a `CostSafeguardController` object exists.

A scenario with `safe_tree_size: 25000.0` but
`cost_safeguard: null` was effectively unsuppressed and overflowed the
36,000-node topology on known tail seed `223741895`.

With an inactive-but-present safeguard state, the same seed completed at
30,742 nodes. The successful profile scenarios in the edited configuration
now explicitly contain:

```yaml
safe_tree_size: 25000.0
cost_safeguard:
  gamma_deviance_max: 1.0
  p95_underprediction_max: 2.0
  consecutive_iterations: 1
  consecutive_successes: 0
  active: false
  observations: 0
```

“Inactive” here means queue-pressure prediction is not yet calibrated; the
object still makes realized frontier cost available to the fixed controller's
25k safety threshold.

## Node distribution

![Distribution of total, preflop, postflop, and leaf nodes](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/node_distribution.png)

| Node class | Minimum | Median | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Total topology | 5,246 | 13,718.5 | 26,543.9 | 27,164.5 | 30,742 |
| Preflop player | 2 | 2 | 4 | 5.5 | 6 |
| Postflop player | 2,006 | 5,249.5 | 10,697.9 | 11,489.7 | 14,582 |
| Chance | 166 | 486.5 | 1,064.6 | 1,159.1 | 1,437 |
| Leaf | 3,071 | 8,013 | 14,650 | 14,691.3 | 14,719 |

The leaf distribution has a visible upper cluster near 14.7k because the 25k
frontier threshold suppresses further player branching after queued work has
already accumulated. It must not be interpreted as the unsuppressed beta-2.4
tail.

## Capacity recommendation

| Capacity | Selected | Observed maximum | Reason |
|---|---:|---:|---|
| `player_node_cap.pre` | 6,804 | 9 retained rows | Required staged envelope $84\times3^4$ |
| `player_node_cap.post_2` | 20,000 | 14,582 | 37.2% row margin |
| `trained_player_cap.pre` | 84 | 6 | One complete family pack |
| `trained_player_cap.post_2` | 20,000 | 14,582 | Covers every self-play decision |
| `action_cap.pre` | 252 | 14 | Exact one-pack action envelope $84\times3$ |
| `action_cap.post` | 35,999 | 27,261 | Bounded by the one-root topology edge cap |
| `folded_range_cap.pre/post` | 252 | 0 | Required staged destination envelope |
| `leaf_cap` | 20,000 | 14,719 | 35.9% row margin |
| `topology_cap` | 36,000 | 30,742 | 17.1% row margin |
| `topology_edge_cap` | 35,999 | 30,741 | Exact one-root relation $E=N-1$ |
| `target_arena_cap` | 56,000 | — | Current generic safety bound `leaf + topology`; must not remain the physical `LayerWork` extent |

For one root:

$$
E=N-1.
$$

The action slabs store player-emitted edges and therefore satisfy:

$$
A_{\mathrm{pre}}+A_{\mathrm{post}}\le E.
$$

This makes 35,999 a closed postflop action envelope once topology is capped at
36,000. The preflop partition separately uses the smaller exact pack
envelope.

The 36k topology margin is intentionally the narrowest selected margin. Raising
topology also raises the exact target arena and the dominant `LayerWork`
allocation. The 32-seed maximum and the three exact final-tail reruns all fit;
a different scientific safeguard or an unsuppressed run needs a new capacity
study.

## Slab allocation

The final manifest allocates 3,976,088,256 bytes (3.703 GiB) across all unique
CUDA slabs over their complete lifetimes. These allocations are not all live
at once. The largest simultaneously live named-slab total is 3,593,615,424
bytes (3.347 GiB) in the target phase.

Largest slabs in the final manifest:

| Slab | Capacity | Tail-validation high-water | Allocated MiB |
|---|---:|---:|---:|
| `LayerWork.post` | 56,000 | 14,719 | 2,267.822 |
| `ActionTarget.post` | 35,999 | 27,261 | 364.187 |
| `LayerWork.pre` | 56,000 | 3 | 290.527 |
| `Leaf` | 20,000 | 14,719 | 203.037 |
| `ActionHead.post` | 35,999 | 27,261 | 182.231 |
| `FamilyWork.post_2` | 20,000 | 14,582 | 119.591 |
| `LearnerIndex.post_2` | 20,000 | 14,582 | 118.637 |
| `SigmaHead.post_2` | 20,000 | 14,582 | 101.318 |
| `SigmaTarget.post_2` | 20,000 | 14,582 | 101.318 |
| `FamilyWork.pre` | 6,804 | 4 | 40.686 |
| All smaller slabs | — | — | 13.942 |

### Required fix: allocate `LayerWork` to the realized layer plan

The shared 56,000-row `LayerWork` allocation is a capacity defect that must be
fixed before materially increasing the tree caps. It is not merely an optional
schema optimization.

The current sequence is:

1. rollout and topology construction finish;
2. `build_layer_plan(...)` assigns the exact ping-pong addresses and returns
   `plan.arena_rows[replica][output]`;
3. only then does `enter_phase(COMPLETION)` allocate `LayerWork`;
4. despite having the exact extents, `GeneratedOwners` uses the immutable
   pre-rollout storage plan and gives every `LayerWork` output partition the
   entire scalar `target_arena_cap`.

The exact physical capacity for an output partition must instead be:

$$
\max_r \operatorname{arena\_rows}[r,\text{output}],
$$

where the maximum is over replicas. This is not always identical to the leaf
count: `arena_rows` also accounts for concurrently live internal-node target
slots and their ping-pong reuse. Under the deliberately all-postflop
single-output envelope below, however, the initial dense leaf placement makes
the exact postflop extent $L$, and the ping-pong planner does not require a
larger extent.

The hard manifest `target_arena_cap` should remain as an overflow/admission
ceiling. It should no longer be the number of rows physically allocated.
After `build_layer_plan`, execution must:

- validate every exact extent against the hard ceiling;
- calculate a runtime-exact completion/target admission;
- allocate each `LayerWork` partition once at its exact extent;
- retain the resulting fixed slab through the target phase without resizing.

This preserves immutable addresses and monotone publication after allocation.
It changes only the late phase's capacity derivation.

The measured waste is material:

| `LayerWork` allocation | Current physical rows | Exact tail-plan rows | Current bytes | Exact allocated bytes |
|---|---:|---:|---:|---:|
| `pre` | 56,000 | 3 | 304,640,064 | 16,384 |
| `post` | 56,000 | 14,719 | 2,377,984,064 | 625,027,680 |
| **Total** | — | — | **2,682,624,128** | **625,044,064** |

The current run therefore reserves about 2.058 GB (1.916 GiB) more physical
`LayerWork` storage than its exact layer plan needs. In the earlier illustrative
96k-topology geometry with 63k leaves, the old shared 159k-row arena would
allocate about 7.617 GB for `LayerWork.pre + LayerWork.post`; an all-postflop
exact 63k-row plan would require about 2.675 GB. The apparent target-phase
ceiling near 96k was therefore an allocator artifact and must not be treated as
the post-fix tree limit.

### Static phase admission

| Phase | Named owners GiB | Bounded workspace GiB | Guard GiB | Required GiB |
|---|---:|---:|---:|---:|
| Rollout | 0.633 | 0.917 | 0.500 | 2.050 |
| Queue drain | 0.633 | 0.009 | 0.500 | 1.142 |
| Completion | 3.132 | 0.390 | 0.500 | 4.022 |
| Align and seal | 3.050 | 0.072 | 0.500 | 3.621 |
| Targets | 3.347 | 0.071 | 0.500 | 3.918 |
| Learner | 0.848 | 0.088 | 0.500 | 1.436 |
| Final backward | 0.848 | 0 | 0.500 | 1.348 |

Every phase is below the 10,500,000,000-byte immutable budget. Static
admission is deliberately conservative accounting; it does not predict
PyTorch caching behavior by itself.

## PyTorch, cache, completion, and non-PyTorch memory

![CUDA memory accounting by execution phase](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/phase_memory_accounting.png)

The following is the maximum-occupancy snapshot for each phase over the three
final-manifest tail validations:

| Phase | Named slabs GiB | Other active PyTorch GiB | Unused PyTorch cache GiB | Non-PyTorch GiB | Driver free GiB |
|---|---:|---:|---:|---:|---:|
| Rollout | 0.633 | 0.202 | 8.298 | 0.221 | 1.213 |
| Queue drain | 0.633 | 0.202 | 8.298 | 0.221 | 1.213 |
| Completion | 3.132 | 0.202 | 1.608 | 0.231 | 5.393 |
| Align and seal | 3.050 | 0.201 | 1.691 | 0.231 | 5.393 |
| Targets | 3.347 | 0.201 | 1.751 | 0.231 | 5.036 |
| Learner | 0.848 | 0.201 | 4.250 | 0.231 | 5.036 |
| Final backward | 0.848 | 0.201 | 4.250 | 0.231 | 5.036 |

The queue-drain snapshot is taken before the explicit synchronization and
`torch.cuda.empty_cache()` boundary. The completion snapshot is after that
reclamation and shows the intended result: more than 5.39 GiB is
driver-visible before evaluator work.

“Other active PyTorch” includes the model, optimizer object state materialized
by the diagnostic, failure state not included in named slab rows, and small
runtime tensors. “Non-PyTorch” is derived as:

$$
\text{device total}-\text{driver free}-\text{PyTorch reserved}.
$$

It includes display processes and CUDA allocations not owned by PyTorch.

## Post-fix pack-64 tree-memory envelope

![Exact-LayerWork player/leaf memory boundary and expected beta rays](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/exact_layerwork_tree_memory_boundary.png)

This graph answers the larger-tree question under the required exact
`LayerWork` fix. Its axes are:

- $P$: total player nodes, conservatively treated as all postflop;
- $L$: total leaves, also treated as postflop output rows.

It uses family pack 64 and the requested decimal reservation boundary:

$$
\max_{\mathrm{phase}} R_{\mathrm{phase}}(P,L)
=10{,}500{,}000{,}000\ \text{bytes}.
$$

The action extent uses the worst structural bound compatible with
`beta_max: 3`:

$$
A(P,L)=\min(3P,\ P+L).
$$

The omitted root subtraction in the second term charges one extra action row,
making the boundary conservative by that row.

For each phase, the measured pack-64 residual is the observed PyTorch
reservation minus the named slabs live in that phase. It includes the model,
forward/workspace cache, and allocator behavior measured by
`pack64_capacity_baseline.json`. Replacing the old capped slabs with the
all-postflop row costs gives:

$$
\begin{aligned}
R_{\mathrm{rollout}}
 &= 8{,}195{,}122{,}176 + 44{,}401{,}856
    + 11{,}582P + 5{,}308A + 10{,}645L,\\
R_{\mathrm{completion}}
 &= 1{,}938{,}952{,}192 + 44{,}406{,}272
    + 11{,}582P + 5{,}308A + 53{,}109L,\\
R_{\mathrm{align}}
 &= 2{,}026{,}929{,}344 + 44{,}406{,}272
    + 11{,}582P + 5{,}308A + 42{,}464L,\\
R_{\mathrm{targets}}
 &= 2{,}091{,}763{,}648 + 1{,}156{,}096
    + 16{,}844P + 15{,}916A + 42{,}464L.
\end{aligned}
$$

All quantities are bytes. The completion coefficient includes both the
10,645-byte retained `Leaf` row and the exact 42,464-byte
`LayerWork.post` row.

Completion is the limiting phase only in the very leaf-heavy corner
$P\lesssim25.1$k. Rollout reservation is the boundary over the expected beta
rays and throughout the practically relevant larger-player region. Once
`LayerWork` is exact, the target phase is not the limiting phase in this
pack-64 model.

### Expected beta rays

For a rooted tree,

$$
L=1+\sum_{\text{internal }v}(d_v-1).
$$

With equal fixed player/chance intensity $\beta$ and no safety suppression,
the expected large-tree slope is therefore:

$$
\frac{L}{P}
\simeq(\beta-1)\left(1+\frac{C}{P}\right).
$$

Across the 26 beta-2.4 audit trees that ended below the 25k safeguard, the
aggregate unsuppressed chance/player ratio was:

$$
\frac{C}{P}=0.08880.
$$

This predicts $L/P=1.5243$ at beta 2.4; the measured through-origin slope was
1.5262. The graph uses the predicted expression for every beta ray. These rays
are expected NLHE geometry, not worst-case bounds; the memory boundary itself
still uses the conservative action extent above.

The intersections with the 10.5 GB boundary are:

| Beta | Expected $L/P$ | Player nodes $P$ | Leaves $L$ | Implied chance nodes | Implied total topology | Limiting phase |
|---:|---:|---:|---:|---:|---:|---|
| 1.5 | 0.5444 | 88,386 | 48,117 | 7,849 | 144,352 | Rollout |
| 1.8 | 0.8710 | 73,426 | 63,957 | 6,520 | 143,903 | Rollout |
| 2.0 | 1.0888 | 65,980 | 71,839 | 5,859 | 143,678 | Rollout |
| 2.2 | 1.3066 | 59,906 | 78,271 | 5,320 | 143,497 | Rollout |
| **2.4** | **1.5243** | **54,856** | **83,618** | **4,871** | **143,345** | **Rollout** |
| 2.6 | 1.7421 | 50,590 | 88,132 | 4,492 | 143,214 | Rollout |
| 2.8 | 1.9598 | 46,941 | 91,997 | 4,168 | 143,106 | Rollout |
| 3.0 | 2.1776 | 44,597 | 97,115 | 3,960 | 145,672 | Rollout |

Thus the beta-2.4 post-fix modeled ceiling is approximately 54.9k all-postflop
player nodes and 83.6k leaves, or 143.3k total topology rows after adding the
expected chance nodes. This is a planning envelope, not a cap that should be
copied directly into production: it sits exactly on the measured-reservation
projection, and it has not been CUDA-validated because the exact `LayerWork`
allocator does not yet exist.

### Family-pack sensitivity: 32, 48, and 64

![Exact-LayerWork player/leaf memory boundaries for family packs 32, 48, and 64](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/family_pack_tree_memory_boundaries.png)

This comparison repeats the capacity projection with matched CUDA baselines
for family packs 32, 48, and 64. All three use seed 223741895, completion pack
8,192, and the same structural-cap allocations. Family pack is the only
execution setting changed, so the curves isolate its observed effect on the
PyTorch reservation. The pack-64 point uses the repeat run, whose rollout
reservation reproduced the first pack-64 baseline byte for byte.

| Family pack | Baseline rollout reserved / allocated, decimal GB | Baseline wall time, s | Beta-2.4 player nodes | Leaves | Implied chance nodes | Implied total topology | Limiting phase |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 4.272 / 0.897 | 120.52 | 69,034 | 105,230 | 6,130 | 180,394 | Targets |
| **48** | **5.820 / 0.897** | **109.59** | **69,069** | **105,283** | **6,133** | **180,485** | **Targets** |
| 64 | 8.875 / 0.897 | 103.80 | 54,856 | 83,618 | 4,871 | 143,345 | Rollout |

Pack 48 increases the modeled beta-2.4 topology ceiling by 25.9% relative to
pack 64, at a 5.6% wall-time cost in this matched whole-rollout comparison.
Pack 32 is another 10.0% slower than pack 48 and produces no credible capacity
gain. Its modeled point is 0.05% smaller only because the independently
measured completion/target residual was 4 MiB higher; the pack-32 and pack-48
boundaries should be treated as coincident at this precision.

The reason is a bottleneck transfer. Pack 64 is rollout-reservation limited at
the beta-2.4 intersection. With pack 48 or 32, completion limits the
leaf-heavy corner, then targets become limiting near $P=37.6$k and remain
limiting across the beta rays. At the pack-48 beta-2.4 intersection, the
target-phase variable bytes are approximately:

$$
1.163\ \text{GB from player-indexed rows}
+2.775\ \text{GB from action-indexed rows}
+4.471\ \text{GB from exact two-arena LayerWork}.
$$

Consequently, lowering the family pack below 48 cannot move this boundary:
the family forward is no longer the resident-memory constraint. Pack 48 is the
practical capacity setting when the pack-64 envelope is too small; pack 64
remains the throughput setting when its approximately 143k-row modeled
topology ceiling is sufficient.

### Alternative: one-arena `LayerWork` with per-layer compaction

The proposed alternative removes the physical ping-pong arena axis. After all
parents in one reverse layer have read their children, each parent result is
written into its first child's slot. A separate device compaction step then
makes the surviving values contiguous before the next reverse layer.

For the postflop output, this changes one `LayerWork` row from:

$$
42{,}464\ \text{bytes}
\quad\text{to}\quad
21{,}232\ \text{bytes}.
$$

The preflop row similarly falls from 5,440 to 2,720 bytes. The rollout equation
does not change. The later phase equations become:

$$
\begin{aligned}
R_{\mathrm{completion}}^{(1)}
 &= 1{,}938{,}952{,}192 + 44{,}403{,}552
    + 11{,}582P + 5{,}308A + 31{,}877L,\\
R_{\mathrm{align}}^{(1)}
 &= 2{,}026{,}929{,}344 + 44{,}403{,}552
    + 11{,}582P + 5{,}308A + 21{,}232L,\\
R_{\mathrm{targets}}^{(1)}
 &= 2{,}091{,}763{,}648 + 1{,}153{,}376
    + 16{,}844P + 15{,}916A + 21{,}232L.
\end{aligned}
$$

The $31{,}877L$ completion coefficient is one 10,645-byte `Leaf` row plus one
21,232-byte single-arena work row.

![One-arena LayerWork boundary compared with the exact two-arena boundary](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/single_layerwork_tree_memory_boundary.png)

The one-arena design makes rollout the limiting phase over the complete
player/leaf graph. Its improvement is confined to the extreme high-leaf,
low-player corner:

| Player nodes $P$ | Two-arena maximum $L$ | One-arena maximum $L$ | Increase |
|---:|---:|---:|---:|
| 0 | 160,361 | 212,350 | 32.4% |
| 10,000 | 155,182 | 186,511 | 20.2% |
| 20,000 | 150,003 | 160,672 | 7.1% |
| 25,000 | 147,413 | 147,752 | 0.2% |
| 30,000 | 134,832 | 134,832 | 0% |

Every expected beta ray intersects the boundary above 44k player nodes.
Consequently, all beta-ray intersections are identical to the exact two-arena
table above. In particular, beta 2.4 remains:

$$
(P,L)=(54{,}856,\ 83{,}618),
$$

with approximately 143,345 expected total topology rows. Halving
`LayerWork` again does not permit a larger normal beta-2.4 tree under pack 64;
the rollout forward/cache envelope must be reduced to move that intersection.

#### Estimated compaction wall time

For a geometric beta-2.4 reverse frontier, the number of rows compacted across
all layers is approximately:

$$
\sum_{\ell\ge1}\frac{L}{\beta^\ell}
=\frac{L}{\beta-1}.
$$

A complete physical compaction reads and writes each surviving 21,232-byte
row. The resulting traffic model is:

| Case | Leaves | Modeled reverse layers | Compacted rows | Read + write traffic |
|---|---:|---:|---:|---:|
| Measured tail seed | 14,669 | 12 | 10,488 | 445,362,432 bytes |
| Beta-2.4 memory intersection | 83,618 | 14 | 59,738 | 2,536,714,432 bytes |

An optimistic gather-copy microbenchmark ran on all three idle RTX 2080 Ti
GPUs. It used the complete 21,232-byte row, two warmups, and twelve timed
repetitions per GPU:

| Physical GPU | Current-tail compaction | Boundary compaction |
|---:|---:|---:|
| 0 | 1.004 ms | 5.442 ms |
| 1 | 0.984 ms | 5.442 ms |
| 2 | 0.988 ms | 5.445 ms |
| **Median** | **0.988 ms** | **5.442 ms** |

This is a bandwidth lower bound, not the implementation estimate. The
microbenchmark used an out-of-place destination, which would reintroduce a
second full buffer and defeat the memory objective. A correct one-buffer
parallel compactor must handle overlapping source/destination intervals with
ordered tiles, cycle decomposition, or an equivalent device-resident scheme.
Allowing a conservative 2--5 times the measured gather-copy cost gives:

- approximately 2--5 ms added wall time for the current 14.7k-leaf tail;
- approximately 11--27 ms at the beta-2.4 memory intersection.

The current tail's target CUDA kernel takes 7.09 ms, so compaction could add
roughly 28--70% to target-kernel time. Scaling existing target work by the
number of internal nodes gives about 26.5 ms at the beta-2.4 intersection, so
the projected compactor could add roughly 41--103% to target CUDA time.

End-to-end impact remains negligible because rollout dominates. The measured
pack-64 tail took 102.34 seconds, while simple player-node scaling projects
about 385.1 seconds at the memory intersection. The estimated compaction cost
is below 0.005% and 0.008% of those wall times, respectively.

Those estimates require precomputed compaction destinations, one
device-resident compaction kernel per nonempty reverse layer, and no
device-to-host synchronization between layers. A host-driven re-compaction
loop is not covered by this estimate and should not be used. Planner setup and
the actual overlap-safe CUDA implementation remain unmeasured.

## Family-pack sweep

![Family-pack memory and latency sweep](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/family_pack_sweep.png)

Each family pack used matched seeds `302484964` and `223741895`. The profiler
synchronized and reset allocator peaks around every model forward, so the
memory columns below cover the forward portion rather than a later target or
completion peak.

| Family pack | Forward peak allocated GiB | Reserved during forward GiB | Incremental forward allocation GiB | Model ms/decision | Wall ms/decision |
|---:|---:|---:|---:|---:|---:|
| 21 | 1.977 | 3.018 | 1.085 | 5.584 | 9.886 |
| 42 | 3.051 | 5.070 | 2.159 | 5.375 | 7.916 |
| 64 | 4.178 | 7.724 | 3.285 | 5.418 | 7.491 |
| 84 | 5.203 | 9.058 | 4.309 | **5.301** | **7.168** |

Pack 84 is the throughput recommendation. Relative to pack 42, it reduced
measured model time by 1.4% and wall time per decision by 9.5%, while using
about 3.99 GiB more reserved memory during the largest forward.

The choice is stable because rollout cache is reclaimed before completion and
all 32 guarded seeds completed. Pack 42 remains the lower-memory fallback if
other work must share a card, but sharing these GPUs during training is not the
validated operating condition.

Tree counts varied slightly with family pack for the same logical seed. For
example, seed `223741895` produced 30,673--30,742 nodes across packs 21--84.
`reproducibility: strict` is repeatability for a fixed execution layout, not
pack invariance.

## Completion-pack sweep

![Completion-pack memory and latency sweep](../../runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/audit_beta24_capacity_20260728/completion_pack_sweep.png)

All completion packs used seed `302484964`, whose tree had 13,099 leaves and
8,595 player decisions. Peaks were reset around each completion call, so they
measure the completion portion.

| Completion pack | Calls | Peak allocated GiB | Reserved during completion GiB | Evaluator workspace MiB | Completion ms/decision |
|---:|---:|---:|---:|---:|---:|
| 512 | 26 | 3.443 | 3.527 | 22.7 | 0.1908 |
| 1,024 | 13 | 3.498 | 3.621 | 45.3 | 0.1764 |
| 2,048 | 7 | 3.608 | 3.826 | 90.6 | 0.1751 |
| 4,096 | 4 | 3.828 | 4.217 | 181.2 | 0.1706 |
| 8,192 | 2 | 4.264 | 4.994 | 362.5 | **0.1698** |
| 10,000 | 2 | 4.459 | 5.344 | 442.5 | 0.1716 |

Pack 8,192 is the measured latency minimum and saves 0.350 GiB of reservation
and 0.195 GiB of peak allocation relative to 10,000. It completes the observed
14,719-leaf maximum in two calls and the 20,000-row leaf cap in at most three.

The exact dynamic evaluator workspace for a full 8,192-row HU call is
380,076,032 bytes. Its half-free-memory guard requires at least:

$$
2\times380{,}076{,}032=760{,}152{,}064\ \text{driver-free bytes}.
$$

The worst final-tail completion boundary exposed 5,790,826,496 free bytes,
over 7.6 times that minimum.

The static completion admission charge is separately:

$$
8192\times17264+268435456+9306112
=419168256\ \text{bytes}.
$$

Static charge and realized evaluator workspace describe different ownership;
both are tracked rather than conflated.

## Final validation

After writing the selected 20k postflop caps, 252-row pre action/folded caps,
and completion pack 8,192, the three largest representative guarded tails
were rerun concurrently:

| Physical GPU | Seed | Total nodes | Post decisions | Leaves | Result |
|---:|---:|---:|---:|---:|---|
| 0 | 223741895 | 30,742 | 14,582 | 14,719 | Success |
| 1 | 969256068 | 27,803 | 12,104 | 14,623 | Success |
| 2 | 586001381 | 26,642 | 10,987 | 14,653 | Success |

Their node counts matched the discovery sweep. No capacity overflow,
completion error, CUDA error, or lingering GPU process occurred.

## Limits of the recommendation

- The 32 seeds estimate guarded beta-2.4 behavior, not a mathematical maximum.
- The 25k threshold materially changes tail trees.
- An unsuppressed beta-2.4 experiment must use a separate, larger manifest.
- The throughput family-pack sweep uses two matched seeds; the controlled
  32/48/64 capacity curves use one matched seed and should not be read more
  finely than the observed allocator noise.
- Completion-pack comparisons use one matched 13,099-leaf tree.
- The diagnostic did not execute learner forward/backward or an optimizer
  update; learner pack sizes 20/60 were retained from the token-pair contract.

The measured result is sufficient to recommend execution settings for the
requested guarded experiment. A production training launch should still start
with one full CUDA iteration before committing to a long run.

## Artifacts

All new raw results, summaries, and plots are under:

```text
runs/experiments/nlhe_hu_32bb_branched_fixed_beta_testing/
audit_beta24_capacity_20260728/
```

Key files:

- `distribution_gpu0.json`, `distribution_gpu1.json`,
  `distribution_gpu2.json`;
- `family_pack21.json`, `family_pack42.json`, `family_pack64.json`,
  `family_pack84.json`;
- `completion_pack512.json` through `completion_pack10000.json`;
- `final_tail_gpu0.json`, `final_tail_gpu1.json`, `final_tail_gpu2.json`;
- `summary.json`;
- `node_distribution.png`;
- `family_pack_sweep.png`;
- `completion_pack_sweep.png`;
- `phase_memory_accounting.png`;
- `exact_layerwork_tree_memory_boundary.png`;
- `exact_layerwork_tree_memory_boundary.json`;
- `pack32_capacity_baseline.json`, `pack48_capacity_baseline.json`,
  `pack64_capacity_baseline_repeat.json`;
- `family_pack_tree_memory_boundaries.png`;
- `family_pack_tree_memory_boundaries.json`;
- `single_layerwork_tree_memory_boundary.png`;
- `single_layerwork_tree_memory_boundary.json`;
- `single_layerwork_compaction_benchmark.json`.

## Code paths used

- fixed branching and safe-tree suppression:
  `pbg/core/branching.py`, `FixedBranchController.decide`;
- staged destination validation:
  `pbg/core/storage.py`, `_DevicePublicationTransaction._validate_requests`;
- chance-closure publication:
  `pbg/games/poker/queued/branching.py`;
- phase ownership and workspace admission:
  `pbg/core/capacity.py`;
- phase-boundary cache reclamation:
  `pbg/games/poker/queued/branching.py`;
- completion FIFO:
  `pbg/core/rollout.py`, `CompletionFIFO`;
- evaluator dynamic workspace:
  `/home/pablob/poker-gpu-evaluator/poker_gpu_evaluator/_backend_cuda.py`.
