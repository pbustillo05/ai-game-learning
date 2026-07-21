Short answer: we have good preventive accounting in sequential modes, but branching mode does not yet have an equivalent closed-loop VRAM controller. Dynamic branching has a tree-size controller, which helps indirectly, but it is not an OOM guarantee.

| Mode | Current mechanism | Main gap |
|---|---|---|
| Normal / sequential RB | Four-phase memory model dynamically adjusts roots per GPU | Actual OOM still aborts; external free-memory changes are not incorporated |
| Fixed branching | Static full-`node_budget` preflight and peak telemetry | No feedback into beta |
| Dynamic branching | Static preflight, $\kappa$ tree-size controller, and a `safe_tree_size` queue safeguard | Controls predicted nodes, not bytes; reacts with a one-iteration lag |
| All modes | Completion and learner batches are sized against available transient memory | Branching policy-forward batches lack equivalent dynamic sizing |

## What exists now

### Normal and sequential RB

These have the most complete mechanism:

1. Before training, an analytical model estimates four separate phase peaks:

   - rollout
   - sealing
   - target/completion construction
   - optimization

   It includes model parameters, optimizer state, tree storage, reach tensors, targets, completion workspace, and learner workspace. An impossible plan is rejected before rollout. See [training.py](/home/pablob/ai-game-learning/pbg/core/training.py:479).

2. Before each iteration, `SequentialMemoryController` chooses a common root count $N$ across ranks. It binary-searches for the largest root width predicted to fit the hardware ceiling. See [memory_control.py](/home/pablob/ai-game-learning/pbg/core/memory_control.py:748).

3. After an iteration, actual allocated peaks from all four phases update conservative one-sided forecasts:

   - positive memory surprises widen the estimate immediately;
   - old pessimistic evidence remains for a bounded window;
   - $N$ decreases immediately when necessary;
   - growth requires a successful streak and is rate-limited.

4. The controller state is checkpointed, so resume preserves its history and selected width.

There are still weaknesses:

- An actual `torch.cuda.OutOfMemoryError` is not caught and recovered from.
- The controller’s hard ceiling is based on the configured device limit. Although `_memory_operating_ceiling()` exists, it is currently unused, so another process consuming VRAM after launch does not immediately contract the next width.
- `record_transaction_failure()` exists but is not wired into the training phases.
- `_validate_memory_calibration()` records a hard-limit violation, but does not stop the run.
- If the minimum configured root width is forecast not to fit, it does cleanly checkpoint and stop before drawing the next iteration. That is the strongest existing last-resort behavior.

### Fixed-beta branching

Current protection is much weaker:

- The same analytical preflight is performed.
- It plans structural memory using the full `node_budget`, one root, dense branching reaches, full event-width targets, and branching heads.
- Actual phase peaks are recorded every iteration.
- `DeviceTreeBuffer` enforces the node capacity transactionally.

But none of those measurements modify beta. The configured beta table remains constant for the entire run.

There is an `branching_upper_cutoff_nodes` latch in the sampler that changes subsequent offspring counts to one, but:

- it is currently experiment-only and is not part of the strict production configuration schema;
- it responds to node count rather than VRAM;
- it does not terminate the tree—it merely stops additional branching while mandatory trajectories continue;
- exceeding `node_budget` still aborts instead of gracefully admitting only safe extra branches.

The dynamic `safe_tree_size` safeguard is inactive in fixed mode.

### Dynamic-beta branching

After warmup, dynamic mode computes local intensities from $T$, $C$, and $\kappa$:

$$
\beta \propto \sqrt{\kappa T/C}.
$$

After each completed tree, it updates $\kappa$ from an EMA of the counterfactual tree size:

$$
\log\kappa_{k+1}
=
\log\kappa_k
+
\eta_\kappa
\log\frac{M}{\overline M_k}.
$$

Thus an oversized tree lowers $\kappa$ and future beta values. See [adaptive_branching.py](/home/pablob/ai-game-learning/pbg/core/adaptive_branching.py:113).

It also has an in-rollout queue safeguard: if stored nodes plus the $C$-predicted queued work exceeds `safe_tree_size`, the current expansion batch is executed with one child per parent. That safeguard can recover and permit later branching again.

However:

- `safe_tree_size` is measured in predicted nodes, not bytes.
- It only runs after dynamic warmup; the fixed-beta warmup is unprotected.
- $\kappa$ updates only after finishing the tree.
- Its distributed input is mean counterfactual tree size, whereas OOM safety must use the worst rank.
- EMA damping intentionally makes it slow to react.
- Bad or temporarily stale $C$ heads weaken its forecast.
- It does not distinguish whether rollout, targets, sealing, or optimization is the memory-limiting phase.

So faster $\kappa$ contraction would help, but $\kappa$ alone should not be the hard safety layer.

## Recommended branching safeguard

I would add a dedicated `BranchingMemoryController`, separate from the scientific beta/KKT controller.

### 1. Byte-aware admission before every tree

Reuse the four-phase forecaster, but parameterize branching memory by:

- node capacity;
- realized nodes and edges;
- leaves;
- decision rows;
- retained target bytes;
- maximum policy-forward family batch;
- completion workspace.

Before any random draw:

- recompute the operating ceiling from current free VRAM and the configured reserve;
- take the minimum safe value across ranks;
- derive a conservative memory-safe tree target $M_{\mathrm{mem}}$;
- cleanly checkpoint and stop if even the mandatory configuration—especially `branching_min_nodes`—cannot fit.

Unlike the sequential controller, this should actually use the existing current-free-memory calculation.

One important limitation is that several branching rollout tensors are preallocated to `node_budget`. Lowering beta does not reduce that base allocation. Therefore the complete structural plan for `node_budget` must always fit; beta control only reduces realized targets, downstream storage, and transient work.

### 2. Dynamically size policy-forward batches

This is the simplest and most important immediate improvement.

`family_batch_rows` affects execution memory but not the sampled probability law. The sampler can safely reduce a queued forward from 64 rows to 32, 16, etc. based on predicted transient bytes and current headroom, without changing logical RNG or tree distribution.

Optimization already performs this kind of row-cap halving. Branching rollout should use the same contract.

This directly addresses the type of inference OOM encountered during the checkpoint validation.

### 3. Fixed-beta memory governor

Keep the configured beta table as the reference schedule and introduce a checkpointed global scale $s_k$:

$$
\beta_{k,b}^{\mathrm{effective}}
=
1+s_k\left(\beta_b^{\mathrm{base}}-1\right),
$$

with configured bounds on $s_k$ or per-bucket beta bounds.

This preserves:

- every beta at or above one;
- the relative street/depth schedule;
- buckets configured at exactly one;
- explicit knowledge of the beta actually used by every target.

The update should be asymmetric:

- contract immediately and without EMA when forecast memory approaches the ceiling;
- allow growth only after several comfortably safe trees;
- cap upward movement per iteration;
- use worst-rank pressure;
- checkpoint and log $s_k$, the limiting phase, and predicted/actual peaks.

I would call this “memory-governed fixed beta,” because silently changing beta while continuing to describe the run as mathematically fixed would be misleading.

### 4. Dynamic-beta memory governor

Retain the existing KKT update for its statistical objective, but add an independent safety envelope.

The effective target becomes:

$$
M_k^{\mathrm{effective}}
=
\min(M_{\mathrm{configured}},M_{\mathrm{mem},k}).
$$

When memory pressure rises:

- update $\kappa$ using the current counterfactual size, bypassing the slow EMA for downward corrections;
- use a larger `eta_down`, potentially $1$, while retaining the existing slow `eta_up`;
- freeze upward $\kappa$ movement for several safe iterations;
- use worst-rank memory pressure rather than average tree size.

Additionally, clamp the proposed excess branching when necessary:

$$
\beta^{\mathrm{executed}}
=
1+s_{\mathrm{mem}}\left(\beta^{\mathrm{KKT}}-1\right).
$$

That independent clamp is necessary because $\kappa$ may react too slowly, may be inactive during warmup, or may be acting through inaccurate heads. Repeated clamping should then drive the faster $\kappa$ update so the KKT controller stops requesting unsafe intensities.

### 5. Hard in-rollout admission

Before admitting extra offspring, reserve enough capacity for every already-live branch to finish with one child per remaining step. Only multiplicity above that mandatory continuation should be suppressible.

The decision must happen before sampling the proposed children. We should retain:

- nominal offspring;
- memory-admitted offspring;
- actual offspring;
- the existing $C/T$ counterfactual corrections.

This provides a true “cannot overflow the tree arena” property rather than discovering overflow at `append_children()`.

### 6. Root-refill interaction

The 1,000-node floor cannot override the memory ceiling.

Because one refill subtree can overshoot dramatically—for example, 3 nodes to 5,435 in one addition—the controller must admit enough memory for the entire configured `node_budget`, not merely 1,000 nodes.

If `branching_min_nodes` is not safely feasible, the correct behavior is a pre-draw memory-stop checkpoint. We should not partially refill or reroll until a smaller subtree appears.

### 7. OOM failure semantics

I would preserve the sequential rule: never catch an OOM, reduce beta, and reroll the same iteration. That would change the sampled law based on which trees exhausted memory.

Instead:

- predicted failure: stop transactionally before draws and write a memory-stop checkpoint;
- unexpected actual CUDA OOM: abort and resume from the last intact checkpoint with stricter controller state/configuration;
- never rely on `empty_cache()` plus retry as the safety mechanism.

So the ideal hierarchy is:

1. full-capacity structural preflight;
2. current-free-memory admission;
3. dynamically reduced execution batch rows;
4. fixed-beta scaling or fast $\kappa$ contraction between trees;
5. in-tree suppression of extra multiplicity with mandatory-continuation reserve;
6. clean pre-draw stop if the minimum safe workload cannot fit.

That would make branching memory control genuinely comparable to sequential control while keeping the statistical beta/KKT logic separate from the non-negotiable hardware safety envelope.