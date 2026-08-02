# Quest H100 launch-blocker mini audit

Date: 2026-07-29

Audited Nova revision: `a5d5fb1dda4d8065c6c3fb49b13676b26974c131`

Successful RunPod revision:
`d9db16cf5f8a02e5059b22a634684b943d1d1608`

Evaluator revision:
`9d35191b396cca5bf07518a07ba2976b11971a5a`

Evidence archive:
`runs/experiments/nlhe_hu_32bb_h100_pilot/harvest/nlhe_hu_32bb_h100_pilot_diagnostics_d9db16cf_20260729.tar`

## Executive status

The H100 pilot established that Normal and RB training both execute in BF16
with one replica, 512 roots, logical batch size 256, and 116 gradient steps.
Normal reached 42.13 player nodes/s and RB with 1,024 completion samples
reached 41.12 player nodes/s. Peak reserved memory was at most 16.38 GiB.

A single-GPU, single-replica Quest run is a **no-go until QH100-1 and QH100-2
are ported to the current branch and retested**.

Post-audit update (2026-07-30): commit `9229734` resolved the QH100-3
implementation defect by removing the validate-only replay and introducing one
transactional learner pass per lane. Same-GPU multi-replica and distributed
Quest runs remain a **performance no-go pending the deferred H100 K1/K2 and
distributed K1 validation gates**.

## Required fixes

| ID | Severity | Defect | Current consequence | Required before |
|---|---|---|---|---|
| QH100-1 | P0 correctness | BF16 transition routing writes FP32 `model_input` into BF16 family storage with strict `index_copy_` | Immediate Normal smoke failure | Every H100 run |
| QH100-2 | P0 capacity | H100 pilot sizes cumulative folded-range storage to one 84-row scheduler pack | Deterministic overflow at logical root 336 for 512-root arms | Every H100 run above one pack |
| QH100-3 | P1 performance — implementation resolved, H100 validation pending | Replica/distributed learner performed a complete validate-only forward/backward pass for every lane, cleared it, then repeated every lane for committed updates | Commit `9229734` removed the duplicate pass; H100 K2 timing and distributed K1 validation remain pending | H100 validation before multi-replica or distributed production |
| QH100-4 | P1 operations | Benchmark reports are emitted only after an arm finishes and omit learner/completion phase timings and progress heartbeats | Timeouts produce empty logs and no partial report; Slurm allocation time cannot be diagnosed safely | Long Quest jobs |

### QH100-1: BF16 route boundary

The current Nova implementation at
`pbg/games/poker/queued/hooks.py::_gather_route` directly calls
`index_copy_` with the gathered source dtype. `direct_model_input()` produces
FP32, while H100 family storage is BF16.

The validated RunPod fix is commit `2ebfae7`: cast only floating-point
`model_input` to the declared destination dtype at the route boundary. All
other field dtype mismatches must remain hard errors.

Acceptance gates:

- CPU and CUDA route tests cover FP32 `model_input` into BF16 storage.
- A non-`model_input` dtype mismatch still raises.
- Normal and RB BF16 smoke gates pass on CUDA.
- At least one complete 512-root K=1 iteration applies all 116 optimizer
  steps.

### QH100-2: cumulative folded-range capacity

The current generator in `pbg/h100_pilot.py` computes:

```python
folded_rows = max(1, min(arm.family_pack, arm.roots))
```

Folded ranges survive for the complete rollout attempt, so this is a
cumulative capacity, not a per-pack workspace. With `family_pack=84`, the
fifth pack starts at logical root $4\times84=336$ and overflows the 84-row
slab.

The validated RunPod fix is commit `d9db16c`:

```python
folded_rows = arm.roots
```

Acceptance gates:

- Generated 512-root configs declare `folded_range_cap.pre=512` and
  `folded_range_cap.post=512`.
- Config tests distinguish cumulative capacities from scheduler pack bounds.
- Normal and RB 512-root K=1 iterations complete without recovery.
- A multiplayer capacity review confirms whether the general bound must be
  larger than one folded row per root outside heads-up poker.

### QH100-3: implementation resolved; H100 validation pending

Status update (2026-07-30): commit `9229734` deleted
`LearnerPreflightOptimizer`, `IterationEngine.preflight_learners()`, and the
`validate_only` path. Sequential and branched modes now prepare their
presentation-specific sampler and enter the same `LearnerTransaction`, which
executes one real `LearnerIteration` per lane. Accelerator-resident rollback
snapshots cover model parameters and buffers, optimizer moments and step
count, scaler state, replica versions, explicit learner RNG state, and run
state.

Focused CUDA validation passed for K=2, K=4, and K=8 one-pass learner call
counts, local late-lane rollback, and transaction-memory reconciliation. The
broader relevant CUDA/two-GPU NCCL suite also passed. QH100-3 is therefore
**resolved at the implementation level**, with these closure gates still
pending:

- run the production K2 arm on H100 and verify exactly 232 committed optimizer
  steps with no hidden dry pass;
- demonstrate H100 K2 wall time no worse than approximately 2x K1;
- benchmark distributed K=1; and
- add a distributed late-failure regression through the production
  `LearnerTransaction`.

The checked-in pilot YAMLs and the generated Normal local-smoke capacity also
need separate repair before they can serve as the deferred H100 validation
path. These launch-path defects do not reinstate the duplicated learner pass,
but they currently prevent end-to-end closure of the H100 gate.

The evidence below describes the original audited implementation.

For K>1 or any distributed session, sequential training at the audited
revision executed:

```text
every lane: validate-only forward/loss/backward, no parameter update
every lane: forward/loss/backward again, then apply optimizer updates
```

The validate-only optimizer suppresses parameter transitions but does not
suppress model forwards or backward passes. Lanes are then processed
sequentially during the committed pass.

Observed H100 evidence:

| Mode | K1 seconds | K2 seconds | Work ratio | Time ratio | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Normal | 39.69 | 158.82 | 2.00 | 4.00 | 0.50 |
| RB | 41.52 | 158.82 | 1.97 | 3.83 | 0.51 |

K4 was active at 93--96% GPU utilization when timed out. Its timeout is
consistent with eight learner passes per measured iteration, not a deadlock
or memory failure.

Required remediation must preserve the all-or-nothing lane/rank update
contract without recomputing the complete learner workload. One acceptable
direction is to retain each lane's gradients until every lane/rank succeeds,
then apply all optimizer transitions. The final design must account for
distributed failure agreement, RNG identity, gradient memory, and cleanup
after a failed lane.

Acceptance gates:

- An instrumented K-lane iteration performs exactly one learner
  forward/backward workload per lane.
- Injecting a late failure in any lane leaves every model, optimizer version,
  run state, and RNG state uncommitted.
- Successful K2 performs exactly 232 committed optimizer steps in total
  without an additional hidden 232-step dry pass.
- Single-GPU K2 wall time is no worse than approximately 2x K1 before claiming
  replica parallel speedup.
- Distributed K=1 is benchmarked separately because it currently enters the
  same preflight path.

### QH100-4: insufficient long-run diagnostics

Successful reports contain aggregate step duration and rollout model timing,
but no learner, completion, target, or optimizer phase timing. The same
aggregate model timing is attached to each lane and must not be summed.
Warmup timing is discarded, and timed-out K4 arms produced empty logs and no
partial reports.

Acceptance gates:

- Emit an atomic partial report after every warmup and measured repeat.
- Record phase timings for rollout, completion, targets, learner
  forward/backward, optimizer, and synchronization.
- Emit periodic progress with lane, phase, logical step, elapsed time, GPU
  memory, and completed/total learner steps.
- On SIGTERM/Slurm preemption, flush progress and the last completed phase
  before exit.

## Validation gaps, not yet classified as code bugs

- Root geometry was not swept. A root count of 512 is safe but not known to
  maximize H100 throughput.
- RB sample counts 256, 512, and 1,024 were not compared. Only 1,024 was
  validated at production-scale roots.
- The logical batch size is 256, but the admitted physical learner pack is
  84; this should not be described as one physical 256-row model batch.
- The pilot proved successful execution and applied optimizer steps, but did
  not retain a trained checkpoint or loss series for numerical-quality and
  checkpoint/resume comparison.
- The RunPod environment used PyTorch `2.12.1+cu126`, CUDA runtime 12.6,
  nvcc 12.4.131, and driver 570.211.01. Quest must run an environment
  preflight against its actual module/container stack.

## Minimum Quest launch gate

Before submitting the long job:

1. Port QH100-1 and QH100-2 onto the intended Quest revision without
   disturbing concurrent work.
2. Run focused CUDA regression tests.
3. Run one Normal and one RB BF16 K=1, 512-root iteration.
4. Verify finite losses, 116 applied optimizer steps, checkpoint creation,
   checkpoint reload, and one resumed iteration.
5. If using multiple GPUs, complete the deferred QH100-3 H100 K1/K2 and
   distributed K1 validation before requesting the full allocation.
6. Start the Quest job with K=1 unless a later replica benchmark demonstrates
   a real aggregate throughput gain.

## Ideal replica learner parallelism

After removing the duplicated dry pass, the ideal single-GPU design is
memory-admitted lane-wave concurrency. Each replica keeps its independent
weights, autograd graph, dense FP32 gradients, optimizer state, RNG, and CUDA
stream; the common learner launches as many lanes concurrently as the H100
activation/workspace budget safely admits. This must remain the same
per-lane learner operation rather than a functional or vmapped backward, and
must never combine gradients between replica lanes.

Within a lane, optimizer step $s$ must still precede forward step $s+1$.
Across lanes, forward/loss/backward work may overlap freely. Each admitted
wave joins at one failure-agreement boundary before its independent optimizer
transitions, while the iteration-wide transaction publishes all lane
versions and run state only after every wave/rank succeeds. Distributed ranks
must launch lane collectives in the same fixed order. H100 experiments should
sweep concurrent lane-wave width, record activation and synchronization
memory, and accept the change only when aggregate learner throughput improves
without numerical, RNG, rollback, or peak-memory regression.
