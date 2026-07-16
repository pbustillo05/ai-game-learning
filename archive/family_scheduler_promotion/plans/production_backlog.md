# FIFO production backlog

This is an implementation/experiment backlog, not mathematical authority.
The algorithm and invariants live in
[`docs/theory/pseudocode/queued_rollout.tex`](../theory/pseudocode/queued_rollout.tex)
and [`distributed_update.tex`](../theory/pseudocode/distributed_update.tex).
Readiness and evidence routing live in
[`docs/audit/fifo_training_readiness_2026-07-16.md`](../audit/fifo_training_readiness_2026-07-16.md).

## Scope and stopping rule

The objective is a maintainable common family-queue executor for normal,
sequential Rao--Blackwell, and branched Rao--Blackwell poker training. Do not
replace it with a faster unrelated branched path. Stop optimization when the
local readiness gates pass: the goal is no material FIFO penalty, not a
state-of-the-art scheduler.

RTX 2080 Ti is the only optimization target. H100 runs are deployment
calibration after acceptance, not evidence required for acceptance.

## Priority 0 — close release evidence before speculative optimization

1. **Complete.** Run the trained P=6/S=10/R=13 checkpoint workload in sequential-RB against
   wavefront: warmup one, three paired seeds, local profile, fixed completion,
   q/admission chosen from a small local calibration. Investigate only if the
   complete-iteration 95% upper ratio exceeds 1.20.
2. **Complete.** Make branched-RB use the same actual-cost family packer and run a checkpoint-
   compatible, fixed-beta workload. First establish law/memory correctness, then
   make the same three-seed paired comparison where wavefront safely fits.
3. **Complete.** Run two ranks on the two idle local 2080 Tis for all modes: unequal realized
   masses/topologies are expected; require finite reduced gradients, exact
   checkpoint/resume, no collective mismatch, and no undeclared syncs.
4. **Complete.** Run a conservative 20-iteration two-rank soak for each mode. After warmup,
   peak reserved memory must remain below the local budget with 25% margin and
   show no monotone allocator growth. Resume one checkpoint for at least five
   further iterations.

These are the shortest path to production confidence. Do not add H100 or
four-rank gates before them.

## Priority 1 — only if a local gate exposes a material FIFO penalty

### A. Amortize the remaining scheduler boundary

Current actual-cost packing has one D2H width vector and CPU greedy span plan
per logical block; physical pack indices are now device-resident. Measure
scheduler/control time separately from model time. If it matters, prototype a
fixed-capacity device launch-plan buffer that emits 2--4 family launches per
control read, with masked descriptors and eager fallback. Preserve logical FIFO
order, queue keys, and addressed RNG.

### B. Harden calibrated execution plans

Require a hardware profile for CUDA launch. Keep a small warmed set of family /
physical-quanta plans; reject an unprofiled dynamic compiled shape that violates
the reserve. Test eager, compile, and CUDA-graph plans independently and retain
only a repeatable end-to-end win with eager tail fallback.

### C. Reduce lifetime and allocator pressure

Profile persistent bytes by phase, then reuse fixed-capacity input/output,
pack, scatter, and completion-interface slabs. Release sequential reaches and
contraction summaries at last use. This is an allocator/lifetime task, not a
reason to alter samples, beta, or completion Monte Carlo count.

### D. Make learner work exposure-based

Choose logical learner draws/epochs from realized decision mass; physical
microbatches remain cost-bounded family packs with exact numerator scaling.
This addresses training usefulness and rank balance without changing rollout
law.

### E. Add overlap only after a component trace proves headroom

Try double staging, completion during unrelated policy work, and early global-Z
SUM overlap one at a time. Accept only a complete-iteration gain; evaluator,
attention, and optimizer work may contend on this hardware.

## Explicitly deferred

- street-keyed logical queues (family/street is an ablation, not a silent
  replacement for `post_A`);
- central actor/learner or stale-policy pipelines;
- adaptive beta, trajectory dropping, or outcome-dependent truncation;
- H100-specific quantum tuning and four-rank optimization;
- a custom C++/CUDA scheduler before local gates show that the remaining
  control boundary is material.

## Reviewed precedents

| Source | Current relevance | Take / do not take |
|---|---|---|
| [Sample Factory](https://www.samplefactory.dev/07-advanced-topics/batched-non-batched/) | Active documentation; its batched mode validates one large device tensor and a small worker count. | Reusable slabs, bounded wait-to-fill, and buffer ownership; do not adopt asynchronous policy lag. |
| [TorchRL MultiSyncCollector](https://docs.pytorch.org/rl/main/reference/generated/torchrl.collectors.MultiSyncCollector.html) | Current collector exposes profiling, compile, CUDA graph, and buffer switches. | Use its measurement discipline and experiment matrix; do not integrate a generic collector. |
| [verl sequence-length balancing](https://verl.readthedocs.io/en/latest/api/utils.html) | Active and directly relevant to cost-bounded physical packs. | Cost-aware partitioning and inverse scatter; retain PBG’s exact global-mass scaling instead of its learner stack. |
| [KataGo](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md) and [LightZero](https://github.com/opendilab/LightZero) | Active batching/tree-engine references. | Outstanding-work accounting, batch histograms, and compact tree handles; no C++ port unless profiling proves Python scheduling is the blocker. |
| [CleanRL multi-GPU PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_multigpupy) | Current, small data-parallel reference. | Keep flat-buffer collective hygiene; do not use equal-rank PPO averaging. |
| [EnvPool](https://github.com/sail-sg/envpool) | Active vectorized environment engine. | Active-pool/backpressure idea only; its C++ environment runtime does not fit poker state/evaluator semantics. |
| [OpenSpiel AlphaZero](https://openspiel.readthedocs.io/en/stable/alpha_zero.html) | Current, but its Python evaluator explicitly lacks inference batching. | C++ queue/accounting reference only; no implementation dependency. |
| [SEED RL](https://github.com/google-research/seed_rl) | Archived/read-only. | Historical batching reference only; reject centralized inference and stale-policy architecture. |
| rlpyt, Acme, RLlib, OpenRLHF, mctx | Useful historical/research references, but not direct fits for this on-policy, evaluator-bound PBG. | No dependency or porting work. |

## Experiment discipline

Use one axis at a time, checkpoint identity plus logical seed grid fixed, and
record wall time, decisions/nodes, model calls, physical rows/padding,
scheduler time, allocator peak, and queue drains. Three paired seeds are enough
for a release decision; use five only to resolve a close or noisy result. Put
the artifact and extended interpretation in the experiment log, not the
readiness dashboard.
