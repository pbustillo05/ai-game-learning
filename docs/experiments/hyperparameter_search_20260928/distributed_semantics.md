# Distributed semantics checked before the two-GPU pilot

The current fixed-step `batch_size` is **per rank**. `pbg/config.py` passes it
unchanged through `TrainingSession` to each `SequentialBatchSampler`. Its local
partition weights sum to one; physical splits preserve that local mean
(`pbg/core/learner.py:561`, `pbg/core/training.py:180`).
`DistributedSUM.sum_gradients` (`pbg/core/replica.py:315`) then sums gradients.
`DenseAdam.step` removes AMP scaling but does not divide by world size.
Consequently, for equal local batch sizes, $g_{implemented}=Wg_{global\ mean}$.

This scales every objective term together, preserving relative coefficients.
It is not a $W$-fold Adam learning rate: constant scaling mostly cancels in
Adam, but epsilon, old moments, overflow and clipping prevent exact equivalence.
The clip of 0.5 corresponds to a global-mean threshold of $0.5/W$.

The two-GPU smoke deliberately preserves production SUM semantics. Batch 128
per rank means 256 global; roots 256 per rank means 512 global; 12 synchronized
updates give 3,072 global presentations. It is an execution/consistency check,
not an exact matched-objective strong-scaling measurement against single-GPU
training. Before promoting a three-GPU quality experiment, explicitly select
rank-invariant global-mean weighting or document and test retained SUM scaling.
Do not compensate by changing only entropy or KL. Three GPUs with batch128
per rank means global384, not256. Adjust update count and compare presentations
per fresh decision when changing this budget.

There is a separate legacy-checkpoint adapter issue. `ReplicaModuleOwner`
constructs `_code = deepcopy(model).to('meta')` (`replica.py:76`) and creates
separate lane objects (`:95`). Functional rollout inference invokes `_code`
(`:241`), so patching only `lane_module(0)._call_preflop` does not preserve the
July policy during distributed rollout. The experiment driver now adapts both
code templates and lane modules. Its distributed run compares direct and
functional pre/post outputs on identical CUDA inputs, and compares final model
hashes across ranks. Matching hashes alone would not detect encoding mismatch.

No production source or authoritative theory was changed for this audit.


## Follow-up: registered campaign uses a global mean

The earlier two-GPU pilot above deliberately retained production SUM. The
registered campaign now installs an experiment-local global-mean wrapper in
experiments/hyperparameter_search/worker.py::install_global_mean: call
session.distributed.sum_gradients and divide the resulting dense gradients by
world size before DenseAdam.step unscales/clips/updates. Production
DistributedSUM is unchanged. Every objective component is normalized together.

The source trace establishing the original scale is:

- pbg/config.py:586–594 preserves the configured statistical batch size.
- pbg/runtime.py:562 passes the same schedule to every rank.
- pbg/core/learner.py:1043–1060 draws that many local rows.
- pbg/core/learner.py:561 weights a local family by its local batch share.
- Cost-aware weights normalize locally at learner.py:899; physical splitting
  preserves them at pbg/core/training.py:180.
- The objective averages sigma rows at learner.py:2122–2125, with weighted
  backward at :2161.
- pbg/core/replica.py:315 performs SUM; :484–488 calls it before Adam.
- pbg/core/optimization.py:299–317 removes AMP scaling, measures norm, and
  clips; it contains no native world-size normalization.

For equal fixed-step local batches the wrapper gives the concatenated-batch
mean. It is not a universal fix for unequal local sample weighting:
epoch-mode partial batches preserve local means, and
pbg/core/training.py:293–299 pads shorter ranks with zero contributions.
The campaign uses equal fixed-step local batches. If unequal batches are
introduced, reduce weighted gradient numerators and actual sample mass
instead of merely dividing by world size.

The registered three-GPU control uses batch128/rank, global384, 24 synchronized
steps and 9,216 global presentations per iteration. Both Normal and RB use
the same common settings. Resume keeps topology fixed.

The worker also applies the historical adapter to both code templates and
materialized lanes for learner, reference, and frozen-opponent owners. Direct
versus functional policy/value comparisons on identical pre/postflop CUDA
inputs remain the appropriate validation; matching hashes alone is insufficient.

A parent-reported three-GPU checkpoint test interrupted after iteration1 and
resumed through iterations2/3, preserving zero entropy and equal model hashes
on all ranks. Exact evidence belongs in the
[new campaign validation ledger](../hyperparameter_campaign_20260928/README.md).
The experiment-specific bundle restore validates each rank's execution
identity separately, accommodating the cards' small physical-VRAM difference
without silently relaxing those checks. These are experiment-driver changes,
not modifications of production or authoritative theory.
