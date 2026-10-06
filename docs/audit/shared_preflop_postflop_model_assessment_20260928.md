# Sharing preflop and postflop weights — assessment, 2026-09-28

## Issue and motivation

Preflop and postflop currently learn separate parameters despite using similar
transformer architectures. Branching supplies very few distinct preflop states,
so preflop receives little direct experience and cannot benefit from postflop
updates to common betting/range computations. Sharing may improve learning per
GPU-hour; positive transfer is a hypothesis, not an established result.

The 169 preflop classes versus 1326 postflop combos justify different
representations, not necessarily different weights. Keep the native $R^2$ and
$\binom{4R}{2}$ domains and share the main transformer and policy/value readout.
Preflop class indexing must continue to enforce suit invariance.

The [timing audit](nlhe_hu_32bb_family_timing_attribution_20260724.md) measured
1.721 s of preflop forwards versus 34.053 s of postflop forwards in sequential
RB. Expanding preflop to combo width would threaten its speed advantage.
Sharing weights at unchanged sequence lengths, width, and depth does not itself
increase transformer work. Actual adapter overhead and learning speed require
measurement. Model mix explained 25.8% of branching's extra rollout cost per
decision; sharing does not fix its tree/data-movement bottlenecks. These timing
measurements excluded learner backward passes.

## Scope and difficulty

This is a localized model refactor with modest integration work. Keep all
`pre`/`post_A` FIFOs, storage schemas, family packing, completion, and targets.
Continue separate family calls in rollout and learning; do not pad preflop to
postflop width.

`LearnerIteration.optimize` already accumulates weighted backward calls sharing
a `logical_step` before one optimizer step. Sequential `steps` and both branched
schedules already provide this grouping. Shared parameters therefore receive
the gradient sum without new accumulation or distributed-reduction machinery.
Sequential `epochs` instead steps family minibatches separately; this remains
valid but can cause long stretches of updates dominated by one street.

Planning estimate, excluding neural model implementation: **75–200 changed
production lines and 250–500 test lines**. Most integration work is in the
queued model wrapper/factory and configuration identity. No queue or fixed-step
learner rewrite is expected. Supporting old split-model frozen opponents with a
new shared learner adds roughly **50–120 production lines plus tests**. These
are inspection estimates, not measured patch sizes.

## Implementation instructions

1. **Introduce an explicit architecture selection.** Update `pbg/config.py`,
   `pbg/games/poker/queued/config.py`, and `build_model_from_config` in
   `pbg/games/poker/queued/model.py`. Preserve the split architecture for baseline
   comparisons and old checkpoints. Record the selection in scientific config
   and its hash. Initially require equal pre/post depths for the shared model;
   leave width, depth, and family batch caps unchanged.

2. **Give shared weights one registered owner.** Under `QueuedPokerModel`, own
   separate pre/post encoders, one shared trunk, and one shared policy/value
   readout. Pass encoded tensors into that trunk. Do not register it again
   under both encoders: `ReplicaModuleOwner` combines deduplicated parameter
   names with `state_dict()` keys, so duplicate module paths can break lane
   checkpoint lookup. Replace the wrapper's separation assumptions and
   street-network type checks while preserving its public call signatures and
   `ModelEvaluation` shapes.

3. **Preserve representation-specific mathematics.** Keep class sizes,
   orbit-averaged incidence, and identical-combo scaling preflop; keep combo
   incidence, board masks, and folded statistics postflop. Share attention,
   incidence-residual weights, compatible public/history embeddings, and
   policy/value readout. First align public token type meanings: their numeric
   IDs currently differ between the two implementations. Retain explicit street
   information and specialized board/folder input maps. Small public branching
   heads can remain family-specific in the first experiment.

4. **Factor each layer's side-feature projection.** The current pre/post FFNs
   have different input widths, so they cannot simply be tied. Implement
   $W_2\operatorname{GELU}(W_z z+A_su_s+b)$: share $W_z,W_2,b$ for common hidden
   and dynamic incidence features $z$; use separate $A_s$ for static class versus
   combo/folder features $u_s$. Preserve required geometry channels. This splits
   the existing first affine map rather than adding a full extra FFN. Check
   workspace bounds in `queued/capacity_compiler.py` against the resulting
   intermediates.

5. **Keep the existing learning objective and update boundaries.** Use `steps`
   for the first comparison. Preserve posterior normalization, statistical
   weights, and physical-chunk weights; do not introduce arbitrary 50/50 street
   weighting. Backward each family chunk immediately, retaining only accumulated
   gradients, then clip/update once per logical step. If mixed-family epoch
   updates are later desired, change `SequentialBatchSampler.batches` and
   `optimizer_steps` together and test presentation counts and weights.

6. **Treat checkpoint conversion separately.** A split checkpoint cannot resume
   exactly into one shared trunk or its optimizer state. Start fresh for the
   initial comparison and test shared-model save/resume. For comparisons against
   old frozen opponents, change `pbg/runtime.py` to reconstruct the opponent from
   its own checkpoint config, not the learner's; reject incompatible
   initialize-from-target requests unless an explicit conversion is provided.

7. **Validate on CUDA before adoption.** Test weighted gradient-sum equivalence,
   one update per logical step, unique parameter ownership, replica lane binding,
   functional inference, checkpoint round trips, and distributed SUM parity.
   Verify unchanged family shapes and preflop suit invariance. Benchmark pre/post
   forward and backward separately, then compare sequential RB and branching
   learning quality versus GPU-hours across seeds. Monitor per-street policy
   movement because postflop updates now affect preflop predictions. Use the
   prescribed venv/toolkit, check `nvidia-smi`, target idle GPUs explicitly, and
   verify they return idle afterward.

Implementation would also require aligning `docs/theory/MHA_model.md` and the
separate-model statement in `docs/theory/core_math.tex`; obtain explicit approval
before editing either. This note changes no architecture or training behavior;
its scope estimates are based on source inspection, not a CUDA experiment.
