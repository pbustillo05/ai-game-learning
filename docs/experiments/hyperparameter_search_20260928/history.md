# Hyperparameter search history audit — 2026-09-28

This is a read-only reconstruction for experiments starting from
`runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt`.  It uses the diagnosis
artifacts, checkpoint/config provenance, and git history.  It does not claim
that a prior short smoke proved poker quality.

## Highest-confidence conclusions

The old and August long runs cannot be compared as “the same recipe trained
longer.”  They differ in utility units, coefficient balance, optimizer
settings, reference cadence, initialization, and work per iteration.  More
fundamentally, all queued August learning evidence before commit `0a09496`
has a completion-to-target row permutation: completion appends rows in Leaf
FIFO order, while `build_layer_plan` assigned slots in topology edge/DFS
order, and `_execute_completion` did not apply the already-declared movement.
The CUDA proof is exact: Normal checkpoint 50 with 128 roots had 120/128
wrong terminal value targets (maximum error 64 bb), and RB checkpoint 1100 had
508/512 misplaced leaf slots and an unrelated posterior mass of 0.08928384.
The production repair `0a09496` passes the 16 layer/iteration identity cases;
the diagnosis suite reports 231 CUDA tests passing (one distributed test
skipped because only GPU 0 was visible).

The direct causal evidence is in
`runs/experiments/nlhe_hu_32bb_rb_training_diagnosis_20260925/mapping_bug_history.md`,
`completion_alignment_proof.md`, and `README.md`.  The activating history is:

| Commit | Relevance |
| --- | --- |
| `880ff86`, `fdffaad` | Introduced/strengthened layered target movement metadata and arena initialization. |
| `8e269a4` | First queued sampled iteration bypassed the movement application; ordinary sequential order could mask it. |
| `7434d41` | Removed the production movement helper while test fixtures retained it. |
| `4616ddb` | Sorted-root DFS topology sealing activated FIFO-versus-topology order divergence in sequential self-play. |
| `35a5a33` | Moved poker completion into generic `_execute_completion`, preserving the omission. |
| `ed4a73e` | Recorded revision for both August runs; contains the defective composition. |
| `0a09496` | Production planner now assigns the actual FIFO completion addresses. |

Treat all pre-`0a09496` continuation, entropy, ratio, and restoration results
as diagnostics of the broken learner signal.  The corrected restoration smoke
shows execution and finite updates, not strong play.

## Objective units and the valid continuity controls

July’s successful distributed RB run used raw big blinds:

```text
entropy 0.05, reference KL 0.2, value coefficient 0.0005,
value clip 0.2 bb, LR 5e-5, AdamW decay 0.01,
fixed reference interval 1380 iterations, 116 updates/iteration,
global batch 256, 1024 completion samples/leaf.
```

The run is `runs/nlhe_hu_32bb_full_sequential_rb`; its command launched
2026-07-18 and its final checkpoint is `ckpt_0001110.pt`.  The historical
initializer was checkpoint 20, loaded as model-only with fresh optimizer and
reference; checkpoint 20 is absent, but the checkpoint-1110
`reference_model` preserves that fixed July initializer/reference.

August Normal/RB used rewards divided by stack size 32, while keeping value
targets and advantages unstandardized.  Objective-continuity settings are:

| Quantity | July raw-bb | Equivalent stack-normalized value | Actual August |
| --- | ---: | ---: | ---: |
| entropy coefficient | 0.05 | 0.0015625 | 0.025 |
| reference-KL coefficient | 0.2 | 0.00625 | 0.05 |
| value coefficient | 0.0005 | 0.016 | 0.125 |
| value clip | 0.2 bb | 0.00625 | 0.2 (= 6.4 bb) |

The exact conversion is documented in
`runs/experiments/nlhe_hu_32bb_rb_training_diagnosis_20260925/hyperparameters.md`.
To continue checkpoint 1110 in stack units, divide the four pre/post value
head affine tensors by 32, keep all policy/trunk tensors unchanged, create a
fresh optimizer, and use the continuity coefficients above.  Commit `1cde6de`
(`docs/audit/nlhe_hu_32bb_normalized_loss_smoke_20260721.md`) performed this
migration on CUDA, used value coefficient `0.016`, value clip `0.00625`, KL
`0.00625`, and entropy `0.001` linearly decayed over 500 iterations.  It chose
`0.016` as the continuity control; the three-iteration test only established
finite updates and gradient/term telemetry.  It did not establish strategy
quality or a cadence choice.

The later August recipe came from fresh-training configs in `3ec332d`
(`configs/experiments/nlhe_hu_32bb_rb_24h_20260722.yaml` and its Normal peer),
then was carried into H100/August configs.  No saved controlled policy-quality
or component-gradient selection was found that justified `0.125/0.025/0.05`
over the continuity control.  At the unopened HU root, entropy is over every
legal event, including 61 half-bb raise sizes plus fold and call.  Its
entropy-only optimum is 61/63 = 96.8254% raising with uniform conditional
size; the raise-size entropy contribution is `0.05 log(61) = 0.206 bb` in
July versus `0.8 log(61) = 3.289 bb` in the August raw-bb-equivalent objective.
This is a real objective pressure, while the target permutation is the proven
pre-repair failure.

## Input encoding and value-head migration

The recovered July network requires the historical exact-hand preflop input
gauge for valid comparison.  Historical input gauges each exact private hand
before summing into 169 classes, yielding root pair/suited/offsuit masses
`6/4/12`; queued preflop inputs use already-summed class reach, yielding
`0.5/0.333/1`.  The network sees raw and log range values and is not invariant
to this scale.  The CUDA `legacy_input_audit.py` found historical/current
network parity within `1.8e-7` in action probability and `9.6e-7` in value at
root and BB facing 3 bb when the adapter is used; postflop parity was
bit-for-bit.  Loading old weights directly with queued inputs changed some
root probabilities by `0.646` and BB probabilities by `0.755`.

The actual normalized restoration path combines these two operations: apply
the historical preflop adapter on learner, frozen-reference, and evaluation
instances, and divide only the four value-head affine tensors by 32. This
preserves the July policy probabilities exactly while expressing value outputs
in stack-normalized units. The value-head conversion does not convert input
encoding and must never be presented as a queued-input migration. The aligned
raw-bb replay proves the adapter path, while `1cde6de` proves the deterministic
value-head migration; the normalized restoration experiment composes both.

An independently trained queued-input model is a different baseline and should
be compared separately. The adapter is an experiment/evaluation compatibility
path, not evidence that scratch-trained August models were wrong; those models
used queued inputs consistently.

## Effective-hand terminology and cadence trap

There are two different quantities that must never share an unlabeled
`effective_hands` field:

* **Production cadence exposure:** `pbg/games/poker/queued/hooks.py:2522-2525`
  sums `completed.n_eff` per leaf. The sibling evaluator contract
  `/home/pablob/poker-gpu-evaluator/docs/api.md` defines
  `n_eff = 1 / sum_j q_j^2`, where `q_j` are normalized positive SIS world
  weights. This is Kish effective completion-world sample size, bounded by the
  completion sample count. The runtime passes this sum into the reference
  cadence (`pbg/runtime.py:811-828`). Normal has no completion posterior and
  uses fresh player-decision count instead (`pbg/runtime.py:814-817`).
* **Diagnostic posterior breadth:** the prior plan in
  `git show 4fd5220:docs/audit/nlhe_hu_32bb_loss_diagnostics.md` proposed
  `exp(-sum_h q_d(h) log q_d(h))` over the exact-hand posterior for each RB
  decision. This is an entropy effective number of private hands, useful as an
  information/breadth diagnostic. It is not evaluator `n_eff`, not an
  independent-sample count, and must not drive the production reference
  threshold unless intentionally introduced as a new schedule.

The historical RB config comment in
`git show 4fd5220:configs/experiments/nlhe_hu_32bb_rb_24h_20260722.yaml` says
`900k matched fresh decisions times measured ~192 effective hands`, giving
`172.8M` cadence exposure. The archived evaluator contract at the same
historical revision defines `n_eff = (sum w)^2 / sum(w^2)` for the SIS world
weights, and archived benchmark fields call the quantity
`completion_mean_n_eff`/`completion_effective_samples`. This is strong direct
evidence that “~192” referred to completion Kish ESS, although the config
comment itself does not spell out the formula and should be cited as an
empirical label rather than a fresh measurement. It is not the entropy
posterior breadth defined above. The Normal threshold `900k` was calibrated
directly to approximately 100 iterations of fresh decisions. The August
queued runs replaced these with 110 iterations for RB and 305 for Normal, and
changed workload from July’s 6144 global roots to 1536 roots and from 116 to
38 updates/iteration. RB’s 110 × 38 = 4,180 optimizer steps/reference, versus
July’s 1380 × 116 = 160,080.

The practical controls are:

* Keep runtime cadence exposure named `completion_n_eff` (RB) and
  `fresh_player_decisions` (Normal), and log both per leaf and aggregate sums.
* If adding private-posterior breadth, call it
  `posterior_entropy_effective_hands` and report its per-decision and
  learner-presentation aggregates separately. Do not replace evaluator
  `n_eff` with it.
* Preserve both public-node/fresh-decision counts and optimizer presentation
  counts; repeated with-replacement learner rows are not independent samples.
* Compare cadence arms at matched exposure and fixed semantics: July fixed
  1380 iterations, evaluator-ESS thresholds, and any entropy-breadth threshold
  are distinct factors. A cadence threshold alone is not evidence of quality.

## Distributed and workload evidence

Historical July RB was two-rank: 2560 roots/rank for iterations 0–9, 2816
for 10–17, then 3072/rank through 1109, i.e. 6144 global mature roots,
29,696 global learner presentations/iteration, and 116 statistical updates.
Across 1110 iterations it recorded 6,805,504 trajectories, 24,619,099 fresh
decisions, 32,962,560 statistical presentations, and 128,703 applied optimizer
steps. August Normal/RB used 1536 roots total, 9728 presentations and 38
updates per iteration; 10,000 iterations therefore hide a much smaller
trajectory/optimizer budget than “10,000” suggests.

The August launch audit says 2048/2560/3072-root candidates failed CUDA OOM
gates on the local 2080 Ti while 1536 completed. This is capacity evidence,
not a scientific preference. The repaired pipeline should first use one lane/
GPU and matched logical work, then scale roots or ranks only after identity and
quality gates pass.

Separate H100 CUDA measurements in `docs/gpu_systems_crash_course.md` support
K=1 and root parallelism for the implementation measured: at 512 roots,
K=2 improved aggregate throughput only 0.95% Normal and 3.10% RB, below a
10% retention threshold; increasing roots 512→4096 gave about 6.68× Normal
and 6.64× RB nodes/s (about 83–84% linear). Earlier K>1 timing had duplicate
learner work and is not a clean learner-scaling result. K=4 at 64 roots passed
stability/memory but was not a fair throughput comparison; K=8 was not run.
The selected H100 baseline was K=1, 4096 roots, rollout/completion/learner
packs 256/512/256, global batch 256, and 116 steps. Treat these as execution
controls, not claims about learning quality.

## Recommended first experiment matrix

After the production repair, use checkpoint 1110’s recovered July reference/
initializer and hold work, seed, reference, input adapter, completion samples,
optimizer, and model fixed. Establish one raw-bb July control and one
stack-normalized continuity arm (value-head `/32`, coefficients
`c_V=.016`, `c_H=.0015625` or the prior smoke’s explicitly labeled `.001`,
`c_KL=.00625`, value clip `.00625`). Add a zero-entropy diagnostic arm only
after the two controls pass identity, support, finite-gradient, full-action-KL,
and value-error gates. Then vary one factor at a time: entropy schedule,
reference cadence convention, value coefficient, and completion sample count.

For every arm report fixed-deal/frozen-opponent evaluation, hand-dependent root
policy statistics, conditional raise-size entropy, full-action KL, weighted
ratio tails, posterior support, value RMSE/explained variance, component
gradient norms, fresh decisions, optimizer presentations, completion samples,
completion `n_eff`, and (RB only) separately named posterior entropy breadth.
Short CUDA smokes can reject broken configurations; they cannot rank poker
strength. Promote candidates to medium runs and multiple seeds only after the
repaired leaf identity and exposure instrumentation remain exact.

## Evidence boundaries

CUDA-backed facts: the `0a09496` leaf identity repair/replays; the Normal/RB
alignment proofs; legacy input parity and direct-scale failure; the normalized
loss smoke’s finite updates and component terms; entropy continuation probes;
the H100/local throughput and OOM gates cited above. Source/config facts are
historical reconstructions, not reruns: July’s exact pre-queued source is
strongly inferred from `537700f` and its launch timing; August RB had an
uncommitted launch delta despite recorded `ed4a73e`; the exact long-run
causal contribution of coefficients, cadence, and input semantics was not
isolated because target corruption was present. No prior run establishes that
the continuity controls or any entropy choice produce strong play.
