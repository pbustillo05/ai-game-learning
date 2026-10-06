# NashPG comparison for the September 2026 hyperparameter search

Inspected 2026-09-28. This is research guidance and experiment design, not a
claim that any candidate is optimal. No training was performed for this note.
The parent experiment uses the available historical **checkpoint 1110**, not
the originally requested checkpoint 1100.

## Primary sources and version provenance

- Local paper: `docs/references/nashpg/paper.pdf` and `source/`, arXiv
  **2510.18183v2**, submitted 2026-04-30.
- Local reproduction repository: `docs/references/nashpg/repo/`, recorded in
  `docs/references/nashpg/NOTES.md` as commit
  `d85314e8b5a72e45f350bdf4324601fe22ecaea6`, 2026-04-29, from
  [easonyu0203/nash_policy_gradient_public](https://github.com/easonyu0203/nash_policy_gradient_public).
- The latest [arXiv record](https://arxiv.org/abs/2510.18183) is **v3**, submitted
  2026-08-03. Its [full paper](https://arxiv.org/html/2510.18183v3) points to
  [ntu-agents/nashpg](https://github.com/ntu-agents/nashpg).
- Latest repository values below were read from the public `main` branch on
  2026-09-28; a latest commit hash was not obtained. They are an observation of
  a moving branch, not a pinned reproduction snapshot.

The latest paper retains 10,000 inner updates and 25 outer rounds, with entropy
0.1 and 64 environments times 64 rollout steps. It explicitly distinguishes
the ideal IMMD convergence theorem from practical neural NashPG, for which
Table 1 lists no convergence guarantee. Its learned best-response evaluation
is a lower bound on true exploitability. These distinctions matter when
interpreting short poker continuations. [NashPG v3](https://arxiv.org/html/2510.18183v3)

## Reported experiments versus defaults

| Quantity | Local paper v2 / latest paper v3 | Vendored code | Latest official code |
|---|---:|---:|---:|
| Learning rate | 0.0003 | 0.0003 | 0.0003 |
| Reference KL coefficient | 0.2 | 0.2 | 0.2 |
| Entropy coefficient | 0.1 | 0.05 | 0.1 |
| Entropy decay | None documented | None | None |
| Environments times steps | 64 × 64 | 32 × 128 | 64 × 64 |
| Transitions per rollout | 4,096 | 4,096 | 4,096 |
| PPO epochs times minibatches | 4 × 4 | 4 × 4 | 4 × 4 |
| Inner rollout updates per reference | 10,000 | 1,000 | 1,000 |
| Outer rounds | 25 | 50 | 50 |
| Advantage standardization | Not explicit in parameter table | Per minibatch | Per minibatch |

The [latest defaults](https://raw.githubusercontent.com/ntu-agents/nashpg/main/conf/algorithm/nash_pg.yaml)
now match the paper's entropy coefficient and rollout shape, but the reference
cadence discrepancy remains. The [latest README](https://raw.githubusercontent.com/ntu-agents/nashpg/main/README.md)
also gives a launch example overriding outer rounds to 10. Examples, defaults,
and reported experiment settings must therefore remain distinct.

Exact local evidence, with line numbers as inspected:

- `docs/references/nashpg/source/experiments.tex:27`: experiment schedule.
- `docs/references/nashpg/source/experiments.tex:56`: selection of KL coefficient 0.2.
- `docs/references/nashpg/source/appendix_eval_metrics_domain.tex:95`: PPO table.
- `docs/references/nashpg/repo/conf/algorithm/nash_pg.yaml:1`: local defaults.
- `docs/references/nashpg/repo/train/core/update_agent.py:69`: masked minibatch
  advantage standardization.
- `docs/references/nashpg/repo/train/core/update_agent.py:91`: categorical
  $\mathrm{KL}(\pi_{current}\|\pi_{reference})$ at sampled observations.
- `docs/references/nashpg/repo/train/core/update_agent.py:105`: actor objective;
  line 112 gives critic coefficient 0.5 and value clipping by PPO epsilon 0.2.
- `docs/references/nashpg/repo/train/core/update_agent.py:149`: shuffled
  minibatches; line 177 repeats epochs.
- `docs/references/nashpg/repo/train/nash_pg.py:146`: AdamW, epsilon `1e-5`,
  global gradient clipping 0.5. Other AdamW arguments use library defaults.
- `docs/references/nashpg/repo/train/nash_pg.py:205`: hard reference replacement
  after the inner loop.
- `docs/references/nashpg/repo/envs/head_up_poker/__init__.py:575`: rewards
  divided by `STARTING_STACK`, 200 chips or 100 bb.

The nominal paper reference interval is 40.96 million collected transitions,
163.84 million transition presentations, and 160,000 optimizer steps. The
default-code interval is one tenth those budgets. These are player-decision
transitions, not complete poker hands; validity masks can reduce useful records.
The paper's total nominal collection budget is 1.024 billion transitions.

NashPG uses 100 bb stacks and 16 categorical actions, while our run uses 32 bb
and individual legal half-bb raise events. Its network uses separate policy
and critic feature extractors. Differences in action entropy, trajectory
length, gradient coupling, and information weighting prevent direct numeric
transplantation of coefficients.

## Mapping to stack-normalized HU 32 bb rewards

For rewards in stack units, one unit is 32 bb and HU terminal utility lies in
$[-1,1]$. The historical continuity control is:

| Parameter | Historical raw-bb objective | Stack-normalized equivalent |
|---|---:|---:|
| Entropy coefficient | 0.05 | 0.0015625 |
| Reference KL coefficient | 0.2 | 0.00625 |
| Squared-value-error coefficient | 0.0005 | 0.016 |
| Value clip | 0.2 bb | 0.00625 stack |

With advantages, critic predictions, and targets divided by 32, these changes
preserve the old objective up to an overall factor of $1/32$. They do not
guarantee identical optimizer trajectories: output-layer reparameterization,
Adam moments, epsilon, and clipping all matter. Controlled continuations should
use a consistent optimizer initialization/conversion policy across arms.

The diagnostic source is
`runs/experiments/nlhe_hu_32bb_rb_training_diagnosis_20260925/README.md:89`.
It also records that later entropy/KL/value coefficients exceeded this
continuity control by 16, 8, and 7.8125 times respectively. The historical target
assignment bug prevents assigning all historical behavior to hyperparameters.

NashPG's [latest loss code](https://raw.githubusercontent.com/ntu-agents/nashpg/main/train/core/update_agent.py)
standardizes advantages per valid minibatch. Our inspected loss at
`pbg/core/learner.py:1992–2139` does not. Thus even with normalized rewards,
NashPG's 0.2 KL and 0.1 entropy act relative to a roughly unit-scale advantage,
whereas ours act relative to the actual normalized poker advantage.

As a rough diagnostic, if normalized advantage standard deviation is
$\sigma_A$, an unstandardized coefficient comparison would be
$c_{KL}\approx0.2\sigma_A$ and $c_H\approx0.1\sigma_A$. This is not an exact
translation or a recommendation to make coefficients batch-dependent:
centering, PPO clipping, posterior weighting, and state sampling differ.
Measure component gradient norms and their alignment with the policy gradient
instead of calibrating from scalar loss magnitudes alone.

## The effective-hand clock currently measures something else

The user's intended quantity $\sum\exp(H(posterior))$ is a measure of posterior
hand coverage. It is not the current queued runtime's reference clock:

1. `pbg/games/poker/queued/hooks.py:2522` accumulates evaluator completion
   `n_eff`.
2. `pbg/runtime.py:815` feeds this quantity to the reference clock for RB,
   while Normal uses player-node count.
3. The sibling contract at
   `/home/pablob/poker-gpu-evaluator/docs/api.md:275` defines sample ESS as
   $1/\sum_j q_j^2$. This is completion-world sample health, not hand-posterior
   perplexity. The sibling API was inspected read-only.
4. Current equal-weight CUDA completion sets `n_eff=n_samples` at
   `/home/pablob/poker-gpu-evaluator/poker_gpu_evaluator/csrc/evaluator_cuda.cu:528`.
5. `pbg/core/learner.py:218–244` supports iteration-based entropy decay only.
6. Reference threshold overshoot is discarded on refresh at
   `pbg/core/learner.py:2222–2226`; large iteration batches change the realized
   cadence.

Consequently increasing completion samples can accelerate KL refresh without
adding independent public histories. Treating the resulting number as learned
hands confounds estimator precision with policy-data exposure.

Keep separate counters for unique public decisions, posterior perplexity,
completion ESS, and learner presentations. Posterior perplexity remains a
useful coverage diagnostic, but many private hands under one public trajectory
are correlated. It cannot by itself establish equivalence to that many IID
sampled trajectories. Normal and RB also retain public-trajectory variance,
opponent nonstationarity, and optimizer noise.

A better empirical bridge is frozen-policy gradient variance: at checkpoint
1110, compare independently sampled Normal and RB batches, estimate their
gradient variance ratio at matched public-state counts, then check improvement
per wall-second. Use a calibrated exposure clock consistently across modes,
with separate optimizer-update counts and reference age visible. A large
collected-exposure count alone does not solve NashPG's inner optimization.

## Concrete normalized-control experiment axes

These are proposed candidates, not selected winners:

- Initialize every paired arm from checkpoint **1110**, using stack-normalized
  rewards and correctly rescaled critic outputs. Hold seed, initial reference,
  optimizer treatment, workload, and evaluation deals fixed where possible.
- Start with value coefficient **0.016**, value clip **0.00625**, and KL
  **0.00625**. Measure actor/critic/entropy/KL gradient norms before widening
  the search.
- Entropy arms: **zero immediately** for the trained checkpoint; or small
  initial **0.0005** / **0.0015625**, decaying to **exactly zero** over a stated
  exposure horizon. Do not silently restart a large exploration bonus.
- KL-strength arms: **0.003125, 0.00625, 0.0125**, with cadence fixed.
- Cadence arms, tested separately: frozen anchor over the short comparison,
  then **0.5×, 1×, 2×** a measured baseline exposure interval. Record both
  public decisions and optimizer steps between replacements; do not copy a
  1,000/10,000 rollout count without mapping its workload.
- Learning-rate arms around a conservative continuation setting:
  **0.000025, 0.00005, 0.0001**, chosen only after measuring clip fraction,
  policy KL movement, and component gradients. NashPG's 0.0003 is a literature
  point, not a justified starting value for our architecture.
- Compare Normal and RB at equal wall-time and at matched public-data budgets.
  Vary completion samples separately from reference cadence and measure target
  uncertainty; otherwise a throughput knob also changes the learning schedule.

NashPG does not supply an entropy-decay schedule. Its appendix experiment
anneals the magnet regularization coefficient, a different term; that finding
does not oppose the user's requirement that exploration entropy vanish. Keep
the reference KL mechanism available while independently removing the entropy
bonus. Evaluate poker quality with paired matches and trained best responses;
lower root raise probability or lower entropy alone is not evidence of stronger
play.
