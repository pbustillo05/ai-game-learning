# Mathematics and model traceability

## Mathematical verdict

The implemented fixed-intensity PPO subset is conformant with the constructive
estimator mathematics in `docs/theory/core_math.tex`. This does not promote the
note's unproved named-method or convergence claims; the qualifications in
`docs/theory/core_math_audit.md` remain in force.

The active theory and notation now consistently distinguish implemented
features from research extensions:

- exact legal-action `KL(new || reference)` and immutable periodic reference
  snapshots are implemented;
- optional public moment output heads exist structurally, but no production
  reducer target, loss, or branching controller consumes them;
- NeuRD, R-NaD/DeepNash, adaptive variance-guided branching, public-moment
  losses, stored all-action Q targets, and suit consistency remain deferred.

## Core estimator trace

| Mathematical requirement | Production implementation | Final assessment |
|---|---|---|
| Curried policy emits every actor private type | $R^2$ preflop classes lift to $\binom{4R}{2}$ combos; postflop compatible rows scatter to the same full width | Conformant for every legal R |
| Public likelihood reach updates on actions/chance | sequential/branched rollout plus poker reach update and board mask | Conformant |
| Sequential mode samples one profile/path per start | one active occurrence/profile per global start | Conformant |
| Normal player-clock lambda return | owner-depth backward recurrence | Conformant |
| RB completes pre-terminal leaves under the posterior | sibling completion facade and streamed reducer | Conformant |
| Preserve posterior mass $P$ and moment $M=PV$ before division | fp32 streamed frontier and guarded conditionalization | Conformant |
| Parent is uniform child mixture | child $P/M$ sums divided by labeled child count | Conformant |
| Leave-one-out sibling baseline | total-minus-child conditionalization, then lambda mix | Conformant |
| Single-child baseline is cached old V | explicit single-child mask | Conformant |
| Bootstrap only on owner clock | owner row alone receives lambda bootstrap | Conformant |
| One distinguished branched root | one root/tree per rank | Conformant |
| Root uses ordinary randomized rounding | root enters the same offspring path as every other branchable node | Conformant |
| Duplicate labels remain distinct children | stable parent-local ordinal and duplicate-preserving CSR | Conformant |
| Path law uses product of $1/N$ | stored path weight drives tree/rank objective mass | Conformant |
| Unequal ranks pool by decision mass | local/global ratio plus one flat fp32 gradient SUM | Conformant |
| Common nonfinite result controls update | extracted finite-step controller gates clip, step, and scaler transition | Conformant; B-03 resolved |

Branched children use independently addressed posterior-predictive private draws;
one shared true profile is never reused across a sibling family.

## Loss-law trace

For `rb_ppo`, production implements

$$
\frac1B\sum_b\left[
\frac1{K_b}\sum_{a\in\mathrm{children}(b)}\sum_x
\widehat P_b(x\mid a)\,\mathcal L_{\mathrm{clip}}(x,a)
+\sum_x\widehat P_b(x\mid b)
  (\mathcal L_V+\mathcal L_H+\mathcal L_{\mathrm{KL}})
\right].
$$

Child posterior mass weights policy terms; parent posterior mass weights value,
entropy, and optional reference KL. Duplicate abstract children remain separate
columns. Normal `ppo` selects the sampled actor combo and uses its scalar
player-clock return. MSE/clipped values, constant/corrected linear entropy, and
reference KL are all covered by tests. Numeric domains now fail at configuration
load, and the reduced finite decision is the sole optimizer gate.

## Model architecture trace

The configured suffix deck has $D=4R$ cards, $H=\binom{D}{2}$ private combos, and
$K=R^2$ canonical preflop classes. Full-deck constants 52/1326/169 are examples,
not tensor contracts.

| Architecture requirement | Production implementation | Final assessment |
|---|---|---|
| Nonshared pre/post weight object | `PokerModelBundle`, `PreflopNet`, `PostflopNet` | Conformant |
| Variable canonical preflop rows | geometry-derived class aggregation and class-native trunk | Conformant for R=4..13 |
| Compatible live-player postflop rows | geometry-derived compact map and full scatter | Conformant for R=4..13 |
| Folded players absent from attention rows | live-seat masks and folder summaries | Conformant |
| Lossless public history | geometry-derived capacity plus binary-lift device gather | Conformant for all P/S boundaries |
| Button-relative seats and exact public arithmetic | poker public batch construction | Conformant |
| Pre-LN reach-biased MHA | learned positive temperature; untempered key-side reach prior | Conformant |
| Exact incidence residual | card accumulators plus identical-combo channel | Conformant |
| Factored passive/call/raise policy | DLogisticTick mixture plus all-in atom and legal normalization | Conformant |
| Actor-private-type V, no learned action Q | actor row value head | Conformant |
| Optional public moments | `MomentHeads` behind configuration, no production target/loss/control | Structural only, documented accurately |

Frozen rollout-policy inference and learner inference now share the configured
autocast precision. A 20-update resume probe found and fixed the former
FP32-versus-FP16 mismatch; post-fix launch/resume identity errors remain within
the declared tolerance.

## Folded-player statistics

The sibling evaluator remains the only owner of folded-card partition math.
Poker stores fold-time public dependencies and sends masked rows to the sibling
facade. Returned unary corrections, signed pair factors, folder mass/card/seat/
line fields, and optional tight-folder terms feed corrected reach, attention
geometry, pressure channels, and folded-seat summaries. Only live players have
postflop hand rows.

The repaired cache is fixed-size, memory-planned, and device-indexed. Cache-hit
rows cross the facade under a false evaluator mask and merge on device; no
Python row key or hit decision is used. Tests cover one and multiple folders,
semantic feature influence, reuse, and fold/street invalidation.

## Deliberately absent capabilities

- NeuRD and R-NaD/DeepNash;
- adaptive/Neyman branching intensity and intensity loss;
- public moment targets/losses/control;
- stored all-action Q targets or per-action value heads;
- suit-consistency, MMD, and sampled-KL losses;
- exploitability or approximate-best-response evaluation.

The last item is M-05 and is explicitly outside this repair scope. Therefore
none of the engineering evidence is a strategy-quality claim.
