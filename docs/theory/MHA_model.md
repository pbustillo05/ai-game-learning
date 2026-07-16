# Poker Player Model

> **Poker-math boundary.** Hand/card-indexed features in this document are
> model inputs, not permission to compute poker-private combinatorics locally.
> Collision-conditioned posteriors, partition/folded-card summaries,
> completion, sampling, and showdown winners are produced by
> `poker-gpu-evaluator` under `archive/POKER_GPU_EVALUATOR_IO.md`. This repository only
> learns transformations of supplied fields, applies the fixed card/class
> representation incidence operator inside neural layers, and navigates the
> public tree. Any later wording that says the model computes folded-card
> kernels, partition corrections, collision-conditioned laws, or private
> assignments is descriptive of desired input semantics and is superseded by
> this boundary; those private-law fields must arrive from the sibling facade.

The deployed object is one weight-only policy object with two networks:

$$
\theta=(\theta_{\mathrm{pre}},\theta_{\mathrm{post}}).
$$

The preflop and postflop networks do not share weights. They share the same
MHA-style sequence architecture. For a run with the highest $R\in\{4,\ldots,13\}$
ranks, let $D=4R$, let $\mathcal D_R=\{52-D,\ldots,51\}$ be the evaluator-facing
suffix deck, and let $\mathcal H_R=\binom{\mathcal D_R}{2}$. The preflop model
runs on $R^2$ canonical hand classes and the postflop model runs on the
$|\mathcal H_R|=\binom{D}{2}$ hands. The full-deck case has 169 classes and
1326 hands. Only $\theta$ is
learned; there are no strategy, regret, or average-policy tables. The
architecture follows:

- transformer sequence layers over player hand/class rows, public tokens, and
  action-anchor tokens;
- an exact incidence residual carrying unnormalized masses and identical-combo
  overlap, as defined by the incidence identities in
  `docs/theory/folder_impact.md`;
- postflop folded-player effects injected through folded statistics rather than
  folded-player hand rows.

## Public Contract

Let $\sigma$ be the public trajectory, $p_\star=o(\sigma)$ the acting player,
$\mathfrak r$ the Bayesian-updated ranges, and
$\mathcal K_R=\mathcal H_R/S_4$ the $R^2$ preflop hand classes.
The overall $F_\theta$ dispatches to the preflop or postflop network according
to the street of $\sigma$.

Preflop:

$$
F^{\mathrm{pre}}_{\theta_{\mathrm{pre}}}
\bigl(\sigma,p_\star,\{\mathfrak r_{p,k}\}_{p\in\mathcal P,\ k\in\mathcal K_R}\bigr)
=
\Big(
\{k\mapsto(\pi^{\mathrm{pre}}_{\theta_{\mathrm{pre}}}(\cdot\mid k,\sigma),
V^{\mathrm{pre}}_{\theta_{\mathrm{pre}}}(k,\sigma))\}_{k\in\mathcal K_R},
\ \text{optional } \mathbf V^{\mathrm{pre}}_{\theta_{\mathrm{pre}}}(\sigma),
\ \text{optional } U^{\mathrm{pre}}_{\theta_{\mathrm{pre}}}(\sigma)
\Big).
$$

Postflop:

$$
F^{\mathrm{post}}_{\theta_{\mathrm{post}}}
\bigl(\sigma,p_\star,
\{\mathfrak r_{q,h}\}_{q\in\mathcal P_{\mathrm{live}}(\sigma),\ h\in\mathcal H_R},
\mathcal F_\sigma\bigr)
=
\Big(
\{h\mapsto(\pi^{\mathrm{post}}_{\theta_{\mathrm{post}}}(\cdot\mid h,\sigma),
V^{\mathrm{post}}_{\theta_{\mathrm{post}}}(h,\sigma))\}_{h\in\mathcal H_R},
\ \text{optional } \mathbf V^{\mathrm{post}}_{\theta_{\mathrm{post}}}(\sigma),
\ \text{optional } U^{\mathrm{post}}_{\theta_{\mathrm{post}}}(\sigma)
\Big).
$$

The model never receives the dealt private hand as a selected input. It
computes one policy row for every actor hand class preflop or actor hand
postflop. The action policy is factored into an action-type distribution and a
conditional raise-size distribution. Board-impossible hands and illegal actions
are masked outside the model. Optional public moment output heads exist
structurally: $\mathbf V^\theta(\sigma)$ has one scalar per live player, and
$U^\theta(\sigma)$ is a single public scalar. The production reducer emits no
targets for them and no registered loss or branching controller consumes them.

## Layer Notation

- $\operatorname{LN}$: layer normalization.
- $\operatorname{MHA}$: multi-head attention as defined in *Definitions of
  Layer Primitives*, including per-head content temperature and additive key
  biases.
- $\operatorname{FFN}_\star$: affine $\to$ GELU $\to$ affine.
- $W_\star x+b_\star$: affine projection.

Any neural map in this file must be one of these layer types or an explicit
algebraic formula.

## Inputs

The public trajectory $\sigma$ packages public history, exact public arithmetic,
street, board, button-relative acting position, legal status, stacks, pot, call
amount, min/max raise, and raise ticks. The only learned-model inputs are
$\sigma$ and Bayesian ranges $\mathfrak r$, plus postflop folded statistics.
There is no privileged equity input, missing-equity flag, or equity-annealing
pretraining path.

Preflop consumes class ranges
$\mathfrak r_{p,k}\in[0,1]$ for every seated player $p$ and class
$k\in\mathcal K_R$. Postflop consumes hand ranges
$\mathfrak r_{q,h}\in[0,1]$ only for live players
$q\in\mathcal P_{\mathrm{live}}(\sigma)$ and hands $h\in\mathcal H_R$.
Normalized ranges mask board-impossible hands and normalize over each player's
available hands; raw ranges remain row features.

**Postflop folded statistics** $\mathcal F_\sigma$, supplied by the evaluator
partition-summary facade using the semantics in
`docs/theory/folder_impact.md` and refreshed
by the caller only on street or fold events:

- per-hand corrections $\Lambda_B(g)$, consumed as the **folder-corrected
  log-reach** $\log\tilde{\mathfrak r}^{\prime}_{q,g}
  =\log\tilde{\mathfrak r}_{q,g}+\Lambda_B(g)$
  (card-Möbius unary hand term);
- signed factors $\{(\lambda_j,b_j)\}_{j\le r_\beta}$ of the pairwise kernel
  $\beta_B$ (folded card-pair kernel), $r_\beta\lesssim16$ loose-dominated;
- tight-folder tilts $\{\delta_c\}$ and $\Sigma_{\rm tot}$ contractions
  (optional third-order folded-stat ingredients);
- per-folder scalars $\log F_k$ and street/line embeddings; unnormalized
  folded card-marginal vectors $w_k=F_kp_k$ for the per-card pressure channel.

All folder conditioning on removed cards happens through
$\Lambda_B,\beta_B$ evaluated at the actual board; nothing downstream
renormalizes on hypothesis-dependent sets. The active rules target is cash
chip-EV, 2–6 players, no rake, no straddles, no antes. Legality masks and
integer raise ticks are applied outside the model.

## Suit Symmetry

**Preflop**: the trunk runs natively on the $R^2$ canonical classes
$(p,k)$, so policy and value are class-indexed by construction.
**Postflop**: do not canonicalize suits before tokenization; the model consumes
the actual board, ranges, and folded statistics in their original card ids.
Optionally add a suit-consistency loss that samples one random suit permutation
$\rho\in S_4$ per training example, runs the permuted public state through the
same model, inverse-permutes the row-indexed outputs, and penalizes disagreement
with the unpermuted outputs.

Class-level geometry preflop: replace the two-hot indicator
$\mathbf 1_h\in\{0,1\}^{D}$ by the orbit-averaged card-distribution vector
$\bar{\mathbf 1}_k=\tfrac1{n_k}\sum_{h:\kappa(h)=k}\mathbf 1_h$
($n_k\in\{4,6,12\}$); overlap geometry becomes its expectation over class
orbits, and class tokens additionally carry $n_k$ and exact
within/cross-class expected-overlap scalars as features.

## Tokens

Token set: player hand/class rows, public tokens, and the actor's action-anchor
tokens. Preflop uses $R^2$ class rows for every seated player. Postflop uses
$|\mathcal H_R|=\binom{D}{2}$ combo rows for every live player by default;
board-impossible
rows have zero/masked range mass and receive no loss.

Postflop may optionally use only the board-compatible combo rows

$$
H_B=\binom{D-|B|}{2},
$$

so for $D=52$, $H_B=1176$ on the flop, $1128$ on the turn, and $1081$ on the
river. This
reduces attention dot products, but creates street/board-dependent ragged
shapes that may complicate batching. If used, gather `valid_combo_ids(B)` before
the trunk, carry `compact_to_combo` / `combo_to_compact` metadata, and scatter
actor outputs back to $\mathcal H_R$ rows after readout.

**Card and combo embeddings** (unchanged):
$c_v=\rho_{\operatorname{rank}(v)}+\sigma_{\operatorname{suit}(v)}$,
$\chi_h=\operatorname{FFN}_\chi([c_a,c_b,c_a\odot c_b])$. Board membership is
not mixed into $\chi_h$.

**Postflop row init.** For live player $q$ and hand $h=\{a,b\}$, the input
feature vector is

$$
\phi^{\rm geo}_h=\Big[\mathbf 1_h,\ \big\{\operatorname{sign-split}\big(\sqrt{|\lambda_j|}\,\langle b_j,\mathbf 1_h\rangle,\ \lambda_j\big)\big\}_{j\le r_\beta}\Big]
\in\mathbb R^{D+2r_\beta},
$$

(the two-hot indicator and the signed $\beta_B$-factor coordinates: positive-
and negative-eigenvalue factors kept in separate slots so a linear Q/K map can
realize $\mathbf 1_h^\top\beta_B\mathbf 1_g$ with correct signs), and

$$
x^0_{q,h}
=
W_x\big[\mathfrak r_{q,h},\,\log(\mathfrak r_{q,h}+\epsilon),\,
\log\tilde{\mathfrak r}^{\prime}_{q,h},\,\chi_h,\,\phi^{\rm geo}_h,\,
P_a{+}P_b,\,\mathbf 1\big]
+\rho_{\operatorname{pos}(q)}+\alpha_{\operatorname{status}(q)}+\zeta_{[q=p_\star]}
\in\mathbb R^{d},
$$

where $P_v=\sum_k w_k(v)E_k$ is the **folded per-card pressure** channel
($E_k=\operatorname{FFN}_{\rm fold0}([\log
F_k,\text{line}_k])$, cost $O(DK)$ per node), which preserves *which* cards
are folder-blocked and by what kind of line, beyond the scalar $\Lambda_B$
already inside $\log\tilde{\mathfrak r}^{\prime}$; the trailing $\mathbf 1$ is
a constant feature so key projections can realize pure additive biases;
$\zeta$ is an actor-flag embedding. Seat embeddings are button-relative.

**Preflop row init.** Preflop uses its own projection on class features
$[\mathfrak r_{q,k},\log(\mathfrak r_{q,k}+\epsilon),n_k,\text{class geometry},
\mathbf 1]$, with seat/status/actor embeddings and no folded-stat channels.

**Public tokens**:
$x^0_{\sigma,i}=W_yy_i+\tau_{\operatorname{type}(i)}+\pi_i$,
$x^0_{\sigma,\operatorname{state}}=W_qq$, and
$x^0_{\sigma,b_j}=W_b[c_{b_j},\beta_j]$.

**Action-anchor tokens.** The actor carries $K_{\rm act}$ learned slot tokens
$a^0_{p_\star,k}=\eta_k+\rho^A_{\operatorname{pos}(p_\star)}$, with mode
anchors
$\mathcal M_{\rm act}=\{\operatorname{passive},\operatorname{call},\operatorname{allin\_ctx}\}\cup\{\operatorname{raise}_m\}_{m\le M_R}$
used only at readout. They are ordinary sequence members — no separate
read/write machinery.

## Geometry Through the Score Function

Every pairwise hand–hand interaction enters either through a bilinear form in
card features, a player/type relation, a key-side reach prior, or the incidence
residual. Let $\ell_j$ be $\log\tilde{\mathfrak r}_{j}$ for preflop rows and
$\log\tilde{\mathfrak r}^{\prime}_{j}$ for postflop rows. The attention logit
for head $m$, query token $i$, key token $j$ is

$$
s^{(m)}_{ij}
=\frac{(W^Q_mx_i)^\top(W^K_mx_j)}{\tau_m\sqrt{d_k}}
\;+\;\ell_j\,\mathbf 1[j\text{ is a hand/class row}],
$$

with masked seats and any batch padding set to $-\infty$ (standard key-padding
mask) and per-head learned content temperature $\tau_m$ **applied to the
content term only** — the reach prior is a log-probability and must not be
tempered. Board-impossible rows are either masked in fixed-shape postflop
batches or absent when optional compact tokenization is used.

Because the token features contain $\mathbf 1_h$, the signed $\beta_B$
coordinates, player one-hots (inside seat/status embeddings), and a constant,
the learned bilinear $Q/K$ span includes, per head:

- **overlap biases** $\propto\langle\mathbf 1_h,\mathbf 1_g\rangle=|h\cap g|$;
- **type biases** $\propto\mathbf 1[p=q]$ and position/status interactions;
- **postflop folded pairwise potential**
  $\Delta_F(h,g)=\mathbf 1_h^\top\beta_B\,\mathbf 1_g$ exactly, via the signed
  factor coordinates from the card-Möbius expansion in
  `docs/theory/folder_impact.md`;
- arbitrary learned low-rank card-pair kernels (learned projections of
  $\mathbf 1_h$).

**No hard blocker masks by default.** Conflicting range mass is strategically
informative; head semantics — compatible-posterior heads with strongly negative
overlap coefficients, blocker-conflict heads with positive ones, same-player
range-shape heads — are **learned roles on shared features, not assigned
head types**. The exact identical-combo channel is routed to the incidence
residual because $\mathbf 1[g=h]$ is nonlinear in overlap
(see the incidence identities in `docs/theory/folder_impact.md`).

## Trunk

A pre-LN transformer of $L_{\rm trunk}$ layers over the full token set:

$$
\widehat X^\ell = X^\ell + \operatorname{MHA}^{(\ell)}(\operatorname{LN}X^\ell),\qquad
X^{\ell+1}_i=\widehat X^\ell_i+\operatorname{FFN}^{(\ell)}\big(\big[\operatorname{LN}\widehat X^\ell_i,\ \psi^\ell_i\big]\big),
$$

where the FFN side-input $\psi^\ell_i$ for hand/class rows concatenates card or
class embeddings, raw reach features, incidence-residual outputs $j^\ell$, and
the postflop bilinear units

$$
\psi^{\rm bil}_i=\big[\,O^{\rm card}_i\odot O^{\rm card}_i,\
O^{\rm card}_i\odot O^{=}_i,\
\{\langle b_j,\mathbf 1_h\rangle^2\}_j,\
\{u^\top\delta_c\}\text{-products}\,\big],
$$

which expose the optional third-order folded-stat products from
`docs/theory/folder_impact.md`.
Preflop uses analogous class-level overlap side inputs and omits folded-stat
terms.

## Incidence Residual

The exact mass path carries unnormalized disjointness-restricted sums and the
identical-combo channel from the incidence identities in
`docs/theory/folder_impact.md`, at
$O(D)$ per query via card accumulators. Per layer, it runs over tokenized range
players: all seated players preflop and live players postflop.
Below, $h,g$ denote postflop hands; preflop uses the orbit-averaged class-row
analogue.

$$
m^\ell_{q,g}=\tilde{\mathfrak r}_{q,g}\,\mathbf 1[g\cap\operatorname{board}=\emptyset]\;
W^S_{J}\operatorname{LN}x^\ell_{q,g},\qquad
C^{(q),\ell}_v=\sum_{g\ni v}m^\ell_{q,g},\qquad S^{(q),\ell}_h=m^\ell_{q,h},
$$

$$
O^{\rm card,\ell}_{q,h}=\sum_{q'\ne q}\big(C^{(q'),\ell}_a+C^{(q'),\ell}_b-2S^{(q'),\ell}_h\big),\qquad
O^{=,\ell}_{q,h}=\sum_{q'\ne q}S^{(q'),\ell}_h,\qquad
O^{\rm own,\ell}_{q,h}=C^{(q),\ell}_a+C^{(q),\ell}_b-2S^{(q),\ell}_h,
$$

$$
j^\ell_{q,h}=W^O_J\operatorname{GELU}\big(W^X_J\operatorname{LN}x^\ell_{q,h}
+O^{\rm card,\ell}_{q,h}+W^C_JO^{=,\ell}_{q,h}+W^{\rm own}_JO^{\rm own,\ell}_{q,h}\big),
$$

fed into the row FFN as part of $\psi^\ell$.
For postflop hand rows these are exact by the inclusion–exclusion identities of
`docs/theory/folder_impact.md`; $O^{=}$ is the only exact identical-combo channel in the
model. Cost $O(L|\mathcal H_R|d)$ postflop per layer, or $O(LH_Bd)$ under optional
compact tokenization.

## Readout

Per-seat summaries are in button-relative order; empty seats are zero-masked.

Actor: $t_{p_\star}=\operatorname{FFN}_{\rm seat}\big(\big[
a^{L}_{p_\star,\operatorname{passive}},a^{L}_{p_\star,\operatorname{call}},
a^{L}_{p_\star,\operatorname{allin\_ctx}},\bar a^{L}_{p_\star,R}\big]\big)$
with $\bar a^{L}_{p_\star,R}$ the mean over raise anchors.

Tokenized non-actor $q$: reach-weighted pooling of refined rows, with
$\tilde{\mathfrak r}^{\prime}=\tilde{\mathfrak r}$ preflop:

$$
t_q=\operatorname{FFN}_{\rm seat}\Big(\Big[
\sum_\xi \tilde{\mathfrak r}^{\prime}_{q,\xi}\operatorname{LN}x^{L}_{q,\xi}\Big/
\sum_\xi\tilde{\mathfrak r}^{\prime}_{q,\xi},\
\operatorname{status}(q),\operatorname{pos}(q)\Big]\Big).
$$

Postflop folded player $k$ (no rows): $t_k=\operatorname{FFN}_{\rm fold}\big(\big[
\log F_k,\ \text{line}_k,\ \operatorname{pool}_v w_k(v)E_k\big]\big)$.

With $\bar x^L_\sigma$ the mean of final public tokens,
$g^L=\operatorname{FFN}_{\rm read}\big([\bar x^L_\sigma,t_{p_\star},t_{q_1},\dots]\big)$.
For actor row $\xi$ ($\xi=k$ preflop or $\xi=h$ postflop), non-raise features
$z^T_{\xi,t}$ and raise-mode features $z^R_{\xi,m}$ are formed from
$[\operatorname{LN}x^{L}_{p_\star,\xi},\operatorname{LN}a^{L}_{p_\star,\cdot},
\chi_\xi,g^L]$. Policy and value heads are evaluated on actor rows. Optional
postflop compact tokenization scatters outputs back to full $\mathcal H_R$ ids.

## Continuous Policy Head

Action types
$\mathcal T=\{\operatorname{passive},\operatorname{call},\operatorname{raise}\}$
with external legality mask; logits $\lambda_{\xi,t}=w_{\lambda,t}^\top
z^T_{\xi,t}+b_{\lambda,t}$ and
$\pi_T(t\mid \xi,\sigma)=\operatorname{softmax}_t(\lambda_{\xi,t}+\log M_{\mathcal T,t})$.
The default 100bb game has $N=200$ global 0.5bb raise-to denominations,
$u_n=n/N$. A state supplies a legal subset $L(\sigma)$ rather than an affine
subsample. Mixture weights
$\omega_{\xi,m}=\operatorname{softmax}_m(w_\eta^\top z^R_{\xi,m}+b_\eta)$, locations
$\mu_{\xi,m}=\operatorname{sigmoid}(w_\mu^\top z^R_{\xi,m}+b_\mu)$, scales
$\sigma_{\xi,m}=\operatorname{softplus}(w_\sigma^\top z^R_{\xi,m}+b_\sigma)+\sigma_{\min}$,
all-in atom
$\iota_\xi=\operatorname{sigmoid}(w_\iota^\top z^T_{\xi,\operatorname{allin\_ctx}}+b_\iota)$:

$$
\pi_R(n\mid \xi,\sigma)
=
\iota_\xi\mathbf 1[n=\max L(\sigma)]
+(1-\iota_\xi)\frac{\mathbf 1[n\in L(\sigma)]}{C_{\xi,\sigma}}
\sum_{m=1}^{M_R}\omega_{\xi,m}
\operatorname{DLogisticTick}(n;N,\mu_{\xi,m},\sigma_{\xi,m}).
$$

$C_{\xi,\sigma}$ normalizes the logistic-mixture mass over the legal
denominations. Thus the head remains a compact continuous density
parameterization while sampling can select every fixed-denomination legal
raise, and the all-in atom follows the spot's actual maximum rather than the
global stack endpoint.

Binomial ticks remain an ablation.

## Value And Public Moment Heads

$$
V_\theta(\xi,\sigma)=w_V^\top\operatorname{FFN}_V\big(\big[
x^{L}_{p_\star,\xi},\chi_\xi,t_{p_\star},\bar x^L_\sigma,g^L\big]\big)+b_V.
$$

There are no learned per-action value heads.

For branched rollout training, either model may optionally emit public moment
heads read from final public/global and seat-summary activations rather than
from actor hand-row states. Let $q_1,\dots,q_{P_{\max}}$ be the fixed
button-relative seat order with empty seats zero-masked, and define

$$
c^L_{\rm pub}=
\big[g^L,\bar x^L_\sigma,t_{q_1},\dots,t_{q_{P_{\max}}}\big],
\qquad
m^L_{\rm pub}=\operatorname{FFN}_{\rm pubmom}(c^L_{\rm pub}).
$$

For every live player $q\in\mathcal P_{\mathrm{live}}(\sigma)$,

$$
\mathbf V^\theta_q(\sigma)
=w_{\mathbf V}^\top
\operatorname{FFN}_{\mathbf V}\big([m^L_{\rm pub},t_q,
\rho_{\operatorname{pos}(q)},\alpha_{\operatorname{status}(q)}]\big)
+b_{\mathbf V}.
$$

The public squared-return-moment head is

$$
U^\theta(\sigma)=
\operatorname{softplus}\!\big(
w_U^\top\operatorname{FFN}_{U}(m^L_{\rm pub})+b_U
\big).
$$

The optional public-head output is

$$
\mathbf V^\theta(\sigma)\in\mathbb R^{|\mathcal P_{\mathrm{live}}(\sigma)|},
\qquad
U^\theta(\sigma)\in\mathbb R.
$$

$\mathbf V^\theta$ has one scalar per live player and $U^\theta$ is a single
scalar estimating the squared-norm return moment used by the variance-based
branching intensities in `core_math.tex`.

## Definitions of Layer Primitives

All named maps are instances of the generic formulas below; a subscript or
superscript names only a private weight set, never a new operation. With
per-coordinate statistics $\mu(x),\sigma(x)$ and learned $\gamma,\beta$:

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu(x)}{\sigma(x)}+\beta,\qquad
\operatorname{GELU}(x)=x\,\Phi_{\mathcal N}(x),\qquad
\operatorname{FFN}_\star(x)=W^2_\star\operatorname{GELU}(W^1_\star x+b^1_\star)+b^2_\star,
$$

hidden width $4\times$ output unless stated; FFN arguments may be
concatenations.

**Attention.** Over queries $Q$ and keys/values $C$, $n_h$ heads of width
$d_k=d/n_h$, per-head content temperature $\tau_m$, additive logit field
$B^{(m)}$:

$$
\operatorname{MHA}_\star(Q,C,C)=[\operatorname{head}_1,\dots,\operatorname{head}_{n_h}]W^O_\star,\qquad
\operatorname{head}_m=\operatorname{norm}_m\!\Big(\frac{(QW^Q_{\star,m})(CW^K_{\star,m})^\top}{\tau_m\sqrt{d_k}}+B^{(m)}\Big)\,(CW^V_{\star,m}),
$$

where $B^{(m)}$ adds $\ell_j$ on hand/class-row keys, $-\infty$ on masked seats,
padding rows, and masked board-impossible rows, and $0$ otherwise; and
$\operatorname{norm}_m$ is row-wise $\operatorname{softmax}$.

Every named attention, FFN, and projection in this file is one of these maps
with private weights.
