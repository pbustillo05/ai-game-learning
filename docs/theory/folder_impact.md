# Card Removal from Folded Ranges: Theory

> **Ownership note.** This is mathematical background for
> `poker-gpu-evaluator`, not an implementation specification for this
> repository. `ai-game-learning` may consume versioned partition/folded-card
> summaries from the sibling facade, but must not implement the partition,
> removal-set, collision, or Monte Carlo machinery described below. Binding
> implementation ownership is defined by `archive/POKER_GPU_EVALUATOR_IO.md`.

**Scope.** This document is self-contained mathematics. It (1) states the problem
of how folded players' ranges bias the joint distribution of live hands and
board cards in no-limit hold'em, (2) derives the exact inclusion–exclusion
solution and explains why it is unusable as-is, (3) constructs a resummed
cluster-expansion estimator for the partition function, with separate treatment
of loose and tight folders, (4) reformulates the result as an **effective
potential on live hands** with an exact card-cluster (finite-difference)
expansion whose orders are the objects a model can consume, (5) specifies
corrected compressed per-folder summaries, and (6) defines the computation
budget, the Monte Carlo validation oracle, and teacher targets for learned
components.

The companion `blocker_model.md` in the sibling evaluator repository specifies
the network that consumes the objects defined here. It is intentionally not a
local file: this document contains no architecture, and the sibling model
document contains no derivations.

---

## 1. The problem

### 1.1 Poker setting

In a hand of no-limit Texas hold'em, each player holds a private 2-card hand
drawn without replacement from a 52-card deck; some cards are public (the
board). Players who fold are out of the pot, but their hole cards remain
*physically removed from the deck*: nobody else can hold them and they cannot
appear on future streets. The folder's cards are unknown, but they are not
uniformly distributed — a player who folded took an action, and the
distribution of hands consistent with that action (their **fold range**) is
informative. A player who open-folded preflop holds, roughly, "any two cards
minus the good ones." A player who called preflop, called the flop, and folded
the turn holds a much smaller, sharply shaped range — typically missed draws
and weak made hands, concentrated on specific ranks and suits.

A belief-state poker agent maintains a joint posterior over the live players'
hands (and, implicitly, the remaining deck from which future board cards are
drawn). The folded players' ranges *reweight* this posterior: a live-hand
hypothesis whose cards overlap heavily with what the folders likely held is
less probable than the naive range product suggests, and the composition of the
residual deck shifts accordingly. Quantifying this reweighting is the problem.

### 1.2 Formal model

Number the deck $\{1,\dots,52\}$. Fix a public board $B$ and a hypothesis for
the live players' hole cards $H_{\rm live}$; let

$$A \;=\; \{1,\dots,52\}\setminus(B\cup H_{\rm live}), \qquad R \;=\; B\cup H_{\rm live}$$

be the **residual deck** and the **removed set**. The $K$ folded players'
unknown hands must occupy *pairwise-disjoint* 2-subsets of $A$.

Each folder $k=1,\dots,K$ carries an unnormalized weight function
$f_k:\{\text{2-subsets of }\{1,\dots,52\}\}\to\mathbb R_{\ge0}$ — the
probability mass their strategy assigns to each hand on the line they took,
evaluated at the moment they folded (hence already conditioned on the board
cards visible at that time; $f_k$ vanishes on hands intersecting that board).
The total posterior weight of the configuration $(B,H_{\rm live})$ factors as

$$\Pr[B,H_{\rm live}] \;\propto\; \Big(\text{live range weights}\Big)\times Z(A),
\qquad
Z(A)\;=\;\sum_{\substack{X_1,\dots,X_K\subseteq A\\ \text{pairwise disjoint},\ |X_k|=2}}\ \prod_{k=1}^K f_k(X_k).$$

$Z(A)$ is the object of study: the total weight of all consistent assignments
of folded hands into the residual deck. It is a partition function with a
hard-core (no-overlap) constraint. Two remarks set the agenda:

1. **Only the $A$-dependence matters.** $Z(A)$ multiplies every
   live-configuration hypothesis; constants common to all $A$ cancel when the
   posterior is normalized. The strategic content is *how $Z$ varies as
   different cards are hypothesized into $H_{\rm live}$* — the
   blocker/card-removal signal.
2. **Exact evaluation is a matching-type partition function.** Identify each
   2-subset with an edge of the complete graph on $A$; then $Z(A)$ is a
   generating function of $K$-edge matchings with player-colored edge weights —
   a monomer–dimer partition function in the sense of Heilmann–Lieb [HL72].
   Such quantities are #P-hard in general, and brute force costs
   $\binom{|A|}{2}^K\sim 10^{3K}$ operations. Since $Z(A)$ must be evaluated
   inside a belief update, across up to $1326$ hand hypotheses per live player,
   per game state, an approximation with controlled error and
   $O(\text{poly}(K)\cdot|R|)$ per-evaluation cost is required.

### 1.3 Sufficient statistics of a range

All formulas below use only the following objects per folder, computed on a
stated base set (see §1.5) of size $a$ and renormalized there:

| Symbol | Definition | Interpretation |
|---|---|---|
| $F_k(A)$ | $\sum_{X\subseteq A} f_k(X)$ | total range mass surviving the removals |
| $M_k$ | $M_k(i,j)=f_k(\{i,j\})/F_k(A)$, $i\ne j\in A$ | normalized range matrix (symmetric, zero diagonal) |
| $p_k$ | $p_k = M_k\mathbf 1$, i.e. $p_k(i)=\Pr[i\in X_k]$ | card-inclusion marginals; note $\sum_i p_k(i)=2$ |
| $\delta_k$ | $p_k - \tfrac{2}{a}\mathbf 1_A$ | tilt: deviation of the marginals from flat |
| $\Sigma_k$ | $M_k+\operatorname{diag}(p_k)-p_kp_k^\top$ | covariance matrix of the card-indicator vector of $X_k$ |

Two identities, both consequences of the fixed hand size $|X_k|=2$, do a
disproportionate amount of work later:

$$\textbf{(I1)}\quad \big\langle \tfrac{2}{a}\mathbf 1,\ \delta_k\big\rangle = 0
\qquad\qquad
\textbf{(I2)}\quad \Sigma_k\mathbf 1 = 0,$$

(I2) because $M_k\mathbf 1 = p_k$, $\operatorname{diag}(p_k)\mathbf 1 = p_k$,
and $p_kp_k^\top\mathbf 1 = 2p_k$. Both hold for *every* range, flat or
concentrated.

A note for later use: $M_k$ is symmetric, nonnegative off-diagonal, and zero on
the diagonal. **It is generally indefinite** — even the uniform complete-graph
matrix has negative eigenvalues — and $\Sigma_k$, while PSD as a covariance,
has an off-diagonal part of both signs. Any factorization used downstream must
be signed (§5).

### 1.4 Probabilistic reformulation

Let $X_1,\dots,X_K$ be **independent** random hands, $X_k\sim f_k/F_k(A)$
supported on 2-subsets of $A$. Then, exactly,

$$Z(A)\;=\;\Big(\prod_{k=1}^{K} F_k(A)\Big)\cdot \Pr\big[X_1,\dots,X_K\ \text{pairwise disjoint}\big].$$

The first factor is the standard single-player card-removal effect (each
folder's range mass shrinks when its likely cards are removed); the second
factor — the **non-collision probability** $P_{\rm disj}(A)$ — is the genuinely
multi-player correction. Sections 2–3 are about estimating $P_{\rm disj}$.

### 1.5 Normalization conventions

Every quantity in this document is anchored as follows, and downstream
consumers must use the same conventions:

- $f_k$ is the raw, unnormalized strategy weight on the folder's line, defined
  on the full deck, zero on hands intersecting the board visible at fold time.
- At a game node with current board $B$, the **base residual set** is
  $A_0=\{1,\dots,52\}\setminus B$. All per-folder sufficient statistics
  ($F_k$, $M_k$, $p_k$, $\delta_k$, $\Sigma_k$) are computed **once per node on
  $A_0$** and renormalized there.
- Removals beyond $B$ — i.e., live-hand hypotheses — are *never* handled by
  re-normalizing on hypothesis-dependent sets. They are handled exclusively
  through the removal-set algebra of §3.4 or, equivalently and preferably,
  through the finite-difference operators of §4, both of which take
  $A_0$-based statistics as input.
- Live-player reach weights $r_{p,h}$ and their board-normalized versions
  $\tilde r_{p,h}$ are defined in `model.md`; this document treats them as
  given and produces *additive corrections in log space* to be combined with
  $\log\tilde r$.

---

## 2. The exact inclusion–exclusion formula, and why it is not ideal

### 2.1 The exact formula

For any finite sets, $\mathbf 1[X\cap Y=\emptyset]=\sum_{T\subseteq X\cap Y}(-1)^{|T|}$.
Applying this to every pair $k<l$ and expanding the product yields the exact
identity

$$Z(A)\;=\;\sum_{(T_{kl})_{k<l}}\ (-1)^{\sum_{k<l}|T_{kl}|}\ \prod_{k=1}^K W_k\Big(\bigcup_{l\ne k}T_{kl}\Big),
\qquad
W_k(T)=\sum_{\substack{X\subseteq A\\ X\supseteq T}} f_k(X),$$

a sum over all families of "claimed-overlap" sets $T_{kl}\subseteq A$. The
inclusion weights are explicit for $m=2$:

$$W_k(\emptyset)=F_k(A),\qquad W_k(\{i\})=F_k(A)\,p_k(i),\qquad W_k(\{i,j\})=f_k(\{i,j\}),\qquad W_k(T)=0\ \ (|T|\ge3),$$

so the series terminates in a useful sense — each folder can absorb at most two
overlap cards. The order-0 term is the naive product $\prod_k F_k(A)$ ("drop
the disjointness rules"); the order-1 term is

$$-\sum_{k<l}\sum_{i\in A} W_k(i)\,W_l(i)\prod_{j\ne k,l}F_j(A)
\;=\;-\Big(\prod_k F_k\Big)\sum_{k<l}\langle p_k,p_l\rangle,$$

minus the expected number of pairwise card collisions, times the naive product.

### 2.2 Why truncating it fails

Define the **expected collision count**

$$\mu \;=\; \sum_{k<l}\langle p_k,p_l\rangle \;\approx\; \binom K2\frac{4}{|A|}\quad(\text{flat ranges}).$$

With realistic numbers — $K=6$–$8$ preflop folders, $|A|=44$–$48$ — one gets
$\mu\approx 1.2$–$2.4$. The exact alternating series is an expansion in powers
of $\mu$: the order-$r$ Bonferroni-type partial sums oscillate with amplitude
$\sim\mu^r/r!$, and at $\mu\gtrsim 1$ the first-order truncation $1-\mu$ is
already negative while the true answer is $e^{-\mu}\approx 0.1$–$0.3$.
Low-order truncations of the raw inclusion–exclusion are therefore
*qualitatively* wrong precisely in the regime of interest, and going to high
enough order to fix this reintroduces combinatorial blowup (the number of
overlap patterns at total order $r$ grows like $\binom K2^{\,r}|A|^r/r!$). The
cure, standard in statistical mechanics, is to resum the series in the
logarithm.

This failure is a statement about the **absolute** value of $P_{\rm disj}$.
Section 4 shows that the strategically relevant object — the *variation* of
$\log Z$ across live-hand hypotheses — admits a different expansion in which
the $\mu=O(1)$ bulk cancels identically, and low-order truncation is sound.

---

## 3. The resummed estimator

### 3.1 Cluster expansion of the log

Index "collision slots" $a=(\{k,l\},i)$ and let
$C_a=\mathbf 1[i\in X_k]\,\mathbf 1[i\in X_l]$, $N=\sum_a C_a$ the total
collision count, so $P_{\rm disj}=\Pr[N=0]$. Expanding $\log\Pr[N=0]$ from the
Bonferroni sums gives, through second order in the slot interactions,

$$\log P_{\rm disj}\;=\;-\,\mathbb E N\;+\;\tfrac12\big(\operatorname{Var}N-\mathbb E N\big)\;+\;O(\text{3rd-order connected clusters}).$$

The leading term is the Poisson heuristic $P_{\rm disj}\approx e^{-\mu}$: if
collisions occurred independently, $N$ would be Poisson and the formula would
stop there. The second term measures the failure of the Poisson picture —
over-dispersion of $N$ (collisions clustering) makes "zero collisions" *more*
likely than $e^{-\mu}$, under-dispersion makes it less likely. Only
*connected* contributions survive in the log: covariances between slots vanish
unless the slots share a folder, and the surviving second-order terms are
exactly two families:

- **within-pair** (same pair $(k,l)$, two cards), determined by
  $\langle M_k,M_l\rangle_F$;
- **shared-hub triples** (folder $k$ against $l\ne l'$), where the covariance
  of $k$'s collision counts is

$$\operatorname{Cov}\big(N_{kl},N_{kl'}\big)\;=\;p_l^\top\Sigma_k\,p_{l'}\;\overset{\textbf{(I2)}}{=}\;\delta_l^\top\Sigma_k\,\delta_{l'}.$$

This identity is the structural heart of the method: *the hub's flat component
is annihilated automatically, and a triple contributes only through the tilts
of its two wing players.* Its consequences are exploited in §3.4 and again in
§4.4.

This is the Mayer cluster expansion [Ma37] for a hard-core interaction, in the
abstract polymer-model formulation of Kotecký–Preiss [KP86]: the polymers are
the hands $X_k$, incompatibility is card overlap, and convergence of the
log-series is governed by the collision density (Kotecký–Preiss criterion;
improved constants in Fernández–Procacci [FP07]). The equivalence between such
hard-core partition functions and non-collision probabilities, and the precise
relation to the Lovász local lemma, is developed by Scott–Sokal [SS05]. The
"compute mean and variance of the collision count, conclude Poisson-type
behavior with explicit error" step is the Chen–Stein method as packaged by
Arratia–Goldstein–Gordon [AGG89].

### 3.2 Three exact resummations available at $m=2$

Rather than truncating the generic series, three pieces can be evaluated
*exactly* and used as the backbone:

**(a) The flat reference.** If all $K$ ranges were uniform on $A$, the answer
is a falling-factorial product:

$$D_{\rm unif}(|A|,K)\;=\;\prod_{j=1}^{K-1}\frac{\binom{|A|-2j}{2}}{\binom{|A|}{2}}.$$

This resums the *bulk packing constraint to all orders* — the dominant,
range-shape-independent part of the effect, which is exactly the part that
makes $\mu=O(1)$ and breaks naive truncation. Note for §4: $D_{\rm unif}$
depends on the removal set only through its **cardinality** $|A|$.

**(b) Exact pairwise disjointness.** For two hands, inclusion–exclusion over
the (at most two) shared cards terminates exactly:

$$q_{kl}\;:=\;\Pr[X_k\cap X_l=\emptyset]\;=\;1-\langle p_k,p_l\rangle+\tfrac12\langle M_k,M_l\rangle_F,$$

with uniform value $q^{(0)}=1-\tfrac{4}{|A|}+\binom{|A|}{2}^{-1}$. (Check at
$|A|=4$: $q^{(0)}=1/6=\binom22/\binom42$.) The Frobenius term
$\tfrac12\langle M_k,M_l\rangle_F$ is the probability the two folders held the
*identical combo* — negligible for wide ranges ($\sim10^{-3}$), but up to
$\sim 1/|\mathrm{supp}\,f_c|$ for two overlapping capped ranges.

**(c) The centering identities (I1)–(I2)**, which guarantee that expanding
around the flat reference produces *no first-order term in the flat direction*:
$\langle p_k,p_l\rangle = \tfrac4{|A|}+\langle\delta_k,\delta_l\rangle$ with no
cross term, and triples reduce to tilt-bilinear forms.

### 3.3 Loose vs. tight folders: the $U/C$ partition

Fold ranges come in two phenomenologically distinct classes:

- $U$ (**loose**, instant preflop folds): $70$–$85\%$ of all combos; marginals
  nearly flat, mildly depleted on A/K/Q ranks and enriched on low cards.
  $\|\delta_u\|$ is small (a few percent of $\|p\|$), and by suit symmetry
  $\delta_u$ lives in a 13-dimensional rank space.
- $C$ (**tight**, acted-then-folded): a player who voluntarily put chips in and
  later released. Tens of combos, sharply shaped by the line and the board
  (missed flush draws on specific suits, folded-out middling pairs). Here
  $\delta_c=O(p_c)$: the tilt *is* the range.

The classes do not need different expansions — identities (I1)–(I2) hold for
both — but they populate *different terms*. The flat parts of everyone,
including tight folders, are absorbed exactly by $D_{\rm unif}$. The tilts
enter the pair factors $q_{kl}$, dominated by pairs involving $C$. And by the
hub identity (I2), third-order clusters survive only when **both wings are
tight**: a triple with a loose wing is suppressed by $\|\delta_u\|$. Tight
folders are thus the *only* players who can activate the exponential correction
below — and there are few of them, so the expensive terms are cheap in
practice.

### 3.4 The estimator

$$\boxed{\;
Z(A)\;=\;
\underbrace{\prod_{k=1}^{K}F_k(A)}_{\substack{\text{1st-order card removal}\\ \text{exact}}}
\;\cdot\;
\underbrace{\prod_{j=1}^{K-1}\frac{\binom{|A|-2j}{2}}{\binom{|A|}{2}}}_{\substack{\text{flat packing}\\ \text{all orders, exact}}}
\;\cdot\;
\underbrace{\prod_{k<l}\frac{1-\langle p_k,p_l\rangle+\tfrac12\langle M_k,M_l\rangle_F}{1-\tfrac{4}{|A|}+\binom{|A|}{2}^{-1}}}_{\substack{\text{pairwise interactions}\\ \text{exact per pair}}}
\;\cdot\;
\underbrace{\exp\Big(\sum_{\{c,c'\}\subseteq C}\delta_c^\top\big(\Sigma_{\rm tot}-\Sigma_c-\Sigma_{c'}\big)\delta_{c'}\Big)}_{\substack{\text{3-player clusters}\\ \text{only tight wings survive}}}
\;}$$

where $\Sigma_{\rm tot}=\sum_{k=1}^K\Sigma_k$ (the sum over hubs collapses to
one precomputed matrix by linearity). Dividing each pair factor by $q^{(0)}$
prevents double counting: $D_{\rm unif}$ already contains the uniform pair
content together with all uniform higher-order clusters, and the ratio swaps
the uniform pair factors for the exact ones while leaving the (correct) flat
higher-order content in place. The exponential's argument is the
genuinely-beyond-uniform triple content; uniform triples vanish there because
$\delta^{(0)}=0$.

**Neglected terms.** Connected triples with a loose wing, suppressed by
$\|\delta_u\|$; and connected 4-clusters, of size
$O\!\big(\|\delta_C\|^2(2K/|A|)^2\big)$. If $|C|\le1$ — the most common
situation — the exponential factor is empty and the formula is *exact at pair
level*; drop it.

**Removal-set algebra.** Because $|R|\le 7+2\cdot\#\text{live}$ is small, every
restricted quantity is the full-deck quantity minus boundary corrections:

$$F_k(A)=F_k-\sum_{r\in R}w_k(r)+\sum_{\{r,r'\}\subseteq R}f_k(\{r,r'\}),
\qquad
\langle w_k,w_l\rangle_A=\langle w_k,w_l\rangle-\sum_{r\in R}w_k(r)w_l(r),$$

(with $w_k=F_k\,p_k$ the unnormalized marginals), and analogously for the
Frobenius products using precomputed per-pair card profiles
$v_{kl}(i)=\sum_j M_k(ij)M_l(ij)$. After an $O(K^2\cdot52)$ precomputation per
game node, each evaluation of $Z(A)$ costs $O(K^2|R|^2)$ floating-point
operations.

**Validation.** An unbiased ground truth is free: sample the hands
independently from $f_k/F_k(A)$ and record the fraction of draws that are
pairwise disjoint — that fraction *is* $P_{\rm disj}(A)$, with acceptance rate
$\gtrsim e^{-\mu}\approx 0.05$–$0.3$, so a few thousand draws calibrate the
truncation error in any regime. Expected accuracy: relative error well under
$1\%$ with loose folders only; the triple term is needed only with $\ge2$ tight
folders holding overlapping shapes. Section 6 upgrades this protocol to the
difference-based form actually consumed by training.

### 3.5 Where this lives in the literature

The estimator is an assembly, tailored to $m=2$ and to the loose/tight
structure of fold ranges; it is not lifted verbatim from a single paper. Its
ingredients are classical, and each has a canonical source:

- **[Ma37]** J. E. Mayer, *The Statistical Mechanics of Condensing Systems. I*, J. Chem. Phys. **5**, 67–73 (1937). DOI: [10.1063/1.1749933](https://doi.org/10.1063/1.1749933). Origin of the cluster expansion: the log of a hard-core/interacting partition function as a sum over connected clusters — the device of §3.1.
- **[HL72]** O. J. Heilmann, E. H. Lieb, *Theory of monomer-dimer systems*, Comm. Math. Phys. **25**, 190–232 (1972). DOI: [10.1007/BF01877590](https://doi.org/10.1007/BF01877590). $Z(A)$ as a 2-set (dimer) packing partition function; the structural theory of such matching polynomials, including analyticity in the relevant density regime.
- **[KP86]** R. Kotecký, D. Preiss, *Cluster expansion for abstract polymer models*, Comm. Math. Phys. **103**, 491–498 (1986). DOI: [10.1007/BF01211762](https://doi.org/10.1007/BF01211762). The abstract polymer formulation used here (polymers = hands, incompatibility = card overlap) and the standard convergence criterion for the log-series.
- **[AGG89]** R. Arratia, L. Goldstein, L. Gordon, *Two moments suffice for Poisson approximations: the Chen–Stein method*, Ann. Probab. **17**, 9–25 (1989). DOI: [10.1214/aop/1176991491](https://doi.org/10.1214/aop/1176991491). Rigorous error bounds for $P[N=0]\approx e^{-\mu}$ from the first two moments of a sum of dependent indicators — exactly the mean/variance structure of §3.1.
- **[SS05]** A. D. Scott, A. D. Sokal, *The repulsive lattice gas, the independent-set polynomial, and the Lovász local lemma*, J. Stat. Phys. **118**, 1151–1261 (2005). DOI: [10.1007/s10955-004-2055-4](https://doi.org/10.1007/s10955-004-2055-4). The dictionary between hard-core partition functions, non-collision probabilities, and Mayer expansions, with the alternating-sign structure of the coefficients.
- **[FP07]** R. Fernández, A. Procacci, *Cluster expansion for abstract polymer models. New bounds from an old approach*, Comm. Math. Phys. **274**, 123–140 (2007). DOI: [10.1007/s00220-007-0279-2](https://doi.org/10.1007/s00220-007-0279-2). Sharper convergence conditions, relevant if one wants certified error bounds at the packing fractions reached with 7–8 folders.

The poker-specific choices — the exact flat reference $D_{\rm unif}$ as the
resummation point, the centering identities (I1)–(I2), the exact pair factors
at $m=2$, and the $U/C$ split — are what turn the generic expansion into a
four-factor formula whose expensive parts activate only for the rare tight
folders.

---

## 4. The effective potential on live hands

This section reorganizes the result of §3 into the form a model consumes. The
move is from "estimate $Z(A)$ at a removal set" to "expand $\log Z$ in the
removed **cards**, anchored at the actual board," which converts folded-player
impact into (i) a per-hand reach correction, (ii) an exact bilinear pairwise
potential between live hands, and (iii) a small, sharply localized higher-order
remainder.

### 4.1 Integrating out folders: a pairwise random field over live hands

Fix the public state with board $B$, live players $1,\dots,L$ with
board-normalized reach weights $\tilde r_{p,h}$, and folders $1,\dots,K$.
Define

$$\Phi(R)\;:=\;\log Z\big(\{1,\dots,52\}\setminus R\big),\qquad R\supseteq B.$$

The exact joint posterior over live hands is

$$P(h_1,\dots,h_L\mid \text{public})
\;\propto\;
\prod_{p=1}^{L}\tilde r_{p,h_p}
\;\cdot\;
\prod_{p<q}\mathbf 1\big[h_p\cap h_q=\emptyset\big]
\;\cdot\;
\exp\Phi\Big(B\cup h_1\cup\dots\cup h_L\Big).$$

Read as a graphical model: this is a **Markov random field with one
$1326$-state variable per live player**. Live players enter the monomer–dimer
system of §1.2 as *conditioned-on* dimers — their mutual exclusion is the
exact hard-core indicator. Folded players are *integrated-out* dimers: their
entire effect on live play is the effective potential $\Phi$, the free energy
of the folded subsystem as a function of which cards the live hypothesis
removes. Inference on this field — marginals, conditional expectations,
free-energy contributions — is precisely the computation a belief-state agent
must perform; `model.md` builds the network as a learned inference machine on
this field.

As in §1.2 remark 1, only the $R$-dependence of $\Phi$ matters; additive
constants cancel in normalization.

### 4.2 The card-cluster (Möbius) expansion at the board

Expand $\Phi$ over the removed cards beyond the board by exact finite
differences anchored at $B$. Define, for cards $u,v\in A_0$, $u\neq v$,

$$\alpha_B(u)\;=\;\Phi(B\cup\{u\})-\Phi(B),$$
$$\beta_B(u,v)\;=\;\Phi(B\cup\{u,v\})-\Phi(B\cup\{u\})-\Phi(B\cup\{v\})+\Phi(B),$$
$$\gamma_B(\{u,v,w\})\;=\;\textstyle\sum_{T\subseteq\{u,v,w\}}(-1)^{3-|T|}\,\Phi(B\cup T),$$

and so on (the Möbius transform of $S\mapsto\Phi(B\cup S)$). Then, **exactly**,
for any set $S\subseteq A_0$ of hypothesized live cards,

$$\Phi(B\cup S)\;=\;\Phi(B)\;+\;\sum_{u\in S}\alpha_B(u)\;+\;\sum_{\{u,v\}\subseteq S}\beta_B(u,v)\;+\;\sum_{\substack{T\subseteq S\\|T|\ge 3}}(\text{higher differences}).$$

The working approximation is truncation at second order. By convention
$\beta_B(u,u)=0$; $\beta_B$ is a symmetric, generally **indefinite**
$52\times52$ matrix supported on $A_0\times A_0$.

### 4.3 Why this expansion succeeds where §2.2's failed

Two structural facts:

1. **The bulk is cardinality-only and cancels.** The dangerous all-orders
   packing content — the reason $\mu=O(1)$ — sits in $D_{\rm unif}(|A|,K)$,
   which depends on the removal set only through $|R|$ (§3.2a). Across
   hypotheses with the same number of live cards (the only comparisons the
   posterior makes), this term is a common constant. The finite-difference
   coefficients $\alpha_B,\beta_B$ therefore contain only the *shape-dependent
   variation* of $\Phi$, not the bulk.
2. **The variation is genuinely small.** Each hypothesis moves $2$–$2L$ cards;
   per-card folded marginal mass is $p_k(u)\sim 2/|A_0|\approx 0.04$, so the
   relevant expansion parameter is $\sum_k p_k(u)\sim 2K/|A_0|\approx 0.3$ per
   card *in the unary term*, and second differences are already
   $\beta_B(u,v)\sim\sum_k|\Sigma_k(u,v)|\approx 10^{-3}$–$10^{-2}$ in typical
   loose-dominated nodes. Third differences carry another factor of a tilt
   norm. This is an expansion of a slowly varying log around its value at the
   actual board, not an expansion of a small probability around $1$.

### 4.4 Order structure: the consumable objects

**Order 0.** $\Phi(B)$: constant, cancels.

**Order 1 — per-hand reach correction.** A live hand $g=\{c,d\}$ contributes
its own unary correction, including the within-hand second difference:

$$\Lambda_B(g)\;:=\;\alpha_B(c)+\alpha_B(d)+\beta_B(c,d).$$

This is the complete *single-hypothesis* folded-impact term. Its consumption
rule: define the **folder-corrected log-reach**

$$\log\tilde r^{\prime}_{p,g}\;=\;\log\tilde r_{p,g}\;+\;\Lambda_B(g),$$

a strict refinement of the raw reach prior. First-order folder impact is a
reach reweighting and nothing else.

**Order 2 — exact bilinear pairwise potential.** For two live hands $h,g$ with
$h\cap g=\emptyset$, the cross term of the expansion is

$$\Delta_F(h,g)\;:=\;\Phi(B\cup h\cup g)-\Phi(B\cup h)-\Phi(B\cup g)+\Phi(B)
\;\approx\;\sum_{u\in h}\sum_{v\in g}\beta_B(u,v)
\;=\;\mathbf 1_h^\top\,\beta_B\,\mathbf 1_g,$$

where $\mathbf 1_h\in\{0,1\}^{52}$ is the two-hot indicator of $h$, and the
approximation drops only third-and-higher differences. **The entire pairwise
coupling between live hypotheses induced by folded players is, to this order, a
single fixed bilinear form in two-hot card indicators**, with the per-node
matrix $\beta_B$. (For $h\cap g\neq\emptyset$ the joint configuration is
impossible; the hard-core indicator of §4.1 governs, and no $\Phi$ value at an
invalid removal multiset is ever needed.) The same statement covers any number
of live players: the second-order expansion of
$\Phi(B\cup h_1\cup\dots\cup h_L)$ is the sum of $\Lambda_B(h_p)$ over players
plus $\mathbf 1_{h_p}^\top\beta_B\mathbf 1_{h_q}$ over pairs — pairwise in
hands *because* it is pairwise in cards. In particular, **no irreducible
three-hand interaction exists at this order**: a trilinear hand coupling first
appears with the third differences.

**Order 3 — the localized remainder.** By §3.3 and the hub identity (I2),
third differences are bilinear in folder tilts and survive at leading size only
when **at least two tight folders** with overlapping shapes are present; with
$|C|\le 1$ the §3.4 estimator is exact at pair level and its third differences
are zero up to the neglected 4-clusters. The remainder's raw ingredients are
the tight tilts $\{\delta_c\}_{c\in C}$ and $\Sigma_{\rm tot}$; a consumer that
wants this content needs those objects and a multiplicative (bilinear-unit)
computational path, not a larger attention pattern.

### 4.5 Leading-order identification of $\beta_B$

Expanding the §3.4 estimator analytically identifies the dominant content of
the second difference. From $\sum_k\log F_k(A)$ with the removal-set algebra,
the coefficient of the pair $\{u,v\}$ in the quadratic Taylor term of
$\log\big(1-\sum_{r}p_k(r)+\sum_{\{r,r'\}}M_k(r,r')\big)$ is

$$M_k(u,v)-p_k(u)\,p_k(v)\;=\;\big(\Sigma_k\big)_{uv}\quad(u\neq v),$$

so

$$\beta_B\;\approx\;\operatorname{offdiag}\,\Sigma_{\rm tot}\;+\;O\!\big(\tfrac{1}{|A_0|}\big)\ \text{cross-folder terms},$$

the off-diagonal of the summed folded covariance. The subleading terms come
from the pair factors (restriction of $\langle p_k,p_l\rangle$ contributes
*unarily* — each removed card $u$ deletes the collision slot $u$, a
$+p_k(u)p_l(u)$ term inside $\alpha_B$ — while renormalization cross terms of
relative size $O(4/|A_0|)$ land in $\beta_B$). Two consequences of the hub
identity carry over: the flat components of all folders cancel out of
$\beta_B$ into cardinality terms, so $\beta_B$ is **tilt-dominated**, and its
effective rank is governed by the folder population — $\lesssim 13$ for
loose-only nodes (rank-space structure by suit symmetry), plus a few sharp
directions per tight folder.

This identification is for understanding and for low-rank structure; the
*numerical* $\alpha_B,\beta_B$ should be obtained by finite differences of the
full estimator (§4.6), which is exact with respect to it and inherits all four
factors' content, not just the $\log F_k$ piece.

### 4.6 Practical evaluation

Compute $\alpha_B,\beta_B$ by brute finite differences of the §3.4 closed form:
$|A_0|+\binom{|A_0|}{2}\le 47+1081=1128$ evaluations of $\Phi$, each
$O(K^2|R|^2)\lesssim 4\cdot10^3$ flops after the once-per-node $O(K^2\cdot52)$
precompute — a total of **a few MFLOPs per game node**, exact with respect to
the estimator (no Taylor error; only the estimator's own truncation enters).
Two operational notes:

- $\alpha_B,\beta_B$ depend only on the board and the folder set. **They are
  constant across all decision nodes within a street with no new folds** and
  need recomputation only on a street change or a fold event.
- For consumption as bilinear features, take a signed eigendecomposition
  $\beta_B=\sum_j \lambda_j\,b_jb_j^\top$ ($52\times52$ symmetric eigh,
  $\sim1.4\cdot10^5$ flops). Because $\beta_B$ is indefinite, the
  factorization **must carry signs**; any consumer splitting it into rank-one
  features must keep $\operatorname{sign}(\lambda_j)$ explicit. Observed
  numerical rank: $\lesssim 16$ in loose-dominated nodes, growing by a few per
  tight folder.

### 4.7 Incidence identities for linear aggregation

A second family of exact facts, independent of folders, governs how
*live-range* statistics aggregate under the hard-core constraint. For any
per-hand weights $w_g\ge0$ and per-hand features (or scalars) $m_g$ of one
player, and a query hand $h=\{a,b\}$ of another, define the card and combo
accumulators

$$C_v=\sum_{g\ni v} m_g\quad(v=1,\dots,52),\qquad S_h=m_h.$$

Since for 2-card hands $\{g: g\cap h\neq\emptyset\}=\{g\ni a\}\cup\{g\ni b\}$
with intersection $\{h\}$, inclusion–exclusion gives, exactly,

$$\sum_{g:\,g\cap h=\emptyset} m_g\;=\;\sum_g m_g\;-\;C_a\;-\;C_b\;+\;S_h,
\qquad
\sum_{g:\,g\cap h\neq\emptyset,\ g\neq h} m_g\;=\;C_a+C_b-2S_h.$$

Interpretation against §1.3: with $m_g=w_g$ the raw range mass, $C_v$ is the
unnormalized card marginal $F\,p(v)$ and $S_h$ the exact-combo mass $f(h)$; the
right-hand sums are the first-order blocker statistic and the
disjointness-restricted total mass against $h$.

Consequences for aggregation operators: **any disjointness-restricted *linear*
aggregation over a 1326-hand range factors exactly through 52 card
accumulators**, at $O(52)$ cost per query instead of $O(1326)$ — and the
identical-combo statistic $S_h$ comes for free. A *normalized* (softmax-style)
aggregation does not factor this way, because the normalizer couples all
entries; and no bilinear score function of two-hot indicators can represent the
indicator $\mathbf 1[g=h]$, since $\mathbf 1[g=h]=\mathbf 1[\langle\mathbf
1_h,\mathbf 1_g\rangle=2]$ is a nonlinear function of the overlap count. These
two facts delimit what attention-style operators can and cannot compute
exactly; the architectural division of labor that follows from them is
specified in `model.md`.

### 4.8 Error budget

The truncation error of the order-2 expansion on a hypothesis set $S$ is the
sum of third-and-higher differences over subsets of $S$. By §4.4, these are
(i) zero up to 4-cluster corrections when $|C|\le1$, (ii) of size
$\sim\delta_c^\top\Sigma\,\delta_{c'}$ per tight pair otherwise — typically
$10^{-3}$–$10^{-2}$ in log space, concentrated on the specific suits/ranks the
tight ranges share. The verification protocol (difference-based, regime-
stratified) is specified in §6.2 and must be run before any learned component
is trained against these targets.

---

## 5. Compressed per-folder summaries

The estimator and the finite-difference operators consume each folder only
through $(F_k,p_k,M_k)$ on $A_0$. For feeding a *learned model* compact
per-folder context — and for the order-3 ingredients — define the following
summary; the numerical $\alpha_B,\beta_B$ themselves do **not** require it
(§4.6 computes them from full $M_k$, which is cheap to hold per node).

**Signed sketch.** Approximate each range matrix by a signed low-rank
factorization

$$M_k\;\approx\;\sum_{j=1}^{r} s_{kj}\,v_{kj}v_{kj}^\top,\qquad s_{kj}\in\{-1,+1\},\ v_{kj}\in\mathbb R^{52},$$

(equivalently a truncated symmetric eigendecomposition with signs kept, or a
rectangular SVD factor pair). $M_k$ is symmetric with zero diagonal and is
generally **not PSD** — even the uniform complete-graph matrix has negative
eigenvalues — so an unsigned factorization is structurally wrong. Combo mass
reconstructs as $f_k(\{a,b\})/F_k\approx\sum_j s_{kj}v_{kj}(a)v_{kj}(b)$, a
signed product, not a squared projection.

**What marginals cannot give.** The card marginals $p_k$ determine
$p_k(a)+p_k(b)$ for a hand $h=\{a,b\}$ but not the exact-combo mass $f_k(h)$
or any entry $M_k(a,b)$. Any pathway that needs exact-combo statistics must
obtain them from the signed sketch, from a sparse exception list of top combos
(appropriate for tight folders, whose support is tens of combos), or from an
exact incidence accumulator (§4.7).

**Exactness status.** A summary is exact with respect to the §3.4 estimator
only if it represents each $M_k$ exactly. With low-rank sketches, a
marginals-only mode for loose folders, or truncated exception lists, the
summary is an *approximation with an inductive-bias justification* —
suit-symmetric preflop ranges live in a $\approx13$-dimensional commutant so
eigenvalue decay is fast, and postflop ranges break suit symmetry only through
the board — and it must be calibrated against the Monte Carlo oracle of §6.2,
never asserted.

**Per-folder summary table.**

| Field | Dim | Role |
|---|---|---|
| $\log F_k$ | 1 | tightness; gates the $U/C$ pathway |
| $p_k$ (rank⊗suit factorized) | 52 | unary terms, $\mu$, all $\delta$'s |
| $\{(s_{kj},v_{kj})\}_{j\le r}$, tight folders only | $\le 4\times52$ + signs | quadratic terms, Frobenius products, $\Sigma_k$ |
| street/line embedding | small | prior over range shape and over trust in the inferred $f_k$ |

**What a downstream model consumes per node** (the complete interface; see
`model.md` for usage):

1. the per-hand corrections $\{\Lambda_B(g)\}_{g}$ — equivalently $\alpha_B$
   plus the diagonal-pair values $\beta_B(c,d)$ (§4.4);
2. the signed factors $\{(\lambda_j,b_j)\}$ of $\beta_B$ (§4.6);
3. the tight-folder tilts $\{\delta_c\}_{c\in C}$ and $\Sigma_{\rm tot}$
   contractions (order-3 ingredients, §4.4);
4. per-folder scalars and embedding inputs: $\log F_k$, line/street metadata;
5. optionally, the unnormalized folded card-marginal vectors $w_k=F_kp_k$
   (per-card folded pressure with folder identity attached).

All items are computed on $A_0$ per §1.5 and refreshed only on street/fold
events.

---

## 6. Computation, validation, and teacher targets

### 6.1 Per-node cost summary

| Object | Cost | Refresh |
|---|---|---|
| Folder statistics $F_k,p_k,M_k,\delta_k,\Sigma_k$ on $A_0$ | $O(K\cdot 1326)$ | street or fold event |
| Removal-set precompute (pair profiles $v_{kl}$) | $O(K^2\cdot52)$ | street or fold event |
| One $\Phi$ evaluation (§3.4) | $O(K^2|R|^2)\lesssim 4\cdot10^3$ flops | per query |
| Full $\alpha_B,\beta_B$ by finite differences | $\le1128$ evaluations $\approx$ few MFLOPs | street or fold event |
| Signed eigh of $\beta_B$ | $O(52^3)\approx1.4\cdot10^5$ flops | street or fold event |
| Rejection-sampling oracle, $n$ draws | $O(nK)$ with acceptance $\gtrsim e^{-\mu}$ | offline / eval only |

Everything above is negligible against a single network forward pass; see the
cost model in `model.md`.

### 6.2 The Monte Carlo oracle, difference-based

The unbiased ground truth of §3.4 remains the rejection sampler. Because the
strategically relevant object is the *variation* of $\Phi$ (§4.1), validation
must be run on **differences**, not absolute values:

1. Sample game nodes stratified by folder regime: loose-only, exactly one
   tight folder, $\ge2$ tight folders with overlapping shapes.
2. For sampled hypothesis pairs $(S,S')$ (single hands, and two-hand joint
   configurations), estimate $\Phi(B\cup S)-\Phi(B\cup S')$ by paired rejection
   sampling (common random numbers across the two removal sets to cancel
   variance).
3. Compare against (a) the §3.4 estimator evaluated directly, and (b) the
   order-2 finite-difference surrogate $\Lambda_B+\mathbf 1^\top\beta_B\mathbf 1$.
   The gap between (a) and the oracle calibrates the estimator's truncation;
   the gap between (b) and (a) isolates the third-difference content.

Acceptance thresholds: relative error of (b) under $1\%$ in log space on
loose-only and single-tight regimes; on the $\ge2$-tight regime, report the
error distribution and the fraction of nodes where order-3 content exceeds the
order-2 signal — this number decides whether the order-3 ingredients (§5 item
3) are worth a computational path at all.

### 6.3 Teacher targets for learned components

Where a learned module is trained to *carry or refine* this computation rather
than receive it as input, the supervised targets are, in increasing order of
difficulty:

1. per-hand $\Lambda_B(g)$ regression (order 1);
2. per-pair $\Delta_F(h,g)$ regression on disjoint pairs (order 2);
3. residual regression $\Phi_{\rm oracle}-\Phi_{\rm order\text{-}2}$ on the
   $\ge2$-tight stratum (order 3), available only via the paired sampler.

The rationale for learning on top of exact features at all: the formula is a
truncation, and it sits downstream of inferred fold ranges $f_k$ that are
themselves a model of opponent strategy. Gradients through the same compressed
statistics let a network correct truncation error where the formula is worst
and learn — via $\log F_k$ and line embeddings — *how much to trust* an
inferred fold range per betting line, which no fixed formula can decide. The
training schedule (pretrain against these targets, anneal, monitor against the
oracle with drift thresholds) is owned by `model.md` and the training
documents; this file owns only the targets and the oracle.
