# Branched naive-policy-gradient rollout variance and the KKT trace-variance proxy

## Purpose and scope

This note studies only the public-rollout contribution to the variance of one
branched policy-gradient estimator. It deliberately uses a simpler loss than
the production branched loss in
[`core_math.tex`](../theory/core_math.tex):

- the rollout profile is on-policy, $\pi=\pi^{\theta_k}$, and every derivative
  is evaluated at $\theta=\theta_k$;
- the policy loss is the naive policy-gradient loss, linear in
  $\widehat Q$, rather than the leave-one-out $\widehat A$ loss;
- $\lambda$ and GAE are absent;
- there is no value-head loss;
- private-state and terminal-completion expectations are exact;
- the minibatch is integrated out.

Thus the completion and minibatch contributions have been removed:
$n=B=\infty$. The only remaining randomness is the sampled public rollout
tree $\mathfrak S^{\beta,\pi}$. Variance means trace covariance:

$$
\operatorname{Var}(Y)
:=
\operatorname{tr}\operatorname{Cov}(Y)
=
\mathbb E\lVert Y-\mathbb EY\rVert_2^2.
$$

The main question is whether the reward trace variance

$$
\operatorname{Var}_{\mathfrak S_\sigma}
\left(
  \mathbb E[\mathbf R(X,\Tau)\mid\mathfrak S_\sigma,\sigma]
\right),
$$

which is the ideal population quantity targeted by the KKT trace-variance
head, controls the actual parameter-gradient rollout variance.

The conclusion is:

> The KKT formalism has the correct qualitative branching geometry, but the
> public reward trace variance is not an upper bound for policy-gradient
> rollout variance, even on-policy with uniformly bounded scores and rewards.
> It is a plausible empirical proxy only when public payoff variation remains
> well aligned with private-state-weighted policy-gradient variation.

## Setup

Let $(\mathfrak S,\ell,N)$ be an ideal sampled tree from
$\mathfrak S^{\beta,\pi}$ as in Definition
`branched-rollout-tree` of `core_math.tex`, without queue-pressure or
lower-size safeguards. For a sampled-tree node $\mathsf s$, define

$$
w_{\mathfrak S}(\mathsf s)
:=
\prod_{\mathsf r\prec\mathsf s}\frac1{N(\mathsf r)},
\qquad
w_{\mathfrak S}(\varnothing)=1.
$$

Conditionally on $\mathfrak S$, this is the probability that the
uniform-child random walk visits $\mathsf s$. For a player node with
$\ell(\mathsf s)=\sigma$ and $p=o(\sigma)$, and for a child
$\mathsf s\mathsf a$ labeled by $\sigma a_{\mathsf a}$, write

$$
z_k(x_p,\sigma,a_{\mathsf a})
:=
\left[
\nabla_\theta\log\pi^\theta
(a_{\mathsf a}\mid x_p,\sigma)
\right]_{\theta=\theta_k}.
$$

Assume there is a finite constant $L_z$ such that

$$
\lVert z_k(x_p,\sigma,a)\rVert_2\leq L_z
$$

for every reachable player decision, private state, and legal action.

With no bootstrapping, the exact tree-conditional action value is

$$
\widehat Q_p(x_p,\mathsf s\mathsf a)
:=
\mathbb E^{\beta,\pi}
\left[
R_p(X,\Tau)
\mid x_p,\mathsf s\mathsf a,\mathfrak S
\right].
$$

The gradient contribution conditional on selecting $\mathsf s$ is

$$
\begin{aligned}
\Psi_Q(\mathfrak S;\mathsf s)
:=
-\frac1{N(\mathsf s)}
\sum_{\mathsf a\in\mathfrak E_{\mathsf s}}
\sum_{x_p\in\mathcal X_p}
&\mathbb P^{\beta,\pi}
(x_p\mid\mathsf s\mathsf a,\mathfrak S)\\
&{}\cdot
\widehat Q_p(x_p,\mathsf s\mathsf a)
z_k(x_p,\sigma,a_{\mathsf a}).
\end{aligned}
\tag{1}
$$

After integrating out the sampled player node, including the usual
$\mathbb E(|\tau|_{\rm player}\mid\mathfrak S)$ scaling, the conditional mean
gradient is

$$
G_Q(\mathfrak S)
:=
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_Q(\mathfrak S;\mathsf s).
\tag{2}
$$

The quantity of interest is exactly

$$
\boxed{
T_{\rm rollout}^{Q,\rm branch}(\beta)
:=
\operatorname{Var}_{\mathfrak S^{\beta,\pi}}
\left(G_Q(\mathfrak S)\right).
}
\tag{3}
$$

There is no $B$ or completion-sample count in (3).

## The actual naive-$Q$ estimator is a leaf functional

For a completed private/public trajectory, define the ordinary
trajectory-sum policy-gradient functional

$$
\Phi_Q(x,\tau)
:=
-\sum_{\substack{\sigma a\preccurlyeq\tau\\
o(\sigma)\in\mathcal P}}
R_{o(\sigma)}(x,\tau)
z_k(x_{o(\sigma)},\sigma,a).
\tag{4}
$$

For a public pre-terminal trajectory $t$, let

$$
F_Q(t)
:=
\mathbb E^\pi
\left[
\Phi_Q(X,\Tau)
\mid\Tau^{\mathsf{pre-c}}=t
\right].
\tag{5}
$$

### Proposition 1: tree/leaf identity

For every realized sampled tree,

$$
\boxed{
G_Q(\mathfrak S)
=
\mathbb E^{\beta,\pi}
\left[
\Phi_Q(X,\Tau)\mid\mathfrak S
\right]
=
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_Q(\ell(\mathsf t)).
}
\tag{6}
$$

#### Proof

For an edge $\mathsf s\mathsf a$, the product
$w_{\mathfrak S}(\mathsf s)/N(\mathsf s)$ is
$w_{\mathfrak S}(\mathsf s\mathsf a)$, the conditional probability that the
uniform-child random walk traverses that edge. By the definition of
$\widehat Q$ and the tower property,

$$
\begin{aligned}
&\sum_{x_p}
\mathbb P^{\beta,\pi}
(x_p\mid\mathsf s\mathsf a,\mathfrak S)
\widehat Q_p(x_p,\mathsf s\mathsf a)
z_k(x_p,\sigma,a_{\mathsf a})\\
&\qquad=
\mathbb E^{\beta,\pi}
\left[
R_p(X,\Tau)z_k(X_p,\sigma,a_{\mathsf a})
\mid\mathsf s\mathsf a,\mathfrak S
\right].
\end{aligned}
$$

Multiplying by the edge weight and summing over sampled player edges gives
the first equality in (6). Expanding the conditional terminal law of
Definition `branched-rollout-tree` gives the leaf mixture in the second
equality. $\square$

By the terminal-node distribution lemma,

$$
\mathbb E_{\mathfrak S}G_Q(\mathfrak S)
=
\mathbb E^\pi\Phi_Q(X,\Tau).
\tag{7}
$$

Consequently,

$$
T_{\rm rollout}^{Q,\rm branch}(\beta)
=
\operatorname{Var}_{\mathfrak S}
\left(
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_Q(\ell(\mathsf t))
\right).
\tag{8}
$$

Thus, for this simplified loss, the `branched.md` matched estimator is no
longer an artificial comparison: after replacing its integrand by (4), it is
the actual branched estimator.

## Exact recursion for the rollout variance

Suppose first that the child count is a deterministic state-indexed profile,
$N(\mathsf s)=n(\ell(\mathsf s))$. Define, backwards on the public tree,

$$
m_G(t)=F_Q(t),
\qquad
v_G(t)=0
\tag{9}
$$

at every pre-terminal state $t$, and

$$
\begin{aligned}
m_G(\sigma)
&=
\sum_eK^\pi(e\mid\sigma)m_G(\sigma e),\\
U_G(\sigma)
&:=
\sum_eK^\pi(e\mid\sigma)
\lVert m_G(\sigma e)-m_G(\sigma)\rVert_2^2,\\
v_G(\sigma)
&=
\frac1{n(\sigma)}
\sum_eK^\pi(e\mid\sigma)
\left[
v_G(\sigma e)
+
\lVert m_G(\sigma e)-m_G(\sigma)\rVert_2^2
\right].
\end{aligned}
\tag{10}
$$

Conditional branch independence proves (10): the $n(\sigma)$ child-branch
estimators are iid, and their average divides trace variance by
$n(\sigma)$. Therefore

$$
\boxed{
T_{\rm rollout}^{Q,\rm branch}(n)
=v_G(\varnothing).
}
\tag{11}
$$

Equivalently,

$$
\boxed{
T_{\rm rollout}^{Q,\rm branch}(n)
=
\sum_{\sigma\notin\mathcal T^{\mathsf{pre-c}}}
\frac{
\mathbb P^\pi(\sigma\preccurlyeq\Tau^{\mathsf{pre-c}})
}{
\displaystyle
\prod_{\substack{\rho\preccurlyeq\sigma\\
\rho\notin\mathcal T^{\mathsf{pre-c}}}}
n(\rho)
}
U_G(\sigma).
}
\tag{12}
$$

For ordinary randomized rounding, replace every $1/n(\sigma)$ in the
recursion by

$$
\kappa_\beta(\sigma)
:=
\mathbb E
\left[
\frac1{N(\mathsf s)}
\mid\ell(\mathsf s)=\sigma
\right].
\tag{13}
$$

The exact randomized-rounding version of (12) is

$$
T_{\rm rollout}^{Q,\rm branch}(\beta)
=
\sum_{\sigma\notin\mathcal T^{\mathsf{pre-c}}}
\mathbb P^\pi(\sigma\preccurlyeq\Tau^{\mathsf{pre-c}})
\left(
\prod_{\substack{\rho\preccurlyeq\sigma\\
\rho\notin\mathcal T^{\mathsf{pre-c}}}}
\kappa_\beta(\rho)
\right)
U_G(\sigma).
\tag{14}
$$

Equation (14) identifies the ideal variance-guided branching statistic:
$U_G(\sigma)$ is the public-event innovation variance of the conditional
trajectory-gradient functional. A controller that knew $U_G$, or its
downstream trace variance $v_G$, could optimize the actual rollout variance
directly.

## The population target of the current trace-variance head

The KKT formalism does not use $U_G$. It propagates the vector reward value

$$
\widehat{\mathbf V}_{\mathfrak S}(\mathsf s)
:=
\mathbb E^{\beta,\pi}
\left[
\mathbf R(X,\Tau)\mid\mathsf s,\mathfrak S
\right].
\tag{15}
$$

For a public state $\sigma$, let $\mathfrak S_\sigma^{(1)}$ denote a sampled
subtree whose root has one sampled child and whose strict descendants use the
configured branching rule. Define

$$
\boxed{
T_R^\beta(\sigma)
:=
\operatorname{trCov}_{\mathfrak S_\sigma^{(1)}}
\left(
\mathbb E[
\mathbf R(X,\Tau)
\mid\mathfrak S_\sigma^{(1)},\sigma]
\right).
}
\tag{16}
$$

This is the precise version of

$$
\operatorname{Var}_{\text{tree sampled starting at }\sigma}
\left(
\mathbb E[\mathbf R(X,\Tau)\mid\text{tree},\sigma]
\right).
$$

With exact completion moments, no minibatch noise, an unbounded amount of
training data, sufficient head capacity, and optimization to the population
minimum, the trace-variance head is intended to converge to (16). The sibling
sample covariance in `core_math.tex` is an unbiased observation of the
one-child branch variance before the documented execution corrections.

Writing

$$
U_R(\sigma)
:=
\operatorname{trCov}
\left(
\mathbb E[\mathbf R\mid\sigma E]
\mid\sigma
\right),
\tag{17}
$$

the corresponding downstream recursion has the same formal shape as (10):

$$
T_R^\beta(\sigma)
=
\sum_{\eta\succeq\sigma}
\mathbb P^\pi(\eta\mid\sigma)
\left(
\prod_{\sigma\prec\alpha\preccurlyeq\eta}
\kappa_\beta(\alpha)
\right)
U_R(\eta),
\tag{18}
$$

with the root count fixed to one in the definition of $T_R^\beta(\sigma)$.
The continuous mean-field KKT calculation replaces $\kappa_\beta(\alpha)$ by
$1/\beta(\alpha)$.

Equations (14) and (18) give the exact link:

- $T_{\rm rollout}^{Q,\rm branch}$ applies the sampled-tree conditional-mean
  variance operator to the complete trajectory-gradient functional
  $\Phi_Q$;
- $T_R^\beta$ applies the same kind of sampled-tree conditional-mean variance
  operator to the public vector reward $\mathbf R$;
- the KKT formalism uses the second as a proxy for the first.

The shared branching geometry is real. The substitution of reward innovations
$U_R$ for gradient innovations $U_G$ is the nontrivial approximation.

## What bounded policy scores imply

### A general but coarse bound

Assume every trajectory contains at most $H$ player decisions and
$|R_p|\leq R_{\max}$. From (4) and Cauchy--Schwarz,

$$
\begin{aligned}
\lVert\Phi_Q(X,\Tau)\rVert_2^2
&\leq
H
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
R_{o(\sigma)}(X,\Tau)^2
\lVert z_k(X_{o(\sigma)},\sigma,a)\rVert_2^2\\
&\leq H^2L_z^2R_{\max}^2.
\end{aligned}
\tag{19}
$$

The law of total variance and Proposition 1 give

$$
\boxed{
T_{\rm rollout}^{Q,\rm branch}(\beta)
\leq
\operatorname{Var}^\pi(\Phi_Q(X,\Tau))
\leq
\mathbb E^\pi\lVert\Phi_Q(X,\Tau)\rVert_2^2
\leq
H^2L_z^2R_{\max}^2.
}
\tag{20}
$$

This is rigorous but does not depend on $\beta$, so it does not justify the
KKT allocation.

### The value-error bound that Cauchy--Schwarz actually gives

At a fixed player state $\sigma$, let $\nu_\sigma(dx_p,da)$ be its on-policy
private-state/action decision law. Suppose an auxiliary sampled subtree
$\mathcal U$ produces a square-integrable estimate
$\widehat Q_{\mathcal U}(x_p,a)$ of $Q(x_p,a)$ under this fixed law. Define

$$
\begin{aligned}
\widehat g_{\mathcal U}(\sigma)
&=
-\mathbb E_{\nu_\sigma}
\left[
z_k(X_p,\sigma,A)\widehat Q_{\mathcal U}(X_p,A)
\right],\\
g(\sigma)
&=
-\mathbb E_{\nu_\sigma}
\left[
z_k(X_p,\sigma,A)Q(X_p,A)
\right].
\end{aligned}
\tag{21}
$$

Then

$$
\begin{aligned}
\mathbb E_{\mathcal U}
\lVert\widehat g_{\mathcal U}(\sigma)-g(\sigma)\rVert_2^2
&\leq
L_z^2
\mathbb E_{\nu_\sigma}
\mathbb E_{\mathcal U}
\left[
\left(
\widehat Q_{\mathcal U}(X_p,A)-Q(X_p,A)
\right)^2
\right].
\end{aligned}
\tag{22}
$$

This follows by applying Jensen or Cauchy--Schwarz to the expectation in
(21), followed by the score bound.

Thus the natural value-based upper-bound proxy is private- and
action-resolved:

$$
T_{Q,\rm private}(\sigma)
:=
\mathbb E_{X_p,A\mid\sigma}
\operatorname{Var}_{\mathcal U}
\left(
\widehat Q_{\mathcal U}(X_p,A)
\right).
\tag{23}
$$

In the branched implementation, where the sampled public event changes the
posterior law, the stable exact objects are the posterior and unnormalised
payoff moments

$$
\widehat P_p(x_p),\qquad
\widehat M_p(x_p)=\widehat P_p(x_p)\widehat Q_p(x_p).
\tag{24}
$$

A rigorous implementation-level version of (22) must retain the joint
fluctuation of both moments. Public reward means alone have already averaged
away this information.

## Public reward trace variance does not bound gradient variance

### Proposition 2: impossibility under the stated assumptions

There is no universal finite constant $C$ such that

$$
T_{\rm rollout}^{Q,\rm branch}(\beta)
\leq
C L_z^2 T_R^\beta(\varnothing)
\tag{25}
$$

for all finite on-policy games satisfying the score bound, even when rewards
are bounded.

#### Proof by counterexample

Let a player observe a private type $X\in\{-1,+1\}$, uniformly distributed,
and choose a public action $A\in\{-1,+1\}$ according to

$$
\pi^\theta(a\mid x)
=
\frac{\exp(\theta xa)}{2\cosh\theta}.
\tag{26}
$$

Afterwards, public chance samples $B\in\{-1,+1\}$ uniformly and the terminal
reward is

$$
R=XB.
\tag{27}
$$

At the on-policy point $\theta=\theta_k=0$, $A$ is uniform and independent of
$X$, while

$$
z_k(X,A)
=
\left[
\frac{\partial}{\partial\theta}
\log\pi^\theta(A\mid X)
\right]_{\theta=0}
=XA.
\tag{28}
$$

Hence $|R|\leq1$ and $|z_k|\leq1$. For every complete public trajectory
$(A=a,B=b)$,

$$
\mathbb E[R\mid A=a,B=b]
=
b\,\mathbb E[X\mid A=a]
=0.
\tag{29}
$$

It follows that the public conditional reward value is identically zero for
every sampled tree and therefore

$$
T_R^\beta(\varnothing)=0.
\tag{30}
$$

For $\beta\equiv1$, however, the exact private-integrated naive policy
gradient conditional on the public trajectory is

$$
F_Q(a,b)
=
-\mathbb E[(XB)(XA)\mid A=a,B=b]
=-ab.
\tag{31}
$$

Therefore

$$
T_{\rm rollout}^{Q,\rm branch}(1)
=
\operatorname{Var}(-AB)
=1.
\tag{32}
$$

The right-hand side of (25) is zero while the left-hand side is one.
$\square$

The failure is caused by private-state cancellation. The reward mean
$\mathbb E[R\mid\text{public tree}]$ averages over $X$ before the reward is
multiplied by the private-state-dependent score. The policy gradient instead
contains the conditional covariance between reward and score.

The ideal score-aware analogue of (16) is

$$
\boxed{
T_G^\beta(\sigma)
:=
\operatorname{trCov}_{\mathfrak S_\sigma}
\left(
\mathbb E[
\Phi_Q(X,\Tau)
\mid\mathfrak S_\sigma,\sigma]
\right).
}
\tag{33}
$$

At one decision, its local analogue is

$$
\operatorname{trCov}_{\mathfrak S_\sigma}
\left(
\mathbb E[
z_k(X_p,\sigma,A)R_p
\mid\mathfrak S_\sigma,\sigma]
\right),
\tag{34}
$$

not the same expression with $z_k$ removed.

## Assessment of the KKT formalism

### What is justified

The KKT formalism captures several correct features of the problem:

1. Child branches are conditionally iid, so averaging $N$ children reduces
   the relevant branch variance through a reciprocal-child-count factor.
2. Branching at an ancestor reduces every downstream innovation carried by
   that branch.
3. The trace-variance-to-cost ratio is the correct qualitative quantity for
   allocating a constrained branching budget.
4. The public reward trace target is cheap relative to parameter-gradient
   variance, is naturally produced by the value backup, and can be learned
   without storing parameter-sized objects.

These facts make the current controller a reasonable engineering heuristic.
In NLHE, public actions and board cards often cause large changes in equity
and payoff. If those changes are strongly correlated with changes in
score-weighted, private-hand-resolved continuation values, $T_R^\beta$ can
rank branching locations similarly to $T_G^\beta$.

### What is not justified

Under only the assumptions of this note, the current KKT rule is not a
mathematical variance controller for $T_{\rm rollout}^{Q,\rm branch}$:

1. Proposition 2 rules out a reward-trace upper bound, even up to an arbitrary
   universal multiplicative constant.
2. The exact rollout variance uses gradient innovations $U_G$, including
   private-state/score correlation and covariance among decisions on one
   trajectory. The current head uses public reward innovations $U_R$.
3. A Cauchy--Schwarz bound naturally produces private/action-resolved $Q$
   error such as (23), not the public reward-mean variance (16).
4. The KKT objective in `core_math.tex` sums value-estimation errors over
   player-node ancestors. This is a useful Cauchy--Schwarz-style surrogate,
   but it is not the exact root variance recursion (14); it discards
   parameter-gradient covariance and can be loose by horizon-dependent
   factors.
5. The continuous KKT calculation also replaces
   $\mathbb E(1/N)$ by $1/\mathbb E(N)=1/\beta$. This mean-field discrepancy
   is secondary to the reward-versus-gradient mismatch but is systematic for
   noninteger intensities.

### Practical verdict

The current KKT formalism is a good *candidate heuristic* for controlling
$T_{\rm rollout}$ in practice, but it is not good as an unvalidated guarantee.
Its success in NLHE is an empirical claim about alignment:

$$
T_R^\beta(\sigma)
\quad\text{must correlate with}\quad
T_G^\beta(\sigma)
\ \text{or}\
T_{Q,\rm private}^\beta(\sigma)
$$

under the reach and cost weights that actually drive the controller.

This should be tested directly. At fixed checkpoints and sampled nodes with
multiple independent child branches:

1. record the existing reward trace target;
2. estimate trace covariance of child gradient-to-go contributions with
   fixed Hutchinson/Rademacher parameter probes;
3. compare weighted rank correlation, large-$T_G$/small-$T_R$ blind spots,
   and the branching allocation induced by the two targets;
4. confirm the conclusion against the directly measured whole-tree quantity
   $G_Q(\mathfrak S)$ from (2), not only nodewise statistics.

If alignment is consistently strong, the reward head is preferable because
it is much cheaper and less noisy. If important blind spots occur, a stronger
controller should use either:

- a private-state-resolved posterior/payoff-moment variance proxy based on
  $(P,M=PQ)$; or
- a projected gradient-to-go trace target approximating (33).

The first retains the Cauchy--Schwarz connection while avoiding
parameter-sized targets. The second follows the exact rollout-variance
recursion but is likely substantially noisier and more expensive.

## Summary

For on-policy naive policy gradient with exact $\widehat Q$, no GAE, no
value-head loss, and $B=n=\infty$:

$$
T_{\rm rollout}^{Q,\rm branch}
=
\operatorname{Var}_{\mathfrak S}
\left(
\mathbb E[\Phi_Q(X,\Tau)\mid\mathfrak S]
\right)
$$

is the exact actual rollout contribution. It admits the ordinary
branch-independent scalar recursion with gradient innovation $U_G$.

The ideal KKT trace head instead targets

$$
T_R^\beta(\sigma)
=
\operatorname{Var}_{\mathfrak S_\sigma}
\left(
\mathbb E[\mathbf R(X,\Tau)\mid\mathfrak S_\sigma,\sigma]
\right).
$$

The two quantities share the same sampling geometry but not the same
integrand. Bounded $\nabla\log\pi$ gives a rigorous bound through
private/action-resolved $Q$ error, or a coarse reward-second-moment bound, but
does not turn public reward trace variance into an upper bound. Therefore the
current KKT formalism is mathematically well motivated as a cheap proxy and
must be validated as such; it does not, under the stated assumptions,
guarantee control of the actual rollout gradient variance.
