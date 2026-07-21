# Gradient variance for branched Rao--Blackwell rollouts

This note is the branched counterpart of
[the sequential variance note](rao_blackwell_variance_experiment.md).
It has two purposes:

1. compare sequential and branched public-rollout variance while holding the
   trajectory-gradient integrand fixed; and
2. give the corresponding variance decomposition for the actual branched
   PG-plus-value loss in [core_math.tex](../theory/core_math.tex), including
   its leave-one-out advantage.

The first comparison isolates the statistical effect of branching. The
second includes both branching and the additional control variate used by
the branched loss.

## Convention and public transition law

Freeze $\pi=\pi^{\theta_k}$ and evaluate all derivatives at
$\theta=\theta_k$. Throughout,

$$
\operatorname{Var}(Y)
:=\operatorname{tr}\operatorname{Cov}(Y)
=\mathbb E\lVert Y-\mathbb EY\rVert_2^2
$$

for every vector-valued random variable $Y$. The same convention applies to
conditional variance.

For a non-pre-terminal public state $\sigma$, let
$K^\pi(e\mid\sigma)$ be its posterior-predictive public-event kernel:

$$
K^\pi(e\mid\sigma)
=
\begin{cases}
\displaystyle
\sum_{x\in\mathcal X}
\mathbb P^\pi(x\mid\sigma)\,
\pi(e\mid x_{o(\sigma)},\sigma),
&o(\sigma)\in\mathcal P,\\[1.2ex]
\displaystyle
\sum_{x\in\mathcal X}
\mathbb P^\pi(x\mid\sigma)\,
\chi(e\mid x,\sigma),
&o(\sigma)=\mathsf c.
\end{cases}
\tag{1}
$$

Thus the actual sequential public pre-terminal law is

$$
\mathbb P^\pi(\Tau^{\mathsf{pre-c}}=t)
=
\prod_{\substack{\sigma e\preccurlyeq t\\
\sigma\notin\mathcal T^{\mathsf{pre-c}}}}
K^\pi(e\mid\sigma).
\tag{2}
$$

This is also the law used independently for every child in
SampleTree, conditional on its public parent.

For the randomized-rounding rule in SampleTree, write

$$
q_\beta(m\mid\sigma)
:=
\mathbb P(N(\mathsf s)=m\mid\ell(\mathsf s)=\sigma)
=
\begin{cases}
1,
&\delta_\sigma=0,\ m=\beta(\sigma),\\
1-\delta_\sigma,
&0<\delta_\sigma<1,\ m=\lfloor\beta(\sigma)\rfloor,\\
\delta_\sigma,
&0<\delta_\sigma<1,\ m=\lceil\beta(\sigma)\rceil,\\
0,
&\textnormal{otherwise},
\end{cases}
\qquad
\delta_\sigma=\beta(\sigma)-\lfloor\beta(\sigma)\rfloor.
\tag{3}
$$

Let $(\mathfrak S,\ell,N)$ be a finite realized sampled tree. If
$e_{\mathsf s,\mathsf a}$ is defined by

$$
\ell(\mathsf s\mathsf a)
=\ell(\mathsf s)e_{\mathsf s,\mathsf a},
$$

then its joint law is

$$
\boxed{
\mathbb P^{\beta,\pi}(\mathfrak S,\ell,N)
=
\prod_{\substack{\mathsf s\in\mathfrak S\\
\ell(\mathsf s)\notin\mathcal T^{\mathsf{pre-c}}}}
\left[
q_\beta(N(\mathsf s)\mid\ell(\mathsf s))
\prod_{\mathsf a\in\mathfrak E_{\mathsf s}}
K^\pi(e_{\mathsf s,\mathsf a}\mid\ell(\mathsf s))
\right].
}
\tag{4}
$$

Pre-terminal sampled-tree nodes have $N(\mathsf t)=0$ deterministically.
The abstract children are distinct even when their public-event labels
coincide.

Strictly, the realized child-count map is node-indexed:
$N=N(\mathsf s)$, not $N=N(\sigma)$. Two sampled-tree nodes with the same
public label may have different randomized counts. A deterministic
state-indexed profile $n(\sigma)$ is a useful special case and is treated
below. If one conditions on a realized node-indexed map $N$, the conditional
label law must be obtained from (4); when $\beta$ is state-dependent, the
$q_\beta$ factors generally prevent it from being just an unweighted product
of the $K^\pi$ factors.

For any sampled-tree node, define its path weight

$$
w_{\mathfrak S}(\mathsf s)
:=
\prod_{\mathsf r\prec\mathsf s}\frac1{N(\mathsf r)},
\qquad
w_{\mathfrak S}(\varnothing)=1.
\tag{5}
$$

For a leaf $\mathsf t$, this is its conditional probability under the
uniform-child random walk on $\mathfrak S$. For a player node, it is its
conditional visitation probability. Hence

$$
\Lambda(\mathfrak S)
:=
\mathbb E^{\beta,\pi}
\left(
|\tau|_{\textnormal{player}}
\mid\mathfrak S
\right)
=
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s).
\tag{6}
$$

Combining the player and stationary-node laws of core_math.tex gives

$$
\mathbb P(\mathsf S=\mathsf s\mid\mathfrak S)
=
\frac{
\mathbf 1_{\{o(\ell(\mathsf s))\in\mathcal P\}}
w_{\mathfrak S}(\mathsf s)}
{\Lambda(\mathfrak S)}.
\tag{7}
$$

If an outgoing abstract child is then sampled uniformly,

$$
\mathbb P(\mathsf S=\mathsf s,\mathsf A=\mathsf a\mid\mathfrak S)
=
\frac{
\mathbf 1_{\{o(\ell(\mathsf s))\in\mathcal P\}}
w_{\mathfrak S}(\mathsf s)}
{\Lambda(\mathfrak S)N(\mathsf s)}.
\tag{8}
$$

Thus $\Lambda(\mathfrak S)$ is the exact branched analogue of the sequential
factor $L/N$.

## Matched comparison: branching alone

To isolate the effect of the public sampling design, use exactly the same
completed-world PG-plus-value integrand as in the sequential note:

$$
\Phi_k(x,\tau;\sigma,a)
:=
-\left(
R_{o(\sigma)}(x,\tau)
-V^{\theta_k}(x_{o(\sigma)},\sigma)
\right)
\left[
\nabla_\theta\log\pi^\theta
(a\mid x_{o(\sigma)},\sigma)
+2c_V\nabla_\theta V^\theta(x_{o(\sigma)},\sigma)
\right]_{\theta=\theta_k}.
\tag{9}
$$

For a public pre-terminal trajectory $t$, its exact Rao--Blackwellized
trajectory-sum gradient is

$$
F_k(t)
:=
\mathbb E^\pi\!\left[
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
\Phi_k(X,\Tau;\sigma,a)
\mathrel{\Bigg|}
\Tau^{\mathsf{pre-c}}=t
\right].
\tag{10}
$$

Its expectation under (2) is the theoretical trajectory-sum gradient

$$
\sum_{t\in\mathcal T^{\mathsf{pre-c}}}
\mathbb P^\pi(\Tau^{\mathsf{pre-c}}=t)F_k(t)
=
\mathbb E^\pi(|\tau|_{\textnormal{player}})
\left[\nabla_\theta l_k(\theta)\right]_{\theta=\theta_k}.
\tag{11}
$$

For $N_{\mathrm{seq}}$ iid sequential public rollouts,

$$
\boxed{
T_{\mathrm{rollout}}^{\mathrm{seq}}(N_{\mathrm{seq}})
=
\frac1{N_{\mathrm{seq}}}
\left[
\sum_t
\mathbb P^\pi(\Tau^{\mathsf{pre-c}}=t)
\lVert F_k(t)\rVert_2^2
-
\left\|
\sum_t
\mathbb P^\pi(\Tau^{\mathsf{pre-c}}=t)F_k(t)
\right\|_2^2
\right].
}
\tag{12}
$$

For one sampled branched tree, the matched conditional trajectory-sum
gradient is the leaf-weighted average

$$
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
=
\mathbb E^{\beta,\pi}
\left(
F_k(\Tau^{\mathsf{pre-c}})
\mid\mathfrak S
\right).
\tag{13}
$$

The terminal-distribution lemma in core_math.tex implies that (13) has
the same expectation as (11). Its public-rollout variance is therefore

$$
\boxed{
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(\beta)
&=
\operatorname{Var}_{\mathfrak S^{\beta,\pi}}
\left(
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
\right)\\
&=
\sum_{(\mathfrak S,\ell,N)}
\mathbb P^{\beta,\pi}(\mathfrak S,\ell,N)
\left\|
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
\right\|_2^2\\
&\quad-
\left\|
\sum_t
\mathbb P^\pi(\Tau^{\mathsf{pre-c}}=t)F_k(t)
\right\|_2^2.
\end{aligned}
}
\tag{14}
$$

For a particular realized child-count map $N=n$, the corresponding
conditional quantity is

$$
\operatorname{Var}\left(
\sum_{\mathsf t\in\partial\mathfrak S}
\left[
\prod_{\mathsf r\prec\mathsf t}\frac1{n(\mathsf r)}
\right]
F_k(\ell(\mathsf t))
\mathrel{\Bigg|}
N=n
\right),
\tag{15}
$$

where the conditional law of $(\mathfrak S,\ell)$ is the normalization of
(4) over trees compatible with $n$. The unconditional variance satisfies

$$
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(\beta)
&=
\mathbb E_N
\operatorname{Var}\left(
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
\mathrel{\Bigg|}N
\right)\\
&\quad+
\operatorname{Var}_N
\mathbb E\left(
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
\mathrel{\Bigg|}N
\right).
\end{aligned}
\tag{16}
$$

The second term in (16) is variance carried by the realized child-count map.
It vanishes when the conditioned count profile is fixed exogenously and
reveals no public labels. Merely disabling randomized rounding is not enough
if a state-dependent profile $n(\sigma)$ makes the realized counts informative
about the sampled public labels.

### Exact recursion for a known state-indexed count profile

Suppose $N(\mathsf s)=n(\ell(\mathsf s))$ deterministically. Define a mean
vector and a scalar trace variance backward on the public game tree. At a
pre-terminal state $t$, set

$$
m(t)=F_k(t),
\qquad
v_n(t)=0.
\tag{17}
$$

At every non-pre-terminal public state $\sigma$, set

$$
\begin{aligned}
m(\sigma)
&=
\sum_{e\in\mathcal E_\sigma}
K^\pi(e\mid\sigma)m(\sigma e),\\
v_n(\sigma)
&=
\frac1{n(\sigma)}
\sum_{e\in\mathcal E_\sigma}
K^\pi(e\mid\sigma)
\left[
v_n(\sigma e)
+\lVert m(\sigma e)-m(\sigma)\rVert_2^2
\right].
\end{aligned}
\tag{18}
$$

Branch independence proves (18): conditional on their public parent, the
$n(\sigma)$ child branches are iid, so averaging them divides their trace
variance by $n(\sigma)$. The desired matched branched rollout variance is

$$
\boxed{
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(n)
=v_n(\varnothing).
}
\tag{19}
$$

For the randomized-rounding rule, the same recursion remains exact after
replacing $1/n(\sigma)$ by

$$
\kappa_\beta(\sigma)
:=
\mathbb E\left(\frac1{N(\mathsf s)}
\mathrel{\Bigg|}\ell(\mathsf s)=\sigma\right)
=
\begin{cases}
\displaystyle\frac1{\beta(\sigma)},
&\beta(\sigma)\in\mathbb N,\\[1.2ex]
\displaystyle
\frac{\lceil\beta(\sigma)\rceil-\beta(\sigma)}
{\lfloor\beta(\sigma)\rfloor}
+
\frac{\beta(\sigma)-\lfloor\beta(\sigma)\rfloor}
{\lceil\beta(\sigma)\rceil},
&\beta(\sigma)\notin\mathbb N.
\end{cases}
\tag{20}
$$

In other words,

$$
v_\beta(\sigma)
=
\kappa_\beta(\sigma)
\sum_eK^\pi(e\mid\sigma)
\left[
v_\beta(\sigma e)
+\lVert m(\sigma e)-m(\sigma)\rVert_2^2
\right],
\qquad
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(\beta)
=v_\beta(\varnothing).
\tag{21}
$$

Unrolling the recursion makes the effect of the count profile completely
explicit. With
$\mathbb P^\pi(\sigma\preccurlyeq\Tau^{\mathsf{pre-c}})$ denoting public
reach probability,

$$
\boxed{
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(n)
&=
\sum_{\substack{\sigma\notin\mathcal T^{\mathsf{pre-c}}}}
\frac{
\mathbb P^\pi(\sigma\preccurlyeq\Tau^{\mathsf{pre-c}})
}{
\displaystyle\prod_{\substack{\rho\preccurlyeq\sigma\\
\rho\notin\mathcal T^{\mathsf{pre-c}}}}n(\rho)
}
\sum_{e\in\mathcal E_\sigma}
K^\pi(e\mid\sigma)
\lVert m(\sigma e)-m(\sigma)\rVert_2^2,\\
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(\beta)
&=
\sum_{\substack{\sigma\notin\mathcal T^{\mathsf{pre-c}}}}
\mathbb P^\pi(\sigma\preccurlyeq\Tau^{\mathsf{pre-c}})
\left(
\prod_{\substack{\rho\preccurlyeq\sigma\\
\rho\notin\mathcal T^{\mathsf{pre-c}}}}
\kappa_\beta(\rho)
\right)
\sum_{e\in\mathcal E_\sigma}
K^\pi(e\mid\sigma)
\lVert m(\sigma e)-m(\sigma)\rVert_2^2.
\end{aligned}
}
\tag{21a}
$$

Thus branching at an ancestor reduces the contribution of every public
innovation below that ancestor. For sequential sampling,
$n(\varnothing)=N_{\mathrm{seq}}$ and $n(\rho)=1$ below the root, so every
term in the first line of (21a) has the common factor
$1/N_{\mathrm{seq}}$.

Equations (18)--(21a) give the sharpest theoretical comparison. A sequential
batch of $N_{\mathrm{seq}}$ trajectories is recovered by branching only at
$\varnothing$ with count $N_{\mathrm{seq}}$ and taking count one at every
other non-pre-terminal node. A genuine branched rollout applies its ordinary
branching rule at $\varnothing$ and at every other non-pre-terminal node.

## Actual branched PG-plus-value loss

The matched estimator above holds the per-trajectory integrand fixed. The
actual branched loss in core_math.tex additionally uses an action-averaged
policy loss and a leave-one-out baseline.

Fix a realized tree $\mathfrak S$. For a player node $\mathsf s$ with
$\ell(\mathsf s)=\sigma$ and $p=o(\sigma)$, use the exact $\lambda=1$
targets and posteriors from the branched formalism:

$$
\widehat Q^1(x_p,\mathsf s\mathsf a),\qquad
\widehat V^1(x_p,\mathsf s),\qquad
\widehat V^{1,\neg\mathsf a}(x_p,\mathsf s),\qquad
\widehat A^1(x_p,\mathsf s\mathsf a)
=
\widehat Q^1(x_p,\mathsf s\mathsf a)
-
\widehat V^{1,\neg\mathsf a}(x_p,\mathsf s).
$$

Here “exact” means that private profiles and terminal completions at the
sampled leaves have been integrated exactly; the hats retain the notation of
core_math.tex because the targets still depend on the sampled public tree.

The exact gradient contribution conditional on sampling node $\mathsf s$ is

$$
\begin{aligned}
\Psi_k(\mathfrak S;\mathsf s)
&=
-\frac1{N(\mathsf s)}
\sum_{\mathsf a\in\mathfrak E_{\mathsf s}}
\sum_{x_p\in\mathcal X_p}
\mathbb P^{\beta,\pi}
(x_p\mid\mathsf s\mathsf a,\mathfrak S)
\widehat A^1(x_p,\mathsf s\mathsf a)
\left[
\nabla_\theta\log\pi^\theta
(a_{\mathsf a}\mid x_p,\sigma)
\right]_{\theta=\theta_k}\\
&\quad-
2c_V
\sum_{x_p\in\mathcal X_p}
\mathbb P^{\beta,\pi}
(x_p\mid\mathsf s,\mathfrak S)
\left(
\widehat V^1(x_p,\mathsf s)
-V^{\theta_k}(x_p,\sigma)
\right)
\left[
\nabla_\theta V^\theta(x_p,\sigma)
\right]_{\theta=\theta_k}.
\end{aligned}
\tag{22}
$$

For $B$ iid sampled player nodes
$(\mathsf S_b)_{b=1}^B$ with law (7), the scaled minibatch gradient is

$$
\frac{\Lambda(\mathfrak S)}B
\sum_{b=1}^B\Psi_k(\mathfrak S;\mathsf S_b).
\tag{23}
$$

Its conditional minibatch mean is

$$
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s).
\tag{24}
$$

The factor $\Lambda(\mathfrak S)$ cancels the denominator in (7), just as
$L/N$ cancels the empirical sequential decision-law denominator. Branch
independence makes the leave-one-out baseline unbiased, and the
terminal-distribution lemma then gives

$$
\mathbb E_{\mathfrak S^{\beta,\pi}}
\left[
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\right]
=
\mathbb E^\pi(|\tau|_{\textnormal{player}})
\left[\nabla_\theta l_k(\theta)\right]_{\theta=\theta_k}.
\tag{25}
$$

The exact public-rollout variance of the actual branched loss is therefore

$$
\boxed{
T_{\mathrm{rollout}}^{\mathrm{branch}}
=
\operatorname{Var}_{\mathfrak S^{\beta,\pi}}
\left(
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\right).
}
\tag{26}
$$

Using the actual tree law (4), this is equivalently

$$
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{branch}}
&=
\sum_{(\mathfrak S,\ell,N)}
\mathbb P^{\beta,\pi}(\mathfrak S,\ell,N)
\left\|
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\right\|_2^2\\
&\quad-
\left\|
\mathbb E^\pi(|\tau|_{\textnormal{player}})
\left[\nabla_\theta l_k(\theta)\right]_{\theta=\theta_k}
\right\|_2^2.
\end{aligned}
\tag{27}
$$

Conditioned on a known realized child-count map $N=n$, replace the outer law
in (26) by the conditional tree-label law obtained from (4), and replace
every path weight by

$$
w_n(\mathsf s)=\prod_{\mathsf r\prec\mathsf s}\frac1{n(\mathsf r)}.
\tag{28}
$$

The law of total variance gives the unconditional result:

$$
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{branch}}
&=
\mathbb E_N
\operatorname{Var}\left(
\sum_{\mathsf s}
w_N(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\mathrel{\Bigg|}N
\right)\\
&\quad+
\operatorname{Var}_N
\mathbb E\left(
\sum_{\mathsf s}
w_N(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\mathrel{\Bigg|}N
\right),
\end{aligned}
\tag{29}
$$

where all sums in (29) are over sampled-tree nodes with player-owned public
labels.

Unlike the matched quantity (14), the actual contribution
$\Psi_k(\mathfrak S;\mathsf s)$ depends on sibling branches through
$\widehat V^{1,\neg\mathsf a}$. Consequently, the simple scalar recursion
(18) does not apply to (26) without augmenting the recursive state by the
posterior and payoff moments needed by the leave-one-out target. Equations
(4), (26), and (27) are nevertheless exact and directly simulable.

## Minibatch and completion variance for the actual branched loss

The exact minibatch term follows from (7):

$$
\boxed{
\begin{aligned}
T_{\mathrm{minibatch}}^{\mathrm{branch}}
&=
\frac1B\,
\mathbb E_{\mathfrak S^{\beta,\pi}}
\Bigg[
\Lambda(\mathfrak S)
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)
\lVert\Psi_k(\mathfrak S;\mathsf s)\rVert_2^2\\
&\hspace{8em}-
\left\|
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\right\|_2^2
\Bigg].
\end{aligned}
}
\tag{30}
$$

For a finite leaf-completion budget $J$, let
$\widehat\Psi_{k,J}(\mathfrak S;\mathsf s)$ denote (22) computed from the
random target table returned by BranchedValue with $J$ completion samples
per sampled-tree leaf. Without assuming independence between node targets,
the exact expectation over multinomial minibatch counts is

$$
\boxed{
\begin{aligned}
T_{\mathrm{montecarlo},J}^{\mathrm{branch}}
&=
\mathbb E_{\mathfrak S^{\beta,\pi}}
\Bigg[
\left(1-\frac1B\right)
\operatorname{Var}\left(
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)
\widehat\Psi_{k,J}(\mathfrak S;\mathsf s)
\mathrel{\Bigg|}\mathfrak S
\right)\\
&\hspace{3em}+
\frac{\Lambda(\mathfrak S)}B
\sum_{\substack{\mathsf s\in\mathfrak S\\
o(\ell(\mathsf s))\in\mathcal P}}
w_{\mathfrak S}(\mathsf s)
\operatorname{Var}\left(
\widehat\Psi_{k,J}(\mathfrak S;\mathsf s)
\mid\mathfrak S
\right)
\Bigg].
\end{aligned}
}
\tag{31}
$$

The first conditional variance in (31) retains every covariance caused by
shared descendant completion samples and backward propagation through the
tree. Completion samples attached to distinct sampled-tree leaves are
independent, but the resulting targets at ancestor nodes are not.

If the finite-$J$ target construction is conditionally unbiased, then

$$
T_{\mathrm{total},J}^{\mathrm{branch}}
=
T_{\mathrm{rollout}}^{\mathrm{branch}}
+T_{\mathrm{minibatch}}^{\mathrm{branch}}
+T_{\mathrm{montecarlo},J}^{\mathrm{branch}}.
\tag{32}
$$

For the production self-normalized importance sampler and the nonlinear
conditional ratios in BranchedValue, conditional unbiasedness is not exact.
The exact finite-$J$ decomposition is obtained by replacing $\Psi_k$ by

$$
\mathbb E\left(
\widehat\Psi_{k,J}(\mathfrak S;\mathsf s)
\mid\mathfrak S
\right)
$$

in (26) and (30), while retaining (31) as written. In particular, one must
not assume
$T_{\mathrm{montecarlo},J}^{\mathrm{branch}}
=T_{\mathrm{montecarlo},1}^{\mathrm{branch}}/J$ for the actual branched
target construction.

## Comparison summary

The theoretically clean public-rollout comparison is

$$
\begin{aligned}
T_{\mathrm{rollout}}^{\mathrm{seq}}(N_{\mathrm{seq}})
&=
\frac1{N_{\mathrm{seq}}}
\operatorname{Var}_{\Tau^{\mathsf{pre-c}}}
\left(F_k(\Tau^{\mathsf{pre-c}})\right),\\
T_{\mathrm{rollout}}^{\mathrm{branch,matched}}(\beta)
&=
\operatorname{Var}_{\mathfrak S^{\beta,\pi}}
\left(
\sum_{\mathsf t\in\partial\mathfrak S}
w_{\mathfrak S}(\mathsf t)F_k(\ell(\mathsf t))
\right),\\
T_{\mathrm{rollout}}^{\mathrm{branch}}
&=
\operatorname{Var}_{\mathfrak S^{\beta,\pi}}
\left(
\sum_{\mathsf s:\,o(\ell(\mathsf s))\in\mathcal P}
w_{\mathfrak S}(\mathsf s)\Psi_k(\mathfrak S;\mathsf s)
\right).
\end{aligned}
\tag{33}
$$

The middle line measures branching alone. The last line measures the actual
branched algorithm, including its leave-one-out policy baseline. For a known
deterministic $n(\sigma)$, the middle line is computed exactly by
(17)--(19) or (21a). For randomized rounding, use (20)--(21a). For the
actual branched loss, use the tree law (4) with (26) or (27).

Sanity checks:

- $\beta\equiv1$ produces one public trajectory, and the matched branched
  formula reduces to the single-trajectory sequential formula.
- At $\beta\equiv1$, the leave-one-out fallback is $V^{\theta_k}$, so the
  actual and matched rollout formulas coincide.
- Sequential mode with $N_{\mathrm{seq}}$ rollouts is branching only at
  $\varnothing$ with count $N_{\mathrm{seq}}$ and count one below it.
- In branched mode, $\varnothing$ follows the same ordinary branching rule
  as every other non-pre-terminal node.
- Duplicate public labels remain distinct sampled-tree nodes and contribute
  separately with their own path weights and independent descendant draws.
- The comparison should control computational cost separately; equal leaf
  count, equal sampled-tree size, and equal wall time are different
  constraints.

## Approved checkpoint-1110 experiment

The executable experiment is configured in
`configs/experiments/nlhe_hu_32bb_branched_rao_blackwell_variance.yaml`.  It
uses only checkpoint 1110 from the completed heads-up, 32 bb, full-deck run.
It does not average checkpoints: the public-tree distribution and frozen
gradient integrand are both those of iteration 1110.

The main comparison is $\beta\in\{2,2.25,2.5\}$.  These values cover the
useful range indicated by the checkpoint-814 geometry runs without spending
GPU time on the already-unattractive low-branching regime.  After the coarse
pilot, the winner is bracketed at spacing 0.125, subject to the fixed interval
$[2,2.5]$.  All candidates use common logical tree seeds.

Every candidate samples one rooted labeled tree.  The root receives the same
ordinary randomized-rounding rule as every other non-pre-terminal node.  The
tree has a 12,000-node allocation bound.  When a proposed expansion would
take the stored tree to 4,000 nodes, the production global cutoff latch turns
on and all subsequent nodes use $\beta=1$; branching is never re-enabled in
that tree.  This is not a stop cutoff.  The 12,000-node hard bound is practical
headroom: prior runs with this 4,000-node latch ended below 6,850 nodes at
$\beta=2.5$, while even $\beta=3$ stayed below 9,700 in the observed sample.

The staged inventory is:

1. 64 geometry-only trees for each coarse or refinement candidate;
2. 8 full moment trees for each candidate;
3. 32 total full moment trees for the selected $\beta$, extended in blocks of
   8 up to 64 only while a component's relative standard error exceeds 10%.

Four fixed Rademacher parameter directions estimate gradient-covariance
trace.  Four independent node-space Rademacher probes estimate the diagonal
completion term without materializing a decision-by-decision covariance
matrix.  Immutable jobs, a manifest fingerprint, common logical seeds, and
resume behavior follow the sequential Rao--Blackwell variance run's format.

For a tree and projected per-decision loss gradients $\psi_s$, the run records

$$
G=\sum_s w_s\psi_s,
\qquad
A=\Lambda\sum_s w_s\psi_s^2-G^2.
$$

Thus the branching-specific minibatch contribution is measured as $A/B$.
It is not imported or extrapolated from the sequential experiment.  The
reported batch sizes are $B\in\{32,64,128,256\}$.  The selection reference is
$B=64$ because that batch size already has completed optimizer probes on the
11 GiB hardware and avoids presuming that the sequential run's larger batch
remains useful after branching improves the estimator.

Completion is not swept.  Each moment tree uses
`complete_linear_moments` with $n_{\rm oracle}=8192$ to estimate the two
tree-conditional coefficients

$$
C_{\rm pair}=n\operatorname{Var}(G\mid\mathfrak S),
\qquad
C_{\rm node}=n\Lambda\sum_s w_s
\operatorname{Var}(\psi_s\mid\mathfrak S).
$$

The implementation differentiates the actual leave-one-out target with
respect to each leaf's posterior/payoff moments and sends those linearized
payoff functionals to the evaluator.  It then reports, analytically,

$$
T(\beta,B,n)
=T_{\rm tree}(\beta)+\frac{A(\beta)}B
+\frac1n\left[
\left(1-\frac1B\right)C_{\rm pair}(\beta)
+\frac{C_{\rm node}(\beta)}B
\right].
$$

The $n$ table $\{64,128,256,512,1024\}$ is therefore a cheap reconstruction
of the explicit $1/n$ law, not five completion experiments.  The selected
$\beta$ receives one 16,384-sample consistency check.

The run selects/refines $\beta$ at the practical reference point
$(B,n)=(64,1024)$ and emits the whole $(\beta,B,n)$ table, so a different
operational constraint can be applied afterward without rerunning trees.
$K$, the number of optimizer steps per iterate, is intentionally not selected
from a frozen one-gradient variance experiment.  Once $\beta$ and $B$ are
known, $K$ should be set by a short wall-time/KL training probe; otherwise a
variance-only run would silently treat correlated reuse of one tree as fresh
independent gradients.

Before any long job, a real-checkpoint CUDA smoke uses a 96-node cutoff and
256 oracle samples.  The measured allocated-memory ceiling is 8.5 GiB on the
11 GiB cards.  A timed full tree must also project the complete campaign below
24 active GPU-hours.  These are launch gates, not statistical hyperparameters.
