# Gradient-variance reduction from Rao--Blackwellization in NLHE

## Question

Using policies saved during
`runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_optimized`, measure:

> For one scaled sequential Rao--Blackwell gradient estimator, how do the
> rollout, minibatch, and completion sources of variance depend on $N$, $B$,
> and $n_{\mathrm{samples}}$? Which feasible triple minimizes total gradient
> variance under specified wall-time and GPU-memory limits?

At every checkpoint, fresh on-policy public rollouts are evaluated with
player-clock $\lambda=1$ and

$$
\mathcal L_{\tt PG}[\widehat A^1]
+\mathcal L_{\tt V}[\widehat V^1].
$$

The gradient is evaluated at $\theta=\theta_k$, where $\theta_k$ is both the
frozen checkpoint and behavior policy. The policy loss is unclipped and the
value loss is ordinary MSE. Entropy, KL, clipping, AMP scaling, gradient
clipping, optimizer state, and parameter updates are excluded.

The object of study is exactly one gradient estimator from one minibatch
drawn with the empirical pre-terminal decision law. Parameter updates,
sequences of gradients, and stochastic-optimization dynamics are outside the
experiment.

## Gradient and variance

Freeze the rollout policy at $\pi=\pi^{\theta_k}$, and evaluate every
derivative at $\theta=\theta_k$. For a vector-valued random variable,
$\operatorname{Var}$ always means trace covariance:

$$
\operatorname{Var}(Y)
:=\operatorname{tr}\operatorname{Cov}(Y)
=\mathbb E\lVert Y-\mathbb EY\rVert_2^2.
$$

The same convention applies to conditional variance.

Let

$$
\mathfrak T^{\mathsf{pre-c}}
=((\Tau_i^{\mathsf{pre-c}},\mathfrak r_i))_{i=1}^N,
\qquad
D_i:=|\Tau_i^{\mathsf{pre-c}}|_{\textnormal{player}},
\qquad
L=\sum_{i=1}^N D_i,
\qquad
\ell:=\frac LN=\frac1N\sum_{i=1}^N D_i.
$$

The empirical mean trajectory length is therefore

$$
\mathbb E(|\tau|_{\textnormal{player}}
\mid\mathfrak T^{\mathsf{pre-c}})=\ell,
$$

where the conditional expectation uses the empirical law that chooses one of
the $N$ public rollouts uniformly. Let
$((I_b,P_b,\Sigma_b,A_b))_{b=1}^B$ be iid samples from the empirical
pre-terminal decision law in `core_math.tex`. Conditional on
$\mathfrak T^{\mathsf{pre-c}}$, every realized player decision
$(i,o(\sigma),\sigma,a)$ has probability $1/L$.

For a terminal completion $(x,\tau)$ of a public pre-terminal trajectory and
a player decision $\sigma a\preccurlyeq\tau$, define the single node-gradient
integrand

$$
\Phi_k(x,\tau;\sigma,a)
:=
-\left(R_{o(\sigma)}(x,\tau)
      -V^{\theta_k}(x_{o(\sigma)},\sigma)\right)
\left[
\nabla_\theta\log\pi^\theta
  (a\mid x_{o(\sigma)},\sigma)
+2c_V\nabla_\theta V^\theta(x_{o(\sigma)},\sigma)
\right]_{\theta=\theta_k}.
\tag{1}
$$

This is the gradient of
$\mathcal L_{\tt PG}[\widehat A^1]+\mathcal L_{\tt V}[\widehat V^1]$ for
one completed private world. At $\theta_k$, the policy importance ratio is
one and both loss gradients are linear in
$\widehat Q^1-V^{\theta_k}$. Consequently, conditioning (1) on the public
pre-terminal trajectory gives the exact Rao--Blackwellized gradient, even
though the value loss itself is quadratic.

For each $i$ and $j$, sample conditionally independently

$$
(X_{i,j},\Tau_{i,j})
\sim
\mathbb P^\pi\!\left(
\mathord\cdot\mid
\Tau^{\mathsf{pre-c}}=\Tau_i^{\mathsf{pre-c}}
\right).
$$

The scaled sampled-private, finite-$n$ Rao--Blackwell, and exact
Rao--Blackwell minibatch gradients are, respectively,

$$
\begin{aligned}
\ell\left[\nabla_\theta\widehat l_{\mathrm{sampled}}(\theta)\right]_{\theta_k}
&=
\frac{\ell}{B}\sum_{b=1}^B
\Phi_k(X_{I_b,1},\Tau_{I_b,1};\Sigma_b,A_b),\\
\ell\left[\nabla_\theta\widehat l_{\mathrm{RB},n}(\theta)\right]_{\theta_k}
&=
\frac{\ell}{nB}\sum_{j=1}^n\sum_{b=1}^B
\Phi_k(X_{I_b,j},\Tau_{I_b,j};\Sigma_b,A_b),\\
\ell\left[\nabla_\theta\widehat l_{\mathrm{RB},\infty}(\theta)\right]_{\theta_k}
&=
\frac{\ell}{B}\sum_{b=1}^B
\mathbb E^\pi\!\left[
\Phi_k(X,\Tau;\Sigma_b,A_b)
\mid
\Tau^{\mathsf{pre-c}}=\Tau_{I_b}^{\mathsf{pre-c}}
\right].
\end{aligned}
\tag{2}
$$

In the first two lines, one completion $(X_{i,j},\Tau_{i,j})$ is shared by
all sampled decisions belonging to public rollout $i$. This sharing is what
creates covariance among their gradient contributions.

The factor $\ell=L/N$ cancels the random denominator in the empirical decision
law. Hence all three estimators in (2) have expectation

$$
\begin{aligned}
&\mathbb E^\pi\!\left[
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
\Phi_k(X,\Tau;\sigma,a)
\right]\\
&\qquad=
\mathbb E^\pi(|\tau|_{\textnormal{player}})
\,
\mathbb E^\pi\!\left[
\Phi_k(X,\Tau;\Sigma,A)
\right]\\
&\qquad=
\mathbb E^\pi(|\tau|_{\textnormal{player}})
\left[\nabla_\theta l_k(\theta)\right]_{\theta=\theta_k},
\end{aligned}
\tag{3}
$$

where $(X,P,\Sigma,A)$ has the ideal decision law of `core_math.tex`, and
$l_k$ is its PG-plus-value loss for rollout profile $\pi^{\theta_k}$. Thus
the $\ell$ scaling removes the finite-$N$ random-length ratio bias: the
expected scaled gradient is exactly the theoretical trajectory-sum gradient.

Applying the law of total variance, first with respect to the public rollout
batch and then with respect to the minibatch, gives

$$
\boxed{
\begin{aligned}
T_{\mathrm{rollout}}
&=
\operatorname{Var}_{\mathfrak T^{\mathsf{pre-c}}}
\left(
\frac1N\sum_{i=1}^N
\sum_{\substack{\sigma a\preccurlyeq
\Tau_i^{\mathsf{pre-c}}\\o(\sigma)\in\mathcal P}}
\mathbb E^\pi\!\left[
\Phi_k(X,\Tau;\sigma,a)
\mid
\Tau^{\mathsf{pre-c}}=\Tau_i^{\mathsf{pre-c}}
\right]
\right),\\
T_{\mathrm{minibatch}}
&=
\mathbb E_{\mathfrak T^{\mathsf{pre-c}}}
\operatorname{Var}_{\mathcal B}
\left(
\frac{\ell}{B}\sum_{b=1}^B
\mathbb E^\pi\!\left[
\Phi_k(X,\Tau;\Sigma_b,A_b)
\mid
\Tau^{\mathsf{pre-c}}=\Tau_{I_b}^{\mathsf{pre-c}}
\right]
\mathrel{\Bigg|}
\mathfrak T^{\mathsf{pre-c}}
\right),\\
T_{\mathrm{montecarlo}}
&=
\mathbb E_{\mathfrak T^{\mathsf{pre-c}},\mathcal B}
\operatorname{Var}
\left(
\frac{\ell}{B}\sum_{b=1}^B
\Phi_k(X_{I_b,1},\Tau_{I_b,1};\Sigma_b,A_b)
\mathrel{\Bigg|}
\mathfrak T^{\mathsf{pre-c}},\mathcal B
\right).
\end{aligned}}
\tag{4}
$$

These definitions immediately imply

$$
\begin{aligned}
T_{\mathrm{sampled}}
&=T_{\mathrm{rollout}}+T_{\mathrm{minibatch}}
  +T_{\mathrm{montecarlo}},\\
T_{\mathrm{RB},\infty}
&=T_{\mathrm{rollout}}+T_{\mathrm{minibatch}},\\
T_{\mathrm{RB},n}
&=T_{\mathrm{rollout}}+T_{\mathrm{minibatch}}
  +\frac1nT_{\mathrm{montecarlo}}.
\end{aligned}
\tag{5}
$$

The last identity is exact for iid conditional completions. The ideal
fraction of variance removed is
$T_{\mathrm{montecarlo}}/T_{\mathrm{sampled}}$, and the ideal
variance-reduction factor is
$T_{\mathrm{sampled}}/T_{\mathrm{RB},\infty}$.

Because the $N$ public rollouts are iid, the rollout term is the variance of
one Rao--Blackwellized trajectory-sum gradient divided by $N$. A further
application of total variance gives the equivalent difference formula

$$
\begin{aligned}
T_{\mathrm{rollout}}
&=
\frac1N
\operatorname{Var}_{\Tau^{\mathsf{pre-c}}}
\left(
\mathbb E^\pi\!\left[
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
\Phi_k(X,\Tau;\sigma,a)
\mathrel{\Bigg|}\Tau^{\mathsf{pre-c}}
\right]
\right)\\
&=
\frac1N\Bigg\{
\operatorname{Var}^\pi\!\left(
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
\Phi_k(X,\Tau;\sigma,a)
\right)\\
&\hspace{5em}-
\mathbb E_{\Tau^{\mathsf{pre-c}}}
\operatorname{Var}^\pi\!\left(
\sum_{\substack{\sigma a\preccurlyeq\Tau\\
o(\sigma)\in\mathcal P}}
\Phi_k(X,\Tau;\sigma,a)
\mathrel{\Bigg|}\Tau^{\mathsf{pre-c}}
\right)
\Bigg\}.
\end{aligned}
\tag{6}
$$

The second equality isolates the public-rollout variance as the part of the
full trajectory-sum gradient variance that remains after integrating the
private profile and terminal completion.

The minibatch and completion terms have more informative exact forms when
written in units of the empirical mean length $\ell$, rather than in units of
the total decision count $L$.  For decision $\nu=(\sigma,a)$ on public
rollout $i$, set

$$
\begin{aligned}
\mu_{i,\nu}
&:=\mathbb E^\pi[\Phi_k(X,\Tau;\nu)
  \mid\Tau^{\mathsf{pre-c}}=\Tau_i^{\mathsf{pre-c}}],\\
G_i&:=\sum_{\nu\preccurlyeq\Tau_i^{\mathsf{pre-c}}}\mu_{i,\nu},
&
Q_i&:=\sum_{\nu\preccurlyeq\Tau_i^{\mathsf{pre-c}}}
       \lVert\mu_{i,\nu}\rVert_2^2,\\
\overline G_N&:=\frac1N\sum_iG_i,
&
\overline Q_N&:=\frac1N\sum_iQ_i.
\end{aligned}
\tag{7a}
$$

All decision sums in this section are restricted to player-owned decisions.
Let $(D,G,Q)$ denote a generic one-rollout copy of $(D_i,G_i,Q_i)$ and let
$\ell_\star:=\mathbb E D=\mathbb E\ell$.  Conditional iid minibatch sampling
then gives

$$
\begin{aligned}
T_{\mathrm{minibatch}}(N,B)
&=\frac1B\mathbb E\left[
   \ell\,\overline Q_N-\lVert\overline G_N\rVert_2^2
   \right]\\
&=\frac1B\left[
  \left(1-\frac1N\right)A_{\mathrm{MB}}^{\ne}
  +\frac1N A_{\mathrm{MB}}^{=}
  \right]\\
&=\frac{A_{\mathrm{MB}}^{\ne}}B
  +\frac{A_{\mathrm{MB}}^{=}-A_{\mathrm{MB}}^{\ne}}{BN},
\end{aligned}
\tag{7b}
$$

where

$$
\boxed{
\begin{aligned}
A_{\mathrm{MB}}^{\ne}
&:=\ell_\star\,\mathbb E Q-\lVert\mathbb E G\rVert_2^2,
&&\textnormal{distinct-rollout, or non-collision, coefficient},\\
A_{\mathrm{MB}}^{=}
&:=\mathbb E\!\left[DQ-\lVert G\rVert_2^2\right],
&&\textnormal{same-rollout, or collision, coefficient}.
\end{aligned}}
\tag{7c}
$$

This split follows by expanding

$$
\ell\,\overline Q_N-\lVert\overline G_N\rVert_2^2
=\frac1{N^2}\sum_{i,j}
  \left(D_jQ_i-G_i^\top G_j\right)
\tag{7d}
$$

and separating $i\ne j$ from $i=j$.  Thus "collision" in (7c) means a
collision of the two public-rollout indices introduced by the second-moment
expansion; it does not mean that iid minibatch draws become dependent when
they select the same deterministic Rao--Blackwellized node.  Both coefficients
in (7c) are nonnegative by Cauchy--Schwarz.  Their difference can have either
sign, so increasing $N$ can approach the limiting coefficient from either
side.  Crucially, $N$ changes only the $1/(BN)$ correction.  The useful
limits are

$$
\begin{array}{rcl}
B\to\infty\textnormal{ at fixed }N
&:&T_{\mathrm{minibatch}}\to0\quad\textnormal{exactly at rate }B^{-1},\\
N\to\infty\textnormal{ at fixed }B
&:&T_{\mathrm{minibatch}}\to A_{\mathrm{MB}}^{\ne}/B,\\
B/N\to\rho\in(0,\infty)
&:&T_{\mathrm{minibatch}}
=A_{\mathrm{MB}}^{\ne}/(\rho N)+O(N^{-2}).
\end{array}
\tag{7e}
$$

The corner $N=1$ gives $T_{\mathrm{minibatch}}=A_{\mathrm{MB}}^{=}/B$,
which is also an immediate check on the collision weighting.

For the completion term, center each node contribution conditionally on its
public rollout and define

$$
\begin{aligned}
\xi_{i,\nu}
&:=\Phi_k(X,\Tau;\nu)-\mu_{i,\nu},\\
U_i
&:=\sum_{\nu\preccurlyeq\Tau_i^{\mathsf{pre-c}}}
  \mathbb E^\pi[\lVert\xi_{i,\nu}\rVert_2^2
  \mid\Tau_i^{\mathsf{pre-c}}],\\
V_i
&:=\mathbb E^\pi\!\left[
  \left\lVert\sum_{\nu\preccurlyeq\Tau_i^{\mathsf{pre-c}}}
  \xi_{i,\nu}\right\rVert_2^2
  \mathrel{\Bigg|}\Tau_i^{\mathsf{pre-c}}\right],\\
\overline U_N&:=\frac1N\sum_iU_i,
&
\overline V_N&:=\frac1N\sum_iV_i.
\end{aligned}
\tag{8a}
$$

$U_i$ keeps only nodewise conditional variances, whereas $V_i$ also keeps
every covariance between decisions sharing rollout $i$'s completion.  The
exact multinomial second moments of the minibatch counts give

$$
\boxed{
\begin{aligned}
T_{\mathrm{montecarlo}}(N,B)
&=\underbrace{\frac1B\mathbb E[\ell\,\overline U_N]}
    _{\textnormal{one-draw (non-pair) term}}+
  \underbrace{\left(1-\frac1B\right)\frac{\mathbb E\overline V_N}{N}}
    _{\textnormal{distinct-draw, same-rollout collision term}}\\
&=\frac1B\left[
  \left(1-\frac1N\right)C_{\mathrm{MC}}^{\ne}
  +\frac1N C_{\mathrm{MC}}^{=}
  \right]
  +\left(1-\frac1B\right)\frac{C_{\mathrm{MC}}^{\mathrm{pair}}}{N}\\
&=\frac{C_{\mathrm{MC}}^{\ne}}B
  +\frac{C_{\mathrm{MC}}^{\mathrm{pair}}}{N}
  +\frac{C_{\mathrm{MC}}^{=}-C_{\mathrm{MC}}^{\ne}
         -C_{\mathrm{MC}}^{\mathrm{pair}}}{BN},
\end{aligned}}
\tag{8b}
$$

where, for a generic one-rollout copy $(D,U,V)$,

$$
\begin{aligned}
C_{\mathrm{MC}}^{\ne}&:=\ell_\star\,\mathbb E U,
&
C_{\mathrm{MC}}^{=}&:=\mathbb E[DU],
&
C_{\mathrm{MC}}^{\mathrm{pair}}&:=\mathbb E V.
\end{aligned}
\tag{8c}
$$

At $B=1$, the distinct-draw collision term vanishes and
$T_{\mathrm{montecarlo}}=(1-1/N)C_{\mathrm{MC}}^{\ne}+C_{\mathrm{MC}}^{=}/N$.
At $N=1$, the formula becomes
$C_{\mathrm{MC}}^{=}/B+(1-1/B)C_{\mathrm{MC}}^{\mathrm{pair}}$.

The $b=b'$ terms in the minibatch second moment produce the first term of
(8b).  Inside its expectation, the random normalization supplies a second
public-rollout index, so it has the same $i\ne j$ versus $i=j$ split as
(7b).  For $b\ne b'$, conditional completion covariance is zero unless the
two sampled decisions collide on the same public rollout; summing those
within-rollout covariances produces $V_i$ and the second term of (8b).
Consequently,

$$
\begin{array}{rcll}
B\to\infty\textnormal{ at fixed }N
&:&T_{\mathrm{montecarlo}}\to C_{\mathrm{MC}}^{\mathrm{pair}}/N,
&\textnormal{shared-completion collision floor},\\
N\to\infty\textnormal{ at fixed }B
&:&T_{\mathrm{montecarlo}}\to C_{\mathrm{MC}}^{\ne}/B,
&\textnormal{independent selected-node floor},\\
B/N\to\rho\in(0,\infty)
&:&T_{\mathrm{montecarlo}}
=N^{-1}(C_{\mathrm{MC}}^{\mathrm{pair}}
       +C_{\mathrm{MC}}^{\ne}/\rho)+o(N^{-1}).
\end{array}
\tag{8d}
$$

Thus the generic joint rate is $\Theta(B^{-1}+N^{-1})$, not merely
$\Theta(B^{-1})$.  Increasing only $B$ cannot average away the one completion
shared by all selected decisions on a rollout, and increasing only $N$ cannot
average away the completion noise of the $B$ selected nodes.  With $n$ iid
completion replicas, every expression in (8b)--(8d) is divided exactly by
$n$.

For parameter-space probes, if $z$ is independent of everything else and
$\mathbb E(zz^\top)=I$, then
$\operatorname{Var}(Y)=\mathbb E_z\operatorname{Var}(z^\top Y)$.  Hence
$U_i$ and $V_i$ can be evaluated from projected node and trajectory-sum
conditional variances without storing covariance matrices.

For a self-normalized importance-sampling integration oracle, with
$q_j=w_j/\sum_\ell w_\ell$, estimate each scalar conditional world variance
and the variance of its weighted Monte Carlo mean by

$$
\widehat v_Z
=
\frac{\sum_jq_j(f_j-\sum_\ell q_\ell f_\ell)^2}
     {1-\sum_jq_j^2},
\qquad
\widehat v_{\mathrm{MC}}
=
\frac{\sum_jq_j^2(f_j-\sum_\ell q_\ell f_\ell)^2}
     {1-\sum_jq_j^2}.
\tag{9}
$$

Use $\widehat v_Z$ in the two variance slots of (8) to estimate the
exact-posterior conditional world variances.  Use
$\widehat v_{\mathrm{MC}}$ to quantify numerical noise in the oracle's
estimated conditional means.  These are finite-sample ratio-estimator moment
estimates, not exact identities for random importance weights.  They become
consistent under the ordinary importance-sampling conditions as the oracle
budget grows.  With uniform weights their algebra reduces to the usual
sample-variance and sample-mean-variance formulas.  Self-normalization is
generally biased at finite sample count; the $\ell$ factor removes the
random-length ratio bias, not this separate numerical-integration bias.

Finally, the Monte Carlo noise in the estimated conditional mean used for
$T_{\mathrm{rollout}}$ is not
$T_{\mathrm{montecarlo},n}$. For each public batch, its trace is estimated by

$$
\mathbb E_z\sum_{i=1}^N
\widehat v_{\mathrm{MC}}\!\left(
\frac1N
\sum_{\substack{\sigma a\preccurlyeq
\Tau_i^{\mathsf{pre-c}}\\o(\sigma)\in\mathcal P}}
z^\top\Phi_k(X,\Tau;\sigma,a)
\right).
\tag{10}
$$

Subtract the public-batch average of (10) from the sample trace variance of
the estimated conditional means when estimating $T_{\mathrm{rollout}}$.

## Experiment

### Operational summary

The implementation and run have five concrete stages:

1. Add `evaluator.complete_linear_moments` in `poker-gpu-evaluator`.  It takes
   the ordinary completion inputs plus bounded $(c^P,c^M)$ hand coefficients
   and returns the weighted mean, conditional-world variance, and numerical
   variance of each mean from the same completion worlds.
2. Add the PBG experiment runner.  For one `(checkpoint, superbatch)` job it
   rolls 3072 public trajectories, streams complete trajectory chunks, builds
   one node functional per decision plus one trajectory-sum functional per
   probe, calls the new API at large $n_{\mathrm{oracle}}$, updates only the
   sufficient statistics in (20), and releases the coefficients immediately.
3. Run the CPU oracle, bounded one-GPU smoke, and oracle-budget calibration.
   No full run is admitted until their numerical, determinism, ESS, and memory
   gates pass.
4. Run the 40 independent variance jobs (five checkpoints times eight
   superbatches) across three GPUs.  The jobs contain no minibatches and no
   DDP: equations (21)--(24) analytically reconstruct every candidate
   $(N,B,n)$ from the streamed moments.
5. Benchmark candidate cost, select the feasible frontier/finalists, and then
   draw actual minibatches only for direct validation.  In that last phase,
   call the functional API only for the unique leaves selected by each
   minibatch and discard the combined functionals after the call.

At no point are 128 minibatches' coefficient tensors saved.  The persistent
scientific data are compact scalar sufficient statistics and short diagnostic
summaries; dense $1326$-hand coefficients exist only for the current bounded
trajectory/probe chunk.

### Exact estimand and the two completion budgets

The experiment studies the theoretical estimator in (2): conditional worlds
are iid draws from the exact posterior and their arithmetic mean is used in
the gradient.  The sibling evaluator's self-normalized importance sampler is
only a numerical integration oracle for the conditional mean and conditional
world variance.  The experiment does **not** claim to measure the complete
finite-sample law of the production SNIS estimator.

Keep two sample counts distinct everywhere in code, configuration, filenames,
and output:

- $n_{\mathrm{oracle}}$ is the large SIS proposal budget used by
  `complete_linear_moments` to estimate conditional moments accurately;
- $n$ is a candidate number of hypothetical iid posterior completions in the
  theoretical gradient estimator.

For the latter estimator,

$$
\boxed{
\begin{aligned}
\mathcal V(N,B,n)
&=\frac{T_{\mathrm{rollout}}(1)}N
+\frac1B\left[
  \left(1-\frac1N\right)A_{\mathrm{MB}}^{\ne}
  +\frac1N A_{\mathrm{MB}}^{=}
  \right]\\
&\quad+\frac1n\left\{
  \frac1B\left[
    \left(1-\frac1N\right)C_{\mathrm{MC}}^{\ne}
    +\frac1N C_{\mathrm{MC}}^{=}
  \right]
  +\left(1-\frac1B\right)\frac{C_{\mathrm{MC}}^{\mathrm{pair}}}{N}
  \right\}\\
&=\frac{T_{\mathrm{rollout}}(1)}N
  +\frac{A_{\mathrm{MB}}^{\ne}}B
  +\frac{A_{\mathrm{MB}}^{=}-A_{\mathrm{MB}}^{\ne}}{BN}\\
&\quad+\frac1n\left[
  \frac{C_{\mathrm{MC}}^{\ne}}B
  +\frac{C_{\mathrm{MC}}^{\mathrm{pair}}}{N}
  +\frac{C_{\mathrm{MC}}^{=}-C_{\mathrm{MC}}^{\ne}
         -C_{\mathrm{MC}}^{\mathrm{pair}}}{BN}
  \right].
\end{aligned}
}
\tag{11}
$$

The reciprocal surface in the last two lines is exact for iid public rollouts,
iid with-replacement minibatch draws, and the shared-per-rollout completion
construction in (2).  In particular, the experiment estimates six
one-rollout coefficients rather than treating three arbitrary functions of
$N$ as free.  Blockwise direct estimates of (7b) and (8b) remain useful checks
of the exact finite-$N$ laws.

The weighted empirical variance $\widehat v_Z$ in (9), evaluated at large
$n_{\mathrm{oracle}}$, estimates the exact-posterior conditional world
variance needed in (11).  The squared-weight quantity
$\widehat v_{\mathrm{MC}}$ estimates only the remaining numerical integration
noise of that oracle call.  It is used for noise correction and diagnostics,
not as the candidate completion term.  Accuracy is governed by effective
sample size and independent-seed agreement, not by nominal
$n_{\mathrm{oracle}}$ alone.

### Source policies

Use the five checkpoints from the completed heads-up, full-deck, 32 bb
sequential-RB source run:

`runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_optimized`

| Stage | Checkpoint |
|---|---:|
| early | ckpt_0000037.pt |
| quarter | ckpt_0000296.pt |
| middle | ckpt_0000555.pt |
| three-quarter | ckpt_0000814.pt |
| terminal | ckpt_0001110.pt |

Hash each checkpoint before evaluation. At every checkpoint,
$\theta_k$ is both the frozen behavior policy and the point at which the
gradient is evaluated.

The source run used $(N,B,n)=(3072,128,1024)$ per GPU. This is an anchor for
the candidate grid, not the geometry at which the variance experiment is
fixed.

The experiment configuration must explicitly fix the following choices rather
than inherit them silently from the source checkpoint:

- heads-up, full-deck, 32 bb NLHE;
- player-clock $\lambda=1$;
- unclipped PG and ordinary MSE value loss with $c_V=0.0005$;
- every sampled/Rao--Blackwell return, advantage, posterior weight, and old
  policy output is a frozen target with no autograd path;
- no entropy, KL, gradient clipping, optimizer, scaler, or parameter update;
- evaluation mode with one documented autocast choice used consistently for
  rollout and probe derivatives; the default scientific run is fp32;
- the ordered set of trainable model parameters included in the gradient;
- eight parameter-space Rademacher probes generated from fixed logical seeds;
  the same probes are reused for every checkpoint and candidate geometry; and
- distinct logical seed domains for public rollout, oracle completion,
  direct-validation completion, and minibatch draws.

The implementation must verify the checkpoint game/model geometry before GPU
allocation and record the checkpoint hash, checkpoint iteration, model
parameter count, probe seeds, evaluator build identity, PBG git revision,
precision choice, and complete experiment configuration in every run manifest.

### Candidate geometry

The initial safe candidate values are

$$
\begin{aligned}
N&\in\{384,768,1536,3072\},\\
B&\in\{32,64,128,256\},\\
n&\in\{128,256,512,1024\}.
\end{aligned}
\tag{12}
$$

Start with $n_{\mathrm{oracle}}=8192$ proposals.  It is an integration-control
setting, not part of the candidate grid.  Increase it geometrically on the
calibration subset until the aggregate conditional means and variances pass
the agreement gates specified below.  A larger value must be evaluated with
smaller leaf microbatches if necessary; it must never be admitted merely
because its nominal count is large while its effective sample size remains
poor.

Larger values are added one axis at a time only after the production memory
benchmark predicts adequate headroom and a smoke run confirms it. If the
constrained optimum lies on the largest tested value of an axis, that axis
must be extended before declaring an optimum.

No full Cartesian-product variance experiment is run. The variance cube is
reconstructed from projected conditional expectations and variances.

### Required `poker-gpu-evaluator` API

The experiment is blocked until the sibling evaluator exposes one batched,
seeded, bounded linear-functional moment operation.  A suitable semantic
contract, extending the bound evaluator API in its authoritative
`docs/api.md`, is:

```python
evaluator.complete_linear_moments(
    *,
    board_cards: uint8[B_leaf, 5],
    range_weights: float32[B_leaf, P, H],
    stratum_level: int32[B_leaf, P],
    payoff_baseline: float32[B_leaf, P],
    stratum_amount: float32[B_leaf, K],
    p_coefficient: float32[B_leaf, F, P, H],
    m_coefficient: float32[B_leaf, F, P, H],
    functional_mask: bool[B_leaf, F],
    n_samples: int,
    seed: int64[B_leaf],
) -> LinearMomentResult
```

Here $H=1326$ for the full deck.  A functional $f$ is the linear form

$$
f(x,\tau)
=
\sum_{p\ \mathrm{dealt}}
\left[
c^P_{f,p}(x_p)
+c^M_{f,p}(x_p)R_p(x,\tau)
\right].
\tag{13}
$$

Consequently its conditional mean is

$$
\mathbb E[f\mid\Tau^{\mathsf{pre-c}}]
=\sum_{p,h}
\left[c^P_{f,p}(h)P_p(h)+c^M_{f,p}(h)M_p(h)\right],
\qquad
M_p(h)=P_p(h)Q_p(h).
$$

This is why two coefficient arrays are requested: the operation reduces the
stable moments $P$ and $M=P Q$ directly and never divides by a small posterior
probability to form $Q$.

The result must contain device-local fp32 tensors

```python
LinearMomentResult(
    mean: float32[B_leaf, F],
    world_variance: float32[B_leaf, F],
    mean_variance: float32[B_leaf, F],
    n_eff: float32[B_leaf],
    functional_valid: bool[B_leaf, F],
)
```

with these semantics:

- `mean` is the self-normalized weighted empirical mean of $f$;
- `world_variance` is $\widehat v_Z$ from (9), estimating the variance of
  $f$ under the exact conditional world law;
- `mean_variance` is $\widehat v_{\mathrm{MC}}$ from (9), estimating the
  numerical variance of `mean` from this finite oracle call;
- `n_eff` is the spot-level effective sample size; and
- padded functionals have `functional_mask=False` and return finite zeros with
  `functional_valid=False`.

`n_samples` in this method is always $n_{\mathrm{oracle}}$.  Candidate $n$
from (12) is never passed to this method; it enters only through the $1/n$
factor in (11) and the cost-proxy benchmark.

The API must satisfy all of the following:

1. It uses exactly the same collision-conditioned SIS target, logical worlds,
   normalized weights, public runouts, and stratified payoff settlement as
   `evaluator.complete`; this is a new reduction of the existing completion
   stream, not a second posterior sampler.
2. All functionals in one leaf row are reduced over exactly the same generated
   worlds and normalized weights.
3. A row is a pure function of its inputs, seed, sample count, and evaluator
   build.  Results are invariant to neighboring rows, batch ordering, and leaf
   microbatching.
4. Coefficients do not alter random-number consumption.  Repeating a row with
   identical completion inputs, seed, and sample count but a different bounded
   functional chunk reuses the same logical worlds.
5. CPU and CUDA implement the same statistical contract.  Cross-device
   bitwise equality is not required, but each backend must be deterministic
   under its documented RNG version.
6. The operation returns only reduced moments.  It must not expose or retain
   the complete sampled-world tensor after return.
7. `F` is runtime-bounded and processed without allocating a covariance matrix
   or any tensor quadratic in `F`.
8. Invalid or degenerate weighted rows fail explicitly; low effective sample
   size is reported rather than silently accepted.

The minimal evaluator change is therefore a public method/result type plus
CPU and CUDA accumulators for, per functional, weighted sums of $f$ and $f^2$
and squared-weight sums of the centered $f$.  It need not construct or return
`p_hat`, `v_hat`, per-hand uncertainty, sampled profiles, or an $F\times F$
covariance matrix.  The evaluator contract/build identity must change in the
ordinary versioned way so the PBG manifest can reject an older build.

The sibling API must include CPU reference tests, CUDA agreement tests,
row-order/microbatch invariance tests, and a hand-checkable test comparing all
three returned moments with direct weighted scalar calculations.

Public fold terminals have fixed payoff but still have posterior hand-sampling
noise.  The PBG adapter must apply the same functional contract to the sibling
evaluator's seeded joint-profile samples with uniform weights.  Fold and
showdown leaves must produce the same `LinearMomentResult` abstraction to the
experiment runner.

### PBG implementation to add

Implement one experiment module and one CLI; do not add an optimizer mode or
modify the production learner's estimator.  The implementation must provide:

1. A rollout-only helper factored from the existing sequential-RB rollout path.
   It returns the public tree, one leaf and sealed leaf reach per trajectory,
   chronological decision-to-trajectory addresses, realized events, owners,
   frozen old values/log probabilities, and compact decision model inputs.  It
   performs no completion, target construction, sealing, or parameter update.
2. A probe-coefficient builder.  For bounded groups of decision rows, it runs
   exact parameter JVPs at $\theta_k$ and produces the coefficients defined
   below.  A test must compare each JVP projection with the dot product of one
   ordinary backward gradient and the same probe.
3. A streamed moment pass over bounded groups of complete trajectories.  It
   builds node and trajectory-sum functionals, calls the new evaluator API,
   updates block-level sufficient statistics, and immediately releases dense
   $H$-wide coefficients and moment outputs.
4. A pure aggregation/analysis layer that reconstructs the variance cube,
   performs oracle-noise corrections and clustered resampling, joins timing and
   memory measurements, and writes tables and plots without requiring CUDA.
5. A CLI runnable under either ordinary Python for one worker or `torchrun` for
   independent multi-GPU workers.  Rank-local result files are immutable and
   rank zero performs the deterministic final merge only after every assigned
   job succeeds.

The intended module names are
`pbg.eval.rao_blackwell_variance` and
`pbg.cli.rao_blackwell_variance`.  The scientific YAML must contain the
checkpoint list, candidate grid, $n_{\mathrm{oracle}}$, probe count/seeds,
superbatch count, trajectory-chunk size, evaluator functional bound, precision,
timing repetitions, memory abort threshold, and output directory.  CLI
overrides are limited to smoke-test bounds, worker/device selection, and a
resume-safe job filter; overrides must be recorded in the manifest.

The complete minimal PBG file boundary is:

```text
pbg/eval/rao_blackwell_variance.py
pbg/cli/rao_blackwell_variance.py
configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml
tests/test_rao_blackwell_variance.py
tests/test_rao_blackwell_variance_cli.py
```

Existing rollout, logical-RNG, checkpoint-loading, poker completion-input,
loss, and model-family batching code must be reused.  A small extraction from
the sequential-RB iteration builder is allowed to expose the rollout-only
object in item 1; it must leave the training behavior unchanged and be covered
by existing training tests.  No other production estimator, learner, storage,
or optimizer code is required for the experiment.

Create the initial YAML with these fixed scientific defaults; field names may
follow the repository's configuration conventions, but the values and meaning
must not change silently:

```yaml
checkpoint_dir: runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_optimized
checkpoints: [37, 296, 555, 814, 1110]
logical_seed: 20260720
candidates:
  N: [384, 768, 1536, 3072]
  B: [32, 64, 128, 256]
  n: [128, 256, 512, 1024]
probes:
  count: 8
  distribution: rademacher
public_sampling:
  superbatches: 8
  maximum_superbatches: 12
oracle:
  initial_samples: 8192
  maximum_samples: 131072
  growth_factor: 2
  independent_seeds: 8
  calibration_blocks: 2
  trajectories_per_calibration_block: 384
  observed_to_predicted_variance_ratio: [0.5, 2.0]
  component_relative_change: 0.05
  component_negligible_total_fraction: 0.01
streaming:
  trajectories_per_chunk: 16
  probes_per_chunk: 1
  maximum_functionals_per_leaf: 40
smoke:
  trajectories: 8
  probes: 2
  oracle_samples: 256
timing:
  warm_repetitions: 5
  measured_repetitions: 20
  additive_relative_tolerance: 0.10
constraints:
  maximum_allocated_gib: 8.5
  maximum_seconds: null
validation:
  initial_replicates_per_checkpoint: 64
  maximum_replicates_per_checkpoint: 256
  predicted_relative_agreement_tolerance: 0.10
analysis:
  confidence_level: 0.95
numerics:
  jvp_backward_rtol: 0.0005
  jvp_backward_atol: 0.00001
  functional_sum_rtol: 0.0001
  functional_sum_atol: 0.00001
  deterministic_rerun_rtol: 0.0
  deterministic_rerun_atol: 0.0
  allocated_cleanup_tolerance_mib: 16
out_dir: runs/rao_blackwell_variance_experiment
```

`maximum_seconds: null` means that the first analysis reports the nondominated
frontier rather than inventing a time budget.  Before any result is inspected,
the user may set this to the desired per-estimator wall-time limit.  The
`numerics` values are implementation-validation tolerances, not adjustable
scientific selection criteria.  All purpose-specific seeds are derived from
`logical_seed` and stable semantic addresses; no mutable global RNG state or
worker rank is part of a scientific address.

### From model JVPs to evaluator functionals

For public trajectory $i$, let $D_i$ be its number of player decisions, so
$\ell=N^{-1}\sum_iD_i$.
For a decision $\nu=(\sigma,a)$ owned by $p$ and a fixed Rademacher parameter
probe $z_r$, compute

$$
D_{\nu,r}(h)
:=
z_r^\top
\left[
\nabla_\theta\log\pi^\theta(a\mid h,\sigma)
+2c_V\nabla_\theta V^\theta(h,\sigma)
\right]_{\theta=\theta_k}.
\tag{14}
$$

Let

$$
P_{i,p}(h)
=\mathbb P^\pi(X_p=h\mid\Tau_i^{\mathsf{pre-c}}),
\qquad
M_{i,p}(h)
=\mathbb E^\pi\!\left[
R_p(X,\Tau)\mathbf1_{\{X_p=h\}}
\mid\Tau_i^{\mathsf{pre-c}}
\right].
\tag{15}
$$

The stable target primitive is $M=P\,Q$.  Do not form $Q=M/P$ merely to
multiply it by $P$ again.  The projected conditional node gradient is

$$
\mu_{i,\nu,r}
:=
\mathbb E^\pi[z_r^\top\Phi_k(X,\Tau;\nu)
\mid\Tau_i^{\mathsf{pre-c}}]
=
\sum_h
\left[
c^P_{i,\nu,r,p}(h)P_{i,p}(h)
+c^M_{i,\nu,r,p}(h)M_{i,p}(h)
\right],
\tag{16}
$$

where the only nonzero player row is the owner row and

$$
c^P_{i,\nu,r,p}(h)=V^{\theta_k}(h,\sigma)D_{\nu,r}(h),
\qquad
c^M_{i,\nu,r,p}(h)=-D_{\nu,r}(h).
\tag{17}
$$

For a compact preflop or other tied-hand model family, compute the JVP in that
family's native output coordinates and expand it through the existing
combo-to-family mapping into canonical combo-ID order before constructing
(17).  Stream one probe at a time: regenerate its Rademacher parameter tensors
from its logical seed, use it for every decision in the chunk, then release it.
Never keep eight full parameter-space probes resident simultaneously.

These coefficients are known after the public rollout and model JVP, before
any minibatch is drawn.  A minibatch only selects these fixed node
functionals, possibly with repeated integer multiplicities.

For every trajectory and probe, send the evaluator:

- one functional with coefficients (17) for every node on that trajectory;
- one additional trajectory-sum functional whose player/hand coefficients are
  the sums of (17) over every node on the trajectory owned by that player.

The evaluator therefore returns, for every node,

$$
\widehat\mu_{i,\nu,r},\qquad
\widehat v_{i,\nu,r},\qquad
\widehat e_{i,\nu,r},
\tag{18}
$$

and for the trajectory-sum functional,

$$
\widehat\mu^{\mathrm{traj}}_{i,r},\qquad
\widehat v^{\mathrm{traj}}_{i,r},\qquad
\widehat e^{\mathrm{traj}}_{i,r}.
\tag{19}
$$

Here $\widehat\mu$ is `mean`, $\widehat v$ is `world_variance`, and
$\widehat e$ is `mean_variance`.  The trajectory-sum variance contains every
covariance between node contributions sharing one completed world.  No
pairwise node covariance is stored.

### Online sufficient statistics and variance reconstruction

Assign every trajectory to its predetermined disjoint block at every candidate
$N$.  For one scalar probe $r$, form the following compact per-trajectory
quantities before releasing a streamed trajectory:

$$
\begin{aligned}
\widehat G_i&:=\sum_\nu\widehat\mu_{i,\nu,r},
&
\widehat Q_i&:=\sum_\nu\widehat\mu_{i,\nu,r}^2,\\
\widehat U_i&:=\sum_\nu\widehat v_{i,\nu,r},
&
\widehat V_i&:=\widehat v^{\mathrm{traj}}_{i,r},\\
\widehat E_i&:=\sum_\nu\widehat e_{i,\nu,r},
&
\widehat H_i&:=\widehat e^{\mathrm{traj}}_{i,r}.
\end{aligned}
$$

The returned trajectory-sum mean must agree numerically with $\widehat G_i$;
its variance outputs $\widehat V_i$ and $\widehat H_i$ retain all
within-trajectory covariances.  Update only these mergeable block scalars:

$$
\begin{aligned}
L&=\sum_iD_i,
&S_G&=\sum_i\widehat G_i,
&S_Q&=\sum_i\widehat Q_i,
&S_{G^2}&=\sum_i\widehat G_i^2,\\
S_{DQ}&=\sum_iD_i\widehat Q_i,
&S_U&=\sum_i\widehat U_i,
&S_{DU}&=\sum_iD_i\widehat U_i,
&S_V&=\sum_i\widehat V_i,\\
S_E&=\sum_i\widehat E_i,
&S_{DE}&=\sum_iD_i\widehat E_i,
&S_H&=\sum_i\widehat H_i.
\end{aligned}
\tag{20}
$$

Also retain a mergeable centered-moment summary of the $\widehat G_i$ values
for the rollout-variance estimate.  Dense $H$-wide coefficients and per-call
outputs are released immediately after (20) is updated.

For one block and probe, estimate the two minibatch coefficients separately:

$$
\begin{aligned}
\widehat A_{\mathrm{MB},r}^{=,\mathrm{raw}}
&=\frac{S_{DQ}-S_{G^2}}N,\\
\widehat A_{\mathrm{MB},r}^{\ne,\mathrm{raw}}
&=\frac{L S_Q-S_{DQ}-S_G^2+S_{G^2}}{N(N-1)}.
\end{aligned}
\tag{21}
$$

Finite $n_{\mathrm{oracle}}$ makes the estimated node means noisy.  If those
row-local mean errors are conditionally unbiased, their exact variance-noise
corrections are

$$
\begin{aligned}
\widehat\Delta_{\mathrm{MB},r}^{=}
&=\frac{S_{DE}-S_H}{N},\\
\widehat\Delta_{\mathrm{MB},r}^{\ne}
&=\frac{L S_E-S_{DE}}{N(N-1)}.
\end{aligned}
\tag{22}
$$

The $S_H$ term retains within-trajectory covariance for the same-rollout
coefficient.  Cross-trajectory oracle errors have zero covariance because
their completion calls use independent logical seeds, which is why no $S_H$
term occurs in the distinct-rollout correction.  For finite self-normalized
importance sampling, (22) is a consistent plug-in correction rather than a
finite-sample unbiasedness identity: residual ratio bias is controlled by the
independent-seed and budget-doubling gates.

Use

$$
\widehat A_{\mathrm{MB},r}^{\square}
=\widehat A_{\mathrm{MB},r}^{\square,\mathrm{raw}}
-\widehat\Delta_{\mathrm{MB},r}^{\square},
\qquad \square\in\{\ne,=\}.
\tag{23}
$$

Do not clamp a negative corrected result.  Treat it as a failed integration
precision gate and increase $n_{\mathrm{oracle}}$ or the number of independent
oracle seeds.

The three completion coefficients are the ordinary mean and order-two
U-statistic estimates

$$
\begin{aligned}
\widehat C_{\mathrm{MC},r}^{\mathrm{pair}}
&=\frac{S_V}{N},\\
\widehat C_{\mathrm{MC},r}^{=}
&=\frac{S_{DU}}{N},\\
\widehat C_{\mathrm{MC},r}^{\ne}
&=\frac{L S_U-S_{DU}}{N(N-1)}.
\end{aligned}
\tag{24}
$$

Substitution into (7b) recovers the direct corrected block formula
$(L S_Q-S_G^2-L S_E+S_H)/(BN^2)$.  Substitution into (8b) recovers
$L S_U/(BN^2)+(1-1/B)S_V/N^2$.  These identities are implementation
checks of both the collision split and the $\ell$ normalization.

Estimate $T_{\mathrm{rollout},r}(1)$ as the sample variance across independent
trajectory means $\widehat G_i$, then subtract the corresponding average
$\widehat H_i$.  Again, do not clamp
a negative corrected estimate.  For the reported checkpoint estimate, pool
the mergeable centered moments from every admitted trajectory and superbatch;
use block-specific estimates only as the diagnostic check of the exact $1/N$
law.

Average the corrected scalar estimates over probes.  Hutchinson uncertainty is
the uncertainty across probe seeds; public-sampling uncertainty is computed
separately across independent superbatches.  Finally evaluate (11) for every
candidate $(N,B,n)$.  No actual minibatches are drawn during this reconstruction.

### Streamed execution of one superbatch

The unit of GPU work is one `(checkpoint, superbatch_id)` job.  A worker must
complete the following steps in order:

1. Load and validate the frozen checkpoint on its assigned GPU.  Put the model
   in evaluation mode and disable every parameter update.
2. Generate exactly $N_{\max}=3072$ fresh iid public pre-terminal trajectories
   with the logical rollout seed derived from checkpoint and superbatch ID.
3. Construct disjoint consecutive blocks for every candidate $N$: eight blocks
   of 384, four of 768, two of 1536, and one of 3072.  Record block membership
   before any trajectory chunking so execution geometry cannot alter the
   statistical grouping.
4. Process complete trajectories in bounded chunks.  A chunk may contain
   multiple model execution families, but no trajectory may be split across
   different completion worlds.
5. Within a trajectory chunk, gather its decision rows by model family.  For
   each fixed probe, run JVPs, restore chronological trajectory order, and
   construct the node coefficients (17) plus one trajectory-sum functional.
6. Pad only to the maximum functional count in that leaf microbatch, set
   `functional_mask`, and call `complete_linear_moments` with
   $n_{\mathrm{oracle}}$.  All node and trajectory-sum functionals for a leaf
   must see the same worlds.  Reuse that leaf's completion seed when the next
   probe is streamed so probes also integrate over the same logical worlds.
7. Update (20) for every candidate block containing those trajectories.  Move
   only compact scalar summaries required for persistence to CPU.  Release the
   JVP workspace, coefficient tensors, evaluator result, and any completed
   worlds before advancing to the next chunk.
8. Write one atomic rank-local result for the completed job.  A partially
   written job is never considered resumable; rerunning the job with the same
   manifest must reproduce its result within the configured numerical
   tolerance.

No minibatch is sampled in this pass.  The reason is exact, not heuristic:
the multinomial second moments in (7b) and (8b) integrate over all minibatches using
only individual-node variances and the variance of the all-node trajectory
sum.  Saving or evaluating 128 separate minibatch functionals would duplicate
work and increase Monte Carlo error.

The implementation must choose trajectory and leaf microbatch sizes from an
explicit memory bound.  For orientation, one probe, 16 trajectories, at most
40 functionals per HU 32 bb trajectory, two players, and two fp32 coefficient
arrays occupies about 13 MiB.  This is only an initial sizing estimate; the
smoke test must measure the actual full-pipeline peak.

### Oracle-accuracy calibration

Before the full run, calibrate $n_{\mathrm{oracle}}$ at the middle checkpoint
on the predetermined 384-trajectory block from each of two independently
rolled public superbatches, using the configured eight independent oracle
seeds on each fixed leaf set.  Compare successive budgets
$8192,16384,\ldots,131072$ subject to the memory gate.

For each candidate oracle budget, report:

- the distribution of spot-level effective sample size and
  $n_{\mathrm{eff}}/n_{\mathrm{oracle}}$;
- paired differences in aggregate node means, node world variances, and
  trajectory-sum world variances between independent oracle seeds;
- the predicted numerical integration contribution obtained from
  `mean_variance`;
- the observed across-seed variance of the returned means; and
- the corrected rollout and minibatch components before any clamping.

Accept the smallest oracle budget for which all of the following hold:

1. no invalid or degenerate functional is returned;
2. the evaluator-reported mean-variance prediction agrees with observed
   across-seed variance within a predeclared tolerance;
3. doubling the oracle budget changes every aggregate variance component by
   less than a predeclared relative tolerance or less than a predeclared
   negligible fraction of total variance; and
4. corrected rollout and minibatch estimates are nonnegative within their
   public-sampling uncertainty intervals.

The tolerances must be present in the YAML before calibration.  If no tested
budget passes, stop and report the effective-sample-size and component that
failed; do not proceed by treating a large nominal budget as exact.

After calibration, repeat the independent-seed agreement check on a smaller
subset at the early and terminal checkpoints.  The accepted
$n_{\mathrm{oracle}}$ is fixed for the full variance run unless one of those
checks fails.

### Public-rollout design

The staged scientific run is:

1. **CPU oracle:** enumerate a tiny finite distribution of public
   trajectories, conditional worlds, and minibatches.  Verify (7)--(8),
   (11), and (20)--(24), including shared-completion covariance and both oracle
   noise corrections.
2. **CUDA smoke:** use the middle checkpoint, one GPU, one superbatch of eight
   trajectories, two probes, a small oracle budget, one trajectory chunk, and
   no timing sweep.  Verify API shapes, JVP/backward agreement, deterministic
   rerun, finite output, complete cleanup, and output-schema validation.
3. **Middle-checkpoint pilot:** run four independent superbatches after oracle
   calibration.  Verify the $1/N$, $1/B$, $1/n$, and affine-in-$1/B$
   identities and inspect probe uncertainty.
4. **Full variance run:** run eight independent superbatches at each of the
   five checkpoints.  Add superbatches in pairs up to twelve if finalist
   confidence intervals overlap materially.
5. **Finalist validation and performance calibration:** run only after the
   variance surface passes all identities.

Treat the superbatch as the cluster for paired resampling across geometries,
but retain its disjoint $N$-blocks as observations inside that cluster.  The
eight-superbatch design is a pilot-scale minimum: if its clustered intervals
cannot distinguish the leading candidates, report that uncertainty or add
superbatches rather than declaring a winner.

### Independent three-GPU scheduling

The three GPUs accelerate the experiment by executing independent jobs.  They
do not jointly form one gradient estimator and perform no gradient all-reduce.
Under `torchrun --nproc-per-node=3`, enumerate jobs in the canonical order
`(checkpoint_index, superbatch_id)` and assign

$$
\operatorname{worker}(j)=j\bmod3.
\tag{25}
$$

Every worker binds only `cuda:LOCAL_RANK`.  Scientific seeds are functions of
checkpoint identity, superbatch ID, trajectory ID, probe ID, and purpose; they
must not depend on worker rank.  Consequently, running the same manifest with
one or three workers produces the same set of job results.

Workers write separate job files and never append concurrently to one file.
After a barrier, rank zero validates the expected job inventory, hashes every
job result, and performs a deterministic merge.  Resume skips only job files
whose manifest fingerprint, checkpoint hash, schema version, and success
marker all match.

### Timing calibration

The variance surface is the theoretical iid-completion proxy, while the time
and memory surfaces measure the existing computation required to request $n$
completion proposals and form one unclipped PG-plus-MSE gradient.  This is an
explicit proxy comparison: it does not assert that finite-$n$ SNIS has exactly
the variance $T_{\mathrm{montecarlo}}/n$.

Wall time is measured for one setup-inclusive estimator without any parameter
update. Benchmark on one explicit idle GPU at the middle checkpoint:

$$
\begin{aligned}
t_{\mathrm{rollout}}(N),\qquad
t_{\mathrm{completion}}(N,n),\qquad
t_{\mathrm{targets}}(N,n),\qquad
t_{\mathrm{gradient}}(B).
\end{aligned}
$$

`gradient` means: draw one iid minibatch of $B$ decisions from the unstratified
empirical decision law, execute its naturally induced mix of model families,
evaluate the exact experiment loss at $\theta_k$, run backward once, and then
discard the gradient.  It performs no optimizer/scaler transition, clipping,
or distributed synchronization.  Because family composition affects cost,
retain the realized family counts for every repetition.

Use five warm repetitions followed by twenty timed repetitions at every
candidate value. Benchmark one axis at a time around the source geometry, then
validate the additive prediction

$$
t(N,B,n)
=
t_{\mathrm{rollout}}(N)
+t_{\mathrm{completion}}(N,n)
+t_{\mathrm{targets}}(N,n)
+t_{\mathrm{gradient}}(B)
\tag{26}
$$

on the candidate triples nearest the predicted optimum.  If the prediction
misses a directly measured triple by more than the predeclared timing tolerance,
use direct triple measurements for selection and do not extrapolate the
additive model into the failed region.

Checkpoint loading, directional probes, the large-$n_{\mathrm{oracle}}$
moment pass, statistical analysis, and diagnostic persistence are experiment
overhead and are excluded from $t(N,B,n)$.  Rollout, ordinary completion at
the candidate $n$, lambda-one target construction, one gradient evaluation,
and required phase-boundary synchronization are included.

The source-run telemetry supplies only initial anchors:

| Production phase per GPU | Source mean | Source p90 |
|---|---:|---:|
| public rollout at $N=3072$ | 44.58 s | 50.17 s |
| completion at $(N,n)=(3072,1024)$ | 1.086 s | 1.107 s |
| target construction at $(N,n)=(3072,1024)$ | 7.685 s | 8.373 s |
| one gradient evaluation at $B=128$ | 0.791 s | 0.848 s |

The last row is the source optimization-phase total divided by its 116
minibatches and must be replaced by the direct benchmark before selection.

### Memory calibration and safety gates

Measure peak allocated and reserved memory over the complete production
pipeline for every one-axis timing benchmark and every finalist triple.
Use

$$
m(N,B,n)
=
\max_{\textnormal{production phases}}
m_{\mathrm{GPU}}(N,B,n)
\tag{27}
$$

as the hard memory response. Do not add phase peaks, but do retain allocator
state between phases so that persistent allocations are represented.

Source telemetry provides the initial bounds:

| Per GPU | Observed |
|---|---:|
| PBG hard ceiling | 9.986 GiB |
| independent hardware guard | 0.581 GiB |
| maximum source iteration allocated | 4.920 GiB |
| maximum source completion allocated | 2.264 GiB |
| maximum source optimization allocated | 4.260 GiB |
| terminal pre-optimization active / reusable | 2.043 / 5.266 GiB |

Before CUDA work, check GPU state and select only an idle device. Stage every
new axis value from the source geometry outward. Abort above 8.5 GiB
allocated, on a nonfinite result, or if memory grows across repetitions.
Clean up and verify idle GPU state after each benchmark family.

Diagnostic probe functionals are generated lazily in bounded chunks. A new
chunk must reuse the same completed worlds and importance weights. Release
directional-pass workspaces before completion evaluation, and never
materialize full dense coefficient tensors, full gradients, completion
worlds, or covariance matrices.

Measure the diagnostic moment pass separately at the accepted
$n_{\mathrm{oracle}}$.  Its time is scientific overhead and does not enter
$t(N,B,n)$, but its allocated/reserved peak must obey the same 8.5 GiB abort
threshold.  Record the selected trajectory chunk, leaf microbatch, functional
bound, and probe packing that produced that peak.

### Constrained selection

Let $t_{\max}$ and $m_{\max}$ be specified before examining the variance
results. For every triple in the safely benchmarked candidate set, compute
(11) and retain it only if

$$
t(N,B,n)\leq t_{\max},
\qquad
m(N,B,n)\leq m_{\max}.
\tag{28}
$$

At checkpoint $k$, select

$$
(N_k^\star,B_k^\star,n_k^\star)
=
\operatorname*{arg\,min}_{(N,B,n)\ \textnormal{satisfying (28)}}
\mathcal V_k(N,B,n).
\tag{29}
$$

Report checkpoint-specific optima, the equal-weight mean variance across the
five checkpoints, and the worst-checkpoint variance. If a single geometry is
required for the whole run, choose the feasible triple minimizing the
predeclared aggregate—equal-weight mean or worst checkpoint. Do not choose
the aggregation after observing which one favors a particular geometry.

If $t_{\max}$ or $m_{\max}$ has not been fixed, report the nondominated
variance--time--memory frontier rather than naming one optimum.

### Direct validation

For the predicted optimum, its nearest lower-cost neighbor, and every
statistically indistinguishable competitor:

1. Generate independent public batches at their actual $N$.
2. Draw one actual iid minibatch of size $B$ from its empirical decision law
   and record the multiplicity of every selected node.
3. Group the selected nodes by trajectory.  For each unique selected leaf and
   probe, form one combined functional by summing the selected node
   coefficients with their minibatch multiplicities.
4. Call `complete_linear_moments` only on those unique leaves, with an oracle
   seed independent of the variance-surface pass.  The shared combined leaf
   functional exactly retains covariance among every selected node using that
   completion.
5. Form the projected conditional minibatch mean

   $$
   \widehat m_r
   =\frac{L}{NB}\sum_{i\ \mathrm{selected}}
   \widehat\mu^{\mathrm{combined}}_{i,r},
   $$

   and its theoretical conditional completion variance at candidate $n$,

   $$
   \widehat c_r
   =\left(\frac{L}{NB}\right)^2\frac1n
   \sum_{i\ \mathrm{selected}}
   \widehat v^{\mathrm{combined}}_{i,r}.
   $$

   Also retain the estimated numerical variance of the oracle conditional
   mean,

   $$
   \widehat e_r
   =\left(\frac{L}{NB}\right)^2
   \sum_{i\ \mathrm{selected}}
   \widehat e^{\mathrm{combined}}_{i,r}.
   $$

6. Across independent public-batch/minibatch replicates, estimate the direct
   total projected variance as
   $\operatorname{Var}(\widehat m_r)-\mathbb E(\widehat e_r)
   +\mathbb E(\widehat c_r)$, then average over probes and compare it with
   (11).  Do not clamp the oracle-noise-corrected term; a materially negative
   result fails the oracle-accuracy gate.

Start with 64 replicates per checkpoint and finalist, paired across finalists
by public/minibatch seed.  Add replicates in blocks of 64, up to 256, while the
direct-minus-predicted 95% interval is too wide to distinguish agreement from
the predeclared 10% relative discrepancy.  Report the interval and stop with
an inconclusive result at the cap; do not convert lack of precision into a
pass.  An interval wholly inside $\pm10\%$ passes; one wholly outside fails;
an interval crossing either boundary is inconclusive and triggers the next
replicate block when available.

This is the only phase that calls the functional evaluator after drawing a
minibatch.  It touches only the leaves selected by that minibatch and stores
only one combined functional per unique selected leaf and probe.  The main
surface does not do this because (20)--(24) integrate over all minibatches
analytically with less work and less sampling error.

This validation concerns one gradient estimator only. It performs no
optimizer step and updates no parameter, optimizer state, scaler, or
reference policy.

A small CPU oracle must enumerate public trajectories, private completions,
and all minibatches in a tiny finite distribution to verify (7)--(8), (11),
and (19)--(24), including repeated node draws, shared-completion covariance,
the exact affine dependence on $1/B$, and both finite-oracle noise corrections.

### CLI phases and persisted artifacts

The CLI must expose idempotent phases so a user can smoke-test before launching
the expensive run:

- `smoke`: CPU oracle followed by the bounded one-GPU CUDA smoke;
- `calibrate-oracle`: select and record $n_{\mathrm{oracle}}$;
- `variance`: execute missing independent superbatch jobs;
- `performance`: run timing and memory calibration;
- `validate`: run the minibatch-first finalist validation;
- `analyze`: merge completed job files and regenerate all tables/plots; and
- `all`: run the admitted phases in the order above, stopping at any failed
  gate.

At minimum, persist:

```text
<out_dir>/
  manifest.json
  calibration.json
  jobs/<checkpoint_sha256>/<superbatch_id>.pt
  performance.jsonl
  validation_jobs/<checkpoint_sha256>/<candidate_id>/<replicate_block>.pt
  validation.jsonl
  summary.json
  tables/*.csv
  plots/*
```

`manifest.json` is written before scientific sampling and includes the
predeclared constraints and tolerances.  `calibration.json` fixes the accepted
oracle budget.  Each job file contains only scalar/short-vector sufficient
statistics, probe IDs, block addresses, ESS summaries, timings, memory peaks,
and identity metadata; it contains no model checkpoint, dense hand
coefficients, rollout tree, or completed worlds.

Validation workers follow the same immutable-job rule under
`validation_jobs/`; only rank zero writes the merged `validation.jsonl` after
a successful inventory check.  No JSONL file has concurrent writers.

The command-line contract is:

```text
python -m pbg.cli.rao_blackwell_variance \
  --config CONFIG \
  --phase {smoke,calibrate-oracle,variance,performance,validate,analyze,all} \
  [--device DEVICE] [--resume]
```

`--device` is valid only for an ordinary single-worker invocation.  Under
`torchrun`, each process must ignore that option and bind to
`cuda:${LOCAL_RANK}`.  `--resume` may skip only manifest-compatible successful
artifacts as specified above.  A phase exits nonzero on a failed scientific or
memory gate; it never writes a success marker and continues.

The smoke command after implementation is:

```bash
./tools/poker-python -m pbg.cli.rao_blackwell_variance \
  --config configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml \
  --phase smoke --device cuda:0
```

After checking `nvidia-smi` and replacing the visible-device list with three
idle physical GPU IDs, the resumable production command has this form:

```bash
CUDA_VISIBLE_DEVICES=GPU0,GPU1,GPU2 \
  /home/pablob/.venvs/poker/bin/torchrun --standalone --nproc-per-node=3 \
  -m pbg.cli.rao_blackwell_variance \
  --config configs/experiments/nlhe_hu_32bb_rao_blackwell_variance.yaml \
  --phase all --resume
```

In `all`, CPU oracle, CUDA smoke, oracle calibration, performance measurement,
and final analysis are rank-zero-only phases with barriers.  The variance jobs
and independent finalist-validation replicates use all three workers under
(25).  Thus the command does not duplicate a scientific job across GPUs and
does not turn the experiment into DDP.

The smoke phase passes only if the CPU oracle identities pass, the CUDA JVP
agrees with backward projection, node means sum to the returned trajectory-sum
mean within the configured fp32 tolerance, two identical logical runs agree,
one-worker and simulated three-worker job assignment produce the same job
inventory, every output validates against its schema, and CUDA memory returns
to the pre-smoke baseline within the allocator tolerance.

### Statistical analysis and required outputs

Use a clustered bootstrap over independent public superbatches, paired across
candidate geometries constructed from the same superbatch. Report
Hutchinson-probe uncertainty separately. Checkpoints are stages of one
training trajectory, not independent training runs.

Required outputs are:

- per-checkpoint estimates and intervals for $T_{\mathrm{rollout}}(1)$,
  $A_{\mathrm{MB}}^{\ne}$, $A_{\mathrm{MB}}^{=}$,
  $C_{\mathrm{MC}}^{\ne}$, $C_{\mathrm{MC}}^{=}$, and
  $C_{\mathrm{MC}}^{\mathrm{pair}}$;
- direct blockwise checks of the exact $1/B+1/(BN)$ minibatch surface and
  $1/B+1/N+1/(BN)$ completion surface;
- oracle-budget calibration tables, ESS distributions, predicted-versus-
  observed oracle mean variance, and raw/corrected component estimates;
- tables of $T_{\mathrm{rollout}}$, $T_{\mathrm{minibatch}}$, and
  $T_{\mathrm{montecarlo}}$ over the candidate cube;
- total theoretical proxy variance $\mathcal V(N,B,n)$;
- measured time and memory response tables;
- the feasible candidate set and selected optimum under each predeclared
  constraint;
- selection frequencies under the clustered bootstrap;
- direct-versus-predicted variance for the finalist triples; and
- plots of variance against each axis, component fractions, probe uncertainty,
  oracle convergence, and the variance--time--memory Pareto frontier.

## Scope

The experiment measures only the variance of one scaled
PG-plus-MSE-value-gradient estimator at $\theta=\theta_k$. It studies the
dependence on public-rollout count $N$, minibatch size $B$, and hypothetical
iid conditional-completion count $n$.  A separately calibrated large
$n_{\mathrm{oracle}}$ SIS call estimates the exact conditional moments.  The
time and memory surface measures the existing implementation at candidate
proposal count $n$ and is reported explicitly as a cost proxy, not as a claim
that finite-$n$ SNIS has the exact iid variance law.

It does not measure stochastic-gradient-descent error, accumulated parameter
error, optimizer dynamics, policy improvement, exploitability, or
seed-to-seed training variation.
