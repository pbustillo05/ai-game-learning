# Advantage-prioritized minibatching in branched mode

## Recommended rule

After the rollout tree and its $\widehat A$ targets have been computed, but before evaluating any gradient, sample player nodes approximately as

$$
q(\mathsf s\mid\mathfrak S)\propto w_{\mathfrak S}(\mathsf s)\,\mathbb E(|\widehat A|\mid\mathsf s,\mathfrak S),
\qquad
w_{\mathfrak S}(\mathsf s)=\prod_{\mathsf r\prec\mathsf s}\frac1{N(\mathsf r)}.
$$

The path mass $w_{\mathfrak S}$ preserves the ordinary branched decision law. The advantage factor samples later streets more often only when their realized advantages are larger; no hand-written street multiplier is needed.
All node sums and normalizations below are restricted to player nodes.

## Why magnitude, and why no square root?

At $\theta=\theta_k$, let

$$
z_k(x_p,\sigma,a)=\nabla_\theta\log\pi^\theta(a\mid x_p,\sigma),
\qquad
\Psi_A(\mathfrak S;\mathsf s)=-\mathbb E[\widehat A z_k\mid\mathsf s,\mathfrak S].
$$

For an unbiased iid node proposal $q$, its proposal-dependent conditional trace variance is

$$
\frac1B\sum_{\mathsf s}
\frac{w_{\mathfrak S}(\mathsf s)^2\lVert\Psi_A(\mathfrak S;\mathsf s)\rVert_2^2}
{q(\mathsf s\mid\mathfrak S)}.
$$

Cauchy--Schwarz shows that the exact minimizer is

$$
q^*(\mathsf s\mid\mathfrak S)\propto
w_{\mathfrak S}(\mathsf s)\lVert\Psi_A(\mathfrak S;\mathsf s)\rVert_2.
$$

Optimal probabilities therefore scale with **gradient magnitude**, not squared gradient magnitude. We do not want to compute or predict that architecture-dependent norm. Instead assume $\lVert z_k\rVert_2\leq L_z$. Then

$$
\lVert\Psi_A(\mathfrak S;\mathsf s)\rVert_2
\leq L_z\,\mathbb E(|\widehat A|\mid\mathsf s,\mathfrak S).
$$

Thus $\mathbb E|\widehat A|$ is an architecture-independent upper-bound proxy available from the target table before backpropagation. There is no square root around it.

A square root is appropriate only when the available statistic is already a **second moment**:

$$
\mathbb E|\widehat A|\leq\sqrt{\mathbb E(\widehat A^2)}.
$$

Consequently, the two dimensionally sensible choices are

$$
f(\mathsf s)=\mathbb E|\widehat A|
\quad\text{or}\quad
f(\mathsf s)=\sqrt{\mathbb E(\widehat A^2)}.
$$

The first is the tighter bounded-score proxy. The second is an outlier-sensitive RMS proxy. We should start with $f=\mathbb E|\widehat A|$ computed directly over sampled children and private-state posterior weights.

## Unbiased implementation

Use a defensive mixture with the ordinary conditional node law:

$$
q_\varepsilon(\mathsf s\mid\mathfrak S)
=(1-\varepsilon)\frac{w_{\mathfrak S}(\mathsf s)(f(\mathsf s)+\delta)}
{\sum_{\mathsf r}w_{\mathfrak S}(\mathsf r)(f(\mathsf r)+\delta)}
+\varepsilon\frac{w_{\mathfrak S}(\mathsf s)}{\Lambda(\mathfrak S)},
\qquad
\Lambda(\mathfrak S)=\sum_{\mathsf r}w_{\mathfrak S}(\mathsf r).
$$

Multiply each sampled loss by the frozen importance weight

$$
\omega(\mathsf s,\mathfrak S)
=\frac{w_{\mathfrak S}(\mathsf s)/\Lambda(\mathfrak S)}
{q_\varepsilon(\mathsf s\mid\mathfrak S)}.
$$

This preserves the original conditional gradient exactly and gives $\omega\leq1/\varepsilon$. Do not backpropagate through the proposal or weight, and do not clip the weight unless bias is explicitly accepted. This changes only $T_{\rm minibatch}$, not $T_{\rm rollout}$.

The possible gain is controlled by heterogeneity in node-gradient magnitudes. For the unavailable exact optimum, the conditional improvement over ordinary sampling is

$$
\frac{\Lambda(\mathfrak S)^2}{B}
\operatorname{Var}_{\mathsf S\sim w/\Lambda}
\left(\lVert\Psi_A(\mathfrak S;\mathsf S)\rVert_2\right).
$$

Large late-street advantages therefore create an opportunity; street rarity alone does not.

## Relation to prior work

This is the policy-gradient analogue of [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), which prioritizes absolute TD error and applies importance correction: magnitude, not the square root of magnitude. [LaBER](https://proceedings.mlr.press/v162/lahire22a.html) derives gradient-norm sampling as the ideal replay law and studies cheaper proxies. [Katharopoulos and Fleuret](https://proceedings.mlr.press/v80/katharopoulos18a.html) similarly use loss-derived upper bounds for gradient-norm importance sampling. [PTR-PPO](https://arxiv.org/abs/2112.03798) prioritizes trajectories using GAE or return statistics, although its off-policy replay and truncated weights differ from this conditionally unbiased setting.

In short:

$$
\boxed{
q(\mathsf s\mid\mathfrak S)\propto
w_{\mathfrak S}(\mathsf s)\mathbb E(|\widehat A|\mid\mathsf s,\mathfrak S)
}
$$

with a defensive ordinary-sampling mixture and exact importance weights.
