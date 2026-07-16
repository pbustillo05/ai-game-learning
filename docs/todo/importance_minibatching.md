NOT YET ADDRESSED - TO KEEP

# The Stationary $\Sigma$ distribution

Typically, the streets are:
- 68.6% preflop
- 14.8% flop
- 10.1% turn
- 6.5% river
but the number of nodes in each street is astronomically larger for later streets.
Branching mode helps sampling more of the later street in rollout, helping improve a lot the accuracy of the $\widehat{A}$ and $\widehat{V}$ estimators (which are typically very bad in early streets).
This does not however adress the fact that:
- there's still a lot of variance coming from seeing only ~6.5% river hands in each minibatch
- there's typically more money involved in later streets so higher variance.
We suggest ways to bias minibacth sampling through an importance sampling.

## IS

Say we sample a fixed number of nodes (say 512) proportionally to $f(\sigma)\mathbb{P}^\pi(\sigma)$:
- What is the optimal $f$ to reduce variance?
- Can we see "normalizing $\widehat{A}$" as a proxy for importance sampling? This is typically done but obviously introduces bias.

I think a good proxy is to use $f(\sigma) \propto \sqrt{\mathbf{Var}(\widehat{V} \mid \sigma, N(\sigma) = 1)}$. But maybe this needs a better theoretical study. Then how should one do this experimentally?