# Ideal Data Storage

Notations:
- $P = |\mathcal{P}|$: number of players
- $X = 1326$ in poker is a private state size upper bound $\max_p |\mathcal{X}_p$
- $N$ is the number of sampled nodes
- $N_spot$ is the number of sampled **player** nodes
- We write branching (resp. sequential) to explain how the two differs.

## Storage per GPU
- BRANCH ONLY: A tree for ALL nodes: matches the PBG tree definition, but truncated to preterminal. Each node points to its parent and all its children. Size is $[N, \max(children_count) + 1]$ if stored densely, and $O(N)$ if stored in a CSR way.
- SEQUENTIAL ONLY: A list of variable-length pointer-based lists (the trajectories). This is ALWAYS stored in a csr way so has size $O(N)$. It gets **discarded** after we're done computing estimators with backward computations.

- A list of all IDs of player-owned nodes $\sigma$ (and only them!) for $o(\sigma) \in \mathcal{P}$. Size is $N_spot \leq N$.
- BRANCHING ONLY: player node weights. Size is $N_spot$.
- A list of all IDs of children of a **sampled** player-node direct children $\sigma a$ for $o(\sigma) \in \mathcal{P}$. Size is $N_actions \leq N$.

Other game data such as: who owns what node, public game history, is it preterminal,... should probably also be stored. Most likely we ONLY needs these for player-owned spots so it adds some $[N_spot, info]$ dataset with $info$ a fixed upper bound.

**Input artifacts**: the model may need some special inputs (reaches and folded stats for us) which are stored per ``queue partition'': for us,
- $[N_spot_preflop, P, 169]$,
- $[N_spot_postflop_A, P, 1326] + [N_spot_postflop_A, folded_stat]$ for each number $2 \leq A \leq P$ of live players with $folded_stat$ the dimension of folded stats (should be small, like only several hundred numbers because only top eigenvalues of $\beta$ are used),
with $N_spot = N_spot_preflop + \sum_A N_spot_postflop_A$.

Note: reach doesn't sound like such a big dataset in prctice, we should completely abandon the "recompute" idea. Instead we should just store this as we go.

**Estimators and Loss terms**: the loss needs some estimators which are stored as follows:
- $\widehat{A}$, $\widehat{P}$, $\log(\pi)$ (and $\widehat{Q}$ if used) are stored as $[N_actions, X]$ (resp. in sequential mode it is $[N_spot, X]$ and we put these numbers at the entry corresponding to the node $\sigma$ itself, not to some $\sigma a$, as each node as a single sampled decision)
- $\widehat{V}$ and $V$ are stored as $[N_spot, X]$.
Note that this means that, when gathering data for a minibatch, one needs to:
- sample player nodes $\sigma$,
- BRANCH ONLY:  look at the tree to find the IDs of the $\sigma a$ for all actions $a$ for $\sigma$ in the sampled rollout tree, then look at $\widehat{A}$, $\widehat{P}$ (and $\widehat{Q}$ if used) at the $\sigma a$ entry. 
- SEQUENTIAL ONLY: just get $\widehat{A}$, $\widehat{P}$ (and $\widehat{Q}$ if used) directly from $\sigma$'s entry in the table
- separate in ``queue partition'' microbatches and compute gradients, accumulate smartly so not to have to store one gradient per partition, sync with other GPU/ranks, update $\theta$.


