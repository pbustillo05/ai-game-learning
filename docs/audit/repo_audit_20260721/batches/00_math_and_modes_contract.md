# Authoritative mathematical and five-mode contract

## Scope and status

This is an implementation-independent baseline extracted from
`docs/theory/core_math.tex`. `docs/theory/MHA_model.md` is used only for the
model output contract that the mathematics directly requires. No production
implementation was used to resolve gaps.

The requested five-mode vocabulary maps onto the authority as follows:

| Requested mode | Authoritative construction | Status in `core_math.tex` |
|---|---|---|
| `BR` | Best-response evaluation | **Not defined**; see ambiguity A-01 |
| `normal` | `SampleTerminal` + `SequentialPG` | Defined by `alg:sample-terminal`, `def:terminal-rollout-decision-law`, `alg:sequential-pg` |
| `sequential_rb` | `SamplePreTerminal` + `PreTerminalMC` + `SequentialRaoBlackwellPG` | Defined by `alg:sample-preterminal`, `def:preterminal-rao-blackwell-decision-law`, `alg:mc-preterminal`, `alg:sequential-rao-blackwell-pg` |
| `branched_rb` fixed beta | `SampleTree` + `BranchedValue` + branched PPO with deterministic fixed intensity | Defined as the fixed regime of `alg:branched-rao-blackwell-pg` |
| `branched_rb` KKT | The same branched pipeline, with fixed-beta warmup followed by the KKT controller | Defined in “Variance-Based Intensities using KKT,” “Queue budget safeguard,” and `alg:branched-rao-blackwell-pg` |

The authority therefore specifies three training estimators, with the branched
one having fixed and adaptive intensity regimes. It does **not** itself assert
that these and BR are the only permitted production modes; that exclusivity is
an audit requirement supplied outside this document.

## Common contract

### Sampling law

- **MATH-LAW-001 — Ideal player-decision law.** A player decision node $Sigma$
  has probability
  $\mathbf 1_{o(\sigma)\in\mathcal P}\mathbb P^\pi(\sigma\preccurlyeq\tau)/
  \mathbb E^\pi|\tau|_{\rm player}$. Conditional private state is the posterior
  given public reach; $P=o(\Sigma)$ and $A$ follows the rollout policy. Player
  weighting is proportional to expected decision count, not uniform over
  players. Source: §“Game-Theoretical Model,” `not:random-variables`,
  equations `eq:stationary-node-law` through
  `eq:ideal-profile-and-player-marginals` (lines 166–211).

- **MATH-LAW-002 — Frozen rollout profile.** Targets and sampled laws are based
  on the frozen current profile $\pi^{\theta_k}$; optimization changes a working
  copy $\theta$, then stores it as $\theta_{k+1}$. Gradients through rollout
  policy and target quantities are stopped. Source: §“Theoretical Model,”
  `eq:pg-loss-expectation`; `thm:policy-gradient-games`; all three training
  algorithms.

- **MATH-LAW-003 — Public reach posterior.** Rao–Blackwell and branched modes
  must maintain factorized reach caches updated by the full policy/chance
  likelihood vector for the realized public event and normalized after each
  step. Posterior/completion sampling is from mass proportional to
  $\mu(x)\prod_q r_q(\sigma\mid x_q)$. Source: `alg:sample-preterminal`
  (lines 567–598), `alg:mc-preterminal` (lines 664–685), and
  `alg:sample-tree` (lines 750–813).

- **MATH-LAW-004 — Player-clock returns.** GAE uses only successive decisions of
  the target player as time steps, with $\gamma=1$; chance and other-player
  nodes do not advance that player's exponent. At $\lambda=1$ the return is
  Monte Carlo. Source: `eq:sequential-lambda-return` and
  `lem:player-clock-gae` (lines 400–412); RB analogue
  `eq:rao-blackwell-lambda-return` (lines 648–658).

### Model and loss

- **MATH-MODEL-001 — Curried actor output.** One model evaluation at public
  state $\sigma$ produces policy distributions for every compatible actor
  private state, rather than consuming one dealt private hand as a privileged
  selected input. Source: `eq:rao-blackwell-curried-policy` (lines 560–565).
  The directly corresponding architecture contract is `MHA_model.md`,
  §“Public Contract”: policy and actor-value rows are class-indexed preflop and
  hand-indexed postflop.

- **MATH-MODEL-002 — No learned per-action Q head.** The general theory assumes
  no Q head; the production model supplies policy and actor-type $V$. Branched
  training additionally requires public positive $T$ and $C$ heads. Source:
  §“Theoretical Model,” paragraph “Assumptions”; `core_math.tex` lines
  1055–1068 and 1352–1356; `MHA_model.md` §“Public Contract” and §“Value And
  Public Branching Heads.”

- **MATH-LOSS-001 — Primitive definitions.** PPO ratio is with respect to the
  frozen rollout policy. The clipped actor loss, value loss/value clipping,
  entropy, and optional exact or sampled KL are exactly the definitions in
  `eq:primitive-losses` (lines 265–329). Raw-Q NeuRD is explicitly
  reward-shift non-invariant.

- **MATH-LOSS-002 — Production poker loss family.** The stated production
  design is clipped PPO advantage + entropy + optional exact legal-action
  reference-policy KL + actor-value loss; branched modes additionally train
  $T/C$ from the first update. NeuRD and stored all-action Q are deferred
  research extensions. Source: §“Poker: a Perfect Example,” lines 1558–1568;
  `alg:branched-rao-blackwell-pg`, lines 1352–1356; lines 1345–1346.

- **MATH-LOSS-003 — Posterior weighting.** RB actor/value losses are summed over
  compatible actor private states. Sequential RB weights each term by the
  posterior at the rollout's pre-terminal state. Branched actor loss weights by
  the child posterior, while branched value loss weights by the parent
  posterior. Source: `alg:sequential-rao-blackwell-pg`, lines 711–728;
  `eq:branched-minibatch-loss`, lines 1370–1395.

- **MATH-LOSS-004 — Nonlinear RB caveat.** Replacing targets by conditional
  means is exact Rao–Blackwellization for the relevant linear identity, but for
  a loss nonlinear in its target it defines a new lower-variance surrogate, not
  the conditional expectation of the original loss. Source:
  `eq:rao-blackwell-q-identity`, `eq:rao-blackwell-loss-identity`, and lines
  645–646.

## Mode-specific contract

### `normal`

- **MATH-NORMAL-001 — Rollout sampling.** Sample $N$ independent full terminal
  rollouts. Each rollout samples one complete private profile $X_i\sim\mu$ and
  then every player/chance event from the corresponding policy/chance law.
  Source: `alg:sample-terminal` (lines 345–369).

- **MATH-NORMAL-002 — Decision sampling.** Select rollout $i$ proportional to
  its number of player nodes, then select uniformly among that rollout's player
  decisions. Equivalently, decisions are uniform over all player decisions in
  the realized batch. Source: `def:terminal-rollout-decision-law`,
  `eq:terminal-rollout-index-law` (lines 371–381).

- **MATH-NORMAL-003 — Targets.** For each visited actor decision, use
  `eq:sequential-lambda-return`; $\widehat A^\lambda=\widehat Q^\lambda-V^{\theta_k}$
  and $\widehat V^\lambda=\widehat Q^\lambda$. Source: lines 400–412.

- **MATH-NORMAL-004 — Required rollout data.** The mathematical consumer needs
  the sampled terminal private profile and public trajectory, terminal reward,
  frozen actor-value predictions at later same-player decisions when
  $\lambda<1$, selected action and frozen/current policy quantities required by
  the configured loss. It does not require posterior vectors, reach caches,
  leaf completion moments, tree topology, $N/\widetilde N$, or $T/C$.
  Source: `alg:sequential-pg` (lines 516–551) and
  `eq:sequential-lambda-return`.

### `sequential_rb`

- **MATH-SRB-001 — Public pre-terminal rollouts.** Sample $N$ independent
  public trajectories only through their pre-terminal state. Temporary hidden
  profiles may draw the next event, but persistent conditioning is the public
  reach cache. Source: `alg:sample-preterminal` (lines 567–598).

- **MATH-SRB-002 — Decision sampling.** Select public rollout proportional to
  its pre-terminal player-node count, then a uniform player decision within it.
  Source: `def:preterminal-rao-blackwell-decision-law`,
  `eq:preterminal-rollout-index-law` (lines 600–610).

- **MATH-SRB-003 — Completion estimator.** At each pre-terminal rollout, Monte
  Carlo must estimate the posterior occurrence of every player private state
  and the conditional reward for each private state. The theory reports SIS as
  empirically preferred over rejection sampling and MCMC. Source:
  `alg:mc-preterminal`, `eq:sequential-preterminal-estimates`, lines 660–705.

- **MATH-SRB-004 — Targets and loss.** Back-propagate
  `eq:rao-blackwell-lambda-return` for every compatible private state; use
  $\widehat A^\lambda=\widehat Q^\lambda-V^{\theta_k}$ and
  $\widehat V^\lambda=\widehat Q^\lambda$; sum loss over private states with
  pre-terminal posterior weights. Source: lines 648–658 and
  `alg:sequential-rao-blackwell-pg` (lines 687–734).

- **MATH-SRB-005 — Required data.** The mathematical consumers require public
  trajectory, per-node/event reach information sufficient to obtain the leaf
  posterior, per-private-state leaf posterior and conditional reward moments,
  frozen $V$ at same-player bootstrap locations, and sampled-decision policy
  quantities. They do not require sibling topology, nominal offspring counts,
  trace/cost correction state, or $T/C$ heads.

### Branched RB: common fixed/KKT contract

- **MATH-BRANCH-001 — One labeled tree and ordinary root rule.** A rollout is a
  single labeled tree rooted at exactly $\varnothing$. Every non-pre-terminal
  node, including the root, obtains nominal child count by randomized rounding
  $\widetilde N=\lfloor\beta\rfloor+\operatorname{Bernoulli}(\beta-\lfloor\beta\rfloor)$.
  There is no special initial root multiplicity in the ordinary law. Source:
  `alg:sample-tree`, especially lines 757–767; `def:branched-rollout-tree`.

- **MATH-BRANCH-002 — Intensity and spine.** $\beta(\sigma)\in[1,\infty)$, so
  randomized rounding always creates at least one ordinary child. At
  $\beta\equiv1$ the ordinary tree is exactly one random trajectory. Source:
  “Practical Implementation,” lines 817–822.

- **MATH-BRANCH-003 — Public independent children.** Every child independently
  samples a temporary private profile from the parent public posterior, then an
  event from policy/chance. Siblings may have identical public event labels but
  remain distinct nodes with independent draws, descendants, and completion
  samples. A single true hidden profile must never be shared across siblings.
  Source: `alg:sample-tree`; `lem:branched-branch-independence`;
  `warn:no-private-tilting` (lines 878–894).

- **MATH-BRANCH-004 — Uniform edge masses and decision law.** Each executed
  outgoing edge has transition mass $1/N(\mathsf s)$. Leaves induce
  `eq:branched-terminal-law`; player and node selection follow
  `eq:branched-player-law` and `eq:branched-stationary-node-law`, including
  inverse products of ancestor child counts. The minibatch action contribution
  averages uniformly over sampled child edges. Source: lines 824–868 and
  `eq:branched-minibatch-loss`.

- **MATH-BRANCH-005 — Leaf moments.** Each leaf needs an independent completion
  estimate of per-player/per-private-state posterior mass and conditional value,
  plus the vector public value moment
  $\widehat{\mathbf V}=(\widehat{\mathbb P}(-_p)\widehat V(-_p))_p$.
  Source: `alg:mc-preterminal-moments` and
  `eq:branched-moment-terminal-estimates` (lines 1177–1198, 1250–1265).

- **MATH-BRANCH-006 — Branched Q/V/A targets.** Parent posterior masses,
  posterior-weighted value moments, public vector values, and cost are uniform
  child reductions. Child Q is the conditional child moment. Parent V is the
  conditional parent moment. Advantage uses a leave-one-out sibling conditional
  value mixed with frozen $V$ at horizon $\lambda$; when there is only one
  target child, the missing sibling term falls back to frozen $V$. Source:
  `eq:branched-targets`; `alg:branched-value`, especially
  `eq:branched-backward-updates` and lines 1296–1336.

- **MATH-BRANCH-007 — Leave-one-out independence.** For $N^{\rm tgt}>1$, the
  sibling baseline must remain conditionally independent of the target branch.
  This is why persistent private tilting/shared hidden hands are forbidden.
  Source: `lem:branched-bucket-moments`, lines 959–963, and
  `warn:no-private-tilting`.

- **MATH-BRANCH-008 — T/C estimands and losses.** $T$ regresses the arithmetic
  conditional mean of a nonnegative corrected trace sample covariance when at
  least two target children exist; $C$ regresses the arithmetic conditional
  mean nominal per-child subtree cost. Both use the Gamma-deviance losses in
  `eq:branched-intensity-head-losses`; $L_T=0$ when the trace target is absent.
  Source: lines 1055–1068 and 1125–1174.

- **MATH-BRANCH-009 — Queue-pressure correction.** If the one-child queue
  forecast exceeds $M_{\tt safe}$ for a player-node forward batch, execute
  $N=1$ while retaining nominal $\widetilde N$. This is pressure control, not a
  hard memory bound. Backward $T/C$ recurrences must correct nominal estimands
  using both counts. Source: `alg:sample-tree`, lines 766–767; §“Queue budget
  safeguard,” lines 1080–1088; `alg:branched-value`.

- **MATH-BRANCH-010 — Lower-size root safeguard.** If the completed ordinary
  tree has fewer than $M_{\min}$ nodes, append independent root children and
  fully expand each by the ordinary beta rule. Recompute final root path masses,
  but preserve $N_0(\varnothing)$ and the original tree size. All root targets
  use only the protected original root-child prefix; refill branches are
  learning data only strictly below root. Source: `alg:sample-tree`, lines
  804–813; lines 1108–1127.

- **MATH-BRANCH-011 — Required data.** Both branched regimes require executed
  topology and distinct child identity, public labels, executed $N$, nominal
  $\widetilde N$, root protected-cohort metadata when refilled, uniform/path
  masses sufficient for decision sampling, node/leaf reach caches, leaf
  posterior/value/vector moments, frozen $V/T/C$, posterior-weighted moments,
  trace correction $\Delta$, cost targets, and per-edge advantages. Stored
  all-action $\widehat Q$ is explicitly unnecessary for production PPO.
  Source: `alg:sample-tree`, `alg:branched-value`, lines 1345–1346, and
  `eq:branched-minibatch-loss`.

### Fixed-beta branched RB

- **MATH-FIXED-001 — Geometry.** $\beta_k=\beta_{\tt fix}$ is deterministic and
  independent of learned $T/C$ outputs. Its realized child count still uses the
  ordinary randomized-rounding, queue-pressure, and lower-size safeguard laws.
  Source: `alg:branched-rao-blackwell-pg`, lines 1358–1366, together with
  `alg:sample-tree`.

- **MATH-FIXED-002 — Heads still train.** Fixed branching still emits and trains
  actor $V$ and public $T/C$ heads from the first update, but $T/C$ do not choose
  rollout geometry. Source: lines 1352–1356 and 1558–1568;
  `MHA_model.md` §“Public Contract.”

- **MATH-FIXED-003 — Sequential equivalence boundary.** The authority says
  $\beta\equiv1$ recovers sequential RB exactly. This claim applies to the
  ordinary tree-valued generalization; enabling a lower-size root refill would
  alter realized geometry and therefore must retain the protected-cohort rules
  to preserve root estimands. Source: lines 821–822, 1108–1127, and 1352.

### KKT branched RB

- **MATH-KKT-001 — Controlled variables.** KKT adapts only player-node
  intensities. Chance-node intensities remain fixed. The continuous intensity is
  a deterministic mean-field approximation to randomized offspring counts and
  is clipped to $[1,\beta_{\max}]$. Source: lines 1009–1013 and the unnumbered
  KKT update equation at lines 1069–1079.

- **MATH-KKT-002 — Intensity equation.** For player state $\sigma$, update
  $\beta_k(\sigma)$ by the square root of $\kappa_k T/C$ times the specified
  ancestor player-clock reciprocal-product sum divided by the full ancestor
  intensity product, then clip. Products and sums must retain the inclusive and
  strict prefix bounds shown in the KKT equation; replacing them by local
  $\sqrt{\kappa T/C}$ is not faithful. Source: KKT update equation, lines
  1069–1079, derived in lines 983–1054.

- **MATH-KKT-003 — Warmup.** Before learned heads control geometry, use fixed
  $\beta_{\tt fix}$. The dynamic regime begins only at $k\ge k_{\tt warm}$.
  Both heads nevertheless train from the first update. Source: lines
  1105–1106, 1352–1366, and 1562–1567.

- **MATH-KKT-004 — Counterfactual budget control.** Use
  $M_k^{\rm cf}=|\mathfrak S|+\sum_s(\widetilde N(s)-N(s))C(\ell(s))$, update an
  EMA, and update $\log\kappa$ by
  $\eta_\kappa\log(M/\overline M)$ with configured log bounds. The KKT
  calculation and controller run in log space. During warmup, neither EMA nor
  $\kappa$ advances. Source: §“Queue budget safeguard,” lines 1088–1105, and
  `alg:branched-rao-blackwell-pg`, lines 1363–1365.

- **MATH-KKT-005 — Approximation boundary.** KKT optimizes a $\lambda=1$ Monte
  Carlo variance proxy under expected tree-size budget. It ignores the
  reciprocal-moment correction caused by randomized rounding, and the trace
  correction is first-order when a shared queue couples siblings. These are
  stated approximations, not implementation freedoms to change the target law.
  Source: lines 967–1008, 1101–1103, and 1144–1149.

## BR contract gap

- **MATH-BR-001 — No authoritative BR algorithm.** `core_math.tex` defines the
  expected payoff $J_p(\pi)$ (`eq:expected-payoff`) and unilateral policy
  gradient (`eq:policy-gradient-games`), but contains no occurrence or
  definition of best response, exploitability, BR search, BR loss, BR sampling,
  or BR stopping criteria. Consequently this source cannot validate any BR
  implementation beyond the underlying game/payoff and strategy semantics.

- **MATH-BR-002 — Shared model semantics only.** If BR consumes the deployed
  poker policy, the only directly applicable model constraints are the public
  policy interface, legal-action masking outside the model, no selected dealt
  hand input, and action distribution defined in `MHA_model.md` §“Public
  Contract” and §“Continuous Policy Head.” No BR-specific tensors or losses are
  authorized by either theory file.

## Ambiguities and internal tensions to carry into the audit

- **A-01 (high) — BR is one of the requested five modes but has no mathematical
  contract in the designated authority.** A separate authoritative BR
  specification is required before implementation fidelity can be adjudicated.

- **A-02 (medium) — “Exactly five modes” is not stated in `core_math.tex`.** The
  document also describes REINFORCE, actor-critic, MMD-style PPO, NashPG-PPO,
  RNaD-PPO, R-NaD/DeepNash, NeuRD, optional suit losses, and other research
  choices. The poker production paragraph narrows the intended loss path and
  defers NeuRD/all-action Q, but it does not formally declare those objects
  non-production or prohibited.

- **A-03 (medium) — Fixed versus dynamic algorithm notation is underspecified.**
  `alg:branched-rao-blackwell-pg` always switches to KKT after warmup; the text
  says the companion exposes distinct fixed and dynamic modes but does not give
  a separate fixed-mode control-flow condition (for example, infinite warmup or
  an explicit switch).

- **A-04 (medium) — Lower safeguard parameter plumbing is incomplete.**
  `SampleTree` declares optional $M_{\min}$ and precisely defines refill
  semantics, but its displayed call sites and `BranchedRaoBlackwellPG`
  hyperparameter list do not pass/list $M_{\min}$.

- **A-05 (medium) — Executed versus target degree in the minibatch equation.**
  Protected root targets use $\mathfrak E^{\rm tgt}$ and $N^{\rm tgt}$, whereas
  `eq:branched-minibatch-loss` displays a sum over $\mathfrak E_{\mathsf S}$ and
  factor $1/N(\mathsf S)$. At a refilled root this conflicts with the preceding
  statement that root posterior, value, and leave-one-out advantage targets use
  only the protected prefix. The intended root sampling/loss denominator needs
  clarification.

- **A-06 (low) — Q notation exceeds the stated production payload.**
  `BranchedValue` and `eq:branched-minibatch-loss` display $\widehat Q$, while
  lines 1345–1346 state production PPO returns $\widehat A$ and does not store
  $\widehat Q$. This is reconcilable if $\mathcal L^\pi$ is PPO-only and its Q
  argument is omitted, but the displayed production algorithm is broader than
  its stated storage contract.

- **A-07 (medium) — GPU exclusivity is not a mathematical invariant.** The
  poker section says rejection sampling and MC “should really be GPU-based” and
  describes GPU-heavy synchronous queues, but even suggests folded statistics
  might run on CPU (lines 1545–1556). A repository-wide NVIDIA-only mandate must
  come from the project/pseudocode contract, not be attributed to
  `core_math.tex`.

- **A-08 (low) — “Unbiased” is qualified.** `lem:branched-bucket-moments` says
  the estimators are unbiased when multiplied by the private-state indicator;
  implementations must not silently strengthen this to unconditional
  per-private-state ratio unbiasedness, especially where `Cond(m,w)` is used.

## Audit-use summary

Any production implementation of the four training modes should be traceable
to the invariant IDs above. A discrepancy against a stated equation or
algorithm is mathematical drift unless it falls inside one of A-01–A-08 and is
resolved by an explicit authoritative decision. BR should be audited for
reachability and implementation quality, but not certified “faithful to
`core_math.tex`” until A-01 is resolved.
