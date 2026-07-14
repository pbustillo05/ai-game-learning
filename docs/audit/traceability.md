# Theory-to-code traceability

## Pseudocode mapping

| Theoretical object | Intended semantics | Current code | Assessment |
|---|---|---|---|
| `SampleTerminal` | Sample one private profile and all public events through terminal reward | `pbg/core/normal.py`, shared sequential sampler, game-owned sampled terminal settlement | Implemented; resolved `AUD-001` |
| Terminal empirical decision law | Pool player decisions with trajectory-length bias | `build_decision_alias` on trajectories | Weight law would be correct, but no normal targets/loss |
| Sequential player-clock lambda return | Bootstrap only at later decisions of the same player | owner-only moment bootstrap in `branched_value` | Correct relative to `NOTATION.md`; TeX display needs correction (`AUD-021`) |
| `SamplePreTerminal` | One private profile per root, carried along one public path; update all-type reaches | `_sample_trees_sequential` | Aligned in the current worktree |
| `PreTerminalMC` | Collision-conditioned profile/runout completion; posterior and actor-type values | synthetic and poker completion backends | Semantically present; active training validates quality and retains only `p_hat`/`v_hat` |
| `SequentialRaoBlackwellPG` | Integrate actor types using leaf posterior; pre-terminal one-path targets | sequential sampler + completion + `branched_value` + `rb_ppo` | Present; decision reaches compact after completion, while target layout remains shared with branching |
| `SampleTree` | Randomized-rounded fixed offspring, public-mixture events, duplicate labels, fresh hidden event draws | `_sample_branched_labeled` | Probability law substantially aligned |
| Branched decision law | Player nodes sampled by conditional path occupancy | `path_weight` + `build_decision_alias` | Aligned |
| Branch independence | Fresh hidden draw and independent descendants/completion per abstract child | child-local profile samples and duplicate-preserving topology | Aligned; policy/value forward runs once per unique parent |
| `BranchedValue` | Average posterior mass and weighted moment; conditionalize at emission; leave-one-out baseline; owner bootstrap | `posterior_values.py` | Substantially aligned |
| Branched PPO loss | child posterior for policy; parent posterior for value/entropy; uniform abstract-child average | `rb_ppo` | Aligned for the shipped fixed-beta PPO objective |
| Fixed beta v1 | game-owned public-stage intensity, no learned controller | `beta_fn` in synthetic/poker backends | Aligned |
| Variable/Neyman beta | public moment heads and allocation controller | dormant heads/hooks only | Deliberately deferred, allowed by user |

## Detailed core-algorithm notes

### Sequential rollout

The in-progress change in `pbg/core/rollout.py` now samples `root_profiles` once and carries them with the frontier. This is the efficient `SamplePreTerminal` construction. It is preferable to mechanically using the branching sampler at beta one, because the latter resamples a public-posterior profile at every event while producing the same marginal one-path public law.

The sampler still maintains all-type reach rows so the curried policy sees the public belief. At a player action it caches the selected action's log probability for every actor type and the value row. This is exactly what sequential RB PPO needs, but more than normal sampled-private training needs.

### Fixed branching

Offspring are randomized-rounded and every child edge gets weight `1/K`. Parent-major append preserves duplicate events as separate nodes. Public event generation samples one posterior private profile per abstract edge. Player and chance edge reach updates are child-local. These properties match the corrected labeled-tree theory.

The sampler evaluates policy/value once per unique player parent, then gathers
the tensor-structured output into child order. Hidden profiles, events, selected
action likelihoods, and reach updates remain edge-local.

### Empirical node law

For each stored player node, `path_weight` equals the product of uniform child
probabilities on its prefix. In the one branched tree, one alias table over all
decision nodes implements the required occupancy law. On sequential
trajectories every path weight is one, reducing to pooling sampled decisions.

The probability law is correct. The historical `AliasTable` API now stores the
normalized categorical law and samples through device-local
`torch.multinomial`; construction no longer copies through CPU/Python lists.

### Backward values

At leaves, code forms `P` and `M=P*V`. Internal nodes average both over abstract children. At a player node it:

- emits `V=Cond(M_parent,P_parent)` before current-node bootstrap;
- emits `Q=Cond(M_child,P_child)` per abstract edge;
- computes sibling totals as `K * parent_average - child`, whose normalization cancels in `Cond`;
- uses `lambda*sibling + (1-lambda)*old_v` when `K>1`, otherwise `old_v`;
- emits `A=Q-baseline`;
- mixes only the owner's moment before ancestors consume it.

That ordering is consistent with the corrected pseudocode and the player-decision clock. Conditional ratios remain finite-sample self-normalized estimates, as the TeX now acknowledges.

### Loss

`rb_ppo` weights each abstract child policy term by `child_p_hat/K`, and weights value/entropy by `parent_p_hat`. Type masks remove padded/impossible rows. PPO ratios use the cached old selected-action log probability. These choices match the fixed-beta RB-PPO companion.

The active composite uses the unclipped value loss even though a clipped value primitive/config field exists. Reference-policy KL, NeuRD/R-NaD, public moments, and Q storage are not active; this is consistent with v1 deferrals but should not be confused with support for every method discussed in the TeX.

## Poker evaluator boundary

The consumer matches the sibling's current `pbg-v2` operation names and major tensor representations:

- lexicographic combo IDs and defensive `COMBO_CARDS` validation;
- `sample_joint_profiles` for collision-conditioned dealt-seat profiles;
- `complete` with all dealt seats, contender mask, affine pot strata, and seeds;
- `evaluate_showdowns` for fixed complete boards and per-stratum shares;
- `partition_many` with the 0.3.0+ signature.

Completion includes folders in `dealt_mask`, and fold-terminal completion samples their blocker-conditioned hands even though payoff is public. This is correct for RB posterior mass.

The folded partition law now receives folded rows only. The evaluator-owned
`pbg-folder-v1` call provides combined/per-folder finite differences, signed
beta factors, pressure, deltas/covariances, and tight contraction. Compact
model-ready summaries are cached by a dependency anchor inherited across
ordinary actions and invalidated exactly by street/fold events (`AUD-009/010`).

## Poker model conformance matrix

| `MHA_model.md` component | Current status |
|---|---|
| Separate preflop/postflop weights | Implemented |
| 169 preflop class rows | Implemented and lifted to combo rows |
| Board-compatible postflop compact rows | Implemented |
| Actor policy for every type, no selected hand input | Implemented |
| Factored passive/call/raise + logistic-mixture ticks | Implemented |
| Actor-type value, no Q head | Implemented |
| Pre-LN transformer | Implemented |
| Per-head content temperature | Implemented |
| Untempered key-side reach prior | Implemented with exact fused/chunked bounded-memory attention |
| Full public history/action arithmetic tokens | Implemented |
| Preflop orbit-averaged card geometry | Implemented |
| Exact incidence residual per layer | Implemented |
| Identical-combo residual channel | Implemented |
| Postflop two-hot geometry | Implemented |
| Folder-corrected log reach `Lambda_B` | Implemented from evaluator finite differences |
| Signed beta factor coordinates | Implemented from evaluator pair field |
| Folder card-pressure channel | Implemented from per-folder evaluator fields |
| Per-folder line/log-mass summary | Implemented through one versioned evaluator call and event-keyed cache (`AUD-009/010`) |
| Folded-seat readout summary | Implemented |
| Bilinear/order-3 side units | Implemented; optional public moments remain dormant |
| Suit-consistency loss | Helper representation transforms exist; training loss is not wired |
| Public moment heads | Implemented as dormant option; not wired to targets/loss/beta |

## File-by-file risk summary

| File / area | Audit assessment |
|---|---|
| `docs/theory/core_math.tex` | Coherent main spine; lambda display and support hypotheses need binding correction |
| `NOTATION.md` | Best current executable mathematical companion; explicitly corrects TeX typos |
| `CONTRACTS.md` | Describes corrected RB v1, not the requested three-mode product; several moved-path references stale |
| `pbg/core/rollout.py` | Correct efficient sequential RB sampling; no normal path; capacity-sized reach allocation |
| `pbg/core/branching.py` | Correct labeled fixed branching with unique-parent model forwards; no byte planner |
| `pbg/core/posterior_values.py` | Corrected mass/moment semantics; byte-planned streamed completion feeds a contracted-tree `P/M` frontier |
| `pbg/core/losses.py` | Correct minimal RB-PPO weighting; normal/RB composites explicitly select MSE or clipped value loss |
| `pbg/core/tree.py` | Sound topology invariants; poker recompute uses child-state chance replay; post-completion decision reaches have a game-owned compact boundary; decision sampling is device-local |
| `pbg/core/training.py` | All three modes and identity-checked resume; one branched tree/rank; completion/leaf/full-reach/frozen-model/topology transients end before optimization |
| `pbg/games/synthetic/family.py` | Useful small finite game oracle; exactly two public dials grow card/chance and action width/depth with exact Kuhn/Leduc anchors; deliberately not advertised as literal NLHE convergence |
| `pbg/games/synthetic/model.py` | Scaled reach-biased MHA with action anchors, exact finite-card incidence, folded pressure, and policy-consistent exact export; MLP retained only as explicit baseline |
| `pbg/games/synthetic/bundle.py` | MC completion only, including backend named `exact`; Python-loop implementation |
| `pbg/games/poker/engine.py` | Broad public NLHE mechanics, side-pot coefficients, and complete 0.5bb fixed-denomination raise set |
| `pbg/games/poker/rollout_backend.py` | Tensor-native state transitions, lossless public history, fold snapshots, exact street/fold partition cache, and ragged live-seat decision reaches |
| `pbg/games/poker/model/*` | MHA-conforming paths and exact bounded-memory attention; H100 ceilings remain unmeasured |
| `pbg/games/poker/mc.py` | Current `pbg-v2` boundary substantially correct; full evaluator DTO is validated, quality-summarized, and collapsed before target construction |
| `pbg/games/poker/folded.py` | Folded-only neutral adapter with exact no-folder identity and combined/per-folder fields; single-call facade extension remains desirable |
| `configs/train/nlhe6.yaml` + `configs/hardware/*` | Scientific/execution split complete; Quest H100 ceilings remain open (`AUD-005`) |
| `tests/` | Strong local shape/contract coverage, synthetic MHA conformance checks, and device/workload-matched performance-gate logic; hardware baselines remain device-local artifacts |
