# Fixed-target paired response-gain protocol v2.1

Protocol identifier: `fixed_target_paired_response_gain_v2_1`.

This protocol supersedes v2 measurements. Protocol v2 correctly separated
target-reference and actual-profile range contexts, but screening exposed a
postflop support defect: exact FP32 softmax underflow could erase possible
private hands, and ranges were not conditioned on newly public cards. The v2
files remain immutable audit evidence and are not valid v2.1 observations.

## Measured quantity

For probe seat $s$, one paired observation is

$$
G_s = U_s(\pi^{\mathrm{candidate}}_s, \pi^{\mathrm{target}}_{-s})
      - U_s(\pi^{\mathrm{target}}_s, \pi^{\mathrm{target}}_{-s}),
$$

on the same initial private deal and coupled logical random addresses. Gains
are signed and reported in bb/hand for both seats and as their arithmetic mean.
The optional sum is called **sum of achieved response gains**. It scales the
mean and its uncertainty by the number of seats; it is not exact exploitability
or an exact NashConv certificate.

The unchanged target is an explicit zero-gain selection candidate. A response
selected on validation remains selected even if its fresh test score is
negative.

## Fixed-target and range semantics

At a matched public history, the target model receives ranges updated only by
its own fixed reference policy. The response model separately receives the
actual mixed-profile ranges. The sampled opponent private cards are not model
inputs.

If an observed action has zero aggregate probability under the target policy,
the deterministic off-support extension retains the prior target-reference
range. If the aggregate action probability is positive, exact zero entries in
the FP32 softmax table are treated as numerical underflow and clamped to the
smallest positive FP32 value before the Bayesian range update. After every
public chance event, private hands conflicting with the board are removed. If
a seat has no remaining finite compatible support, its extension is uniform
log-weight over all board-compatible private hands. No response-dependent
fallback is used.

Existing candidates were trained under `adaptive_mixed_profile_range_v1`, with
one shared range context conditioned by the actual acting policy. Their v2.1
scores are valid fixed-target evaluations of those saved candidate policies,
but do not retroactively change their training objective.

## Randomness, sampling, and selection

Every deal identity contains protocol version, split namespace, evaluation
seed, and logical deal ID. Stable SHA-256 addressing makes private deals and
public chance independent of batching, worker order, and session call order.
The button is explicit. Mixed and baseline arms share the initial deal and
logical random addresses. Screening, validation, and test have fresh v2.1
namespaces distinct from each other and from v2.

A trajectory is one probe-seat/button assignment on one logical deal, run once
under the mixed profile and once under the target baseline. Complete $P^2$
blocks balance probe seat and button. Paired differences are the observations;
replays sharing a deal are clustered. Reported 95% intervals quantify
conditional Monte Carlo uncertainty for one trained response, not variation
across training seeds or uncertainty about exact exploitability.

The common candidate grid is response iterations 50, 100, 150, 200, 250, and
300. Each gets 1,000 screening trajectories; the top three get 5,000 validation
trajectories. Validation selection jointly covers both seats, includes the zero
control, and breaks exact ties in favor of zero and then the lower iteration.
The choice is frozen before 20,000 fresh test trajectories. Legacy and v2
results must be shown as separate, unconnected, visibly superseded series.

## Numerical semantics verification

The CUDA audit
`range_semantics_numerical_audit_v2_1_r2.json` (SHA-256
`15d4c9a731a89e0c7a00269bb692ecbdafa2e089a0f3bd92949dfae90399b95d`)
uses one fixed logical deal and records the following distinct cases:

- All 150 private hands conflicting with the three-card board have negative
  infinite log reach. Each seat retains finite support on all 1,176 compatible
  hands.
- An action with positive aggregate support but exact zeros on most FP32 rows
  retains all 1,326 prior hands; the lowest finite update is exactly
  $\log(\mathrm{float32\ tiny})=-87.3365$.
- An action with zero probability for every private row is genuinely outside
  the fixed policy's support and retains the prior range exactly.

The same audit compares target-reference and legacy response-conditioned range
bookkeeping at one matched reachable public history. For the dealt acting hand,
the observed check/call probability is 0.002137 under the target and 0.000320
under the response, so the history is reachable under both. The root actor is
seat 0, the deviation/response seat. At the next decision the actor is seat 1,
the frozen target opponent; the reported TV therefore measures the opponent's
policy, not the response policy. With the public history and target private
hand held fixed, the two range contexts change that target action distribution
by L1 0.3164 (total variation 0.1582; maximum component difference 0.01132).
Thus frozen weights did not define a fixed behavioral opponent under the legacy
bookkeeping. The v2.1 fix can materially change the target's policy on reachable
histories; that is a semantic correction, not a target-training change.

A separate float64 log-space calculation constructs the underflow-case update
as prior log reach plus the log of the selected action probability clamped only
at `float32` tiny, then normalizes with `logsumexp`. It agrees with the evaluator
within $3.11\times10^{-6}$ in both unnormalized log reach and normalized log
posterior (predeclared tolerance $10^{-5}$). This is distinct from impossible
hands, which are masked to negative infinity by public-card compatibility, and
genuinely off-support histories, for which every compatible private hand gives
the action zero probability and the deterministic extension retains the prior.
The immutable result is `range_semantics_numerical_audit_v2_1_r3.json`.
