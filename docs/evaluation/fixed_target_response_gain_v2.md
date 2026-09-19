# Fixed-target paired response-gain protocol v2

Protocol identifier: `fixed_target_paired_response_gain_v2`.

## Measured quantity

For probe seat $s$, one paired observation is

$$
G_s = U_s(\pi^{\mathrm{candidate}}_s, \pi^{\mathrm{target}}_{-s})
      - U_s(\pi^{\mathrm{target}}_s, \pi^{\mathrm{target}}_{-s}),
$$

on the same initial private deal and coupled logical random addresses. Payoffs and
gains are in bb/hand. The report retains the signed observations, the mean for
each seat, and the arithmetic mean over seats. Negative candidate gains are not
truncated.

The optional aggregate is named **sum of achieved response gains** and equals the
sum of the per-seat estimates. Its standard error and confidence interval are
scaled from the mean by the number of seats. It is neither exact exploitability
nor an exact NashConv certificate: each candidate is a finite trained response.

The unchanged target is an explicit zero-gain selection candidate. A response
selected on validation can have a negative held-out test gain; that result
remains the reported test result.

## Fixed-target policy semantics

A frozen target means fixed weights plus a fixed policy definition. At a matched
public history, the target model receives ranges conditioned only with the
target's own reference policy, including when the observed action was generated
by the response. The response model separately receives the actual mixed-profile
ranges required to describe the policy that generated the trajectory.

If an observed public action has zero total probability under the target
reference range, protocol v2 deterministically retains the target's prior
reference range at that history. It never substitutes the response policy.
Target policy tables contain all private-hand rows; sampled opponent private
cards are not fed into the target model input.

Existing response checkpoints were trained by
`adaptive_mixed_profile_range_v1`: training used one shared range context
conditioned by the actual acting policy. Protocol-v2 evaluation can measure
those checkpoints as candidate policies, but it does not retroactively change
their training objective. Correcting response training requires dual retained
range contexts throughout rollout, leaf completion, and learner publication;
that is separate work.

## Randomness and sampling units

Every deal has an explicit identity containing protocol version, split
namespace, evaluation seed, and nonnegative logical deal ID. A stable SHA-256
namespace feeds the counter-based GPU RNG; Python `hash`, batch position,
worker order, and session call order are not inputs. The button is explicit and
drives button, blinds, first actor, and commitments.

A trajectory is one probe-seat/button assignment on one logical deal, evaluated
once under the mixed profile and once under the target baseline. Complete
$P^2$ blocks balance probe seat and button. Mixed and baseline arms use the same
private deal and coupled action/chance addresses. Screening, validation, and
test use distinct split namespaces and therefore distinct deal streams even if
their numeric logical IDs overlap.

Uncertainty is computed from paired payoff differences. Replays sharing a
logical deal are one cluster. The current main design uses one seat evaluation
per logical deal and stratifies by seat; a complete all-seat replay design is
also supported and clusters all seats together. The 95% interval is conditional
Monte Carlo uncertainty for one selected trained response. It does not include
variation across response-training seeds or claim uncertainty about exact
exploitability.

## Selection protocol

The main benchmark uses response iterations 50, 100, 150, 200, 250, and 300 for
every target. Screening uses 1,000 trajectories per trained candidate.
The three highest screening candidates receive 5,000 validation trajectories.
The unchanged zero-gain target is included in validation selection without
consuming a response-training slot. Selection maximizes validation mean paired
gain; an exact tie prefers the zero control, then the lower response iteration.
One checkpoint is selected jointly for both seats and frozen before a fresh
20,000-trajectory test.

A denser 25-iteration grid is a separately labeled search diagnostic and is
never mixed into the six-candidate main selection budget. Legacy results use a
different protocol and must appear as a separate, unconnected series.
