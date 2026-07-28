# Canonical poker policy CUDA operator

Status: implementation proposal.

## Contract

One repository-owned operator composes `{type_logits, omega, mu, s, iota}` with public legality. It is the numerical
authority for:

- logical-key action sampling inside `sample`;
- selected `pi(action|h)` and `log_pi(action|h)` during action-conditioned sampling and `commit`;
- differentiable selected log-probability, entropy, and reference-KL terms in the learner.

Arithmetic is fp32. Stored model fields may use the configured inference dtype. Rollout never materializes
`<H,event_count>`. Both selected-action and full learner-reduction entry points are required; learner reductions may
use bounded transient legal-action rows internally.

`model` writes the raw policy head to family workspace. `sample_actor` returns an action label without writing retained
arrays. Once the action is known, `commit` evaluates the selected action over its output domain and writes
`ActionHead.old_log_pi`. The learner evaluates current-policy reductions from sealed `model_input` and action labels.

## CUDA entry points

Let `policy_raw` have shape `<...,H,F>`. For each batch prefix $b$, acting hand $h$, and selected action $a$:

```text
sample_actor(policy_raw, legality, actor_hand, logical_key)
  -> action, selected_log_pi

selected_log_pi(policy_raw, legality, action)
  -> log_pi <...,H>

policy_reductions(policy_raw, legality, reference_raw?)
  -> entropy <...,H>, optional reference_kl <...,H>
```

`selected_log_pi` and `policy_reductions` register autograd through every policy field. Sampling has no backward.
All entry points share the discretized-logistic, legal-grid, short-all-in, all-in-atom, and zero-mass fallback device
implementation.

## Acceptance

- Forward agreement among sampling, `ActionHead.old_log_pi`, materialization, and learner paths.
- Gradient agreement with a high-precision differentiable reference.
- No rollout action-axis expansion.
- CUDA coverage for contiguous legal raises, singleton short all-in, no-raise states, and numerical tail cases.
