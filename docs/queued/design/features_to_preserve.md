# Residual features to preserve during the queued-core rewrite

Status: preservation baseline.

## Scope and rule

This file lists only desired, implemented behavior that is not already a
contract in another document in this folder. The storage, queue, sampling,
topology, target, poker-policy, folded-range, dtype, replica, and cleanup rules
specified elsewhere are deliberately not repeated here.

Preserve behavior, not old classes, module boundaries, configuration names, or
storage representations. Every item below needs a new owner and an acceptance
test in the migration ledger. Removal or semantic change requires explicit
user approval.

## Training and objective surface

- The update schedule supports a fixed number of gradient steps per rollout
  and shuffled multi-epoch reuse for the sequential modes that currently allow
  it. Tail batches remain exact; a physical batch size never changes the
  requested number of statistical presentations.
- The loss surface retains clipped PPO policy updates, MSE or clipped value
  loss, constant or linear-decay entropy regularization with an explicit
  independent decay horizon, reference-policy KL, and the configured
  coefficients and clipping thresholds.
- Reference-policy refresh remains selectable by iteration interval or by
  accumulated effective rollout hands. The reference weights, reference
  iteration, exposure counter, and overshoot/reset behavior survive resume.
- Optional reference-output caching remains an admitted optimization chosen by
  predicted time saved versus resident and build bytes. Failure to admit a
  cache uses the exact uncached path without changing sampled rows, RNG state,
  objective mass, or updates.
- Cost-aware statistical allocation gives every positive-mass stratum at least
  one draw, then deterministically apportions the remainder using lagged mass,
  variance, and positive per-draw cost inputs, with the current exact fallback
  proxies when calibrated inputs are unavailable. Global stratum normalization
  remains exact across ranks.
- Cost-safeguard activation retains its Gamma-deviance and p95
  underprediction tests, required consecutive-success streak, reset behavior,
  activation state, and checkpointed observations.

## Model and poker behavior

- Configurable equal-stack NLHE retains every valid player count, half-integer
  stack depth, compact rank count, blind, deck-fit, history-capacity, and legal
  action boundary. Rake, antes, and side pots remain absent.
- The current preflop/postflop model architecture and parameter separation
  remain supported. Lossless history capacity is derived from the game bound,
  and the worst legal history attention footprint is checked against the
  hardware profile before execution.
- Preflop class-order equivariance and empty-seat masking remain exact.
  Postflop execution remains ragged over live seats: folded seats do not create
  attention rows, while compact card-incidence and evaluator folder features
  retain their declared semantics.
- Supported training precision profiles remain FP32, FP16 AMP, and BF16 AMP
  where the hardware allows them. Autocast batch subdivision must not change
  retained old-policy/value outputs beyond the accepted precision tolerance;
  nonfinite FP16 updates apply one common scaler backoff.
- Hardware attention implementations may differ for performance, but their
  supported forward and backward results, token/token-pair admission, and
  query-chunk bounds remain equivalent to the authoritative model operation.
- Utility units remain explicitly selectable between raw big-blind payoff and
  stack-normalized payoff. Approved checkpoint conversion changes only the
  final preflop/postflop value-head affine weights and biases.

## Best response and scientific controls

- A probe target is identified by checkpoint content hash, not path. Probe
  initialization remains selectable between independent random initialization
  and compatible target weights, and the choice plus initial-model identity is
  checkpointed.
- Training probe seats are uniform logical-key draws assigned to roots. Evaluation retains the
  balanced seat/button pairing design, multiplayer early-fold cutoff semantics,
  paired per-seat gain and standard error, and the approximate-NashConv lower
  bound. If the number of roots is a multiple of the number of player, we prefer a cyclic assignmenet guaranteeing the same amount of roots for each seats.
- The `strict`, `seeded`, and `throughput` reproducibility profiles remain
  explicit and recorded. Backend determinism, TF32, and benchmarking choices
  may differ by profile, while logical scientific work and declared RNG
  addresses do not.

## Configuration, persistence, and public products

- Configuration loading remains closed-world and reports unknown or invalid
  nested values with their full dotted paths. Scientific configuration is
  separated from hardware scheduling, launch limits, and checkpoint cadence;
  CUDA training requires a compatible hardware profile.
- Checkpoints are atomic and strict-schema. They preserve model, optimizer,
  scaler, explicit and ambient RNG state per rank, controller states,
  reference state and cadence, probe identity and initialization, logical work
  counters, run identity, and the gap-free metrics prefix.
- Resume rejects scientific drift but permits non-scientific hardware and
  checkpoint-cadence changes. Metrics beyond the selected checkpoint are
  backed up and trimmed before new gap-free records are appended.
- A rank-common terminal failure checkpoints the last completed optimizer
  state when the audited phase boundary permits it. `SIGUSR1`/`SIGTERM`
  preemption records only a notice in the handler, then makes one synchronized
  iteration-boundary checkpoint-and-stop decision.
- Inference artifacts remain model-only, content-hashed, and strict about game,
  model, deck, and source identity. The public stateful singleton and batch
  session preserves legal-action, range, compact-card parsing, and
  preflop/postflop behavior on CUDA and supported MPS paths.
- Performance gates compare matched logical work and reject baselines with
  incompatible code/config identity, device/environment, RNG version, mode, or
  required metrics. Warmup and diagnostic kernel probes do not count as
  measured end-to-end throughput.
- Distributed initialization starts every rank from one recorded model state.
  Diagnostics retain global/per-rank logical work, decision counts and
  fractions by street, trajectory-length summaries, family/batch model timing,
  completion health, posterior effective hands, objective/loss components,
  clipping and gradient statistics, and initial-model/config/code identity.
