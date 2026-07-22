# Runtime, configuration, and losses

## Active surface

NLHE cash poker is the only active game. `pbg.runtime` constructs it directly;
there is no game registry, rollout implementation selector, active retired game,
or local private-hand evaluator.

The game schema has exactly three run dials:

```yaml
game:
  n_players: 2       # P in 2..6
  stack_bb: 8.0      # S in 8..128 by 0.5 bb
  rank_count: 13     # R in 4..13
```

Loading enforces finite half-big-blind stacks, `2P+5 < 4R`, equal stacks,
0.5/1 blinds, and the no-ante/no-rake cash rules. `DeckGeometry` then owns
$D=4R$, $H=\binom{D}{2}$, $K=R^2$, compact/global card maps, class maps, and
category statistics. Production model, rollout, evaluator, completion,
showdown, and folded-summary tensors all derive their widths from that object.

## Strict configuration

Recursive loading rejects missing fields, wrong types, unknown fields,
nonfinite numbers, invalid signs/intervals, empty work schedules, and invalid
cross-field combinations with dotted paths. Focused tests cover:

- positive rollout budgets, queue/arena sizes, completion samples, iterations,
  minibatches, learning rates, update counts, and checkpoint intervals;
- lambda, PPO/value clips, coefficients, entropy schedule, grad clip, Adam
  betas, optimizer choice, seeds, and output paths;
- model dimensions, head divisibility, player capacity, compact-deck widths,
  folded-summary geometry, and the lossless history bound;
- hardware token/token-pair limits, memory fraction, reserve, device count,
  compute capability, VRAM range, worker count, and precision;
- mode/loss/beta compatibility, global sequential-start quotas, reference-KL
  refresh requirements, and worst-case model-token preflight.

CUDA training through either the CLI or Python API requires a hardware profile.
Hardware profiles remain outside scientific checkpoint identity so a compatible
profile can change query chunking or other execution limits on resume.

## Rollout and residency

Normal, sequential RB, and branched RB share the family scheduler. Logical
families are `pre` and `post_A`; street is row metadata. Logical RNG addresses
depend on seed/iteration, global trajectory or rank tree, abstract path, child
ordinal, and draw purpose—not queue timing or physical packing.

Public history uses a binary-lifting ancestor table and one batched device
gather. CUDA materializes the preflighted lossless width without reading a
realized length. Folded summaries use a fixed device-indexed cache. The only
remaining mixed-row scheduler-control transfer is fixed-size and counted.

The hardware precision applies consistently to frozen behavior-policy inference,
identity replay, and learner inference. Completion and posterior/value
accumulators remain fp32 outside autocast.

## Losses and optimizer update

The only registered composites are:

- `ppo` for normal mode;
- `rb_ppo` for sequential and branched RB.

Both support MSE or clipped actor-type value loss, exact legal entropy with a
constant or linear schedule, and optional exact legal-action
`KL(new || reference)`. The immutable reference snapshot refreshes at
`kl_ref_interval` and is checkpointed. Optional moment heads have no registered
loss.

For unequal distributed work, each family/rank contributes its exact local mass
over global mass. Ranks reduce one flat fp32 gradient buffer with `SUM` and no
world-size division. The common reduced-buffer finite decision alone controls
unscale, clipping, optimizer mutation, and loss-scale transition. Rejected FP16
steps apply one common backoff; disabled scalers retain unit scale. Metrics
separate applied and nonfinite-skipped steps.

Tests cover Adam/AdamW, FP16/BF16/FP32, enabled/disabled AMP, finite and injected
nonfinite buffers, one rank, unequal two-rank work, constant/linear entropy,
MSE/clipped values, reference refresh, checkpoint, and resume.

## Checkpoint and evaluation

Atomic checkpoints include model, optimizer, scaler, completed iteration,
scientific configuration, explicit per-rank Torch/NumPy/process RNG state, run
identity, and optional reference state. Resume requires the same scientific
configuration/world size, checkpoint containment in the run directory, and an
exact gap-free metrics prefix. A changed compatible hardware profile is allowed
and recorded separately.

`pbg.eval.arena` remains read-only duplicate-deal checkpoint comparison using
the independent public engine and sibling showdown facade. It is not an
exploitability or best-response measurement.
