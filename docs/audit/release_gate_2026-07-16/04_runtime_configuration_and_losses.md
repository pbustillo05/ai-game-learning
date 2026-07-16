# Runtime, configuration, and losses

## Active game surface

NLHE cash poker is the only active game.

- `pbg.runtime` imports and constructs the poker bundle directly.
- There is no game registry, selector, or generic active-game dispatch.
- No module under `pbg/` implements Kuhn, Leduc, or a synthetic game.
- Retired games exist only under `archive/synthetic_games/`, are not packaged,
  and are excluded from active import/tests.
- `pbg.core` does not import `pbg.games` or `pbg.eval`; poker depends on the core
  protocols, and top-level composition owns the concrete connection.

The game schema has exactly the intended three public game fields:

```yaml
game:
  n_players: 2       # P
  stack_bb: 8.0      # S
  rank_count: 13     # R, highest ranks retained with all four suits
```

`PokerConfig` correctly validates:

- integer `P` in 2--6;
- finite `S` in 8--128 and an exact 0.5 bb multiple;
- integer `R` in 4--13 through `DeckGeometry`;
- strict `2P+5 < 4R`;
- fixed 0.5/1 bb blinds, no ante/rake, and equal stacks.

The rule engine and tensor backend implement fold, check/call, every legal 0.5 bb
raise-to denomination, all-in, short-all-in reopening, public chance streets,
fold settlement, and showdown cuts. Equal starting stacks make active side pots
impossible, although generic payoff strata remain capable of representing test
fixtures.

End-to-end production support is narrower than the schema: `R=13` only, and
deep multiplayer histories require a manually increased model capacity. See
B-01 and B-02.

## Training modes

| Mode | Public sample per rank | Completion | Target | Composite |
|---|---|---|---|---|
| `normal` | rank-local quota of sequential paths, each with one sampled joint profile | fixed-profile terminal settlement | sampled player-clock lambda return | `ppo` |
| `rao_blackwell` | rank-local quota of sequential paths stopped at pre-terminal leaves | sibling posterior/equity Monte Carlo | posterior-weighted player-clock targets | `rb_ppo` |
| `branched_rao_blackwell` | one distinguished randomized-rounded labeled tree | sibling completion at every sampled leaf | posterior `P/M`, child Q, leave-one-out A, parent V | `rb_ppo` |

All modes use the same family scheduler. Poker has logical families `pre` and
`post_A`, where `A` is live-player count. Street remains row metadata. Actual
model work may be physically split beneath hardware token and token-pair caps
without changing logical sampling order or target multiplicity.

Sequential starts are a global count partitioned into collision-free rank
quotas and must be at least world size. Branched mode requires
`sequential_starts: 1` and creates one tree on each rank. Unequal local decision
mass is normalized by global mass; one fp32 flat gradient buffer is SUM-reduced
without world-size division.

The audit confirmed all three modes on one GPU and two ranks for an `R=13`
acceptance game. It does **not** confirm all advertised NLHE configurations
because B-01 and B-02 are counterexamples.

## Currently selectable composite losses

There are exactly two entries in `pbg.core.losses.COMPOSITES`.

### `ppo`

Allowed mode: `normal` only.

Per sampled decision row, the composite contains:

1. PPO clipped policy loss on the sampled actor private combo:

   $$
   -\min\bigl(\rho\widehat A,
   \operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\widehat A\bigr).
   $$

2. Actor-combo value loss, chosen with `loss.value_loss`:

   - `mse`: `c_v * (V - V_target)^2`;
   - `clipped`: maximum of the current and PPO-clipped old-value squared errors.

3. Exact entropy over the externally masked legal factored poker action law,
   weighted by `c_h`.
4. Optional `c_kl * KL(new || reference)` over exactly the legal events when
   `c_kl > 0`.

The advantage/value target is the sampled terminal reward propagated backward
on the acting player's decision clock using `loss.lam` and the frozen old V.

### `rb_ppo`

Allowed modes: `rao_blackwell` and `branched_rao_blackwell`.

Per decision row, the composite contains:

1. The same PPO clipped edge loss for every stored abstract child and actor
   private type, weighted by that child's posterior mass and divided by the
   stored child count.
2. MSE or clipped actor-type value loss weighted by parent posterior mass.
3. Exact legal entropy weighted by parent posterior mass.
4. Optional exact legal-action `KL(new || reference)` weighted by parent
   posterior mass.

Invalid/private types and padded children are masked. In branched mode duplicate
event labels are separate abstract child samples. In sequential RB the single
child posterior shares the parent posterior storage.

### Entropy schedules

- `const` works as intended.
- `linear_decay` is exposed but currently broken by B-04 and must not be used for
  a production run until fixed and tested.

### Reference KL

When `c_kl > 0`, training keeps an immutable no-gradient reference model,
refreshes it every `training.kl_ref_interval`, evaluates exact masked
`KL(new || reference)`, and checkpoints/restores its state and reference
iteration. This is implemented in all three modes.

It is a reference-regularized PPO option, not proof that the full IMMD/NashPG or
R-NaD outer algorithm is implemented. `core_math.tex` still describes this
boundary inconsistently.

### Optimizer/update options

- optimizer: `adam` or `adamw`;
- precision: profile-selected FP16, BF16, or FP32, conditioned on `training.amp`;
- logical updates: repeated sampled `gd_steps_per_iter`, or complete shuffled
  `ppo_epochs` for normal/sequential RB only;
- global norm clipping through `training.grad_clip`.

These are update choices, not separate losses. The finite-step and numeric
validation defects in B-03/B-05 apply.

## Losses not available now

The following cannot be selected as production composites:

- unclipped REINFORCE/vanilla policy gradient;
- a distinct actor-critic composite;
- NeuRD;
- R-NaD/DeepNash losses or modified returns;
- IMMD outer-loop logic;
- MMD or sampled-KL objectives;
- public moment `W`, `S`, `U`, or player-vector V losses;
- learned/adaptive branching-intensity losses;
- all-action Q regression;
- suit-consistency/permutation loss;
- approximate-best-response/exploitability loss.

Primitive helper functions do not make a loss selectable. Optional
`MomentHeads` can be instantiated, but no registered composite consumes their
outputs, so they receive no task loss and are not a usable production feature.

## Configuration assessment

### Correct strictness

The loader recursively rejects unknown keys, missing required fields, wrong
primitive types, malformed tuples/maps, retired executor keys, invalid enums,
mode/loss mismatch, illegal beta placement/intensity, invalid completion sample
semantics, and many hardware-profile errors. All 23 current non-hardware YAML
presets loaded successfully against the local profile.

### Missing strictness

The loader needs a complete numeric validation layer. At minimum:

- `node_budget > 0`;
- `0 <= lam <= 1`;
- finite positive iterations, update counts, batch size, learning rate, and
  checkpoint interval;
- finite nonnegative `c_v`, `c_h`, `c_kl` and positive sensible clip epsilons;
- `0 <= beta1,beta2 < 1`, conventionally `beta1 < beta2` if required;
- finite positive grad clip;
- model/player and history-capacity cross-checks;
- deck/model hand-width cross-check;
- CUDA requires a hardware profile and world size no larger than its limit;
- output/checkpoint schedule cross-field checks.

Validation must occur before model/evaluator allocation and report the dotted
configuration path.

## Checkpoint and evaluation

Checkpoint payloads include model, optimizer, scaler, completed iteration,
scientific config, per-rank Torch/NumPy/process RNG state, run id, and optional
reference model. Resume checks path containment, world/config identity, and an
exact gap-free metrics prefix. The tests exercise corruption and identity
failures.

`pbg.eval.arena` is read-only duplicate-deal checkpoint comparison using the
independent CPU public engine and sibling showdown facade. It is useful for
relative head-to-head evaluation but is not an exploitability measurement or
best response.
