# Findings and required fixes

## Summary

| ID | Severity | Finding | Release effect |
|---|---|---|---|
| B-01 | Release blocker | Production rejects every reduced deck | Advertised `(P,S,R)` domain is false |
| B-02 | Release blocker | Legal histories exceed model capacity | Shipped deep/multiplayer configs can fail mid-rollout |
| B-03 | Release blocker | Nonfinite decision does not gate optimizer | FP32/BF16 can step with nonfinite gradients |
| B-04 | Release blocker | Linear entropy decay reads the wrong config field | Exposed loss schedule is mathematically wrong |
| B-05 | Release blocker | Strict config omits critical value validation | Invalid science/optimizer configs pass loading |
| M-01 | Major | Hot-path host reads and Python loops are unaccounted | Residency/observability contract is not met |
| M-02 | Major | CUDA hardware profile is not required by Python API | Safety contract can be bypassed |
| M-03 | Moderate | Active documentation contradicts implementation | Existing readiness and feature claims are stale |
| M-04 | Moderate | Core production modules are oversized | Change risk is concentrated in two files |
| M-05 | Moderate | No approximate best-response release probe | Strategy quality is not a release acceptance signal |

## B-01: reduced decks are not implemented end to end

**Repair status (2026-07-16): resolved in commit `5c3c5ef`.** Production now
derives card, combo, and preflop-class widths from `DeckGeometry` throughout
range state, model inputs, both networks, rollout/reach storage, chance draws,
completion, showdown, and folded-player summaries. Compact IDs remain local
while evaluator-facing card IDs retain the sibling `pbg-v3` suffix-deck
contract. Tests exercise every `R=4,...,13`, every legal `(P,R)` pair,
preflop/postflop/folded evaluator boundaries, and one-iteration normal and
Rao--Blackwell training for every rank count. The compact suite passes 123
tests, and the broad CPU regression passes 335 tests with one unrelated
version-conditional skip.

**Contract.** `AGENTS.md`, `README.md`, `CONTRACTS.md`, and `docs/poker.md` all
advertise `rank_count` in 4--13, subject to `2P+5 < 4R`. The run is supposed to
have exactly three game dials `(P,S,R)`.

**Evidence.** `PokerConfig` and `DeckGeometry` accept and construct compact decks,
and the sibling `pbg-v3` API supports compact `H = C(4R,2)`. The production
backend immediately rejects them:

```text
pbg/games/poker/rollout_backend.py:103-106
if cfg.rank_count != 13:
    raise RuntimeError(
        "compact-deck model tensors are not yet migrated to variable hand width"
    )
```

The test suite codifies the gap rather than support:
`tests/test_compact_deck_geometry.py:148-150` expects a reduced-deck rollout to
fail. A direct `R=12` construction in this audit produced that RuntimeError.

The fixed constants propagate through production model and rollout boundaries:
1326 combo rows, 169 classes, 52-card embeddings, and full-deck compact sizes.
Configuration loading does not reject `R != 13`, so failure occurs during bundle
construction rather than at the documented strict validation boundary.

**Required fix.** Choose and document one release contract:

- migrate all preflop/postflop/model/adapter/reach widths to `DeckGeometry` and
  add boundary training tests for every supported `R`; or
- narrow the minimal release schema and all documentation to `rank_count: 13`,
  rejecting other values with a dotted `ConfigError` at load time.

The second option is the smaller production-ready release if compact decks are
not essential for v1, but it changes the user's stated required scope.

## B-02: legal histories exceed the model's positional capacity

**Repair status (2026-07-16): resolved in the working repair series.** The
lossless capacity is derived as
`ceil(SP + 3P - S + 1) + 3`, is installed in both networks and the state-store
bound when omitted, and is rejected with `root.model.max_history_tokens` when
an explicit value is smaller. The worst legal one-row token and token-pair
footprint is checked against both hardware-profile phases before allocation.
Adversarial smallest-full-raise trajectories attain the exact bound for every
`P=2,...,6` at `S=8` and `S=128`, and the resulting packed histories pass the
postflop public-token construction at full length.

**Contract.** `AGENTS.md:44` gives the theoretical maximum number of player nodes
in a trajectory:

$$
N_{\max}(P,S)=SP+3P-S+1.
$$

The model architecture consumes the lossless public trajectory. All configured
`P in {2,...,6}` and `S in [8,128]` are advertised as supported.

**Evidence.** `PokerModelConfig.max_history_tokens` defaults to 256
(`pbg/games/poker/model/preflop.py:43`). Both networks raise when packed history
exceeds it (`preflop.py:295-296`, `postflop.py:579-580`). None of the shipped
configs sets a larger value, and bundle/config validation does not compare this
capacity with game geometry.

At `S=128`, the documented maxima are:

| Players | Maximum player nodes |
|---:|---:|
| 2 | 135 |
| 3 | 266 |
| 4 | 397 |
| 5 | 528 |
| 6 | 659 |

The shipped `configs/train/nlhe6.yaml` is 6-player 100 bb and can reach 519
player nodes before adding the three street-separator history entries. Thus the
default production preset itself has a legal failure path.

The tensor state allocates a larger conservative action-history capacity, so the
failure is specifically at the model position embedding/validation boundary,
not public navigation.

**Required fix.** Derive and validate lossless history capacity from `(P,S)`.
The bundle should fail before allocation if an explicitly smaller model capacity
is requested. Add adversarial maximum/minimum-raise trajectory tests at the
largest supported configurations and ensure attention token/token-pair preflight
uses that maximum.

## B-03: finite-gradient result is observed but not enforced

**Repair status (2026-07-16): resolved in the working repair series.** The
common reduced-buffer decision now solely controls unscale, clipping, and the
optimizer step. A rejected FP16 step applies one explicit common scale backoff;
disabled scalers retain unit scale. Metrics report applied and nonfinite-skipped
steps, and tests cover one-rank FP32/BF16/FP16 parameters plus an unequal
two-rank injected nonfinite gradient.

**Contract.** `docs/theory/pseudocode/distributed_update.tex:167-170` requires an
identical nonfinite decision from the reduced flat gradient buffer to gate
unscale, clipping, optimizer step, and next loss scale. `CONTRACTS.md` requires
finite distributed gradients.

**Evidence.** `_sync_scaled_gradients` correctly SUM-reduces one flat fp32 buffer
and returns `finite` (`pbg/core/training.py:371-418`). The caller stores the
boolean, then unconditionally executes:

```text
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

at `training.py:1564-1574`. The result only becomes a metric at line 1580.

An enabled FP16 `GradScaler` performs its own local inf check and normally skips
the step, but the repository's common decision still does not control it. More
importantly, the scaler is disabled for FP32, BF16, CPU, and `training.amp:
false` (`training.py:234-238,1773`). In those supported modes `scaler.step` calls
the optimizer even when the common reduced buffer is nonfinite.

**Required fix.** Make the returned common boolean the single explicit branch.
On false, do not clip or step; update/reset scale consistently under an explicit
policy; log the skipped step; and add FP32, BF16, FP16, one-rank, and unequal
two-rank injected-inf tests that compare parameters and scaler state.

## B-04: `linear_decay` entropy decays over one iteration

**Repair status (2026-07-16): resolved in the working repair series.** Linear
decay now reads `cfg.training.iters`, starts at `c_h` for zero-based iteration
0, reaches zero at `training.iters`, and rejects loss-like direct calls that do
not provide an explicit positive horizon. Focused tests exercise iteration 0,
the midpoint, and the endpoint through both `ppo` and `rb_ppo`.

**Contract.** `entropy_schedule` accepts `const` and `linear_decay`; the latter is
expected to decay across `training.iters`.

**Evidence.** `_entropy_coefficient` uses:

```text
getattr(getattr(cfg, "train", cfg), "iters", 1)
```

at `pbg/core/losses.py:89-94`. The dataclass field is `training`, not `train`.
For a 100-iteration config with `c_h=0.01`, this audit observed coefficient
`0.0` at iteration 50; the intended value is `0.005`. Entropy becomes zero from
iteration 1 onward.

This affects both `ppo` and `rb_ppo` because both call the same helper.

**Required fix.** Read `cfg.training.iters`, retain support for a direct
`LossConfig` only if intentionally public, define endpoint semantics, and add
iteration 0/midpoint/final tests for both composites.

## B-05: strict configuration is not value-strict

**Repair status (2026-07-16): resolved in the working repair series.** The
loader now rejects nonfinite, sign-invalid, interval-invalid, and zero-work
values across rollout, loss, training, model, and hardware fields with dotted
paths. Tests cover every listed counterexample plus PPO/value epsilons,
coefficients, Adam betas, seeds/output paths, model dimensions, and optional
hardware limits. Cross-field checks now also cover lossless history capacity,
attention/profile limits, the configured player count, and compact-deck folder
geometry.

**Contract.** `CONTRACTS.md:40-43` says missing fields, wrong types, invalid
values, and unknown keys fail during recursive strict loading with a dotted path.

**Evidence.** Unknown keys, required fields, types, enums, branching beta,
completion samples, several optional positive integers, optimizer name,
reference interval, and hardware limits are validated. The following invalid
values were accepted by `_validate_config` in this audit:

| Field | Accepted invalid example | Consequence |
|---|---:|---|
| `rollout.node_budget` | `0` | late backend/tree failure |
| `loss.lam` | `2.0` | invalid normal lambda return; late RB rejection |
| `training.iters` | `0` | no training work |
| `training.gd_steps_per_iter` | `0` | empty/invalid optimization schedule |
| `training.lr` | `-1.0` | late optimizer error |
| `training.grad_clip` | `-1.0` | invalid clipping rule |
| `training.checkpoint_every` | `0` | modulo-by-zero at checkpoint condition |

Additional finite/range validation is absent for PPO epsilons, value/entropy
coefficients, Adam betas, minibatch size positivity as a general schema rule,
and several hardware optional bounds.

**Required fix.** Validate all numeric fields at load time, including finiteness,
sign, interval, cross-field, game/model history, deck/model width, and CUDA
profile requirements. Add one dotted-path rejection test per boundary.

## M-01: hot-path host work violates residency and observability claims

**Repair status (2026-07-16): resolved in the working repair series.** Public
history packing now uses a logarithmic binary-lift ancestor table and one
batched device gather; CUDA tensors use the preflighted lossless bound without
reading a realized device length. Folded summaries now live in a fixed,
memory-planned device-indexed cache. Cache-hit rows cross the sibling facade
with a false evaluator row mask and finite dummy inputs, then merge by tensor
mask, so no cache key, hit decision, or field concatenation crosses the host.
CUDA policy metrics use host-known family bounds, actual-cost family calls no
longer transfer an unhelpful constant width vector, and the remaining mixed-row
planner transfers one explicitly counted fixed control block. A sync-debug
test covers complete preflop construction, a one-folder postflop construction,
and a cache hit; the static guards also cover the actual folded-adapter input
validation. The test passes on an RTX 2080 Ti with no requested synchronization.

**Contract.** `CONTRACTS.md:193-196` and `docs/architecture.md:109-114` say hot
numeric tensors remain on device, declared scheduler-control synchronization is
counted, and undeclared synchronization is forbidden.

**Evidence.** At least these production paths escape that accounting:

- `_packed_history` reads `lengths.max().item()` on every model input and then
  reconstructs history in a Python loop over the dynamic width, issuing indexed
  assignments for every history position (`rollout_backend.py:587-630`).
- `_partition_summary` evaluates a CUDA boolean and copies every cache key to a
  Python list, uses Python dictionaries, and concatenates cached fields row by
  row (`rollout_backend.py:1579-1621`). It is reached for postflop model calls
  because `PostflopNet.requires_partition_summary = True`.
- policy planning/recording contains additional `.tolist()` and `.item()` calls;
  some actual-cost width transfers are intentionally declared, but model-call
  metric reads and the two paths above are not recorded as scheduler control.

The audit smoke reported `execution_undeclared_host_synchronizations: 0`, which
therefore reflects missing instrumentation, not absence of synchronization. The
CUDA sync-debug test covers alias compaction and attention only; it does not
exercise full policy input construction.

The history loop is also an obvious scaling inefficiency: its kernel-launch and
Python work grows with trajectory length, precisely where B-02 requires support
up to hundreds of events.

**Required fix.** Replace parent-walk packing with a device-resident batched
gather/scan or materialized bounded history representation, and replace the
Python folder cache with device-indexed cache metadata/batched gather. Until
then, explicitly instrument every unavoidable host control transfer and extend
sync-debug coverage to full preflop/postflop policy forward construction.

## M-02: the Python training API can launch CUDA without a hardware profile

**Repair status (2026-07-16): resolved in the working repair series.**
`run_training` rejects a selected CUDA device before reproducibility setup or
bundle construction when `cfg.hardware` is absent. A Python-API regression test
proves the bundle builder is not reached.

`CONTRACTS.md:187-190` says CUDA launch requires a hardware profile. The CLI
supplies `local_3x2080ti` by default, but `pbg.runtime.train(load_config(path))`
can choose CUDA with `cfg.hardware is None`. `_hardware_metadata` records null
limits and training proceeds with fallback precision/default behavior.

Require a profile in `run_training` whenever the selected device is CUDA, or
narrow the contract to CLI-only enforcement. The safer production choice is a
runtime check so library callers cannot bypass memory/token limits.

## M-03: active documents contain stale or contradictory claims

**Repair status (2026-07-16): resolved in the working repair series.** The
active theory now parameterizes cards, combos, and canonical classes by
$D=4R$, $\binom{D}{2}$, and $R^2$ while retaining full-deck values only as
examples. `core_math.tex`, `NOTATION.md`, and the model document agree that
exact legal-action reference KL is implemented, whereas optional public moment
output heads are structural only and have no registered target, loss, or
branching controller. Poker/architecture/contracts document the bounded device
history gather, device-indexed folded-summary cache, and single counted
mixed-row control block. The older readiness report is explicitly a superseded
evidence snapshot, and the top-level README limits this release to engineering
and estimator correctness rather than strategy-quality claims.

- Existing readiness says production is accepted; B-01 through B-05 disprove
  that release conclusion.
- `README.md` and `docs/poker.md` advertise reduced decks without disclosing the
  production rejection.
- `CONTRACTS.md` overstates invalid-value rejection and device residency.
- `core_math.tex:1437` says reference-policy KL is deferred, while the code and
  `NOTATION.md` implement it.
- `NOTATION.md:326-327` says public moment heads are deferred, while optional
  `MomentHeads` exist. They have no registered loss or branching control, so
  they are structurally present but not a trainable production feature.

Resolve after the implementation boundary is chosen; documentation should not
be edited to declare success before matching tests exist.

## M-04: change risk is concentrated in oversized modules

**Repair status (2026-07-16): resolved in the working repair series.** Common
reduced-gradient synchronization and finite-step control now
live in `pbg/core/optimization.py`; device history materialization and folded
summary caching now live in `pbg/games/poker/model_inputs.py` and
`pbg/games/poker/partition_cache.py`; hardware-bounded physical policy planning
lives in `pbg/games/poker/policy_planning.py`; and atomic save/load, complete RNG
capture, scientific-config identity, and metrics-prefix validation live in
`pbg/core/checkpointing.py`. Compatibility imports preserve the existing public
training API, and checkpoint/training plus planner behavior tests pass after the
extraction.

The package boundary is good: generic core does not import poker, evaluator
ownership is clean, and composition roots are explicit. However,
`pbg/core/training.py` is 1,934 lines and
`pbg/games/poker/rollout_backend.py` is 1,730 lines. Each combines construction,
planning, instrumentation, runtime state, and multiple algorithm phases.

This is not a rewrite request for v1. After blockers are fixed, extract cohesive
units for optimization/finite-step control, checkpoint lifecycle, model input
packing, and partition-cache handling. Keep behavior tests around the extraction.

## M-05: no strategy-quality release probe exists

`docs/todo/BR_probe.md` correctly records that duplicate-deal evaluation is not
an approximate best-response or exploitability probe. The current arena can
compare checkpoints head to head, but a broken or stagnant self-play strategy
can still checkpoint, resume, and pass execution tests.

This need not block an engineering-only MVP if explicitly accepted, but it
blocks any claim that the shipped trainer is learning a strategically useful or
improving policy. Add a bounded approximate-BR trend probe before using strategy
quality as a production acceptance claim.
