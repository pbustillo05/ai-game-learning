# `ai-game-learning` ↔ `poker-gpu-evaluator` IO contract

**Status:** implemented `pbg-v2` integration contract. This document mirrors the
shipped facade documented by `../poker-gpu-evaluator/docs/api_pbg.md`. The
evaluator document is authoritative for its implementation; this copy records
the consumer-side ownership rules and adapter obligations.

## 1. Scope and ownership

`poker-gpu-evaluator` must provide exactly four production operations to this
repository:

1. batched Monte Carlo completion of a hold'em spot, including private-hand
   posterior marginals and payoff moments; and
2. batched rejection sampling of exact joint private-hand profiles;
3. batched partition/blocker summaries for model conditioning; and
4. batched exact settlement of fixed final-board showdowns, stratified by
   pot eligibility.

The evaluator owns card/combo validation, collision-conditioned private
assignment and posterior/partition computations, uniform public runouts inside
completion, hand ranking, tie splitting, per-stratum winner determination,
Monte Carlo weights/reductions, CUDA dispatch, and estimator diagnostics.

This repository owns public game-state navigation, action-likelihood/reach-row
bookkeeping, policies, pot amounts and commitment accounting, conversion from
engine state to the tensors below, rollout RNG allocation, chip-EV assembly,
and all learning logic. It may store or transform evaluator-produced
hand-indexed tensors, but it must not implement private-card collision laws,
partition/folded-card approximations, hand ranking, tie resolution, or
per-stratum winner selection. Evaluator-produced partition summaries may cross
the boundary as model inputs; no local implementation of their poker math is
permitted.

The sibling repository must be installable as a normal Python package (editable
installation is sufficient during development), exposing:

```python
from poker_gpu_evaluator.pbg import (
    API_VERSION,
    COMBO_CARDS,
    CompletionConfig,
    CompletionResult,
    JointProfileResult,
    PartitionSummaryResult,
    ShowdownResult,
    complete,
    evaluate_showdowns,
    partition_many,
    sample_joint_profiles,
)
```

`API_VERSION` is the literal string `"pbg-v2"`. The local adapter checks it
before its first production call and raises an actionable incompatibility
error rather than attempting a legacy fallback. It also compares the exported
`COMBO_CARDS` against the local representation map before making a production
call.

Importing the package must not initialize CUDA or compile an extension. CUDA
loading/JIT may happen on the first call with `backend="cuda"`.

## 2. Shared representation

Dimensions:

- `B`: independent public spots.
- `P`: padded seats, `2 <= P <= 9`; it is constant within a call.
- `H = 1326`: unordered two-card combos.
- `K`: side-pot strata, `1 <= K <= P` for `complete`.
- `M`: requested joint profiles for `sample_joint_profiles`.

Cards use `card_id in [0, 51]`, with `rank = card_id // 4` and
`suit = card_id % 4`. A board is always `[B, 5]` `torch.uint8`; undealt suffix
slots are `255`. Valid real-card counts are 0, 3, 4, or 5. Real cards are unique
and precede every sentinel.

Combos use the following frozen lexicographic enumeration:

```python
combo_id = 0
for a in range(52):
    for b in range(a + 1, 52):
        combo_cards[combo_id] = (a, b)
        combo_id += 1
```

This is the enumeration used by `poker-gpu-evaluator`. `COMBO_CARDS` is a
defensive copy of its frozen uint8 `[1326,2]` table. The local
`hand_index`/`hand_cards` implementation is checked against that exported table
for all 1326 rows.

All tensors are `torch.Tensor`. Inputs must be contiguous. All tensor inputs in
one call must be on one device, except `seed`, which may be CPU or on that same
device. Outputs are contiguous and returned on the primary input's device.
There must be no NumPy conversion or device round trip in the CUDA path.

Floating inputs are `torch.float32`; floating outputs and accumulations are at
least float32. Indices are `torch.int32`, masks are `torch.bool`, board cards are
`torch.uint8`, and seeds are `torch.int64` interpreted as their 64-bit bit
patterns. NaN, infinity, negative range weight, duplicate cards, bad padding,
and wrong dtype/shape are hard errors.

## 3. Monte Carlo completion API

### 3.1 Signature

```python
@dataclass(frozen=True)
class CompletionConfig:
    n_samples: int
    n_partials: int = 8
    backend: str = "auto"       # "auto" | "cpu" | "cuda"

@dataclass
class CompletionResult:
    p_hat: torch.Tensor          # float32 [B, P, 1326]
    v_hat: torch.Tensor          # float32 [B, P, 1326]
    vvec_hat: torch.Tensor       # float32 [B, P]
    u_hat: torch.Tensor          # float32 [B]
    n_eff: torch.Tensor          # float32 [B]
    equity_share: torch.Tensor   # float32 [B, P, 1326, K]
    output_valid: torch.Tensor   # bool [B, P, 1326]
    sample_count: torch.Tensor   # int64 [B]
    estimator_stddev: torch.Tensor  # float32 [B, P, 1326], NaN if unavailable
    metadata: dict[str, object]

def complete(
    *,
    board_cards: torch.Tensor,       # uint8 [B, 5]
    range_weights: torch.Tensor,     # float32 [B, P, 1326], every dealt seat
    dealt_mask: torch.Tensor,        # bool [B, P], includes folded seats
    contender_mask: torch.Tensor,    # bool [B, P], live/all-in showdown seats
    stratum_players: torch.Tensor,   # bool [B, K, P]
    payoff_baseline: torch.Tensor,   # float32 [B, P]
    stratum_amount: torch.Tensor,    # float32 [B, K]
    seed: torch.Tensor,              # int64 [B]
    config: CompletionConfig,
) -> CompletionResult: ...
```

`backend="auto"` chooses CUDA iff the primary tensors are CUDA tensors;
otherwise CPU. `backend="cuda"` with CPU primary inputs is an error, as is
`backend="cpu"` with CUDA primary inputs. The call operates on the current
PyTorch CUDA stream and must not globally synchronize the device.

### 3.2 Input semantics

`dealt_mask[b,p]` means seat `p` has a private hand and participates in the
collision-conditioned target distribution. It includes live, all-in, and folded
seats and excludes only empty/padded seats. `contender_mask[b,p]` selects the
live/all-in seats for showdown eligibility; it does not suppress posterior or
value outputs for folded dealt seats. Every
contender is dealt; each spot needs at least two contenders. Folded rows retain
their fold-time action-likelihood weights and block private/runout cards even
though their payoff is deterministic. Undealt rows are ignored and normally
zero.

`range_weights` are nonnegative, unnormalized posterior/reach weights for every
dealt seat. The implementation first zeros combos intersecting the board. Every
dealt row must then have positive mass. Multiplying any dealt row by an independent
positive scalar must not change any result except floating roundoff.

`stratum_players[b,k,p]` says whether contender seat `p` is eligible to win
stratum `k`. Every positive-amount stratum must have at least one eligible
contender. Zero-amount strata are allowed and return zero contribution.
Strata need not be sorted, though this repository will provide nested
side-pot strata.

The evaluator computes a showdown share `S[p,k]`: zero if ineligible; otherwise
1 divided by the number of tied best eligible hands when `p` ties for best,
and zero when it loses. The per-sample payoff consumed by the learning code is
defined exactly as:

```text
R[p] = payoff_baseline[p] + sum_k stratum_amount[k] * S[p,k]
```

Normally `payoff_baseline[p] = -committed_total[p]`; stratum amounts include
chips contributed by folded players. Supplying these affine coefficients does
not transfer pot-accounting ownership to the evaluator.

### 3.3 Target distribution and estimator

For a spot with current board `D`, sample mutually disjoint dealt-seat combos
`x` and the ordered missing board runout `t` with target mass

```text
q(x,t) proportional to
    product_{p dealt} range_weights[p, x_p]
    * 1[x is pairwise disjoint and disjoint from D]
    * Unif(t | deck minus D minus cards(x)).
```

Runout order is street order; only the resulting final five-card board affects
showdown. The promoted estimator is sequential importance sampling: order
dealt seats deterministically by descending concentration, sample each row
from its currently legal masked distribution, multiply the standard masked
normalizer importance factors, then sample a uniform residual runout. A
statistically equivalent unbiased/consistent implementation is acceptable,
but `metadata["estimator"]` must name it and `n_eff` must correspond to its
actual weights. Samples with no legal continuation have zero weight.

The result fields are the self-normalized estimates:

```text
p_hat[p,h] = sum_i w_i 1[x_i,p=h] / sum_i w_i
v_hat[p,h] = sum_i w_i 1[x_i,p=h] R_i[p]
             / sum_i w_i 1[x_i,p=h]
vvec_hat[p] = sum_i w_i R_i[p] / sum_i w_i
u_hat = sum_i w_i sum_p R_i[p]^2 / sum_i w_i
n_eff = (sum_i w_i)^2 / sum_i w_i^2
equity_share[p,h,k] = E[S[p,k] | x_p=h]
```

`u_hat` includes all `P` payoff entries. Folded seats participate in private
sampling as blocker rows but retain their deterministic engine payoff through
`payoff_baseline` (normally negative committed chips); padded seats have a zero
baseline. Thus `vvec_hat` includes folded-seat expected/deterministic payoff.

For every dealt seat, including folded seats, `p_hat` sums to one over valid
combos (within 1e-5), and
`vvec_hat[p]` must agree with `sum_h p_hat[p,h] * v_hat[p,h]` within MC/reduction
roundoff. For a folded seat, `v_hat[p,h] = payoff_baseline[p]` on every
observed/valid combo because its payoff is deterministic, while its `p_hat`
remains the collision-conditioned posterior needed by earlier decision nodes.
Only undealt/padded seats have zero combo-indexed outputs and zero
`vvec_hat`. Invalid/unobserved combo cells are zero;
`output_valid` identifies combo cells whose conditional estimates have a
positive denominator. `estimator_stddev` is for `v_hat`; unavailable cells are
NaN, while `v_hat` itself remains zero there.

`sample_count[b]` is the number of attempted estimator lanes contributing to
diagnostics (normally `n_samples`). A spot with zero total estimator weight is
a hard `DegenerateRangeError` naming its batch indices; returning NaNs as a
successful result is forbidden.

### 3.4 Reproducibility and metadata

`seed[b]` defines an independent spot-local stream. Results must be repeatable
bit-for-bit for identical inputs, build, device model, backend, and seed.
CPU/CUDA and different GPU architectures need only agree statistically. A
batch permutation, with inputs and seeds permuted together, must permute
outputs without changing per-spot values.

Metadata must contain only cheap host scalars/strings (no device tensors), at
least: `backend`, `estimator`, `n_samples`, `n_partials`, `kernel_ms`,
`rng_algorithm`, `api_version="pbg-v2"`, and evaluator package/git version.

## 4. Partition-summary API

The sibling's existing neutral partition API is promoted into the stable facade:

```python
@dataclass
class PartitionSummaryResult:
    base_log_z: torch.Tensor                    # float32 [B]
    single_card_log_ratio: torch.Tensor         # float32 [B,52]
    pair_card_log_ratio: torch.Tensor | None    # float32 [B,52,52]
    output_valid: torch.Tensor                  # bool [B]
    metadata: dict[str, object]

def partition_many(
    *,
    board_cards: torch.Tensor,       # uint8 [B,5], 255 padded
    range_weights: torch.Tensor,     # float32 [B,S,1326]
    row_mask: torch.Tensor,          # bool [B,S]
    field_order: int = 2,
    backend: str = "auto",
) -> PartitionSummaryResult: ...
```

It computes collision-conditioned assignment mass and finite-difference blocker
summaries. `single_card_log_ratio[c]` is `log Z(D∪{c})-log Z(D)` and the pair
field is the non-additive two-card correction under the sibling's documented
neutral partition semantics. Live and folded action-likelihood rows may be
supplied as rows; their public status affects local model metadata, not the
partition law. Scaling each active range row independently by a positive scalar
may change `base_log_z` but not the ratio fields.

This repository may cache, reshape, scatter, decompose, or learn projections of
these returned tensors. It may not compute them from private-card collision
rules, enumerate disjoint assignments, or provide a local mathematical oracle.
Validation, device, metadata, and CPU/CUDA conventions match the other facade
operations. Partition summaries are deterministic and take no seed.

The initial implementation accepted `query_card_sets`, `query_combo_ids`, and
`seed` but ignored them. Those arguments are not part of the shipped signature
and were removed in evaluator package 0.3.0. Consumers must not pass them.

## 5. Exact fixed-showdown API

### 5.1 Signature

```python
@dataclass
class ShowdownResult:
    winner_mask: torch.Tensor      # int32 [B, K], original-seat bit mask
    showdown_share: torch.Tensor   # float32 [B, K, P]
    metadata: dict[str, object]

def evaluate_showdowns(
    *,
    board_cards: torch.Tensor,       # uint8 [B, 5], all five cards real
    combo_ids: torch.Tensor,         # int32 [B, P], -1 for inactive seats
    active_mask: torch.Tensor,       # bool [B, P]
    stratum_players: torch.Tensor,   # bool [B, K, P]
    backend: str = "auto",
) -> ShowdownResult: ...
```

This is the stable facade over the sibling repository's existing exact batch
evaluator. The implementation should rank every active hand once, then derive
the winner set independently for each eligibility stratum; it must not rerank
once per stratum. The existing `batch_eval` CUDA/Metal/reference implementations
are the implementation seed, not APIs this repository may import directly.

`winner_mask[b,k]` uses original padded seat indices: bit `p` is set iff seat
`p` ties for the best hand among `stratum_players[b,k]`. A positive stratum must
have at least one eligible active seat. For a one-player stratum that player's
bit is set without requiring a rank comparison. Inactive, folded, padded, or
ineligible seats can never win that stratum.

`showdown_share[b,k,p]` is `1 / popcount(winner_mask[b,k])` for set winner bits
and zero otherwise. It is returned so tie splitting also remains evaluator-owned.
For every valid stratum it sums to one over players. This repository multiplies
these shares by its locally owned `stratum_amount` and adds its locally owned
`payoff_baseline`; neither pot amounts nor chip payoffs cross into this API.

All cards across the final board and active combos must be unique. Validation,
device, stream, contiguity, reproducibility, lazy CUDA initialization, and error
conventions match `complete`. Metadata includes at least `backend`, `kernel_ms`,
`api_version`, and evaluator package/git version. The CPU and CUDA paths expose
this identical tensor signature.

### 5.2 Ownership requirement

No code in this repository may compare poker hand ranks, categories, kickers,
or rank integers, including in tests or a "reference" backend. Tests use the
sibling CPU implementation or a fake facade returning stipulated masks/shares.
Local code may only assemble chip payoffs arithmetically from
`showdown_share`, `stratum_amount`, and `payoff_baseline`.

## 6. Exact joint-profile rejection API

### 6.1 Signature

```python
@dataclass
class JointProfileResult:
    combo_ids: torch.Tensor       # int32 [B, M, P], -1 for undealt seats
    accepted: torch.Tensor        # bool [B, M]
    attempt_count: torch.Tensor   # int32 [B, M]
    acceptance_rate: torch.Tensor # float32 [B]
    metadata: dict[str, object]

def sample_joint_profiles(
    *,
    board_cards: torch.Tensor,    # uint8 [B, 5]
    range_weights: torch.Tensor,  # float32 [B, P, 1326]
    dealt_mask: torch.Tensor,     # bool [B, P], includes folded seats
    n_samples: int,               # M > 0
    seed: torch.Tensor,           # int64 [B]
    max_attempts: int = 100_000,  # per requested profile
    backend: str = "auto",
) -> JointProfileResult: ...
```

Validation, device, combo, board, scaling, backend, and stream conventions are
the same as `complete`. One dealt seat is allowed here; padded/undealt rows are
ignored.

### 6.2 Distribution

For every `(b,m)`, independently propose one combo per dealt seat from that
seat's board-masked normalized `range_weights` row. Accept the whole profile
iff all proposed private cards are pairwise disjoint. Retry that `(b,m)` only.
Consequently every accepted profile is an exact unweighted draw from

```text
Pr(x) proportional to product_{p dealt} range_weights[p,x_p]
                 * 1[x pairwise disjoint and disjoint from board].
```

Do not use sequential masked sampling here: unless corrected, it has the wrong
joint law for branching. Seat proposal order must not affect the accepted law.

`attempt_count[b,m]` includes the accepted attempt, or equals `max_attempts`
on failure. Failed `combo_ids[b,m,:]` are all `-1` and `accepted[b,m]` is
false. `acceptance_rate[b]` is accepted proposals divided by total proposals
over that spot. The evaluator function returns partial failure rather than
raising; this repository's adapter raises `AcceptanceError` if any requested
profile failed, including failed indices and measured acceptance rates.

Identical seeds have the reproducibility guarantee in §3.4. Streams must be
indexed by spot, requested sample, and attempt so retrying one lane cannot
perturb another lane.

Metadata must include `backend`, `max_attempts`, `kernel_ms`, `rng_algorithm`,
`api_version="pbg-v2"`, and evaluator package/git version.

## 7. Adapter responsibilities in this repository

`pbg/games/poker/mc.py` is a thin adapter, not a second CUDA implementation.
It must:

1. convert `-1` local board padding to evaluator `255` padding without changing
   real card ids;
2. verify `COMBO_CARDS` matches the local lexicographic combo representation;
3. preserve or stably pack every dealt seat, including folded blocker rows, and
   scatter every dealt-seat output back to `P_max`;
4. construct nested `stratum_players`, `payoff_baseline`, and
   `stratum_amount` from `NLHEEngine` commitments/status;
5. exponentiate local log-reach rows after subtracting each row maximum, pass
   float32 unnormalized weights, and never normalize hypothesis-by-hypothesis;
6. allocate one int64 seed per spot by consuming the supplied local
   `torch.Generator`; no global RNG is used;
7. map `CompletionResult` field-for-field into `BatchedCompletionStats`;
8. map `JointProfileResult.combo_ids` into the PBG profile tensor, retaining
   sampled combos for folded dealt seats and keeping `-1` only for
   undealt/padded seats;
9. carry `PartitionSummaryResult` fields into model DTOs without recomputing
   poker quantities;
10. pass all fixed-showdown eligibility strata in one facade call and map
   `ShowdownResult` shares into affine local payoff assembly; and
11. turn evaluator validation/degeneracy/failure into the local typed errors
   with batch indices and diagnostic context.

The adapter must not import private modules such as
`poker_gpu_evaluator.pbg.cuda`, mutate `sys.path`, shell out, load shared
objects by path, or copy CUDA sources into this repository.

## 8. Acceptance tests across repositories

Before `poker_gpu` may be the shipped poker backend, CI must cover:

- all 1326 combo ids round-trip identically between repositories;
- board padding conversion for preflop/flop/turn/river;
- sharp-support exact enumeration on turn and river for `p_hat`, `v_hat`,
  `vvec_hat`, `u_hat`, and each stratum of `equity_share`;
- multiway histories with a folded dealt seat, verifying that its `p_hat`
  remains a normalized collision-conditioned marginal and its `v_hat` is the
  deterministic fold payoff rather than a zeroed output;
- tie boards, board-plays ties, and 2–4 nested side pots with folded chips;
- fixed-showdown winner masks/shares agree across CPU/CUDA for varied
  eligibility strata, and each active hand is ranked only once per spot;
- partition-summary ratios agree with sibling exact/reference tests and cross
  the local adapter field-for-field;
- row-scale invariance over scales from `1e-20` through `1e20`;
- independent-rejection marginals and collision legality versus exact
  enumeration, including nonuniform overlapping ranges;
- seeded repeatability, batch-permutation independence, and no use of global
  torch/NumPy RNG state;
- CPU/CUDA statistical agreement using confidence bounds fixed before the run;
- inactive/padded seats, zero-mass rows, impossible assignments, invalid cards,
  partial rejection failure, and explicit CUDA-without-device errors;
- CUDA outputs remain CUDA tensors, run on the caller's current stream, and do
  not introduce a host-device-host copy in a profiler smoke test; and
- an end-to-end poker rollout/completion smoke in this repository using only
  the public `poker_gpu_evaluator.pbg` import.

## 9. Implemented facade notes

The installable `poker_gpu_evaluator.pbg` facade now provides the four
operations and result types above on CPU and CUDA. Legacy evaluator APIs are
not integration surfaces for this repository. In particular, local code must
not import private evaluator modules, use NumPy-returning range-equity helpers,
or recover only an acting-hand marginal when a full joint profile is required.

Evaluator package 0.3.0 removed the three formerly ignored partition arguments
listed in §4 and added the defensive public `COMBO_CARDS` export. `API_VERSION`
remains `"pbg-v2"`; therefore consumers must follow the exact shipped signature
rather than infer optional arguments from the API-version string alone.
