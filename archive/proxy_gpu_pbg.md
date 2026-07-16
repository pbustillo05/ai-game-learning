# `poker_gpu_evaluator.pbg` API (`pbg-v2`)

This is the authoritative evaluator contract intended to be copied or linked by the consuming repository. The evaluator owns deterministic or explicitly seeded poker math; the consumer owns game navigation, policies, pot construction, and chip-EV assembly.

## Imports and representations

```python
from poker_gpu_evaluator.pbg import (
    API_VERSION, FOLDER_API_VERSION, COMBO_CARDS,
    CompletionConfig, CompletionResult, DegenerateRangeError,
    JointProfileResult, PartitionSummaryResult, ShowdownResult,
    FolderSummaryResult, complete, evaluate_showdowns, partition_folders,
    partition_many, sample_joint_profiles,
)
assert API_VERSION == "pbg-v2"
assert FOLDER_API_VERSION == "pbg-folder-v1"
```

Cards are `uint8` IDs `0..51`, with `rank = card // 4` and `suit = card % 4`. Boards are `[B,5]`; undealt suffix positions are `255`, and valid real-card counts are 0, 3, 4, or 5. `COMBO_CARDS` is a defensive copy of the frozen lexicographic enumeration of every `(a,b)` with `a < b`.

All inputs are contiguous torch tensors. Floating inputs are `float32`, combo indices are `int32`, masks are `bool`, and seeds are `int64`. Inputs share one device except that seeds may remain on CPU. `backend="auto"` follows the primary tensor; explicit `cpu`/`cuda` requires that device. Importing the package never initializes CUDA or compiles an extension.

## Completion

```python
complete(
    *, board_cards,                 # uint8 [B,5]
    range_weights,                  # float32 [B,P,1326]
    dealt_mask, contender_mask,     # bool [B,P]
    stratum_players,                # bool [B,K,P]
    payoff_baseline,                # float32 [B,P]
    stratum_amount,                 # float32 [B,K]
    seed,                           # int64 [B]
    config=CompletionConfig(n_samples, n_partials=8, backend="auto"),
) -> CompletionResult
```

Every dealt seat, including a folder, participates in the collision-conditioned private assignment. Contenders are a subset of dealt seats. Strata may contain contenders only. Completion returns `p_hat[B,P,1326]`, `v_hat[B,P,1326]`, `vvec_hat[B,P]`, `u_hat[B]`, `n_eff[B]`, `equity_share[B,P,1326,K]`, `output_valid[B,P,1326]`, `sample_count[B]`, per-hand `estimator_stddev`, and metadata. A folder's value is its deterministic baseline while its sampled hand still blocks cards. Positive independent rescaling of a dealt range row does not change conditioned outputs.

## Exact joint profiles

```python
sample_joint_profiles(
    *, board_cards, range_weights, dealt_mask,
    n_samples: int, seed, max_attempts: int = 100_000,
    backend: str = "auto",
) -> JointProfileResult
```

The result contains `combo_ids[B,M,P]` (`-1` for undealt or failed lanes), `accepted[B,M]`, `attempt_count[B,M]`, `acceptance_rate[B]`, and metadata. Each attempt independently proposes every dealt row and accepts only a pairwise/board-disjoint whole profile, producing the exact product-range law conditioned on collisions. Exhausting one lane returns partial failure rather than raising.

## Neutral partition summaries

```python
partition_many(
    *, board_cards,                 # uint8 [B,5]
    range_weights,                  # float32 [B,S,1326]
    row_mask,                       # bool [B,S]
    field_order: int = 2,           # 1 | 2
    backend: str = "auto",
) -> PartitionSummaryResult
```

The deterministic result contains `base_log_z[B]`, `single_card_log_ratio[B,52]`, optional symmetric `pair_card_log_ratio[B,52,52]`, `output_valid[B]`, and metadata. `query_card_sets`, `query_combo_ids`, and `seed` were accepted but ignored by the initial implementation and were removed in package 0.3.0; consumers must delete those arguments.

### Model-ready folded summaries (`pbg-folder-v1`)

```python
partition_folders(
    *, board_cards,                 # uint8 [B,5]
    range_weights,                  # float32 [B,F,1326]
    row_mask,                       # bool [B,F]
    beta_rank: int = 16,            # 0..52
    backend: str = "auto",
) -> FolderSummaryResult
```

This versioned extension performs the combined and per-folder partition work
behind one facade call. It returns the combined base/unary/pair fields; signed
top-magnitude beta eigenvalues/vectors; per-folder log mass, unary/pair fields,
unnormalized card marginals, centered deltas, and covariance matrices; the
summed covariance and tight-cluster scalar; validity masks; and metadata with
`folder_api_version="pbg-folder-v1"`. The consumer may cache a compact subset
until the next street or fold event, but does not derive these poker statistics.

## Exact stratified showdown

```python
evaluate_showdowns(
    *, board_cards,                 # complete uint8 [B,5]
    combo_ids,                      # int32 [B,P], -1 iff inactive
    active_mask,                    # bool [B,P]
    stratum_players,                # bool [B,K,P]
    backend: str = "auto",
) -> ShowdownResult
```

All active private and board cards must be unique. Each stratum is nonempty and contains active seats only. The evaluator ranks each active hand once, then returns original-seat `winner_mask[B,K]`, equal `showdown_share[B,K,P]`, and metadata. Every valid share row sums to one.

## Errors and reproducibility

Wrong shapes, dtypes, devices, padding, cards, duplicates, negative/nonfinite weights, inconsistent masks, and zero board-legal active rows are hard errors. Completion raises `DegenerateRangeError` with batch indices when sampled legal mass is zero. Identical inputs and seeds are repeatable within the same backend and package version; CPU and CUDA RNG sequences are not promised to match each other. Metadata always includes backend, `api_version`, evaluator package version, and git version, plus operation-specific estimator, RNG, timing, and limit fields.
