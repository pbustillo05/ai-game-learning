# P04 — Evaluator-owned partition/folded-card summary boundary

**Depends on:** P00, P02, P03.
**Files:** `pbg/games/poker/folded.py`; `tests/test_p4_folded.py`.
**May touch:** append `IMPLEMENTATION_NOTES.md`.

## Goal

Provide typed DTOs and a thin adapter for partition/blocker summaries computed
by `poker_gpu_evaluator`. This repository must not implement disjoint private
assignment partition functions, removal-set expansions, folded-player blocker
approximations, or Monte Carlo validation oracles.

## API

```python
@dataclass
class FoldSnapshotBatch:
    reach_weights: Tensor   # float32 [B,F,1326], unnormalized action likelihoods
    board_at_fold: Tensor   # uint8 [B,F,5], 255 padded
    line_meta: Tensor       # float32 [B,F,D]
    folder_mask: Tensor     # bool [B,F]

@dataclass
class PartitionSummaryBatch:
    fields: dict[str, Tensor]
    metadata: dict[str, object]

class PokerGPUPartitionAdapter:
    def summarize(
        self,
        *,
        board_cards: Tensor,
        live_range_weights: Tensor,
        folded: FoldSnapshotBatch,
        query_combo_ids: Tensor | None,
    ) -> PartitionSummaryBatch: ...
```

The adapter lazily imports only `API_VERSION`, `PartitionSummaryResult`, and
`partition_many` from `poker_gpu_evaluator.pbg`, checks `API_VERSION`, preserves
device and contiguity, maps padding conventions, and returns sibling fields
without mathematical recomputation. Refresh on street/fold events remains
caller-owned.

The model may consume evaluator fields as learned features. Local code may
reshape, scatter, mask, normalize for neural-network numerics, or project those
fields; it may not derive them from card-collision laws.

## Tests

- Fake-facade contract tests for exact names, shapes, dtypes, devices, metadata,
  lazy imports, version errors, and no NumPy/CPU round trip on CUDA.
- Mapping tests prove every output field is copied/scattered from a stipulated
  sibling result, not recomputed.
- Static scan forbids local `Phi`, `alpha_B`, `beta_B`, `lambda_g`, partition
  enumeration, removal-set algebra, disjoint-assignment Monte Carlo, and local
  imports of sibling internals.
- Real-sibling smoke is dependency-skipped when unavailable. Mathematical
  correctness and CPU/CUDA agreement are tested in the sibling repository.

## Do not

- Do not reproduce `Docs/folder_impact.md` as executable code.
- Do not add a local reference backend or small-deck brute-force oracle.
- Do not sample private hands or decide whether private assignments collide.
