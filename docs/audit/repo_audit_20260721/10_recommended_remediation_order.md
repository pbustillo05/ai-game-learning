# Recommended remediation order

## Verdict

Do not start with bulk deletion or optimization. First settle the missing user
policy choices, then repair the two blockers and authoritative storage path,
then migrate/delete compatibility and non-core products, then establish strict
validation and performance gates.

## User decisions required before implementation

1. **BR authority (C03):** choose shared versus independent per-seat response,
   seat sampling/weighting, objective, stopping rule, cutoff semantics, and
   reported exploitability/NashConv estimator. Add this to authoritative math
   and pseudocode before certifying the fifth mode.
2. **Evaluator v2 (C10):** confirm v2 as the sole contract and correct the
   missing `AGENTS.md` authority path (`api_pbg.md` is absent; sibling
   `docs/api.md` says v2). The sibling repository remains read-only.
3. **CPU/MPS/inference scope (C13):** confirm no external supported consumer
   requires CPU/MPS inference or checkpoint-versus-checkpoint arena. Without an
   explicit scope expansion, archive them under the stated GPU-only mandate.
4. **Compatibility policy (C15):** identify checkpoints/configs that must still
   resume, approve one-time offline conversion, and choose a deadline after
   which v2–v4 checkpoint, old config/model, v1 evaluator, and permissive
   benchmark shims are deleted.

## Implementation order

1. Freeze a closed five-semantic selector and one canonical config per
   combination; preserve three shared storage/training labels.
2. Repair C01/C02: direct normative compact seal under one source+destination
   hard transaction, and common-order terminal failure agreement.
3. Repair C05/C06/C07/C09: sole immutable schemas, natural-width storage,
   pre-reserved logical addresses/RNG, bounded child slabs, authoritative queue
   order/accounting, common fixed/KKT pressure correction, and sparse-once
   folded anchors.
4. Enforce CUDA/NCCL at all production and BR entries (C08); fix multi-player
   BR early-fold and selected-owner completion (C04/C16).
5. Require evaluator v2 and current config/checkpoint/model schemas; offline
   convert only approved artifacts (C10/C15).
6. Rebuild benchmark/manifests as strict five-semantic, CUDA-only, required-
   metric gates (C12). Add direct-seal, lifecycle, physical-plan, queue,
   asymmetric-failure, evaluator-v2, and five-mode CUDA tests (C18).
7. Archive C13/C14 products and their configs/tests/docs/tools; delete the
   immediate and post-migration objects in the linked inventory. Repeat symbol,
   import, config, and path audits.
8. Rewrite current README/contracts/architecture/algorithm/file-reference/
   poker/training docs from verified behavior; move historical ledgers and
   concluded TODOs to archive.
9. Remove C11 synchronization risks and run equal-work target/learner gates.
10. Pursue C17, reference-cache, fused rollout, communication overlap, and
    graph optimizations only after ownership and physical shapes are stable.

## Exit criteria

- Exactly five semantic identities classify every retained preset, artifact,
  manifest, test, and benchmark; only three internal storage/loss families
  remain.
- No full legacy tree, post-hoc whole-tree conversion, schema-on-first-write,
  unaccounted queue/folded owner, or logical-work-sized transient remains.
- Every production entry requires NVIDIA CUDA; every semantic workflow has a
  tiny CUDA train/eval/resume gate.
- Failure injection proves all ranks terminate in common order.
- Active package exports contain only the five workflows and required shared
  support; approved historical evidence lives in `archive/`.
- Performance claims use matched work/hardware and clearly distinguish raw,
  documentary, and inferred evidence.
