# Scope, contracts, and method

## Verdict and scope

This is a repository-wide, read-only implementation audit at commit
`b55df976e80b82c474c86202c90f3b32072af03c`, compared historically from
`537700f5b94873c3daa6e6a03ad00cfef4b8e88b`. It covers the active Python
package, configs, tools, tests, documentation, `runs/`, Git history, and the
read-only sibling evaluator boundary. Archive content was used as provenance,
not as production authority.

## Authority order

1. `docs/theory/core_math.tex` governs sampling laws, estimators, targets, and
   losses. The extracted contract is [batch 00](batches/00_math_and_modes_contract.md).
2. `docs/theory/pseudocode/` governs ownership, schemas, physical planning,
   queues, memory, distributed order, and failure behavior. The extracted
   `STORE-001`–`STORE-036` and `TEST-STORE-*` contract is
   [batch 01](batches/01_pseudocode_storage_contract.md).
3. `archive/refactor/implementation_notes.md` is provenance only. Where it and
   current pseudocode differ, current pseudocode controls.
4. The historical improvement contract and run provenance are in
   [batch 02](batches/02_historical_regression_contract.md).
5. [Batch 13](batches/13_adversarial_finding_reconciliation.md) controls
   conflicts among findings. [Batch 12](batches/12_validation_evidence.md)
   controls claims about what actually ran.

BR is not specified by `core_math.tex`: no response class, seat law, stopping
rule, cutoff semantics, or reported estimator is authoritative. C03 is
therefore a certification blocker, not evidence by itself that the implemented
equation is wrong.

## Classification vocabulary

- **Confirmed defect:** current call graph contradicts an applicable contract.
- **Authority gap:** implementation cannot be certified because the governing
  contract is absent or ambiguous.
- **Static performance risk:** code structure implies synchronization,
  unbounded work, or extra allocation; magnitude is not measured.
- **Measured historical baseline:** artifact-backed historical observation,
  not a current comparison.
- **Validation gap:** required execution evidence was not obtained.
- **Optimization:** a semantics-preserving opportunity whose benefit remains
  unmeasured unless explicitly stated otherwise.

## Method

Specialized passes extracted contracts, traced config/entry points, inspected
rollout/storage/queues, targets/learner/distribution, poker/evaluator, model,
evaluation, reachability, five end-to-end modes, history, and performance. A
final adversarial pass rechecked high-severity claims against source and the
production call graph. Static reachability combined imports, entry points,
direct calls, dynamic hooks, configs, tests, and documentation; name search
alone was not accepted as proof.

The audit intentionally made no production changes. Tiny validation followed
the repository's GPU hygiene rules; no large run or benchmark was attempted.

## Important rejected overclaims

- Fixed-beta production **does** require and train T/C. Only the lower-level
  rollout helper/schema remains locally permissive.
- Production `post_A` planning and execution are both conservative flop-width.
  Street-aware packing is C17, an optimization—not proven under-admission.
- `capture_probe_leaf` is dynamically reached and must not be deleted.
- Three storage labels do not violate the five-semantic-combination
  requirement.
- No current measured performance regression was established.
