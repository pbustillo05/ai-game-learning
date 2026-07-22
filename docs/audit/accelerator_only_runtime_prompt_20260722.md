# Implement accelerator-only PBG execution

## Goal

Make `pbg/` accelerator-only. A user must never be able to run training,
best-response evaluation, performance benchmarks, or model inference with CPU
as the execution device. CPU may still perform small orchestration tasks such as
configuration parsing, checkpoint I/O, hashing, deterministic seed creation,
and report formatting, but it must not own rollout state, queues, model
forwards, losses, backward, optimizer state, completion, posterior sampling, or
showdown evaluation.

Supported execution devices are `cuda` and `mps`. Do not expose `cpu` or
`auto`, and never fall back to CPU when an accelerator is absent.

## Authority and current constraint

Before changing core paths, follow `AGENTS.md`,
`docs/theory/core_math.tex`, and the implementation contracts in
`docs/theory/pseudocode/`. Preserve immutable schemas, bounded physical work,
last-consumer ownership, exact estimators, logical RNG laws, and memory
accounting.

The sibling evaluator is read-only. Its current authoritative contract is
`/home/pablob/poker-gpu-evaluator/docs/api.md` (the older `api_pbg.md` path is
absent). Contract v2 supports MPS for only part of the API: batched
`sample_joint_profiles`, `complete`, `complete_linear_moments`, and
`summarize_folders` are currently deferred. Consequently:

- CUDA must support every production workflow.
- MPS should support inference where all required operations exist.
- MPS training must fail clearly at startup until every evaluator operation
  required by the selected training workflow is available on MPS.
- Distributed training must be CUDA/NCCL only. Do not stage MPS gradients or
  collectives through CPU/Gloo.

Do not modify the sibling evaluator as part of this task.

## Required implementation

1. Replace the current completion-backend selector with one explicit execution
   device contract whose only values are `cuda` and `mps`. Prefer one device
   field for the entire run; evaluator dispatch should follow the primary
   tensor device rather than a separate backend string.
2. Validate the requested device and workload capabilities before allocating a
   model, optimizer, rollout store, or evaluator workspace. Reject unavailable
   devices and unsupported MPS workflows with actionable errors.
3. Enforce that model parameters, rollout/storage tensors, forward and backward
   tensors, optimizer state, and all evaluator inputs and outputs use the same
   selected accelerator.
4. Remove CPU selection, CUDA-absent CPU fallback, CPU workspace substitutes,
   CPU memory-admission bypasses, and Gloo training from `pbg/core/training.py`
   and related configuration/runtime code.
5. Make training, BR evaluation, performance benchmarking, variance runners,
   and their CLIs reject CPU and avoid implicit device selection.
6. Rework inference to accept explicit CUDA or MPS, use the tensor-native poker
   state/backend, and perform batched model/evaluator computation on that
   device. Small Python-facing hand/action objects may remain on the host.
7. Delete or archive the parallel CPU/list-backed production implementation
   after migrating every live consumer:
   - `pbg/eval/arena.py`;
   - `pbg/games/poker/ranges.py`;
   - `ChanceDraw` and `PokerGame` in `pbg/games/poker/adapter.py`;
   - `PublicHistoryEvent`, `_State`, `StateTable`, and `NLHEEngine` in
     `pbg/games/poker/engine.py`;
   - the engine-based legacy completion route in `pbg/games/poker/mc.py`.
8. Preserve or relocate shared production objects that the tensor-native path
   still needs, including `PokerConfig`, action/status constants, `PotStrata`,
   the affine showdown-payoff formula, `PokerPolicyAdapter`, and
   `PokerModelBundle`.
9. Remove CPU-specific algorithm and shape branches once no production consumer
   needs them, including CPU-only history trimming, compact-width selection,
   live-seat-width selection, storage checks, and evaluator backend choices.
   Retain necessary CUDA and MPS implementations.
10. Remove obsolete configuration fields, CLI choices, exports, entry points,
    tests, fixtures, documentation, and compatibility code. Do not retain a
    parallel legacy path “for safety.” Update pseudocode/storage documentation
    if any authoritative schema, ownership, or lifetime contract changes.

Start with a fresh reachability inventory because line numbers and the worktree
may have changed. Preserve all unrelated user changes and do not rewrite or
discard them.

## Testing and acceptance

- Add tests proving every public production entry rejects CPU and never falls
  back when CUDA/MPS is unavailable.
- Add static/import tests proving production workflows cannot import or reach
  `NLHEEngine`, `PokerGame`, or `RangeState`, and that public CLIs do not
  advertise `cpu` or `auto`.
- Add same-device assertions around bundle construction and evaluator calls.
- Retain CPU tests only for small device-independent mathematics,
  configuration, checkpoint metadata/I/O, and serialization. CPU execution of
  a production workflow is not valid smoke or acceptance evidence.
- Add CUDA acceptance for all production training semantics and BR evaluation.
  Add MPS inference acceptance. Add an MPS-training startup-rejection test until
  the sibling capability gap is closed.
- Run focused tests first, then the complete relevant suite. Do not substitute
  CPU smoke tests for accelerator tests.
- Before CUDA work, run `nvidia-smi`, select only an idle GPU explicitly, start
  with tiny workloads, preflight memory, avoid OOM, and verify afterward that
  the GPU is idle with no lingering process or unexplained allocation.
- If no suitable accelerator is available, report the unrun accelerator gates;
  never silently run them on CPU.

Acceptance requires one sole accelerator-native production implementation,
clean deletion/reachability searches, passing focused and broad tests, and no
CPU/Gloo execution route for training or inference.

## Completion and commit

Summarize the final device contract, deleted paths, retained host-only work,
tests run, accelerator hardware used, and any deliberately blocked MPS
workflow. Review the diff for unrelated changes, then commit only the changes
made for this task with a clear commit message. Do not include pre-existing
user changes in the commit.
