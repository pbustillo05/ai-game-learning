# Final core pipeline

Status: Phase 11 cutover closed against validated and audited executable
candidate `71a92378cf89a81ed5724ba61515fa6c9f441706`.

## Authority

The repository has one production engine. Generic execution lives under
`pbg/core/`; NLHE-specific composition lives under
`pbg/games/poker/queued/`. The former isolated replacement package and former
generic core implementation are deleted. Package initializers do not retain
compatibility exports.

The final ownership chain is:

```text
strict RootConfig
  -> immutable ModeSchema + CapacityManifest
  -> generated fixed CUDA owners
  -> semantic family FIFOs
  -> model / sample / transition publication
  -> one topology and LayerPlan
  -> evaluator-v2 Leaf -> LayerWork completion
  -> one layered target executor
  -> aligned learner presentation
  -> immediate loss/backward
  -> dense FP32 optimizer transition
  -> strict checkpoint / metrics / consumer report
```

`pbg.core.contracts` generates the twelve rows from:

```text
{self_play, best_response}
× {sequential, branched.fixed, branched.KKT}
× {sampled, rao_blackwell}
```

No aggregate mode, compatibility alias, runtime engine selector, implicit
device, attention fallback, or capacity growth path is accepted.

## Generic/game boundary

The core receives only generic execution bounds through `ExecutionGeometry`
and game-supplied opaque physical fields through `PhysicalLayout`. It does not
import poker. NLHE owns players, stacks, ranks, cards, actions, retained
history, private profiles, folded summaries, posterior sampling, evaluator
bindings, and model architecture.

The poker hook publishes final opaque leaf names:

- `leaf_public_state`;
- `leaf_private_profile`;
- `trained_actor`; and
- `response_cutoff`.

Evaluator contract-v2 names such as `board_cards`, `combo_ids`, and
`selected_player` exist only at the direct binding boundary.

## Storage and execution

`SoASchema`, `MonotoneSlab`, `ExactBlock`, and `GeneratedOwners` are the sole
allocation vocabulary. Capacities are immutable and admitted before rollout.
Reservations publish atomically, failure state is lane-local, retries discard
whole unpublished attempts, and every owner releases at its declared last
consumer.

`RolloutTransaction` is the sole model/sample/commit path. Sequential and
branched chance loops both use `BoundedChanceClosure`; no independent chance
generation loop remains. `Topology`, `LayerPlan`, and
`LayeredTargetExecutor` are the sole structural and target authorities.

Sampled leaves call showdown evaluation directly. Rao--Blackwell leaves use
the evaluator completion contract, including selected completion for trained
roles. Posterior/profile sampling and folded-summary construction remain in
this repository; they do not pass through an evaluator compatibility facade.

## Model and accelerator contract

Production training, evaluation, benchmarks, variance, and evaluator-dependent
inference require CUDA. Distributed training is CUDA/NCCL-only. Supported MPS
inference remains accelerator-resident and fails before evaluator-dependent
work that MPS does not implement. CPU is limited to small orchestration,
configuration, hashing, checkpoint I/O, and reporting.

Attention execution has exactly two explicit strategies: `sdpa` and
`chunked_dense`. CUDA and MPS honor the configured strategy identically. CPU
attention rejects before projection or neural work. Retired `auto` and `flex`
selectors are rejected.

Scientific model construction requires the canonical fields `players` and
`epsilon`. It rejects the retired or redundant inputs `n_players`, `eps`,
`k_act`, `use_branching_heads`, and `max_history_tokens`; the latter three are
derived once inside the final model factory.

## Consumers and persistence

Ordinary, distributed, and replica training all compose
`pbg.runtime.TrainingSession`. Best-response evaluation, variance, training
benchmarks, inference, export, and inference benchmarks consume that same
strict configuration and final model authority. Checkpoints and inference
artifacts accept only their current schemas and canonical scientific mapping.
Conversion is offline only.

## Deletion and recovery

Phase 11 deleted:

- the old core and isolated replacement namespace;
- legacy poker implementation modules and superseded tests/configuration;
- adapters, aliases, fallbacks, stale checkpoint paths, and test-only
  production helpers;
- comparison and migration tools;
- tracked prototype/benchmark result trees and stale implementation notes.

Historical audit reports remain as evidence only. Recovery is Git history, not
a second engine or runtime compatibility path.

## Verified closure

The final candidate passed:

- 441 of 441 indexed production definitions reachable from 31 accepted roots,
  with zero review candidates;
- 312 collected tests, with 310 passing and two expected environment skips on
  one visible CUDA device;
- the explicit two-GPU NCCL asymmetric-failure gate;
- strict attention and artifact-alias adversarial probes;
- a deterministic matched-work public benchmark with 30,920,192 bytes peak
  allocated and 52,428,800 bytes peak reserved; and
- initial/final GPU baselines of 6/6/16 MiB with no compute processes.

Independent validation and a different read-only authority audit both passed
the exact executable SHA. Detailed commands and timings are recorded in
`docs/audit/queued_phase11_report_20260728.md`.
