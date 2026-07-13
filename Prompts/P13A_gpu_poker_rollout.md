# P13A — GPU-native poker rollout backend

**Depends on:** P02–P07, C08, C09, C11.
**Files:** `pbg/games/poker/rollout_backend.py`,
`pbg/games/poker/tensor_state.py`; `tests/test_p13a_gpu_rollout.py`.
**May touch:** poker adapter/model routing, `IMPLEMENTATION_NOTES.md`.

## Fresh-context bootstrap

This prompt is self-contained; prior prompt text is not required. Before
editing: run `git status --short`; preserve unrelated changes; read
`PREAMBLE.md`, `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, `CONTRACTS.md`,
`POKER_GPU_EVALUATOR_IO.md`, and the poker implementation notes at the end of
`docs/latex/core_math.tex`; compare the binding contract with the shipped API
in `../poker-gpu-evaluator/docs/api_pbg.md`; inspect the completed C08/C09
interfaces and all P02–P07 poker sources/tests. The implemented 0.3.0+
partition call has only `board_cards`, `range_weights`, `row_mask`,
`field_order`, and `backend`; never pass the removed query or seed arguments.
Run the CPU oracle and focused poker baselines first. Implement the backend,
parity/device tests, and notes in this session, then run focused and full
suites.

## Goal

Implement the poker `RolloutBackend` without Python state-table lookups or host
transfers. Retain `NLHEEngine` as the independent CPU correctness oracle.
The tensor backend itself must run on both CPU and CUDA using the same Torch
code: CPU exercises it in ordinary CI, while CUDA validates residency and
throughput when available.

## Tensor state

Implement `PokerTensorStateStore(capacity, max_rollout_steps, device)` as
preallocated structure-of-arrays storage addressed by the C08 int32 state
handles. Required fields are:

```text
stacks, committed_street, committed_total     float32 [capacity,P]
status, acted, reopen_mask                    bool/int8 [capacity,P]
actor, street, button, n_raises, phase        int32 [capacity]
board                                         int16 [capacity,5], -1 padded
last_full_raise, current_bet                   float32 [capacity]
action_history                                int32 [capacity,max_rollout_steps]
history_length                                int32 [capacity]
fold_payoff                                   float32 [capacity,P]
```

Every append allocates rows with device prefix sums and returns int32 handles;
no Python list or host-side free list is allowed. `action_history` preserves the
full public action sequence rather than relying on a collision-prone hash.

Implement batched tensor operations for:

- roots/blinds and button rotation;
- legal fold/check-call/affine raise ticks;
- full-raise reopening and short all-ins;
- player transitions and next-actor selection;
- chance transitions and complement board draws;
- fold terminals, completion-cut classification and pot strata;
- reach updates, public encoding and fixed stage-dependent `beta_fn`.

`beta_fn` maps player states to `preflop|flop|turn|river` and chance states to
`chance_to_flop|chance_to_turn|chance_to_river`; it applies an exact
`cfg.rollout.beta_fix` override or the required `default`. Leaves are never
queried. The returned tensor is fp32 on the state device.

Poker `EventBatch` rules:

- player event: `public_id` is the existing fold/check-call/raise-tick id and
  `payload={}`;
- chance event: `payload["cards"]` is padded dealt cards `[B,3]` (`-1` suffix for turn/
  river), and `public_id` is a deterministic int32 colex label of the sorted
  new card set with a street-disjoint offset. Equal card sets therefore have
  equal labels without requiring host lookup.

Completion-cut means no future player action. Incomplete-board all-ins and
closed betting are PRETERM; completion owns the remaining runout. Fold wins are
TERMINAL.

## Policy routing and queues

`policy_forward(frozen_model, states, reaches)` constructs all model inputs and
routes them without host access.

It gathers state-store rows by handle, constructs `PokerPublicBatch`, range and
partition inputs on device, invokes exactly one street-family model per queue,
and returns combo-indexed policy/value rows aligned to the input handles.

- Maintain one preflop queue because it uses the 169-class model.
- Share one postflop queue across flop/turn/river using board masks.
- Maintain one postflop queue for each `n_live` value from 2 through P. At each
  dispatch choose the largest ready queue; when a frontier wave ends, flush all
  nonempty queues. Do not create separate street queues.
- Accumulate frontiers from many trees before forwarding; never perform a
  model call per tree or node.
- Preflop outputs are lifted to 1326 combo rows at this boundary.
- Folded summaries, joint profiles and completion use only public sibling
  facade tensors and preserve device/stream. Use the existing facade adapter's
  `COMBO_CARDS` compatibility check and exact evaluator 0.3.0+ partition
  signature; do not duplicate either boundary in the rollout backend.

Equal public labels may share immutable tensor payload indices, but their C09
sampled-tree nodes and descendant RNG streams remain distinct.

## Multi-GPU and RNG

- One independent backend/tree pipeline per DDP rank/GPU.
- Explicit device-local generators and rank-derived seeds.
- Never move trees or rollout states between GPUs.
- CPU is permitted only for test-oracle comparison and exceptional diagnostics.

## Tests

- Random legal trajectories match `NLHEEngine` step-for-step for legal masks,
  raise targets, states, completion cuts, contributions and chip conservation.
- Cover short all-ins, reopen rules, side pots, folds, all-in runouts and every
  street transition.
- CUDA tests assert all outputs remain on device and sibling calls receive the
  original device tensors.
- Fake-facade partition instrumentation rejects any obsolete
  `query_card_sets`, `query_combo_ids`, or `seed` keyword.
- AST/static scan rejects `.cpu()`, `.numpy()`, `.tolist()` and hot-path `.item()`.
- Instrumented model records large preflop/postflop batch calls and no per-node
  forwards.
- Report achieved batch-size histograms by preflop and postflop `n_live`; do
  not impose hardware-specific throughput thresholds in CI.

Required commands:

```bash
pytest -q tests/test_p2_engine.py tests/test_p5_completion.py tests/test_p7_adapter.py
pytest -q tests/test_c08_device_rollout.py tests/test_c09_labeled_branching.py
pytest -q tests/test_p13a_gpu_rollout.py
pytest -q
```

## Do not

- No local collision sampler, hand evaluator, winner logic, Python-engine
  fallback inside the tensor backend, or
  flop/turn/river queue split without measured justification.
