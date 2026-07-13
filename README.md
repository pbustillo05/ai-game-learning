# Prompt Suite README — how to run GPT 5.6 through this build

## Protocol

1. **Fresh context per prompt.** Start a new workspace-aware context and submit
   exactly one active prompt file. Each active prompt contains a
   `Fresh-context bootstrap` section telling the agent which repository files
   to inspect; no prior prompt text or chat transcript is required. The agent
   must read those files from the workspace rather than asking you to paste
   them. `docs/latex/core_math.tex` remains authoritative if a transcription
   discrepancy is found, and `POKER_GPU_EVALUATOR_IO.md` is binding whenever a
   prompt names the sibling facade.
2. **Correction order for the current repository:** C00 → C08 → C09 →
   C10 → C11 → P12 → P13A → P13B. P09–P11 remain as historical
   implementation prompts; C09–C11 explicitly supersede their obsolete
   Poisson/dedup/value/loss contracts.
3. **After each prompt:** run `pytest -q` yourself; skim the diff against the
   review checklist below; resolve any `CONTRACT-QUESTION` comments (edit
   CONTRACTS/NOTATION if the question is right, tell the next session via
   IMPLEMENTATION_NOTES if it's wrong); commit. Never start the next prompt
   with a red suite.
4. **If a session goes off the rails** (renamed interfaces, invented
   abstractions): discard the diff and re-run the prompt; do not repair
   drifted code by hand — repaired drift compounds.

## Active versus historical files

- Launch now, in order: `C00_math_contract.md`, `C08_device_rollout.md`,
  `C09_labeled_branching.md`, `C10_posterior_moments.md`,
  `C11_minimal_training.md`, `P12_eval.md`, `P13A_gpu_poker_rollout.md`, and
  `P13B_integration.md`.
- `P00`–`P11` document the already-built foundation. They are retained for
  provenance and tests, but are not part of the current launch sequence.
- In particular, never run P08–P11 after their C08–C11 corrections; the C
  prompt is the complete replacement contract.

## Contract deltas vs the signed-off CONTRACTS v0.1 (flagging per our rule)

- **D1.** `games/synthetic/model.py` added (SyntheticNet + adapter + policy
  table exporter): the generic loop needs *some* model for synthetic games;
  the poker model is poker-owned. (P01.)
- **D2.** `PBGGame.sample_profiles(state_refs, reach_rows, n, gen)` added to
  the Protocol: branching needs joint posterior profile draws generically;
  poker routes it to the rejection backend, synthetic implements it exactly.
  (P01/P09.)
- **D3.** `OldOutCache` is a compact policy cache aligned to one stable
  decision-row order. Reference-policy KL is deferred. (P08/C11.)
- **D4.** `ReachCache` lives in `core/tree.py` (not `completion.py`) to keep
  file ownership clean; `completion.py` keeps only `ReachView` ferrying.
  (P08.)
- **D5.** Engine deals chance cards via a `deal(state_refs, cards)` side
  channel rather than enumerating chance outcomes as engine events (the
  outcome space is huge); the adapter samples the cards. (P02/P07.)
- **D6.** Node-budget overflow is a loud capacity error; it never changes the
  sampled offspring law. (C08/C09.)
- **D7.** `branched_value` takes `max_resident_rows` kwarg instead of a new
  config field. (P10.)
- **D8.** All poker-private combinatorics use `poker_gpu_evaluator.pbg`:
  completion, exact joint profiles, partition/folded-card summaries, and exact
  fixed-showdown winner shares per pot stratum. There are no local poker
  mathematical reference implementations. This repo retains public-tree
  navigation, policies, pot amounts, and chip-EV assembly. (P00–P13.)

If any delta is unwelcome, say so before its prompt runs; each is localized.

## Per-prompt review checklist (what to eyeball in the diff)

- **P00**: config strictness actually rejects unknown keys; import test walks
  files, not import side effects.
- **P01**: Kuhn/Leduc arise from pure configs (grep for "kuhn" in family.py —
  must be absent); exact Kuhn equilibrium test uses the closed form.
- **P02**: min-raise/no-reopen rule; pot-stratum fixture with 3 unequal stacks;
  no hand-rank/category/payoff evaluator symbols.
- **P03**: fold snapshots taken before masking; no per-hypothesis renorm.
- **P04**: partition-summary DTO/adapter maps sibling fields without deriving
  poker quantities locally; no removal algebra or local oracle.
- **P05**: thin sibling adapters have no NumPy/device round trip; rejection
  returns full profiles; fixed showdown returns masks/shares per stratum.
- **P06**: temperature on content only (find the division site); dlogistic
  normalization exact.
- **P07**: compact scatter round-trip; adapter entropy exact over composed
  events; sampled actor combo only selects a row; partition inputs are opaque.
- **P08**: three reach modes agree test; PolicyCache vs recompute test; W1 —
  profiles drawn per event draw and discarded.
- **C09**: randomized rounding, duplicate-labeled abstract children retained,
  uniform `1/K` edges, and terminal-distribution test.
- **C10**: P/M rather than direct conditional-value averaging; O(K)
  leave-one-out oracle; Q absent unless explicitly requested.
- **C11**: policy uses child posterior and emitted A; value/entropy use parent
  posterior; stable decision-row indexing; checkpoint resume exactness.
- **P12**: duplicate deals actually swap seats; CI paired; every fixed showdown
  crosses the evaluator facade.
- **P13A**: CPU-engine parity plus no host transfers in the GPU rollout path.
- **P13B**: typed GameBundle, fixed-beta presets, device-residency acceptance.

## When the codebase is done

Bring the final `IMPLEMENTATION_NOTES.md`, the T5/T3/T4 outputs, and the
Leduc exploitability curve back to Claude for the review pass (step 5 of the
original plan), together with any CONTRACT-QUESTIONs that were deferred.
