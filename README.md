# Prompt Suite README — how to run GPT 5.6 through this build

## Protocol

1. **Fresh context per prompt.** Paste, in order: `PREAMBLE.md`, then the
   current repo's `IMPLEMENTATION_NOTES.md` (it stays short — it's the only
   cross-prompt memory), then the prompt file. Nothing else. Do NOT paste
   `core_math.tex` / `blocker_model.md` / `folder_impact.md` — every needed
   formula is already restated inside the prompts.
2. **Order:** P00 → P01 → P02 → P03 → P04 → P05 → P06 → P07 → P08 → P09 →
   P10 → P11 → P12 → P13. Safe parallel tracks if you want to run two
   sessions: {P02→P03→P04→P05→P06→P07} (poker) and {P01→P08→P09→P10→P11}
   (generic core, tested on synthetic) after P00; they only meet at P12/P13.
3. **After each prompt:** run `pytest -q` yourself; skim the diff against the
   review checklist below; resolve any `CONTRACT-QUESTION` comments (edit
   CONTRACTS/NOTATION if the question is right, tell the next session via
   IMPLEMENTATION_NOTES if it's wrong); commit. Never start the next prompt
   with a red suite.
4. **If a session goes off the rails** (renamed interfaces, invented
   abstractions): discard the diff and re-run the prompt; do not repair
   drifted code by hand — repaired drift compounds.

## Contract deltas vs the signed-off CONTRACTS v0.1 (flagging per our rule)

- **D1.** `games/synthetic/model.py` added (SyntheticNet + adapter + policy
  table exporter): the generic loop needs *some* model for synthetic games;
  the poker model is poker-owned. (P01.)
- **D2.** `PBGGame.sample_profiles(state_refs, reach_rows, n, gen)` added to
  the Protocol: branching needs joint posterior profile draws generically;
  poker routes it to the rejection backend, synthetic implements it exactly.
  (P01/P09.)
- **D3.** `OldOutCache` concretized as `PolicyCache` living in `core/tree.py`,
  storing per-stored-child per-type log-probs + old V only (not full
  PolicyOutput — full raw payloads at 1326 rows would be prohibitive). KL
  reference terms re-forward a frozen ref checkpoint at minibatch time
  instead of caching ref outputs. (P08/P11.)
- **D4.** `ReachCache` lives in `core/tree.py` (not `completion.py`) to keep
  file ownership clean; `completion.py` keeps only `ReachView` ferrying.
  (P08.)
- **D5.** Engine deals chance cards via a `deal(state_refs, cards)` side
  channel rather than enumerating chance outcomes as engine events (the
  outcome space is huge); the adapter samples the cards. (P02/P07.)
- **D6.** Node-budget overflow in branched mode degrades to spine-only
  continuation with a metrics flag (deterministic), rather than erroring.
  (P09.)
- **D7.** `branched_value` takes `max_resident_rows` kwarg instead of a new
  config field. (P10.)

If any delta is unwelcome, say so before its prompt runs; each is localized.

## Per-prompt review checklist (what to eyeball in the diff)

- **P00**: config strictness actually rejects unknown keys; import test walks
  files, not import side effects.
- **P01**: Kuhn/Leduc arise from pure configs (grep for "kuhn" in family.py —
  must be absent); exact Kuhn equilibrium test uses the closed form.
- **P02**: min-raise/no-reopen rule; side-pot fixture with 3 unequal stacks;
  chip conservation test actually randomizes.
- **P03**: fold snapshots taken before masking; no per-hypothesis renorm.
- **P04**: raw-weight accumulators in the removal algebra (the
  double-normalization bug is THE risk here); small-deck brute-force test
  present and not weakened; q_kl ≤ 0 raises.
- **P05**: fp32 accumulators; SIS weight formula matches the prompt; n_eff.
- **P06**: temperature on content only (find the division site); dlogistic
  normalization exact.
- **P07**: no [1326,1326] tensors (grep allocations); compact scatter
  round-trip test; adapter entropy exact over composed events.
- **P08**: three reach modes agree test; PolicyCache vs recompute test; W1 —
  profiles drawn per event draw and discarded.
- **P09**: T3 lemma test present with real statistics; sequential
  byte-identity guard; controller state in checkpoint.
- **P10**: reference implementation is genuinely independent (path
  enumeration, not the same recursion copy-pasted); lam sweep in T4;
  capture-before-mix order untouched.
- **P11**: three weighting levels in rb_ppo (P̂ν / P̂ / bare); T5 variance
  ratio reported; checkpoint resume exactness.
- **P12**: duplicate deals actually swap seats; CI paired.
- **P13**: only the marked core edit; acceptance checklist numbers pasted.

## When the codebase is done

Bring the final `IMPLEMENTATION_NOTES.md`, the T5/T3/T4 outputs, and the
Leduc exploitability curve back to Claude for the review pass (step 5 of the
original plan), together with any CONTRACT-QUESTIONs that were deferred.
