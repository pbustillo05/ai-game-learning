# Reachability cleanup — 2026-07-22

## Result

The high-confidence immediate-deletion inventory from the 2026-07-21 audit is
implemented. The static audit now indexes 492 production definitions, of which
477 are reachable, and reports 15 review candidates. Before cleanup it indexed
526 definitions and reported 43 candidates.

This pass deleted only surfaces for which repository-wide references,
entry-point reachability, tests, and the authoritative audit disposition agreed.
It did not archive complete supported products or remove migration paths whose
replacement is not yet the sole production implementation.

## Deleted or moved out of production

- Deleted the unused `pbg.core.distributed` and `pbg.core.logging_util` modules.
- Deleted predecessor learner allocation, mode/target compatibility aliases,
  unused memory helpers, completion views, observability counters, and the
  unused staging and categorical wrappers.
- Moved `DiagnosticTreeBuilder` into a test-only fixture and deleted the unused
  tree alias sampler implementation.
- Deleted obsolete full-deck card/class wrappers and constants where active
  consumers already use `DeckGeometry` directly.
- Deleted unused model diagnostics and ablation helpers: tick probability
  reconstruction, suit-row permutation, explicit attention-weight inspection,
  the private SDPA duplicate, and the redundant storage-width assertion.
- Deleted the unused branched-variance geometry loader and stale engine
  constants.

Tests were updated to exercise the canonical production APIs directly rather
than keeping predecessor APIs alive for test convenience.

## Deliberately retained candidates

- `occurrence_schema`: required by the authoritative storage design and slated
  to become the sole schema.
- `pbg.eval.arena`: a documented and directly tested offline checkpoint
  comparison product, not dead code merely because training does not call it.
- `weighted_ratio_moments` and `parameter_jvp`: public experiment utilities
  supporting the active variance-analysis surface.
- `pbg.games.poker.model.__getattr__`: dynamic implementation of the declared
  lazy model exports.

The arena-dependent `hand_cards`, `board_conflict_mask`, and
`sample_uniform_complement` wrappers also remain until an explicit decision is
made to archive that complete product.

## Validation

- Focused cleanup regression set: 144 passed, 11 skipped.
- Full CPU suite in the restricted sandbox: 448 passed, 32 skipped; the only
  three failures were local Gloo socket creation denied by the sandbox.
- Distributed tests rerun outside that restriction: 4 passed.
- Production and test trees compile successfully, `git diff --check` passes,
  and the post-cleanup reachability audit has no unclassified immediate-delete
  candidate.

No CUDA behavior changed and no GPU benchmark was needed for this static
deletion pass.
