# Implementation notes

## P00 — repository skeleton and configuration

- Added the complete package, config, test, and evaluation directory skeleton.
- Kept all future-prompt modules to ownership docstrings only.
- Implemented strict recursive YAML-to-dataclass loading with dotted-path errors.
- Kept synthetic and poker sub-configurations as game-owned mappings.
- Poker root configs require the `poker_gpu` completion boundary.
- Converted YAML optimizer beta lists to the frozen tuple representation.
- `seed_all` returns explicit Torch and NumPy generators; NumPy globals are untouched.
- Metrics are append-only JSONL; wandb remains an import-guarded optional sink.
- Added permanent AST-based import-direction test T12.

## P01 — synthetic family and exact tools

- Added colex private types, collision-free deals, and a deduplicated SoA state table.
- Implemented capped limit betting, public chance events, fold/showdown chip rewards.
- Final chance prefixes are pre-terminal; fold endings remain terminal only.
- Added collision-conditioned reach updates and exact/rejection posterior sampling.
- Exact tools enumerate public trees and profiles in float64 with the 2M guard.
- Best responses share actions across opponent profiles at each information set.
- Closed-form Kuhn equilibria at alpha 0, 1/6, and 1/3 evaluate at numerical zero.
- Added the synthetic embedding/MLP, categorical adapter, and table exporter.
- CONTRACT-QUESTION: Standard specified Kuhn rules give 9 total public states and 5 terminal histories, not the prompt's 9 non-root decisions and 6 terminals.
- CONTRACT-QUESTION: P00 required core/policy_adapter.py to remain empty, so P01 cannot use the frozen core PolicyOutput without violating file ownership; a local identical DTO is used.

## P02 — public NLHE engine and card representations

- Added the frozen lexicographic combo map and the standard AA-to-22 13x13 class map.
- Complement draws use random-key permutations, giving uniform ordered samples without replacement on CPU or CUDA.
- Added an immutable public NLHE state table with blind/action order, affine raise-to ticks, full-raise reopening, short-all-in handling, street chance transitions, and fold terminals.
- Showdowns remain pre-terminal; public contribution levels produce nested affine pot strata and no local private-hand settlement logic.
- The engine accepts caller-supplied chance cards and canonicalizes the flop as one unordered event.

## P03 — ranges and Bayesian bookkeeping

- Stored one fp32 log-likelihood row per rollout slot and player, initialized uniformly.
- Action updates gather only the observed event and stabilize row gauges below -30.
- Board deals apply compatibility masks without modifying surviving hypotheses.
- Fold snapshots clone linear weights, the fold-time board, and an 8-float public metadata row.
- Metadata is caller-populated because ranges have no dependency on the public engine.
- Branch clones append independent reach, live, board, metadata, and stability rows.
- Class aggregation normalizes combo rows on demand and scatter-sums the standard 169 map.

## P04 — evaluator partition-summary boundary

- Added typed batches for fold snapshots and opaque evaluator summary fields.
- The adapter lazily imports and version-checks only the stable sibling facade.
- Local -1 board padding is mapped to facade-owned uint8/255 padding on-device.
- Live and enabled folded likelihood rows are concatenated without normalization.
- Evaluator result tensors are shape/device/dtype checked and returned unchanged.
- No partition, blocker, collision, removal-set, or private-assignment math is local.

## P05 — poker-private evaluator adapters

- Added game-neutral single/batched completion DTOs, an opaque reach view, and the completion protocol.
- Completion maps public engine boards, dealt/contender status, stabilized unnormalized reach rows, affine pot strata, and explicit-generator seeds to the sibling facade.
- Evaluator completion statistics and diagnostics are validated and retained without local reconstruction.
- Exact profile sampling makes one batched facade call, preserves folded blocker seats, and raises a typed diagnostic error on any partial failure.
- Fixed showdown validation checks only shape, eligibility, tie shares, and mask consistency; poker winner selection remains entirely evaluator-owned.
- All three operations lazily import and require the stable `pbg-v2` public facade, with no local evaluator or CUDA fallback.

## P07 — postflop model and poker public-tree adapter

- Added generic padded compact-row maps with differentiable gather/scatter and representation-row permutation helpers.
- Added the combo-level postflop transformer with supplied compatibility maps, fixed-width output scatter, opaque evaluator-summary projection, optional fold-line features, autocast support, and public moment heads.
- Added consistent suit relabeling for boards and combo-indexed tensors; it performs only representation transforms.
- Added exact composition of contextual passive/call/raise-tick laws over engine-supplied legal events, including full-row probabilities, stipulated-profile row selection, sampling, entropy, and KL.
- Added mixed-street model routing with preflop class-to-combo lifting at the bundle boundary.
- Added the public poker facade for exact-profile delegation, complement chance draws, public encoding/reach updates, owned range slots, and evaluator-mediated fixed-showdown settlement.
- P07 focused tests: 9 passed (`tests/test_p7_postflop.py`, `tests/test_p7_adapter.py`, seed values 70/73 where stochastic).
- Full suite after P07: 60 passed, 3 skipped in 10.20s on CPU; skipped tests are optional sibling-boundary/device cases.
