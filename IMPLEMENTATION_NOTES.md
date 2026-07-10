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
