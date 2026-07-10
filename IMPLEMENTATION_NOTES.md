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

