# P00 — Repository skeleton, config system, logging, test scaffolding

**Depends on:** nothing (first prompt).
**Files:** the entire directory skeleton of the PREAMBLE layout (empty modules
with module docstrings only), plus full implementations of `pbg/core/config.py`,
`pbg/core/logging_util.py`, `pbg/core/game.py` (constants + `PBGGame` Protocol
only), `pyproject.toml`, `configs/kuhn_smoke.yaml` (stub values),
`tests/test_p0_config.py`, `tests/test_p0_imports.py`, `IMPLEMENTATION_NOTES.md`.
**May touch:** nothing else.

## Goal

A repo that imports cleanly, loads configs strictly, seeds deterministically,
logs metrics to jsonl, and enforces the import-direction rule — before any
algorithmic code exists.

## Spec

1. **Packaging.** `pyproject.toml` with `pbg` as an installable package
   (setuptools, `pip install -e .`), deps: `torch>=2.2`, `numpy`, `pyyaml`,
   `pytest`. No other dependencies without a CONTRACT-QUESTION.

2. **`pbg/core/config.py`.** Implement the nested dataclasses exactly as named
   in PREAMBLE §Config. Requirements:
   - `load_config(path: str | Path) -> RootConfig`: strict — any YAML key that
     does not map to a dataclass field raises `ConfigError` naming the key and
     its path; missing keys use dataclass defaults; missing keys without
     defaults raise.
   - Nested parsing is recursive over dataclass fields; dicts-of-floats
     (`beta_fix` etc.) pass through as `dict[str, float]`.
   - `seed_all(seed: int, rank: int = 0) -> torch.Generator`: seeds torch (CPU
     and CUDA if available), numpy default_rng is NOT seeded globally — instead
     return `(torch_gen, np_gen)` tuple, both seeded with `seed*1000+rank`.
     Signature: `seed_all(seed, rank) -> tuple[torch.Generator, np.random.Generator]`.
   - `RootConfig.model` is `dict` (game-owned; parsed later by game code).

3. **`pbg/core/game.py`.** Constants `CHANCE=-1, TERMINAL=-2, PRETERM=-3` and
   the `PBGGame` Protocol verbatim from PREAMBLE (use `typing.Protocol`,
   `runtime_checkable`). No implementations.

4. **`pbg/core/logging_util.py`.**
   ```python
   class MetricsLogger:
       def __init__(self, out_dir: Path, use_wandb: bool = False): ...
       def log(self, step: int, metrics: dict[str, float]) -> None: ...
       def close(self) -> None: ...
   ```
   Always appends one JSON object per call to `out_dir/metrics.jsonl`
   (`{"step": ..., **metrics}`); wandb is optional and import-guarded (repo must
   work without wandb installed).

5. **Skeleton modules.** Every file in the PREAMBLE layout exists with a
   one-paragraph module docstring stating which prompt owns it (e.g. `"""Owned
   by P10. Backward value propagation. See CONTRACTS §5-P10."""`) and no code.
   Copy `CONTRACTS.md`/`NOTATION.md` placeholders: create empty files with a
   header line `# populated at repo setup` (the real documents are added by the
   maintainers).

6. **`configs/kuhn_smoke.yaml`.** A syntactically complete config exercising
   every field (values are placeholders; `game.name: synthetic`).

## Tests

- `test_p0_config.py`: round-trip load of `kuhn_smoke.yaml`; unknown-key raises
  `ConfigError` mentioning the key path; `seed_all` determinism (two generators
  with same seed produce identical `torch.rand` / `np` draws; different ranks
  differ).
- `test_p0_imports.py` (**this is T12**, permanent): walk `pbg/` with `ast`,
  parse imports, assert the PREAMBLE import-direction rule. Must be written so
  it keeps passing as files gain content.
- Logger test: two `log` calls produce two valid JSON lines with the given keys.

## Do not

- Do not implement any game, tree, or training logic.
- Do not add config fields beyond the PREAMBLE list.
- Do not make wandb a hard dependency.
