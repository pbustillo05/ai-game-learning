# Rao--Blackwell variance experiments

This directory preserves the checkpoint-specific sequential and branched
Rao--Blackwell variance experiments run in July 2026. They are historical
research workflows, not production estimators and not part of the installed
`pbg` package.

The modules intentionally retain their original formulas and checkpoint/run
contracts for reproducibility. Production code must not import this package.
Future variance measurements should consume sealed production outputs instead
of adding another target or rollout implementation here.

The original scientific designs are preserved under `docs/`.

Run the preserved unit tests explicitly from the repository root:

```bash
/home/pablob/.venvs/poker/bin/python -m pytest -q experiments/rao_blackwell_variance/tests
```

The server-safe historical launchers are under `tools/`. Their output remains
under the original `runs/` paths so existing manifests and evidence retain
their meaning.
