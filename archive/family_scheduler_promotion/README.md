# Family-scheduler promotion archive

This directory preserves the migration notebook, retired implementation plans,
raw benchmark JSON, and one-off launch tooling used to promote the sole current
rollout implementation on 2026-07-16.

Names such as `fifo`, `queued`, `wavefront`, executor versions, Kuhn, Leduc,
and removed configuration fields are historical experiment identities. They do
not describe supported runtime APIs. Nothing here is imported by `pbg` or
collected by the configured test suite.

- `experiment_log_2026-07-16.md` explains the promotion measurements.
- `results/` contains their immutable raw JSON artifacts.
- `plans/` contains superseded migration and optimization plans.
- `tools/` contains a superseded one-off timed launcher.
