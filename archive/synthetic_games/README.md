# Retired synthetic games

This directory is an inert historical snapshot of the former synthetic-game,
Kuhn, and Leduc implementation. These games were retired because their scale
is not representative of the GPU-oriented NLHE research target.

The snapshot preserves the old source, exact exploitability evaluator, configs,
dedicated tests, design notes, and implementation prompt in recognizable
subdirectories. It is deliberately outside the `pbg` package and `tests`
directory:

- it is not importable as `pbg.games.synthetic`;
- it is not registered by active configuration or training code;
- it is not collected by normal CI; and
- it receives no compatibility or maintenance work.

The files may contain imports referring to their former active locations. They
are historical evidence, not a runnable secondary distribution. Third-party
papers remain under `docs/references/` and were not moved here.
