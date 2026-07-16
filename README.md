# GPU NLHE strategy learning

`pbg` is a GPU-oriented framework for learning strategies in configurable
imperfect-information no-limit Hold'em cash games. NLHE is the sole active
game. Retired Kuhn, Leduc, and synthetic implementations are inert history
under `archive/synthetic_games/`.

## Production surface

- equal-stack NLHE with 2--6 players, 8--128 bb stacks, and 4--13 highest ranks;
- normal PPO, sequential Rao--Blackwell PPO, and branched Rao--Blackwell PPO;
- one family-scheduled rollout implementation across all three modes;
- logical RNG addressing invariant to queue order and physical batching;
- streamed evaluator completion and posterior-mass value reduction;
- exact global decision-mass scaling and flat distributed gradient `SUM`;
- strict configuration, memory preflight, metrics, atomic checkpoint, and resume;
- preflop class and compact postflop attention models; and
- duplicate-deal checkpoint evaluation and same-implementation benchmarking.

This repository owns public poker state, policy, likelihood reaches, and chip
accounting. The read-only sibling `poker_gpu_evaluator.pbg` API owns all joint
private-profile sampling, collision-conditioned completion, folded-card
partition summaries, and showdown comparison. There is no local poker-math
fallback.

## Architecture

```text
pbg.config / pbg.runtime     strict NLHE composition
  pbg.core                   generic execution, rollout, values, loss, training
  pbg.games.poker            NLHE engine, model, tensor backend, evaluator API
  pbg.eval                   offline duplicate-deal arena
```

The active iteration is:

```text
frozen model -> family-scheduled trajectories or one labeled tree/rank
             -> sampled settlement or streamed evaluator completion
             -> compact aligned targets -> globally scaled learner update
```

There is no game selector, rollout executor selector, wavefront compatibility
path, or active legacy-game package.

## Environment

Use Python 3.10+, PyTorch 2.2+, NumPy, PyYAML, pytest, and the sibling evaluator.
On the shared GPU server use `/home/pablob/.venvs/poker`; do not install into
system Python. `tools/poker-python` verifies the checked CUDA toolkit and sibling
API before launching Python.

## Train

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti
```

Resume with the identical scientific config and existing run directory:

```bash
pbg-train --config configs/train/nlhe6.yaml \
  --hardware-profile local_3x2080ti \
  --resume runs/nlhe6/ckpt_0000100.pt
```

The Python entry points are `pbg.config.load_config` and `pbg.runtime.train`.
Hardware profiles change execution limits only. For multiple ranks use
`torchrun`; global sequential starts are partitioned across ranks, while
branched mode creates one root/tree per rank.

Before CUDA work, inspect `nvidia-smi`, select only idle devices, start with a
tiny smoke, retain the configured reserve, and verify clean teardown.

## Validate

```bash
/home/pablob/.venvs/poker/bin/python -m pytest -q
```

Tests cover strict retired-key rejection, all three modes, labeled topology,
randomized rounding, logical RNG invariance, old-policy identity, real evaluator
integration, completion, unequal distributed mass, finite gradients,
checkpoint/resume, and GPU memory safety. CUDA and sibling-dependent tests skip
when prerequisites are unavailable.

These are engineering and estimator-correctness checks. They do not establish
exploitability, approximate-best-response performance, or strategic improvement;
the bounded BR trend probe remains intentionally outside this release gate.

## Documentation

- [`CONTRACTS.md`](CONTRACTS.md) — active runtime/configuration invariants.
- [`NOTATION.md`](NOTATION.md) and
  [`docs/theory/core_math.tex`](docs/theory/core_math.tex) — mathematics.
- [`docs/README.md`](docs/README.md) — implementation documentation index.
- [`docs/training_and_evaluation.md`](docs/training_and_evaluation.md) —
  configuration, commands, checkpoints, benchmarks, and GPU procedure.
- [`docs/audit/README.md`](docs/audit/README.md) — current readiness and cleanup
  reachability evidence.
- [`archive/`](archive/) — retired games, superseded audits, migration plans,
  raw promotion comparisons, and historical prompts.

`IMPLEMENTATION_NOTES.md` is a historical ledger, not a current contract.
