# Validation evidence

## Environment and GPU hygiene

- Revision: `dde65f6d470cd4d890ec1b19293c22e83139cc24`
- Python: shared `/home/pablob/.venvs/poker` through `tools/poker-python`
- CUDA/PyTorch: checked CUDA 13 environment used by the repository wrapper
- Evaluator: installed sibling API asserted as `pbg-v3`
- GPUs: three RTX 2080 Ti devices

Before CUDA work, GPU 0 had an unrelated 2,908 MiB process. It was not selected
or touched. GPU 1 was at 9 MiB and GPU 2 at 19 MiB. Single-device work used only
GPU 1; distributed work used GPUs 1 and 2. Final state returned to exactly 9 and
19 MiB with zero utilization. No OOM occurred.

## Static and configuration checks

| Check | Result |
|---|---|
| `python -m compileall -q pbg tests` | pass |
| Import-direction/retired-game tests | included in full pass |
| Load every non-hardware YAML with local profile | 23 loaded, 0 failed |
| Root/project Markdown local-link scan | no genuine missing local target |
| `ruff check pbg tests` | not run: `ruff` is not installed in the shared venv |

The configuration probe intentionally mutated a valid parsed config and called
the same `_validate_config` function. It demonstrated that zero node budget,
lambda 2, zero iterations, zero update steps, negative learning rate, negative
clip norm, and zero checkpoint interval are all accepted.

## Complete automated suite

Command, with CUDA restricted to the idle device:

```bash
CUDA_VISIBLE_DEVICES=1 ./tools/poker-python -m pytest -q
```

Result:

```text
226 passed, 1 skipped, 1 warning in 14.00s
```

The warning states that PyTorch synchronization debug mode is a prototype and
does not detect all synchronizing operations. That qualification matters to
M-01: the focused sync-debug test cannot substantiate the stronger whole-runtime
claim.

The one skipped test required a prerequisite outside the one-visible-GPU full
suite context. Distributed behavior was exercised separately on two visible
GPUs.

## Targeted evaluator/model checks

Before the complete run, a focused set covering host-sync primitives,
completion moments, folded-player logic, postflop architecture, execution
foundations, and real GPU rollout produced:

```text
69 passed, 10 skipped in 4.73s
```

The skips in that sandboxed invocation were expected CUDA/prerequisite skips;
the complete escalated suite later ran with CUDA available.

The socket-dependent DDP test file was also run in an environment permitting
local process-group sockets:

```text
3 passed in 5.22s
```

## Real single-GPU training matrix

Temporary audit configs cloned the repository's one-iteration heads-up 8 bb,
`R=13`, small-model acceptance settings and wrote only to `/tmp`. Each command
used the production CLI/runtime and sibling evaluator on GPU 1.

| Mode | Result | Decisions | Optimizer steps | Peak reserved |
|---|---|---:|---:|---:|
| normal | pass, finite checkpoint/metrics | 56 | 1 | 396,361,728 B |
| sequential RB | pass, finite checkpoint/metrics | 47 | 1 | 469,762,048 B |
| branched RB | pass, finite checkpoint/metrics | 11 | 1 | 419,430,400 B |

All reported `reduced_gradients_finite: 1.0`, old-policy identity within the AMP
tolerance, nonzero decision mass, one optimizer step, and a checkpoint at
iteration 1. These passes demonstrate the ordinary finite path; they do not test
the injected-nonfinite branch missing in B-03.

## Real two-GPU training matrix

The same three temporary configs were launched through two ranks with
`CUDA_VISIBLE_DEVICES=1,2`. All completed with exit code zero.

| Mode | Global decision mass | Rank-0 objective scale | Optimizer steps | Collective calls | Collective bytes |
|---|---:|---:|---:|---:|---:|
| normal | 42.0 | 0.5238096 | 1 | 2 | 7,589,136 |
| sequential RB | 46.0 | 0.4565217 | 1 | 2 | 7,589,136 |
| branched RB | 8.5048828 | 0.5878976 | 1 | 2 | 7,589,136 |

Each reported finite reduced gradients. The unequal objective scales show that
the run exercised global-mass normalization rather than trivial equal-rank
averaging.

## Boundary probes that fail the release contract

### Reduced deck

Constructing the production backend with an otherwise legal
`PokerConfig(n_players=2, stack_bb=8, rank_count=12)` produced:

```text
RuntimeError: compact-deck model tensors are not yet migrated to variable hand width
```

This is also an explicit expected-failure unit test for `R=4`.

### Entropy decay

For `training.iters=100`, `loss.c_h=0.01`, and iteration 50, the coefficient was:

```text
0.0
```

The documented linear schedule implies `0.005`.

### History capacity

The failure is established by exact static bounds rather than a large adversarial
rollout: the documented maximum `SP+3P-S+1` is 519 for the shipped 6-player
100 bb preset and 659 at 6-player 128 bb, while both network validators reject
history width above 256. All values are deterministic configuration/source
constants.

## What the evidence does and does not establish

Established:

- the full-deck ordinary path is operational in every training mode;
- real sibling sampling/completion/folder/showdown boundaries execute;
- two-rank NCCL training performs globally scaled SUM updates;
- checkpoints and metrics are produced;
- the tested workloads stay well inside RTX 2080 Ti memory;
- GPUs return cleanly to idle.

Not established:

- any `R < 13` production training;
- lossless execution of every legal deep multiplayer trajectory;
- safe nonfinite behavior in FP32/BF16/AMP-disabled modes;
- correct `linear_decay` entropy;
- exhaustive `(P,S,R)` execution;
- large-run throughput or memory stability after the newly found hot-path issues;
- strategic quality, exploitability improvement, or convergence.

The successful tests therefore support a strong full-deck prototype, not the
advertised general production release.
