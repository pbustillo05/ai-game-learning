# Validation evidence

## Environment and hygiene

- Validated revision: `5c10366ffaa330eeab5af73a6761b50874100246`
- Python: `/home/pablob/.venvs/poker` through `tools/poker-python`
- Evaluator: installed read-only sibling `poker_gpu_evaluator.pbg`, API `pbg-v3`
- Hardware: three RTX 2080 Ti devices, CUDA 13 environment
- Hardware profile: `configs/hardware/local_3x2080ti.yaml`

Every CUDA phase was preceded by `nvidia-smi` and used explicit
`CUDA_VISIBLE_DEVICES`. When an unrelated 182 MiB compute process appeared on
GPU 0, it was left untouched and work moved to GPUs 1/2. No OOM occurred. At
the final check, the test devices were back at their display-only 9/19 MiB
baselines with zero utilization and no owned compute process.

## Automated regression

Complete post-fix command, with CUDA confined to an idle device and socket
permissions available for the DDP tests:

```bash
CUDA_VISIBLE_DEVICES=1 ./tools/poker-python -m pytest -q
```

Result:

```text
382 passed, 1 skipped, 1 warning in 49.38s
```

The skip is version-conditional. The warning is PyTorch's standard notice that
synchronization debug mode is a prototype. The tested full-policy path itself
reports zero undeclared host synchronizations.

A focused CUDA/evaluator/geometry gate also passed:

```text
158 passed, 1 warning in 25.97s
```

It covers real sibling calls, CUDA rollout, full policy input construction,
sync debug, and all compact-deck tests. A CUDA-hidden run had 367 passes and 13
expected skips; its only three failures were sandbox-denied Gloo sockets, and
all three passed in the escalated complete suite.

## Geometry and boundary coverage

- Every `R=4,...,13` passes geometry, preflop/postflop models, evaluator mapping,
  masked folded summaries, chance, completion, and showdown checks.
- Every legal `(P,R)` pair satisfying `2P+5<4R` is enumerated.
- One-iteration normal and RB training coverage spans every rank count.
- `P=2,...,6` at `S=8` and `S=128` attains the exact legal player-node bound;
  the derived capacity plus three separators passes model packing.
- Histories beyond 256 are exercised for 3--6 players.
- One- and multiple-folder continuation, semantic folder influence, cache hit,
  fold invalidation, and street invalidation pass.

## Final one-GPU matrix

Temporary configs clone the repository's heads-up 8 bb acceptance settings and
write only to `/tmp`. Each mode uses the production CLI/runtime and sibling
evaluator.

| Mode | Decisions | Identity max logp error | Optimizer | Peak reserved | Result |
|---|---:|---:|---:|---:|---|
| normal | 41 | $3.86\times10^{-5}$ | 1 applied | 455,081,984 B | pass |
| sequential RB | 42 | 0.0011735 | 1 applied | 442,499,072 B | pass |
| branched RB | 11 | 0.0004773 | 1 applied | 415,236,096 B | pass |

Every mode reports finite reduced gradients, a checkpoint at iteration 1, and
zero undeclared synchronization. Branched mode sampled one distinguished tree,
used fractional beta 1.5, and observed two children at a node.

## Final unequal-work two-GPU matrix

Commands used two visible devices and `torch.distributed.run --nproc-per-node=2`.

| Mode | Rank-0 local mass | Global mass | Rank-0 scale | Collectives/bytes | Result |
|---|---:|---:|---:|---:|---|
| normal | 23 | 41 | 0.5609756 | 2 / 7,467,280 | pass |
| sequential RB | 29 | 56 | 0.5178571 | 2 / 7,467,280 | pass |
| branched RB | 3.5 | 7.75 | 0.4516129 | 2 / 7,467,280 | pass |

All three runs applied one finite flat-SUM update, wrote checkpoints, and
reported zero undeclared synchronization. Scales differ from 0.5, demonstrating
that the test exercised unequal mass rather than equal-rank averaging. The
17-start sequential quota also exercises the 9/8 world-size split.

## Checkpoint/resume soaks

Each mode ran a two-rank 25-update workload with checkpoints every five
iterations. The metrics file was restored to the exact checkpoint-20 prefix and
resumed through 25 under a compatible hardware profile whose query chunk changed
from 128 to 96.

| Mode | Metrics history | Finite/applied updates | Profiled peak reserved | Resume artifact |
|---|---|---:|---:|---|
| normal | exact 0--24 | 25 / 25 | 429,916,160 B | `hardware_resume_0000020.json` + checkpoint 25 |
| sequential RB | exact 0--24 | 25 / 25 | 390,070,272 B | same |
| branched RB | exact 0--24 | 25 / 25 | 415,236,096 B | same |

Every record has a finite reduced gradient, no nonfinite skip, and zero
undeclared synchronization. Execution-reserved accounting was constant across
the normal and sequential soaks and zero-residual for the branched scheduler.

The first sequential resume attempt exposed an additional release defect: the
frozen rollout policy ran in FP32 while identity/learner forwards used FP16.
After 20 updates this produced a 0.03547 log-probability mismatch and failed the
identity gate. Commit `5c10366` applies the configured precision consistently to
behavior and learner policy inference while keeping completion/value reduction
in fp32. The same checkpoint-20 resume then passed. The identity failure path
now raises a normal diagnostic exception instead of poisoning the CUDA context.

## Realistic six-player checkpoint gate

After the tiny gates, the repository's 6-player 10 bb normal workload ran on one
RTX 2080 Ti with 128 starts, a 96-wide/two-layer model, AdamW, clipped values,
active reference KL, and checkpoint interval 20.

| Measure | Result |
|---|---:|
| Iterations | exact 0--19 |
| Finite/applied optimizer updates | 20 / 20 |
| Profiled-iteration decisions | 1,038 |
| Reference KL range | 0 to 0.0753732 |
| Undeclared synchronizations | 0 |
| Calibrated peak reserved | 3,913,285,632 B |
| Calibrated safe limit | 9,199,091,712 B |
| Final checkpoint | 79 MiB at iteration 20 |

The 3.91 GiB peak retained more than 5.2 GiB of headroom under the profile's
safe limit.

## Loss/update coverage

The complete suite additionally covers:

- `ppo` and both RB uses of `rb_ppo` with MSE and clipped values;
- constant and linear entropy at start/midpoint/endpoint;
- reference KL refresh, checkpoint, and resume;
- Adam and AdamW;
- one-rank FP16/BF16/FP32 plus disabled AMP;
- injected finite and nonfinite gradients, including unequal two-rank injection;
- parameter/scaler immutability on rejected updates; and
- dotted-path rejection for every audited invalid numeric boundary.

## Evidence boundary

This evidence establishes the repaired execution, estimator, safety, and
checkpoint contracts. It does not establish exploitability, approximate-best-
response performance, strategic improvement, or convergence. M-05 remains
explicitly outside this release scope.
