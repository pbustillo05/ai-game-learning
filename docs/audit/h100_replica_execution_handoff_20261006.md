# Shared-model H100 rental handoff

The shared-model implementation and rental harness are integrated. CUDA execution
was deferred at the user's request because the shared GPUs are busy. No numerical
correctness, capture compatibility, memory ceiling, learning-quality or speedup
claim follows from the offline checks. The first rental session must run the gates
below before interpreting performance results.

## Environment

Use Python 3.10.21 and `requirements-h100.lock` in a clean environment. The exact
package, CUDA runtime and evaluator source pins are in
[`configs/h100_environment.json`](../../configs/h100_environment.json). The
shared development environment contains duplicate CUDA toolkit distribution
metadata; the lock records the distributions resolved by Python and the actual
`nvcc` reports 13.0.88. Do not copy that environment's overlapping installations
or modify it while other users have jobs running. Reconstruct the clean rental
environment, run `python -m pip check`, and let preflight reject package drift.
Package installation on a fresh rental has not been exercised here.

Install this repository and the sibling evaluator from its pinned revision with
`python -m pip install --no-deps --no-build-isolation -e .` in each respective
checkout, after installing the lock. The evaluator source must be at
`caeeb976045a17454d12293811469863e5e1ce61`; its imported build revision is checked.
The checkout must be clean and committed before preparing a plan. Plans and
completed evidence are bound to the code revision and content hashes.

Activate the rental venv, then set its toolkit paths (the shared development venv
is `/home/pablob/.venvs/poker`):

```bash
source /home/pablob/.venvs/poker/bin/activate
source tools/setup-h100-toolkit.sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_IDLE_GPU_UUID
```

Select one idle full H100. Preflight records UUID, SKU, MIG, driver, clocks, power,
reported memory, native precision policy, libraries and evaluator identity. It
runs the production fused attention forward/backward in native BF16. Admission
uses reported memory with at least 10 GiB reserve; it does not assume 80 GiB from
an “80 GB” product label. No command changes clocks, power limits or driver policy.

The setup script also creates the missing `libcudart.so -> libcudart.so.13`
link in a freshly installed CUDA runtime wheel. PyTorch extension builds link
with `-lcudart`; without this link, the clean locked environment fails to build
the alias, branching, and layered-target operators. It leaves an existing
linker name untouched.

## Commands

Offline preparation needs neither CUDA nor an old checkpoint:

```bash
python -m pbg.cli.h100_pilot prepare \
  --directory runs/experiments/nlhe_hu_32bb_h100_pilot \
  --configs-directory configs/experiments/nlhe_hu_32bb_h100_pilot
```

Preflight and the complete correctness gate can be run separately:

```bash
python -m pbg.cli.h100_pilot preflight \
  --report runs/experiments/nlhe_hu_32bb_h100_pilot/reports/preflight.json
python -m pytest -q
```

The orchestrator runs those gates itself, then Normal/RB K=1/2 smoke tests and
matched K=1/16/32 scaling, with K=8 fallback if neither larger count completes:

```bash
python -m pbg.cli.h100_pilot run \
  --plan runs/experiments/nlhe_hu_32bb_h100_pilot/plan.json
```

Resume interrupted orchestration with the same command plus `--resume`. Completed
reports and correctness logs must retain their recorded hashes. After a code fix,
commit it and prepare a new run directory; old evidence cannot silently become
results for the new code. Failed arm configuration/runtime scratch is removed;
partial reports, logs and failure records are retained. The parent owns no CUDA
context, and arm child exit is followed by a selected-GPU process/memory check.
The deadline reserves time for shutdown; it does not certify benchmark success.

Production arms retain batch size 256 and 116 logical steps. Learner physical
caps are pre=32/post=8; rollout/completion caps are 8/64. Reference caching is off
for the matched experiment. All arms use the same shared architecture. The
smallest materially faster admitted K is validated with two further seeds.
Repeated samples restore model, optimizer, scaler, RNG, controllers, reference
state and logical counters into the same storage while retaining warmed caches.
Reports separate setup/checkpoint, reset and warmup time from iteration timing.
Production samples that admit new compiled/captured shapes are rejected as cold.

For attribution, use a separate instrumented run of the chosen config:

```bash
python -m pbg.cli.benchmark \
  --config configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_normal_h100_replica_k16.yaml \
  --warmup 2 --repeats 1 --phase-telemetry \
  --output runs/experiments/nlhe_hu_32bb_h100_pilot/reports/phase-attribution.json
```

Use Nsight Systems on the same entry point to find gaps, then inspect dominant
operators with a short Nsight Compute run if counters are permitted:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output runs/experiments/nlhe_hu_32bb_h100_pilot/reports/timeline \
  python -m pbg.cli.benchmark \
  --config configs/experiments/nlhe_hu_32bb_h100_pilot/nlhe_hu_32bb_normal_h100_replica_k16.yaml \
  --warmup 2 --repeats 1 \
  --output runs/experiments/nlhe_hu_32bb_h100_pilot/reports/profiled.json
```

Keep profiler results separate from unprofiled medians. Nsight is not installed
on this development environment; the profiler command and all CUDA command
bodies remain rental validation items. Offline preparation and both CLI help
interfaces were exercised here. Never enable counters by changing shared driver
policy. Check `nvidia-smi` after standalone commands as well.

## Evidence and interpretation

The work ledger derives shapes from the sole model configuration: Normal major
forward/dX/dW work is 12.219/142.262 GFLOP per pre/post HU presentation; factored RB
is 12.665/145.779. These are conservative full-width matrix envelopes, excluding
minor projections/scalar operations and separately listing attention score
recomputation. They are neither issued FLOPs nor measured MFU. Do not use them
alone to predict full-pipeline time.

Reports distinguish fresh player nodes, reused learner sigma/action presentations,
reference forwards/cache gathers, optimizer steps, aggregate throughput and
per-run latency. Before/after program counters identify physical cohort work.
Instrumented phase reports give synchronized wall time, allocator bytes at the
boundary, cumulative allocator high-water marks and named live owner bytes.
Cumulative peaks are not independent phase peaks. Routing traffic, completion
sample counts, topology work and model-family timings remain separately visible.
Do not sum overlapping per-lane model event intervals into GPU wall time.

Offline validation: 36 metadata/import/config/ownership checks passed, with 27
CUDA gates skipped in the targeted suite. Python compilation, static undefined-
name checks and full test collection passed. Tests now include shared pre/post
gradient sums, exact K cohorts, independent controls, selected/factored heads,
native FP16/BF16 output/gradient tolerances, capture restoration and bounded-cache
contracts. These numerical tests have not been executed after this rewrite.
Architecture, correctness and CUDA reviewers resolved their concrete findings;
review does not replace execution.

Required rental work includes the full CUDA suite, native BF16 dispatch, capture
repair if needed, measured memory reconciliation and stable repeated teardown,
reference-refresh/resume behavior, branched/multi-way consumers, and distributed
SUM/failure gates when multiple idle GPUs are available. A single H100 does not
validate multi-rank NCCL. Tune only after these checks; learning quality across
independent seeds and all performance expectations remain unproven.
