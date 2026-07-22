# Validation evidence

## Scope and outcome

This batch validated the audited tree without changing production code or tests.
It first read every existing report in this audit directory (`00` through `08`
and `10`), inspected collection and CUDA guards, ran the whole suite with CUDA
hidden, and then staged only tiny CUDA workloads after checking GPU occupancy.
All Python commands used `/home/pablob/.venvs/poker/bin/python`.

The broad suite collected **489 tests**. With CUDA deliberately hidden, **454
passed, 32 skipped, and 3 failed** in 109.70 seconds. All three failures were
local two-process Gloo tests whose socket initialization was denied by the
sandbox (`Operation not permitted`); the exact three tests passed in 8.20
seconds when rerun outside that network sandbox. Thus all **457 collected
non-CUDA tests passed** once the infrastructure restriction was removed. The
32 skips were 29 CUDA cases and 3 Apple-MPS cases.

Tiny CUDA evidence is mixed but bounded:

- Sequential RB, fixed branched RB, and KKT branched RB each completed one real
  rollout/target/learner update on physical GPU 0. Reported iteration reserved
  peaks were only 22--25 MiB.
- The tiny normal-mode case and the two-iteration fixed-refill case reached the
  sibling evaluator's `sample_joint_profiles_cuda` JIT path and could not build
  it because this host has no CUDA toolkit/`CUDA_HOME`. This is an environment
  blocker, not an OOM or a numerical assertion failure. The shared venv does
  contain `ninja`; after putting its executable on `PATH`, the failure advanced
  from “Ninja is required” to the definitive missing-`CUDA_HOME` error.
- The CUDA queue/arena/RNG transaction, public KKT head and Gamma-loss path,
  dynamic scientific safeguard, and real sibling execution boundary all
  passed.
- One tiny two-GPU NCCL learner test passed on physical GPUs 0 and 1.
- There is no existing BR CUDA acceptance case. No new test or ad-hoc training
  workflow was invented during this read-only validation.

No large benchmark, soak, or production-size training job was run.

## Collection and broad non-CUDA suite

The repository defines no global CPU/CUDA marker in `pyproject.toml`; CUDA tests
use individual `skipif(torch.cuda...)` guards. Hiding CUDA for the broad run was
therefore the only reliable way to prevent a full-suite invocation from
accidentally launching hardware work.

| Command | Result | Duration |
|---|---:|---:|
| `env CUDA_VISIBLE_DEVICES= /home/pablob/.venvs/poker/bin/python -m pytest --collect-only -q` | 489 collected | 1.56 s pytest / 2.3 s wall |
| `/usr/bin/time -f 'WALL_SECONDS=%e' env CUDA_VISIBLE_DEVICES= /home/pablob/.venvs/poker/bin/python -m pytest -q -ra` | 454 passed, 32 skipped, 3 sandbox failures | 109.70 s pytest / 110.75 s wall |
| `env CUDA_VISIBLE_DEVICES= /home/pablob/.venvs/poker/bin/python -m pytest -q -ra tests/test_p13b_ddp.py::test_distributed_metric_summaries_allow_rank_local_optional_metrics tests/test_p13b_ddp.py::test_unequal_rank_flat_sum_matches_concatenated_rollout_oracle tests/test_p13b_ddp.py::test_unequal_rank_nonfinite_decision_skips_every_optimizer` (outside network sandbox) | 3 passed | 8.20 s |

The original three traces all ended in
`ProcessGroupGloo ... RuntimeError ... Operation not permitted` during
`dist.init_process_group`. None reached an implementation assertion. Their
successful exact rerun establishes that they were infrastructure failures.

The broad run emitted 18 `torch.jit.script` deprecation warnings from
`test_rao_blackwell_variance_cli.py`. This is consistent with the audit's
classification of that variance workflow as stale/archival rather than a core
production mode.

## GPU selection and hygiene

Initial `nvidia-smi` at 18:06:53 showed three idle RTX 2080 Ti devices:

| Physical GPU | Memory | Utilization | Temperature | Processes |
|---:|---:|---:|---:|---|
| 0 | 6 MiB / 11264 MiB | 0% | 29 C | display-only Xorg, 4 MiB |
| 1 | 6 MiB / 11264 MiB | 0% | 28 C | display-only Xorg, 4 MiB |
| 2 | 16 MiB / 11264 MiB | 0% | 29 C | display-only Xorg/GNOME, 13 MiB |

Physical GPU 0 was selected explicitly with `CUDA_VISIBLE_DEVICES=0` for all
single-GPU work. GPUs 0 and 1 were selected explicitly only for the one tiny DDP
case. A state check followed every test stage. After each stage the devices
returned to the same baseline; there was no unexplained memory growth.

The state commands were `nvidia-smi` for the initial/final full process tables
and
`nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader`
between stages.

Final `nvidia-smi` at 18:09:24 showed GPU 0 at 6 MiB/0%, GPU 1 at 6 MiB/0%, and
GPU 2 at 16 MiB/0%. The process list contained only the same display processes
and no compute workload.

## Tiny single-GPU mode acceptance

The selected acceptance function uses the smallest active game geometry:
$P=2$, $S=8$ bb, $R=4$, at most two roots, four completion samples, an
8-dimensional one-layer model, and one optimizer step. This is far below a
material OOM risk.

Command:

```text
env CUDA_VISIBLE_DEVICES=0 /home/pablob/.venvs/poker/bin/python -m pytest -q -ra -s \
  tests/test_refactor_gpu_acceptance.py::test_tiny_gpu_training_modes_report_bounded_phase_memory
```

Result: **3 passed, 1 failed in 4.82 s**.

A preliminary invocation of the identical command inside the filesystem
sandbox saw CUDA as unavailable and produced **4 skips in 1.40 s** plus an NVML
initialization warning. Per the repository instructions, the meaningful run
above was then repeated with GPU access outside that sandbox; the four skips
are not counted as validation evidence.

| Semantic mode | Result | Peak reserved | Evidence |
|---|---|---:|---|
| normal | environment-blocked | not reported | sibling `sample_joint_profiles_cuda` attempted JIT build |
| sequential RB | passed | 22 MiB | finite loss, two trajectories, sealed pages, target/phase accounting |
| fixed branched RB | passed | 24 MiB | finite loss, one action, T/C learner path and phase accounting |
| KKT branched RB | passed | 24 MiB | finite loss, one action, dynamic geometry and phase accounting |

The first invocation reported “Ninja is required” because the venv executable
directory was not on `PATH`. The venv was verified to contain Ninja 1.13.0 at
`/home/pablob/.venvs/poker/bin/ninja`. The normal node was rerun alone with that
directory on `PATH`:

```text
env PATH=/home/pablob/.venvs/poker/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CUDA_VISIBLE_DEVICES=0 /home/pablob/.venvs/poker/bin/python -m pytest -q -ra -s \
  'tests/test_refactor_gpu_acceptance.py::test_tiny_gpu_training_modes_report_bounded_phase_memory[normal-normal-fixed-2]'
```

It failed in 3.48 seconds with
`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA
install root.` Inspection found neither `/usr/local/cuda*` nor an `nvcc`
executable. The failure occurred before the normal rollout could complete and
does not validate or refute its target/loss equations.

This partially confirms the audit's GPU-readiness concern: a current supported
mode is not launchable on this otherwise CUDA-capable host without an undeclared
sibling-evaluator build-time dependency. It does not confirm a production-code
logic regression.

## Queue, memory, refill, KKT, and evaluator smoke

Command:

```text
env CUDA_VISIBLE_DEVICES=0 /home/pablob/.venvs/poker/bin/python -m pytest -q -ra -s \
  tests/test_execution_foundations.py::test_cuda_queue_arena_and_rng_execute_transactionally \
  tests/test_adaptive_kkt_branching.py::test_tiny_cuda_branching_heads_and_gamma_losses \
  tests/test_refactor_gpu_acceptance.py::test_tiny_branched_gpu_safety_soak_exercises_scientific_controls \
  tests/test_p13a_gpu_rollout.py::test_real_sibling_execution_poker_tiny_cuda
```

Result: **4 passed, 1 failed in 6.63 s**.

- Passed: schema/page-accounted queue/arena/RNG transaction on CUDA.
- Passed: positive public T/C heads and Gamma-deviance losses on CUDA.
- Passed: two-iteration KKT dynamic-safeguard soak, including suppressed-child
  evidence and no hard-memory-limit breach.
- Passed: tiny real sibling execution boundary on CUDA.
- Environment-blocked: fixed-refill two-iteration soak. Its later refill work
  reached the same joint-profile CUDA JIT path and failed for unavailable Ninja
  on the command `PATH`; the normal-mode rerun above establishes the deeper
  missing-CUDA-toolkit blocker.

The low-level protected-root refill tests did pass in the broad non-CUDA suite,
including `test_root_refill_reaches_minimum_and_finalizes_uniform_weights` and
the compact target-prefix tests. The GPU evidence remains incomplete because
the end-to-end refill soak did not finish. No failure here confirms the audit's
fixed-beta queue-safeguard defect: protected lower-size refill and the missing
fixed-beta queue-pressure safeguard are different mechanisms.

## Tiny two-GPU validation

Before launch, GPUs 0 and 1 were both at 6 MiB and 0% utilization. The chosen
test uses only tiny in-memory learner fixtures, a scalar model parameter, and
3--4 sampled rows per rank; it does not invoke rollout or the evaluator.

Command:

```text
env CUDA_VISIBLE_DEVICES=0,1 /home/pablob/.venvs/poker/bin/python -m pytest -q -ra \
  'tests/test_distributed_learner.py::test_two_cuda_unequal_strata_exact_sum[normal]'
```

Result: **1 passed in 6.46 s**. Both ranks produced bit-identical updated
weights while preserving unequal exact objective masses of 0.25 and 0.75. This
supports the retained SUM-gradient/global-mass path (`STORE-035`, `STORE-036`),
but it does not exercise the asymmetric mandatory-failure condition in
`AUD-TL-001`; that blocker remains untested and statically confirmed.

## Finding reconciliation and remaining gaps

- **Supported by execution:** sequential RB, fixed branched RB, and KKT are
  CUDA-reachable through real tiny updates; T/C training and the KKT dynamic
  safeguard execute; queue/page transactions and exact distributed mass/SUM
  execute; devices return cleanly to idle.
- **Not disproved by tiny success:** the post-hoc `SampledTree` conversion,
  unadmitted seal overlap, inferred source schemas, unbounded logical child
  waves, queue ordering/accounting, postflop worst-width execution, and folded
  summary ownership findings are structural or production-scale issues. Tiny
  tests cannot establish their safety.
- **Normal validation gap:** the real normal CUDA update is blocked by the
  sibling evaluator joint-profile extension's undeclared local CUDA-toolkit
  build requirement. No target/performance conclusion should be drawn from the
  blocked test.
- **Fixed-refill GPU gap:** protected refill is numerically covered on CPU, but
  its tiny end-to-end CUDA soak is blocked by the same evaluator extension.
- **BR validation gap:** no repository CUDA acceptance node exists for BR, and
  the reports identify a static multi-player early-fold defect. The broad CPU
  BR tests passed, but they are heads-up and do not exercise `TRACE-001`.
- **Failure-agreement gap:** no test injects a mandatory failure asymmetrically
  before the collective sequence. The successful DDP test does not address
  `AUD-TL-001`.
- **Performance gap:** no historical-size target or learner benchmark was run.
  Therefore the per-decision CUDA scalar synchronization risk and backward-pass
  regression risk remain unquantified.
- **Evaluator authority gap:** the tiny real sibling boundary passes, but the
  active facades still accept contract v1 per the static audit, and the named
  `api_pbg.md` authority is missing. Validation did not modify the read-only
  sibling repository.

## Overall validation disposition

The current tree has a strong passing unit/contract baseline and representative
tiny GPU reachability for three training semantics plus the dynamic safeguard
and distributed learner. It cannot yet be called five-mode NVIDIA-acceptance
complete: normal and fixed-refill are environment-blocked at the evaluator CUDA
extension, BR has no CUDA gate and a static multi-player defect, and no test
covers common-order terminal failure. These gaps align with, rather than clear,
the corresponding audit findings.
