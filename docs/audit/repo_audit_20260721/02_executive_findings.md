# Executive findings

## Re-audit status — 2026-07-22

This status reflects the working tree at `32adb8c` after the post-audit
changes, including the current uncommitted `pbg/core/storage.py` optimization.
The current `AGENTS.md` accelerator contract and the sibling evaluator's
`/home/pablob/poker-gpu-evaluator/docs/api.md` are used as the operative
repository policy.

## Verdict

The refactor has resolved most of the original structural and mathematical
blockers. Of the 18 canonical findings, 13 are fixed, 2 are partially fixed,
2 remain open defects, and 1 remains an unmeasured optimization opportunity.

The repository is nevertheless **not production-ready**. The five-mode
performance gate is still incomplete and fail-open (C12), active documentation
and workload manifests are stale (C18), and the full CUDA test suite now exposes a stateful-inference
shape regression (N01). Performance preservation against historical equal-work
runs has not been established.

## Canonical findings

| ID | Current status | Re-audit assessment | Source finding IDs |
|---|---|---|---|
| C01 | **Fixed** (`95343dc`) | Production rollout now writes direct page-backed topology/decision/leaf stores and seals immutable `SealedRolloutSample` objects. The former `SampledTree` conversion path and overlapping legacy representations are absent and guarded by migration tests. | `AUD-RSQ-001/002`, `AUD-RD-003`, seal part of `HIST-OOM-01`, `PERF-002/007` |
| C02 | **Fixed** (`f40b13c`) | Mandatory distributed phases use `DistributedFailureAgreement`, including singleton-feasibility agreement before exchange collectives. Focused tests pass; the current one-visible-GPU run did not re-execute multi-rank CUDA coverage. | `AUD-TL-001` |
| C03 | **Fixed** (`d407d8e`) | Shared-response BR now has authoritative response semantics, seat laws, objective, stopping rules, paired evaluation, and focused contract tests. | `A-01`, `EVAL-001`, authority part of `MODE-007` |
| C04 | **Fixed** (`7ac14dc`) | Multi-player probe-fold ownership and settlement are implemented. The three-player train/resume/evaluate CUDA acceptance test passes. | `TRACE-001`, `EVAL-002`, correctness/test part of `MODE-007` |
| C05 | **Fixed** (`3625e0a`) | Rollout stores are created from explicit natural schemas, preflop carriers use class width, and stable logical child addresses are reserved before stochastic materialization. | `AUD-RSQ-003/004/006` |
| C06 | **Fixed** (`c723115`) | Branched offspring receive logical addresses first and are physically materialized through queue-bounded child subspans. The physical-plan invariance and bounded-materialization gates pass. | `AUD-RSQ-005`, part of `AUD-RSQ-006`, `PERF-003` |
| C07 | **Fixed** (`a78deab`) | Device queues have explicit capacity/staging accounting. The queue-pressure safeguard is batch-local and applies to fixed branching and dynamic warmup as well as active KKT; focused tests pass. | `AUD-RSQ-007/008/009`, fixed/warmup `HIST-OOM-01`, `PERF-007` |
| C08 | **Fixed under current accelerator contract** (`2625160`) | Device resolution accepts only `cuda` and `mps`; training, distributed work, BR, benchmarks, and variance work reject unsupported backends, and distributed training is CUDA/NCCL-only. MPS inference is now an explicitly supported product rather than stale code. | `MODE-001`, `AUD-TL-005`, CUDA part of `EVAL-002` |
| C09 | **Fixed** (`48b66e4`) | Folded summaries have sparse anchor ownership, reject the forbidden all-live `post_P` summary, and include cache bytes in memory accounting. The CUDA accounting gate passes. | `POKER-001/002`, `AUD-MODEL-002`, `PERF-007` |
| C10 | **Fixed** (`2625160` plus cleanup) | Poker completion/folded bindings accept evaluator contract v2 only, and the authoritative sibling binding document exists at the path named by `AGENTS.md`. | `POKER-003`; `POKER-008` documented the former path gap |
| C11 | **Fixed statically and by focused CUDA gates; timing unverified** | Sequential targets use one bounded structural transfer and segmented device contraction; audited evaluator/model/learner controls no longer perform the identified repeated scalar synchronizations. CUDA synchronization-debug tests pass. No production-scale equal-work timing establishes the size of the speedup or excludes other regressions. | `HIST-PERF-02`, `AUD-TL-002/004`, `POKER-005/006`, `PERF-001/005` |
| C12 | **Open — high** | `pbg/benchmark.py` still exposes only `normal`, `rao_blackwell`, and one `branched_rao_blackwell` label; it does not gate BR, fixed-beta branched RB, and KKT branched RB as five distinct semantics. Mode mutation is not strictly revalidated, checkpoint loading remains permissive, and comparison still tolerates missing baseline modes and metrics. | `MODE-003/004`, `EVAL-010`, `PERF-008` |
| C13 | **Fixed / superseded by current scope** (`2625160`) | The obsolete CPU arena, list engine, and range products were removed. Accelerator-native inference remains installed because current policy explicitly supports CUDA inference and the supported MPS inference path. Its newly observed correctness failure is tracked separately as N01. | `MODE-002`, `POKER-004`, `EVAL-005/006`, `AUD-RD-001` |
| C14 | **Partially fixed** (`9f6297b`) | Variance modules were removed from the installed `pbg` package and moved under `experiments/rao_blackwell_variance`. They are not archived, however, and the branched runner still owns parallel tree-layout, moment, decision-edge, and functional-propagation algorithms rather than consuming one production implementation. | `EVAL-008/009`, `AUD-RD-002` |
| C15 | **Fixed in working tree** | Runtime checkpoint migration, legacy fingerprints/private aliases, permissive training/benchmark model loads, obsolete model/config aliases, evaluator-v1 summary fallbacks, and stale config keys were removed. Training, BR, benchmarking, and inference export now require current v6 checkpoints; inference artifacts contain only exact game/model configuration. | `AUD-RD-005`, `EVAL-007/011`, `MODE-008` |
| C16 | **Partially fixed** | Selected-owner identity and compact `[B,1,H]` BR leaf/target storage are implemented, and fold terminals avoid dense all-player outputs. Pre-terminal BR completion still calls evaluator-v2 `complete`, receives `[B,P,H]` `p_hat`/`v_hat`, validates all players, and only then projects the selected player. The sibling v2 API currently offers no selected-player batched completion endpoint. | `EVAL-003`, `PERF-006` |
| C17 | **Open optimization — unmeasured** | CUDA family planning still charges the shared flop-width `post_A` shape on all postflop streets. Street-aware physical packing could reduce turn/river work, but neither the potential gain nor a regression has been measured. | revised `POKER-009`; `AUD-MODEL-001` rejected as stated |
| C18 | **Open — high; broken paths medium** | Active docs still describe three modes, CPU rollout workers, deleted modules such as `pbg.eval.arena`, and obsolete paths. `benchmarks/workloads.yaml` still names missing `configs/benchmarks/nlhe_hu_short.yaml` and retired CLI arguments. Tests have improved materially, but the full CUDA suite is not green (N01). | `AUD-RD-006/007`; revised `AUD-RD-008` |

## New finding from this re-audit

### N01 — Stateful accelerator inference is broken — high

Both CUDA stateful-inference acceptance tests fail at
`pbg/inference.py:596`. The preflop model/adapter returns class-space policy
rows shaped `[16,18]` for the test artifact, while `_evaluate_full_device`
attempts to store them directly in combo-space output shaped `[1,120,18]`.
The missing class-to-combo expansion prevents `predict`, `predict_many`,
`advance`, and consequently the tested stateful CUDA inference flow from
operating. This is independent of the uncommitted storage optimization.

Required disposition: restore the explicit preflop class-to-combo projection
at the accelerator-native inference boundary and retain CUDA and MPS coverage
for singleton, batched, action-advance, and postflop paths.

## Validation performed

- C15 strict-schema/config/model focused suite after the fix: **68 passed,
  6 skipped**. Final CUDA BR train/resume/evaluate acceptance passed on physical
  GPU 1, and the GPU returned to its 6 MiB display-only idle state.
- Focused structural/CUDA audit suite: **94 passed, 7 skipped**. This covered
  direct sealing, distributed failure contracts, BR authority and probe-fold
  acceptance, no-host-sync gates, labeled branching, KKT/safeguards, accelerator
  policy, refactor acceptance, and performance-gate unit contracts.
- The post-C15 full repository run reached **270 passed, 10 skipped, 3 failed**;
  one failure was a test import regression fixed immediately afterward and its
  focused rerun passes. The two remaining failures are the pre-existing N01.
- The run used the required `/home/pablob/.venvs/poker` environment and CUDA 13
  toolkit. Physical GPU 1 was exposed as `cuda:0`; the existing GPU 0 workload
  was not touched.
- No equal-work benchmark was run. C12 makes the current benchmark incapable of
  certifying all five required modes, so historical efficiency preservation is
  still unproven rather than passed.

## Preserved strengths

- KKT primitives/controller wiring, fixed/warmup safeguards, protected-root
  refill, ordinary-prefix targets, and controller exclusion survive with
  focused tests.
- Cost-aware allocation, fused policy terms, sampled-row reference cache,
  biased SDPA, flat fp32 gradients, SUM synchronization, and direct backward
  remain present.
- Page transactions roll back atomically for their owned stores, and the direct
  compact seal removes the former overlapping legacy tree representation.
- Tiny CUDA acceptance covers sequential RB, fixed branched RB, KKT branched
  RB, and multi-player BR probe-fold flow. Prior audit evidence covers a tiny
  two-GPU exact-mass/SUM learner run; this one-visible-GPU re-audit did not
  repeat it.

These strengths do not clear the open readiness defects or establish
equal-work performance preservation.
