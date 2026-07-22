# Adversarial finding reconciliation

## Scope and disposition rules

This pass read batch reports `00` through `10` in full and rechecked the
highest-severity and conflicting claims against the cited authority and current
production call graph. It did not run benchmarks and does not turn a static
allocation or synchronization difference into a measured performance result.

The dispositions below mean:

- **upheld**: the claim and severity survive review;
- **upheld, revised**: the underlying issue survives, but its wording, scope, or
  severity must change;
- **duplicate-merge**: the evidence is valid but belongs to one canonical
  finding with other reports;
- **rejected**: the stated production behavior is not supported by the current
  call graph.

## Required conflict resolutions

### Fixed-beta T/C: production requires it; the helper is locally permissive

The broad reading of `AUD-RSQ-011` is rejected. Validated fixed-beta production
requires `model.use_branching_heads=true` and positive `c_t/c_c`
(`pbg/core/config.py:380-388`), and `_record_branched_controller_metrics`
rejects missing frozen T/C before returning even in fixed mode
(`pbg/core/training.py:983-1004`). Targets and the branched learner then train
the Gamma-deviance losses. This agrees with `core_math.tex:1352-1366` and
`:1562-1567`.

The narrower claim is upheld at **medium** severity: `sample_branched_tree`
only rejects missing T/C after KKT becomes active, while fixed/warmup capture
appends the fields conditionally (`pbg/core/branching.py:438-460,683-699`). A
direct helper caller outside the validated training frame can therefore omit
both fields. The remedy is unconditional branched schemas and a model-boundary
assertion, not a claim that installed fixed production currently skips T/C.

### `post_A` width: optimization opportunity, not proven under-admission

`AUD-MODEL-001` is **rejected as stated**. Its cited street-specific planner is
not called by production: repository call search finds no caller of
`plan_policy_batches`, and CUDA `policy_forward_family_actual_cost` deliberately
returns the family path without actual-row planning
(`pbg/games/poker/rollout_backend.py:2100-2127`). The active family forward,
learner planning, transient estimate, and model all use the flop-width
`_family_maximum_width` (`:1861-1880`, `:2188-2230`;
`pbg/core/training.py:2339-2370`). Therefore the inspected production path is
conservative rather than under-admitted.

`POKER-009` is upheld with revised wording: **CUDA execution and active planning
are both worst-width**. Turn/river rows pay flop padding, but this is an
optimization opportunity whose current speedup and memory reduction are
unmeasured. The historical 97% late-street/18.67-billion-pair figures in
`docs/todo/optimization.md` describe an earlier measured workload; they do not
prove a current-HEAD regression or a current 11--13% gain.

### Evaluator v1/v2: code defect survives, authority provenance is ambiguous

`POKER-003` is upheld with revised authority wording. The path named as binding
by `AGENTS.md`, `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`, does not
exist. The sibling instead contains `docs/api.md`, which declares
`CONTRACT_VERSION == "2"` at line 23, plus an explicitly stale
`docs/stale_api_pbg.md`. The repository launcher also requires v2
(`tools/poker-python:20`). Production facades nevertheless advertise v1 and
accept both versions (`pbg/games/poker/mc.py:21-60` and
`pbg/games/poker/folded.py:20-104`).

Thus accepting v1 is a confirmed high-severity compatibility defect relative
to the current sibling package/launcher. It should not be described as a
violation of the missing `api_pbg.md` until the user updates the authority path
or the sibling owner restores a maintained alias. `POKER-008` remains a
separate medium authority-path decision.

### Five semantic modes versus three training/storage labels

There is no sixth training estimator and no requirement for five duplicated
storage implementations. Strict validation currently yields exactly these five
semantic combinations:

1. `normal`;
2. `rao_blackwell` without a probe;
3. `branched_rao_blackwell` with `branching_mode=fixed`;
4. `branched_rao_blackwell` with `branching_mode=dynamic`;
5. `rao_blackwell` with a non-null probe.

The three internal labels are consistent with the three authoritative storage
families (`docs/theory/pseudocode/family_scheduled_rollout.tex:49-64`). Fixed
and KKT correctly share branched targets/losses. `TRACE-002` remains medium:
the absence of one closed derived semantic selector causes benchmark, preset,
manifest, and acceptance coverage to collapse or omit combinations. It is a
control-plane clarity defect, not proof of extra estimators.

### BR authority gap versus BR implementation failure

`A-01`/`EVAL-001` are upheld only as a **certification blocker**, not as proof
that the code implements the wrong BR equation. `core_math.tex` contains no BR
sampling law, response class, stopping rule, or NashConv estimator, so the
current shared-policy sequential-RB probe cannot be certified as the requested
fifth mode. The user must decide whether the authority should specify this
approximate shared response or independent per-seat responses.

`TRACE-001` is a separate, confirmed high implementation defect. Rollout writes
`PROBE_FOLD` only into tree ownership (`pbg/core/rollout.py:860-868`), but leaf
capture passes no cutoff identity/seat and reclassifies the poker state
(`pbg/core/rollout.py:565-610`;
`pbg/games/poker/rollout_backend.py:2590-2610`). For $P>2$, a selected player
can fold while the hand remains live; the state is then sent to terminal
settlement, which rejects non-terminal/non-preterminal rows
(`pbg/games/poker/rollout_backend.py:772-788`). Heads-up tests mask this.

The reachability inventory in batch 06/09 that calls `capture_probe_leaf`
uncalled is rejected: it is reached dynamically by the named `getattr` hook in
`chance_close`. It must be repaired, not deleted.

### `SampledTree`: reachable does not mean retainable

`AUD-RSQ-001`, `AUD-RSQ-002`, and `AUD-RD-003` are upheld and duplicate-merged
as one blocker. `PagedTreeSource.finalize` gathers full node/edge columns and
constructs `SampledTree` (`pbg/core/tree.py:555-587`); only afterward do the
sequential and branched seal paths allocate the normative compact topology and,
for branched, whole-tree reorder maps (`pbg/core/training.py:1725-1826`). Those
allocations are not a source-plus-destination hard-ledger transaction.

`SampledTree`, `PagedTreeSource`, `RolloutDecisionRows`, and `StoppedLeafRows`
are therefore **compatibility-reachable stale owners**. They cannot be deleted
before direct compact sealing is implemented, but current reachability is not
an argument to retain them.

### CPU/MPS classification

Core CPU/Gloo training (`MODE-001`, `AUD-TL-005`) is upheld at high severity:
config accepts `auto/cpu/cuda`, `run_training` selects or falls back to CPU, and
multi-process CPU uses Gloo (`pbg/core/config.py:278-287`;
`pbg/core/training.py:2543-2558,2708-2717`). This is a substantial parallel
production path, not the permitted negligible device-agnostic helper case.

The CPU arena, CPU/MPS inference product, and list engine/range implementation
are not sixth core modes. They are active installed non-core products and are
properly classified for archive under the user's GPU-only/production-only
scope. Before removal, the user must decide whether an unrecorded external
deployment consumer expands repository scope. Small pure CPU formula/oracle
tests may remain only when they share the production formula and add negligible
runtime code.

### Performance evidence classification

No report establishes a measured current-HEAD performance regression against a
same-hardware, same-workload pre-refactor run. Canonical language must be:

- **measured historical baseline/opportunity**: old `runs/` timings and the
  old postflop workload counts;
- **static algorithmic regression risk**: per-decision CUDA scalar reads in
  `pbg/core/targets.py:169-182,251-312`, repeated evaluator/learner host checks,
  unbounded child waves, full-tree conversion, and allocator flushes;
- **static memory-accounting defect**: allocations provably outside the hard
  ledger, without claiming an observed OOM;
- **unverified preservation**: KKT, refill, SDPA, flat-gradient backward, and
  cache machinery survive statically/focused tests but lack a current equal-work
  benchmark;
- **measurement gap**: the three-label benchmark omits KKT/BR and can silently
  omit modes/metrics.

`HIST-PERF-02` and `AUD-TL-002` are upheld as a high-confidence static risk,
not a measured regression. `HIST-OOM-01` is upheld as a high fixed-beta closure
gap, not as evidence that a current production run OOMed.

## Per-finding blocker/high disposition

| Batch finding | Disposition | Canonical destination |
|---|---|---|
| `A-01` | upheld, revised | C03 BR certification authority gap |
| `HIST-PERF-02` | duplicate-merge | C11 static CUDA synchronization regression risks |
| `HIST-OOM-01` | upheld, revised | C01/C07 static memory closure and fixed safeguard gaps |
| `MODE-001` | duplicate-merge | C08 core CPU/Gloo production path |
| `MODE-002` | duplicate-merge | C13 non-core CPU/MPS product surface |
| `MODE-003` | duplicate-merge | C12 incomplete/fail-open performance gate |
| `MODE-004` | upheld | C12; overrides bypass strict revalidation |
| `MODE-007` | duplicate-merge | C03/C04/C08 BR authority, correctness, and CUDA gate |
| `AUD-RSQ-001` | duplicate-merge | C01 forbidden legacy seal/handoff |
| `AUD-RSQ-002` | duplicate-merge | C01 unadmitted seal overlap |
| `AUD-RSQ-003` | upheld | C05 inferred/handwritten rollout schemas |
| `AUD-RSQ-004` | upheld | C05 preflop natural-width violation |
| `AUD-RSQ-005` | upheld | C06 logical-offspring-sized child transient |
| `AUD-RSQ-006` | upheld | C05/C06 late logical address reservation |
| `AUD-RSQ-007` | upheld | C07 queue scheduler differs from authority |
| `AUD-RSQ-008` | upheld | C07 queue clone/concatenation outside accounting |
| `AUD-RSQ-009` | upheld | C07 common fixed/warmup scientific safeguard absent |
| `AUD-TL-001` | upheld | C02 distributed terminal-failure agreement blocker |
| `AUD-TL-002` | duplicate-merge | C11 static target host-sync risk |
| `POKER-001` | duplicate-merge | C09 forbidden neutral `post_P` summary |
| `POKER-002` | duplicate-merge | C09 per-pack summary duplication before seal |
| `POKER-003` | upheld, revised | C10 v1 compatibility plus authority-path ambiguity |
| `POKER-004` | duplicate-merge | C13 list engine/range parallel product |
| `POKER-005` | upheld, revised | C11 static sync risk; performance magnitude unmeasured |
| `AUD-MODEL-001` | **rejected** | Replace with C17 worst-width optimization only |
| `AUD-MODEL-002` | duplicate-merge | C09 folded-summary cache omitted from accounting |
| `EVAL-001` | upheld, revised | C03 certification blocker, not an implementation equation bug |
| `EVAL-002` | duplicate-merge | C04/C08 BR correctness/domain and CUDA acceptance |
| `EVAL-003` | upheld | C16 all-player BR completion violates minimal projection |
| `EVAL-005` | duplicate-merge | C13 stale CPU/full-deck arena |
| `EVAL-006` | duplicate-merge | C13 CPU/MPS inference product |
| `EVAL-008` | duplicate-merge | C14 historical variance implementations in active package |
| `EVAL-010` | duplicate-merge | C12 incomplete/fail-open benchmark |
| `AUD-RD-001` | duplicate-merge | C13 non-core products |
| `AUD-RD-002` | duplicate-merge | C14 historical variance implementations |
| `AUD-RD-003` | duplicate-merge | C01 legacy tree blocker |
| `AUD-RD-005` | upheld, revised | C15 legacy compatibility; inventory/convert approved artifacts first |
| `AUD-RD-006` | duplicate-merge | C18 stale tests and missing structural gates |
| `AUD-RD-007` | upheld | C18 current docs contradict implementation/mode scope |
| `AUD-RD-008` | upheld with severity reduced to **medium** | C18 broken operational references/commands |
| `TRACE-001` | upheld | C04 multi-player BR early-fold failure |

`AUD-RSQ-011` was medium rather than high, but is included because it was a
named reconciliation tension: broad production omission rejected; narrow
helper/schema weakness upheld at medium.

## Canonical deduplicated finding list

1. **C01 — Blocker: direct compact seal and hard seal transaction are absent.**
   Replace the production `SampledTree` handoff and unadmitted conversion; merge
   `AUD-RSQ-001/002`, `AUD-RD-003`, and the seal portion of `HIST-OOM-01`.
2. **C02 — Blocker: rank-local mandatory failures lack common terminal
   agreement.** Reorder mandatory singleton feasibility before mass exchange
   and add asymmetric NCCL failure tests (`AUD-TL-001`).
3. **C03 — Certification blocker: BR has no authoritative algorithm.** User
   decision required for response class, seat law, objective, stopping, cutoff,
   and reported estimator (`A-01`, `EVAL-001`, authority part of `MODE-007`).
4. **C04 — High: multi-player BR early-fold settlement is broken and lacks a
   CUDA acceptance gate.** Merge `TRACE-001` and correctness/testing parts of
   `EVAL-002`/`MODE-007`.
5. **C05 — High: rollout schemas/domains/addresses violate immutable natural
   storage.** Merge inferred schemas, preflop H-width carriers, and late stable
   address reservation (`AUD-RSQ-003/004/006`).
6. **C06 — High: branched child materialization transient scales with logical
   offspring.** Bounded physical subspans are absent (`AUD-RSQ-005`, part of
   `AUD-RSQ-006`).
7. **C07 — High: queue scheduling, queue accounting, and common scientific
   safeguard differ from authority.** Merge `AUD-RSQ-007/008/009` and the
   fixed/warmup portion of `HIST-OOM-01`.
8. **C08 — High: core training/BR accepts CPU/Gloo instead of requiring
   NVIDIA.** Merge `MODE-001`, `AUD-TL-005`, and CUDA enforcement in
   `EVAL-002`.
9. **C09 — High: folded-summary ownership and accounting violate the sparse
   anchor contract.** Merge neutral `post_P`, per-pack duplicates, and omitted
   mandatory cache bytes (`POKER-001/002`, `AUD-MODEL-002`).
10. **C10 — High: production accepts evaluator v1.** Require the current v2
    package after resolving the missing binding-document path (`POKER-003`;
    `POKER-008` remains a medium documentation decision).
11. **C11 — High static performance risk: repeated CUDA-to-host controls exist
    in targets and evaluator/model boundaries.** No current regression magnitude
    is measured (`HIST-PERF-02`, `AUD-TL-002`, `POKER-005`, plus medium
    `POKER-006`/`AUD-TL-004`).
12. **C12 — High: performance evidence gate is incomplete and fail-open.** It
    exposes three labels, forces fixed branching, omits BR/KKT, mutates without
    strict revalidation, and tolerates missing modes/metrics (`MODE-003/004`,
    `EVAL-010`).
13. **C13 — High: active installed CPU/MPS/list-engine products are outside the
    five GPU workflows.** Archive after external-scope confirmation; merge
    `MODE-002`, `POKER-004`, `EVAL-005/006`, and `AUD-RD-001`.
14. **C14 — High: checkpoint-specific variance algorithms remain active rather
    than archived.** Merge `EVAL-008` and `AUD-RD-002`.
15. **C15 — High: indefinite compatibility paths remain active.** Inventory and
    offline-convert approved checkpoints/configs first, then delete old
    checkpoint, config, model, evaluator, and benchmark shims (`AUD-RD-005`).
16. **C16 — High: BR completion computes/stages all players although only the
    selected owner is consumed.** Project immediately and expose selected-owner
    evaluator output if supported (`EVAL-003`).
17. **C17 — Optimization, unmeasured: CUDA `post_A` uses conservative flop
    width for flop/turn/river.** Preserve one statistical partition and test a
    street-aware physical plan (`POKER-009`); do not claim current under-admission
    or a measured speedup.
18. **C18 — High documentation/test-readiness drift, with broken command paths
    medium.** Rewrite false architecture/mode claims and replace stale tests
    with authority-defined structural gates (`AUD-RD-006/007`; revised
    `AUD-RD-008`).

## Unsupported overclaims and required user decisions

- Do not claim a measured current performance regression; none is backed by a
  comparable current run.
- Do not claim production fixed-beta omits T/C; it does not.
- Do not claim production postflop admission uses turn/river width while the
  model executes flop width; the cited narrower planner is not production
  reachable.
- Do not classify `capture_probe_leaf` as dead; dynamic rollout dispatch reaches
  it.
- Do not call three storage labels a violation of the five-mode requirement;
  the problem is missing canonical semantic identity in cross-mode consumers.
- Do not certify BR math until the user supplies an authoritative contract.
- The user must confirm whether CPU/MPS inference or checkpoint-versus-checkpoint
  arena evaluation has an intentionally supported external consumer; absent
  that scope expansion, archival follows the stated repository mandate.
- The user must choose which historical checkpoints/configs need one-time
  conversion before compatibility deletion.
- The user or sibling owner must resolve the missing evaluator authority path;
  current runtime/package evidence supports v2, but the named binding document
  is absent.
