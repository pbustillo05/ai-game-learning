# Queued Phase 7 validation and audit

Date: 2026-07-27

Status: passed.

Clean phase start:
`fbaeb1fc60a89a11197e5ff65f107473903ec981`.

Final executable candidate:
`0d035afc6c300c382d22f336f7ea3be52b138e70`.

The candidate worktree was clean before and after independent validation and
the final read-only audit. This report, the branched-execution contract, and
the ledger update were added only after both gates passed.

## Delivered contract

Phase 7 completes sampled fixed/KKT branching for self play and best response
through one queued engine. It includes keyed randomized rounding, exact
nominal/executed/target degrees, protected root refill, shared tree profiles,
duplicate-child identity, sampled completion, layered trace/cost targets,
additive CUDA decision log mass, linear CUDA alias sampling, ragged learner
gathers, cost-aware allocation, optional bounded reference caching, and
whole-attempt rank-common recovery.

The audit cycle additionally closed multiplayer folded summaries, inactive
chance draws and path identity, configured rollout precision/autograd
lifetime, compact learned/frozen role selection, complete memory admission,
integer controller/manifest identity, exact checkpoint root count, and
explicit final-consumer release.

## CUDA environment

Independent final validation used:

```text
venv                 /home/pablob/.venvs/poker
CUDA toolkit         nvidia/cu13 from that venv
CUDA_VISIBLE_DEVICES 0
GPU                   NVIDIA GeForce RTX 2080 Ti
driver/toolkit        driver 580.173.02 / CUDA 13.0
```

GPU 0 was idle before testing and returned to its 6 MiB, 0% utilization
display baseline with no compute process afterward.

## Test evidence

Focused:

```bash
python -m pytest -q tests/test_q07_branched_sampled_modes.py
# 32 passed in 80.76s
```

Adversarial selection:

```bash
python -m pytest -q tests/test_q07_branched_sampled_modes.py \
  -k 'phase7_config or branch_controller_requires or phase7_admission or \
  decisions_are_pack_invariant or whole_attempt_backoff or \
  failed_branched_attempt or healthy_branched_attempt or \
  untrained_unary_root_refill or inactive_fixed_chance or \
  full_branched_iteration_is_family_pack_invariant or multi_pack_roots or \
  kkt_feedback or best_response_rollout or multiplayer_best_response or \
  branched_allocator_and_reference_cache or three_player_nonterminal_fold or \
  chance_branching'
# 21 passed, 11 deselected in 52.74s
```

Cumulative:

```bash
python -m pytest -q \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py \
  tests/test_q03_layered_targets.py \
  tests/test_q04_rollout_transactions.py \
  tests/test_q05_sequential_sampled_training.py \
  tests/test_q06_sequential_modes.py \
  tests/test_q07_branched_sampled_modes.py
# 184 passed in 121.26s
```

Ruff, import checks, candidate/full-phase diff checks, an ineligible/eligible
reference-cache construction probe, and an explicit sampler-release probe
also passed. Embedded deterministic logical seeds cover controller decisions,
recovery, scheduling, four-mode optimizer runs, multiplayer roles, folded
summaries, and cache allocation.

## Adversarial coverage

Validation demonstrated:

- optimizer steps for all four sampled branched modes;
- shared fixed/KKT engine and deterministic pack-invariant decisions;
- protected root prefixes and multi-child refill;
- inactive chance rows with no ROUND draw, deal, or path advance;
- one sampled private profile shared by siblings;
- duplicate action labels with distinct child identities;
- four-player best-response root actor 3 using the frozen model for selected
  seat 0;
- one compact learned/frozen authority shared with sequential execution;
- three-player nonterminal folded-range publication and sealed summaries;
- finite additive log mass through a 700-node degree-four chain;
- deterministic linear alias construction through 65,536 rows;
- cost-aware sample allocation and exact cached reference gathers;
- first-chunk cache rejection and bounded `[2,2,1]` admitted construction;
- exact smaller/larger root-count mismatch rejection;
- whole-attempt discard, disjoint retry identity, controller backoff, and
  rank-common failure agreement; and
- alias, log-normalizer, probability, and store-reference release immediately
  after optimization.

## Memory and lifetime evidence

Rollout admission adds concurrent family, best-response merge, chance
frontier, expanded folded-statistics, compact topology plus dummy row, and
library workspace. Align-and-seal adds topology-sized CUDA log-mass metadata.
Learner admission adds sigma/action workspaces, the 44-byte-per-decision alias
construction peak, reference cache build/resident overlap, and library
workspace.

The cache builder performs one bounded first forward before deciding whether
to continue. Full payload size is inferred from that chunk, `resident +
payload + payload` must fit, and every subsequent forward remains capped by
`learner_sigma_pack_cap`. The normal learner model workspace is admitted
separately and concurrently.

Failed attempts release all generated owners before replacement. Successful
attempts release compact topology, completion, alignment, targets, caches,
alias tables, log normalizers, stratum probability, and store references at
their exact final consumers.

## Independent audit

The audit iterated over successive exact executable SHAs until every
critical/high and ownership/memory blocker closed. The final read-only audit
of `0d035afc6c300c382d22f336f7ea3be52b138e70` reported PASS.

It independently confirmed one fixed/KKT controller path, one compact
learned/frozen selector, one topology/target/learner path, exact CUDA role
state, complete folded-range semantics, inactive chance identity, bounded
cache construction, exact root/controller compatibility, admitted linear
alias/log-mass operations, explicit last-consumer release, whole-attempt
rank-common recovery, and no queued production import of `pbg/core`.
