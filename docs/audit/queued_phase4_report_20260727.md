# Queued Phase 4 validation and audit

Date: 2026-07-27

Status: passed.

Executable candidate:
`de6b429ce5c174df1af21e5709bfd9d51e007429`.

The candidate worktree was clean before and after independent validation and
read-only audit. Documentation was added only after the executable candidate
passed both gates.

## Delivered contract

Phase 4 delivers:

- semantic `FamilyWork` FIFOs and stable compatible-cohort scheduling;
- lane-local fixed/KKT schema identity without inactive padding;
- schedule-independent logical RNG with authoritative retry epochs;
- one generic `model -> sample -> commit` state machine;
- descriptor-derived device counts, scans, reservations, exact coverage, and
  rollback-safe atomic publication;
- declared cross-family branched fan-out;
- replica-local asynchronous topology drafts, reservation-bound endpoints,
  nonblocking polling, and bounded aggregate sealing;
- bounded chance closure and exact `Leaf -> LayerWork` completion.

Production owner and bound authority comes from
`RolloutOwnerRegistry.from_generated()`. The registry binds the complete live
owner set, rollout/completion workspace admissions, and manifest pack,
topology, descriptor, and geometry bounds.

## CUDA environment

Validation used:

```text
venv                 /home/pablob/.venvs/poker
CUDA toolkit         nvidia/cu13 from that venv
CUDA_VISIBLE_DEVICES 0
GPU                   NVIDIA GeForce RTX 2080 Ti
driver/toolkit        driver 580.173.02 / CUDA 13.0
```

GPU 0 was idle before testing and returned to its 6 MiB, 0% utilization
baseline with no compute process afterward.

## Test evidence

Focused:

```bash
python -m pytest -q tests/test_q04_rollout_transactions.py
# 20 passed
```

Cumulative:

```bash
python -m pytest -q \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py \
  tests/test_q03_layered_targets.py \
  tests/test_q04_rollout_transactions.py
# 91 passed
```

The targeted rollback, cross-family, topology, chance, completion, and
terminal-contract selection passed 13 tests.

Adversarial evidence included:

- every one of 1,052 declared active route rows covered exactly once in FIFO
  order;
- RNG agreement for 4,097 rows across uneven subdivisions and alternating
  streams, with 586 inactive rows drawing zero;
- exact three-lane scalar/vector materialization at a capacity collision;
- `int64.max` extent rejection across two destinations with no publication
  and minimum failure address 9;
- post-publication rollback followed by a real
  `GeneratedOwners.abort_failed_lane()` reset of every live owner;
- one-to-two branched `pre -> post2` FIFO fan-out;
- divergent replica reservation tails and equal per-replica topology
  sequences;
- chance-generation exhaustion and skipped coverage as terminal failures;
- illegal non-`LayerWork` completion rejection.

Topology ordering received two final isolated probes:

1. `device_valid=False` and `target_ordinal=-1` were produced on another
   nonblocking stream, then changed to true/zero before `validation_event`.
   The commit waited and published exactly one leaf/source row and one topology
   packet.
2. A permanently negative ordinal terminated the subprocess with the expected
   validation message and never reached the post-publication sentinel.

## Profiler and timing

For 1, 2, 4, and 8 packs, every measured hot-path trace reported:

```text
D2H copies             0
H2D copies             0
aten::nonzero          0
host local scalar      0
cudaStreamSynchronize  0
```

Each trace had one expected profiler-boundary device synchronization.

Matched work was 327,680 rows:

| Shape | Median | Per row | Peak temporary bytes |
|---|---:|---:|---:|
| 16,384-row packs | 26.716 ms | 0.0815 us | 1,217,024 |
| 1,024-row packs, 16-way subdivision | 422.909 ms | 1.2906 us | 80,384 |

The measurement demonstrates that scheduling subdivision changes launch
overhead, not row identity, labels, RNG, or retained results.

## Legacy context

The broader legacy selection remained:

```text
92 passed, 5 failed
```

All five failures are unchanged stale references to the already deleted
`configs/train/nlhe6.yaml`; no Phase 4 executable regression caused them.

## Independent audit

The audit iterated until every critical/high finding was closed. The final
read-only audit of `de6b429ce5c174df1af21e5709bfd9d51e007429`
confirmed:

- endpoint and topology validation are ordered on the commit stream before
  staging or reservation;
- coverage-derived routing cannot skip or reorder a source/destination row;
- cross-family queue effects are declared and unique;
- rollback records a shared terminal failure and supports generated-owner
  reset;
- completion is restricted to matching `LayerWork` partitions;
- topology polling, transfer slots, cumulative storage, and sealing are
  bounded;
- rollout and completion use their own admitted workspace bounds;
- retained owners and failure records are guarded around hooks.

No remaining critical or high-severity executable issue was found.

## Closure

Phase 4 replacement evidence is closed. Legacy definitions and callers remain
open in the migration ledger until their consumer phases land and the old
engine is deleted at cutover.
