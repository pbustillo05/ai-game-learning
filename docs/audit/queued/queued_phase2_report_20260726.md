# Queued Phase 2 closure report

Date: 2026-07-26

## Revisions

- clean phase start: `803221ed12cee902503d0a3e31a58c46afba8cca`
- first candidate: `e778c73d72dde57ec46cf180ed27a6a8966b873b`
- validated and audited candidate:
  `5bd7d0ebb900ec4dddc6a951646d6e1e62622a3b`
- closure commit: the commit containing this report; its parent must be the
  candidate above

The audit of the first candidate found that a transaction could be constructed
across different lanes/failure authorities and that whole-attempt reset did
not preflight every live owner before mutating any. The fix candidate requires
one lane/common failure state and uses a two-pass all-owner reset. All
validation and audit evidence below is for the fix candidate.

## Environment

| Item | Identity |
|---|---|
| selected device | CUDA GPU 0, NVIDIA GeForce RTX 2080 Ti |
| device memory | 11,346,575,360 bytes |
| driver / reported CUDA | 580.173.02 / 13.0 |
| Python | 3.10.12 from `/home/pablob/.venvs/poker` |
| PyTorch / toolkit | 2.12.1+cu130 / CUDA 13.0 |
| cuDNN | 92000 |
| evaluator boundary | contract 2; repository `9d35191b396cca5bf07518a07ba2976b11971a5a` |

The venv CUDA toolkit, `PATH`, and `LD_LIBRARY_PATH` required by `AGENTS.md`
were set for every CUDA command. `nvidia-smi` showed GPU 0 idle with only its
6 MiB display allocation before validation. No evaluator operation was invoked
because Phase 2 validates only its bounded-workspace manifest contract.

## Exact-candidate CUDA validation

With `CUDA_VISIBLE_DEVICES=0`:

```bash
python -m pytest -q \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py
```

Result: `25 passed in 2.22s`.

The Phase 2 subset was also repeated independently:

```bash
python -m pytest -q tests/test_q02_storage_transactions.py
```

The pre-fix repeat was `11 passed in 1.69s`; the final combined run includes
the additional whole-owner retry test and passes all 12 Phase 2 tests.

Coverage includes:

- immutable schema/field/capacity/extent records;
- one aligned backing arena and exact tensor-storage accounting;
- every declared CUDA owner for all twelve mode rows;
- exact field absence/presence inherited from `ModeSchema`;
- every phase allocation and last-consumer release exactly once;
- configured, analytical, allocated, high-water, and observed byte
  reconciliation in every phase;
- successful and injected-failure multi-destination publication;
- overflow before any reservation and minimum-address failure selection;
- deterministic batched minimum failure selection for multiple lanes;
- disjoint retry epochs and nonrepeating monotone row addresses;
- complete failed-lane discard across all live owners while a completed lane
  remains byte-identical;
- exact `ExactBlock` realized extents and explicit release;
- transactional rollback of partially completed phase allocation;
- rejection of uncovered partitions, geometry, tokens, workspaces, and memory
  budgets; and
- queued import and forbidden-storage static gates.

The cumulative legacy storage/memory regression command was:

```bash
python -m pytest -q \
  tests/test_execution_foundations.py \
  tests/test_memory_control.py \
  tests/test_refactor_memory_width.py \
  tests/test_refactor_storage_targets.py \
  tests/test_storage_lifecycle_migration_gates.py
```

Result: `66 passed in 2.94s` on CUDA GPU 0.

Phase 2 introduces no rollout sampler, stochastic law, or delivered training
mode, so it consumes none of the three matched-work scientific seeds and makes
no end-to-end throughput claim. Its performance-relevant gate is exact fixed
allocation and bounded phase admission; rollout timing begins with its first
consumer phase.

## Read-only ownership audit

The exact candidate audit found:

- queued code imports neither `pbg.core` nor poker;
- all twelve rows are generated through the same physical-layout function;
- every `ModeSchema` owner is accounted as a CUDA allocation or one of the two
  explicitly deferred compact CPU topology owners;
- there is one field/byte authority, one fixed slab, one exact block, one
  failure state, one immutable capacity manifest, and one aggregate lifetime
  owner;
- no page, recyclable pool, free list, growth method, wraparound queue,
  inferred schema, convenience store, compatibility loader, or runtime engine
  selector exists;
- failure recording uses preallocated CUDA state and selects the minimum
  logical address per lane;
- publication and retry are lane-local; a failed lane cannot alter a completed
  lane;
- phase overlap includes source, destination, fixed scratch, external
  workspace, and guard bytes before allocation;
- every generated allocation is released at its declared last consumer and
  exactly once; and
- no old core, poker, configuration, runtime, evaluator, or authoritative
  theory file changed.

The old core remains reachable only from the legacy production roots because
the plan requires isolated replacement until cutover. Phase 2 ledger rows have
verified replacement evidence but remain open for final old-definition and
caller deletion.

The mandatory plan calls for separate validation and audit agents. Session
instructions prohibited spawning sub-agents without an explicit request for
delegation, so exact-revision validation and a separate read-only audit pass
were performed locally. No gate was waived.

## Verdict

Phase 2 passes. The closure commit contains only this report, the verified
storage/lifetime contract, and migration-ledger evidence. Any later executable
change to the Phase 2 mechanisms requires fresh CUDA validation and audit.
