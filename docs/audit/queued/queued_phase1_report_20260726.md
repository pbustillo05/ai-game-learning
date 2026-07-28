# Queued Phase 1 closure report

Date: 2026-07-26

## Revisions

- clean repository start: `2e19bd89279df31d5491d089ab4cb5f2a5a52ff1`
- separately approved authority prerequisite:
  `80a9e23917ef5e70d36b7d7bdcc3d49c6a3c4b7c`
- validated and audited candidate:
  `7e371b8daf9f978ad2e479edfda6ae87fdaadb6b`
- closure commit: the commit containing this report; its parent must be the
  candidate above

The candidate worktree was clean. This closure contains documentation,
ledger/report evidence, and status wording only.

## Environment

| Item | Identity |
|---|---|
| host GPU | NVIDIA GeForce RTX 2080 Ti, GPU 0 explicitly selected |
| driver / reported CUDA | 580.173.02 / 13.0 |
| Python | 3.10.12 from `/home/pablob/.venvs/poker` |
| PyTorch | 2.12.1+cu130 |
| cuDNN | 92000 |
| evaluator | contract 2, package 0.6.0, revision `9d35191b396cca5bf07518a07ba2976b11971a5a` |

The venv CUDA toolkit location and `PATH`/`LD_LIBRARY_PATH` required by
`AGENTS.md` were exported. `nvidia-smi` showed all three GPUs idle before work.
GPU 0 held only the 6 MiB display allocation before and after validation; no
worker or strange lingering allocation remained.

## Candidate validation

Command, with `CUDA_VISIBLE_DEVICES=0`:

```bash
python -m pytest -q tests/test_q01_mode_contracts.py
```

Result on the exact candidate: `13 passed in 1.97s`.

The test matrix covered:

- the exact twelve-row Cartesian product and lookup identity;
- immutable rows and nested owner records;
- rejection of invalid/aggregate legacy identities;
- absence of every representative inapplicable field/config entry;
- fixed/KKT physical compatibility with distinct semantic identity;
- rejection of incompatible/empty replica groups;
- closed partitions, mappings, conversions, and malformed-contract failure;
- the 18-name capacity vocabulary;
- import and legacy-vocabulary firewalls, including injected failures;
- transitive dual-engine root detection, including an injected failure;
- legacy module and active configuration inventory coverage;
- three representative real CUDA schema round-trips;
- explicit rejection of CPU schema validation.

Phase 1 contains no stochastic production rollout or benchmark, so it consumes
no logical scientific seed and makes no throughput claim. The three frozen
matched-work seeds and later-phase thresholds are recorded in the prerequisite
closure.

## Broader baseline

The additional legacy subset:

```bash
python -m pytest -q \
  tests/test_p0_config.py tests/test_checkpoint_schema.py \
  tests/test_p0_imports.py tests/test_accelerator_only_runtime.py
```

reported `49 failed, 13 passed`. These are pre-existing baseline failures
outside the isolated candidate:

- configuration tests and one accelerator test require the already-deleted
  `configs/train/nlhe6.yaml`;
- the legacy import-direction test reports the existing core-to-poker replica
  imports.

The candidate adds no import to an accepted production root and changes no
legacy config, core, poker, checkpoint, or runtime file. The Phase 1 static
gate independently passes against the actual current roots.

## Read-only ownership audit

The exact candidate audit found:

- 12 unique semantic identities from one `ModeSchema` generator;
- 8 physical signatures: fixed and KKT pair only within the same
  regime/estimator, while the other axes remain incompatible;
- exactly four sampled-branched negative-control rows;
- no nullable owner field and no alternative schema class;
- no import from queued core to old core or poker;
- no production root reaching both engines;
- no poker label in generic partitions/hooks;
- every old core/poker module classified in the migration ledger;
- every accepted root, active config family, persistence schema, selected
  artifact family, capacity, and preservation family assigned a disposition,
  owner, phase, and gate;
- no saved H100 hunk applied or copied into the queued implementation.

No production storage, queue, target, learner, optimizer, or compatibility
object was introduced in Phase 1.

The mandatory plan calls for separate validation and audit agents. Session
instructions prohibited spawning sub-agents without an explicit user request,
so validation and the read-only audit were performed locally against the same
candidate SHA. This constraint caused no code or gate waiver.

## Verdict

Phase 1 passes. Later migration rows remain open; only the vocabulary,
authority closure, inventories, firewall, and generated schema contract are
closed here.
