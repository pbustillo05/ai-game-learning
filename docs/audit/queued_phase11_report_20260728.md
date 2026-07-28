# Phase 11 cutover report

Date: 2026-07-28

Starting revision: `4af4ba6c3370930e7b2b402a1e6355160cb75c5d`

Validated and audited executable revision:
`71a92378cf89a81ed5724ba61515fa6c9f441706`

## Result

Phase 11 passes. The old core was atomically replaced by the queued
implementation, imports and public roots were moved to `pbg.core`, and the
isolated replacement namespace was deleted. Legacy poker modules, comparison
and migration scaffolding, stale benchmarks/prototypes, compatibility
surfaces, test-only production helpers, and superseded configuration were
deleted.

Independent validation and a different read-only authority audit both passed
the exact executable revision above. The worktree remained clean throughout
those final checks.

## Environment and GPU hygiene

Python work used `/home/pablob/.venvs/poker`. CUDA work used:

```bash
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Initial and final `nvidia-smi` readings were identical:

| GPU | Baseline memory | Utilization | Compute processes |
|---:|---:|---:|---:|
| 0 | 6 MiB | 0% | none |
| 1 | 6 MiB | 0% | none |
| 2 | 16 MiB | 0% | none |

Only idle GPUs were selected. The two-GPU test used GPUs 0 and 1 explicitly.

## Static and reachability gates

The independent validator ran:

```bash
git diff --check
/home/pablob/.venvs/poker/bin/ruff check --select F401,F821 pbg tests tools
/home/pablob/.venvs/poker/bin/python tools/audit_reachability.py --fail-on-high
/home/pablob/.venvs/poker/bin/python -m pytest --collect-only -q
```

Results:

- 441 definitions indexed;
- 31 accepted production roots;
- 441 definitions reachable;
- zero review candidates;
- exactly 312 tests collected;
- all 13 public/consumer import probes passed;
- no stale generic `pbg.queued` import or directory remained; and
- `git diff --check` and selected Ruff gates passed.

Manual and scripted history/similarity audits found no retained migrated
implementation body. The only old-body matches were the intentional
accelerator-device move and a trivial optimizer zeroing operation. Internal
similarity groups were limited to small rank-boundary and timing helpers, not
parallel authorities.

## CUDA suite

The final cumulative command was equivalent to:

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v \
  python -m pytest -q -rs
```

Result:

- 310 passed;
- two expected skips: unavailable MPS hardware and the two-GPU test hidden by
  the single visible GPU;
- pytest time: 101.45 seconds;
- wall time: 1:42.87; and
- maximum host RSS: 2,271,636 KiB.

The attention primitive suite separately passed all six tests in 2.62 seconds.
Its three neural equation/oracle/backward tests explicitly placed modules and
all tensors on the selected CUDA device. Only retired-selector validation and
CPU fail-fast validation remained CPU-side.

## Distributed gate

The explicit two-GPU command selected GPUs 0 and 1 and ran:

```bash
python -m pytest -q \
  tests/test_replica_identity.py::test_distributed_nccl_sum_and_asymmetric_failure_agreement
```

Result:

- one passed in 5.48 seconds;
- wall time: 6.29 seconds; and
- maximum host RSS: 1,273,140 KiB.

## Matched-work benchmark

The public benchmark used the sole retained smoke configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pbg.cli.benchmark \
  --config configs/train/nlhe_hu_8bb_normal_smoke.yaml \
  --warmup 1 \
  --repeats 2 \
  --output /tmp/phase11-benchmark.json
```

Seed `104830` produced identical logical work, telemetry, non-timing
structure, and five nodes in each sample:

| Sample | Seconds | Nodes/s | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|
| 1 | 0.31242 | 16.004 | 30,920,192 B | 52,428,800 B |
| 2 | 0.31199 | 16.026 | 30,920,192 B | 52,428,800 B |

The command exited successfully in 4.56 seconds wall time.

## Adversarial closure

Validation and audit specifically proved:

- CPU attention rejects before validation, projection, or neural work;
- CUDA dispatch calls only the configured `sdpa` or `chunked_dense` path;
- CUDA and MPS share that dispatch without a device-specific override;
- production contains no explicit-attention oracle, `auto`, `flex`, or
  fallback;
- hash-valid strict artifacts reject `n_players`, `eps`, `k_act`,
  `use_branching_heads`, and `max_history_tokens`;
- sequential and branched chance paths use one `BoundedChanceClosure`;
- `GeneratedOwners` is the sole allocation/lifetime reconciliation authority;
- all twelve mode rows remain closed;
- the generic core imports no poker implementation;
- no production object exists solely for tests; and
- no executable prototype, comparison benchmark tree, or rollback engine
  remains.

## Commit sequence

- `63c269154d4e5823ce3f9e5352ed03a730c523ef` — atomic core cutover;
- `7434d41d88157c3163963fad512764a37ce995b7` — remove residual
  compatibility/test-only surfaces;
- `0950cb33b4bb54ffaaf3d0fbe016d90759263bc7` — delete implicit attention
  and artifact aliases;
- `f08afe1e06ffc7ecb6d6a540a9b08cb6b75bee24` — delete CPU attention
  fallback;
- `8e3970f620b8651828c80f744ef4d0b98f1051e0` — correct the source
  attention contract; and
- `71a92378cf89a81ed5724ba61515fa6c9f441706` — move all retained
  attention equation tests to CUDA.

This report, the ledger closure, and queued-document path updates form the
separate documentation-only closure commit whose parent is the exact
validated/audited executable revision.
