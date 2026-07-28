# Queued Phase 3 closure report

Date: 2026-07-26

## Revisions

- clean phase start: `0fabbd42d9e30af89e7ddd3f18a7a429cace4f18`
- initial candidate: `880ff860d0159fe7358893f0b6d97c9e1547ad01`
- compiled one-kernel-per-layer candidate:
  `452e24bb2e2587d9dcf7120f71cfe15b1934fd40`
- successive audited fix candidates:
  `0d501707151859f40da2777479b2bdd9331ae40b`,
  `fdffaad265601ae541cef9cd416b85d8b9c06237`,
  `7833a9cc7309cf3e92fe6a96e465b153b2b2fd49`,
  `81f2bbcc57f23fcac8121dbef61c334fa70443b0`, and
  `c8ee86ba7dd91b0c5ef872c4a09ac1807fd2ca0a`
- validated and audited executable candidate:
  `ec9ef6f725a8f986c4b2910b348b1c2a09b804ea`
- closure commit: the commit containing this report; its parent must be the
  executable candidate above

The iterative audits closed implicit multi-kernel execution, signed sampled
returns, incomplete RB validation, unchecked plan addresses and conversions,
unary-label corruption, carry surfaces, scratch/traffic accounting, completion
schemas, child-span partitioning, sequential degree and row identity, target
destination coherence, leaf movement bijection, and nominal-degree admission.

## Environment

| Item | Identity |
|---|---|
| selected device | physical CUDA GPU 1, NVIDIA GeForce RTX 2080 Ti |
| device memory | 11,346,575,360 bytes |
| driver / reported CUDA | 580.173.02 / 13.0 |
| Python | 3.10.12 from `/home/pablob/.venvs/poker` |
| PyTorch / toolkit | 2.12.1+cu130 / CUDA 13.0 |
| cuDNN | 92000 |
| evaluator boundary | contract 2; repository `9d35191b396cca5bf07518a07ba2976b11971a5a` |

The venv CUDA toolkit, `PATH`, and `LD_LIBRARY_PATH` required by `AGENTS.md`
were set for every CUDA command. GPU 1 was idle with only its 6 MiB display
allocation before and after validation, at 0% utilization and with no compute
process.

## Exact-candidate validation

With `CUDA_VISIBLE_DEVICES=1`:

```bash
python -m pytest -q tests/test_q03_layered_targets.py
```

Result: `45 passed in 3.18s`.

```bash
python -m pytest -q \
  tests/test_q01_mode_contracts.py \
  tests/test_q02_storage_transactions.py \
  tests/test_q03_layered_targets.py
```

Result: `70 passed in 4.09s`.

```bash
python -m pytest -q \
  tests/test_execution_foundations.py \
  tests/test_memory_control.py \
  tests/test_refactor_memory_width.py \
  tests/test_refactor_storage_targets.py \
  tests/test_storage_lifecycle_migration_gates.py
```

Result: `66 passed in 2.89s`.

The targeted static/firewall set passed `4/4` in `2.20s`. Compilation,
diff-check, queued firewall, forbidden-literal, dual-engine-root, and production
carry searches were clean.

## Profiler and adversarial evidence

The independent two-layer profile reported plan/counter/profile kernel counts
of `2/2/2`. It observed no CUDA activity or device-to-host copy between the
layer kernels; all 27 copies were confined to prelaunch staging or the final
publication boundary.

All twelve modes replayed bitwise identically, including six negative sampled
and six zero-mass RB cases. Maximum replay allocation was 30,720 bytes.

Coverage included sampled and RB multi-layer recurrences, trace/cost/delta,
raw RB laws before conversion, internal H-to-K, protected prefixes, duplicate
labels, unary paths, unbalanced trees, maximum degree three and depth eight,
exact child/leaf/target bijections, sequential aligned row identity, completion
schema crossings, malformed-plan rejection, headless rows, and replica-local
atomic failure. The depth-eight fixture produced root sigma `2.0625` and leaf
sigma `10`. Aligned two-row self-play and best-response fixtures produced sigma
`[2, 4]` and advantage `[0, 2]`.

## Exact memory and traffic

| Quantity | Bytes |
|---|---:|
| analytical owners | 8,000 |
| named allocated owners | 8,000 |
| CUDA current before / after | 16,384 / 16,384 |
| transient peak increment | 10,752 |
| executor scratch required / reported | 1,324 / 1,324 |
| movement indices | 24 |
| admitted requirement | 1,058,624 |
| configured budget | 67,108,864 |

The exact RB traffic was four protected edges, 256 bytes read, 64 bytes
written, and 96 target bytes written.

## Required selection benchmarks

### Direct sibling sum versus total-minus-child

| Implementation | Median CUDA time | Peak incremental bytes | Result |
|---|---:|---:|---:|
| direct logical-order sibling sum | 0.02430 ms | 4,073,472 | 0 |
| total-minus-child | 0.01436 ms | 1,357,824 | -1 |

Direct summation is slower, but it preserves unit terms lost by
total-minus-child. It is the sole production implementation.

### Physical copies versus last-consumer retention

| Implementation | Four-layer time | Payload traffic |
|---|---:|---:|
| physical copy | 0.65830 ms | 173,801,472 bytes |
| last-consumer retention | 0.000252 ms | 0 bytes |

Last-consumer retention is the sole production implementation. All carry
planning, schemas, uploads, kernels, and counters are deleted.

## Read-only audit

The final audit passed after independently tracing topology, plan admission,
completion movement, CUDA recurrences, publication, ownership, and stale
surfaces. It confirmed one topology owner, one plan, one compiled executor,
exact address/conversion/child/leaf/target/aligned-row coherence, closed CUDA
completion schemas, exact accounting, atomic lane-local publication, no carry
surface, no parallel target path, and no forbidden import or CPU production
state.

## Verdict

Phase 3 passes. This closure commit contains only this report, the verified
topology/target contract, and migration-ledger evidence. Any later executable
change to these mechanisms requires fresh CUDA validation and audit.
