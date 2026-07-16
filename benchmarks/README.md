# Comparative performance gates

`pbg-benchmark` runs the same frozen scientific configuration through normal,
sequential Rao--Blackwell, and branched Rao--Blackwell training. It reports
median phase times, realized trees/nodes/leaves/decisions and offspring,
completion effective samples, retained target bytes, normalized throughput,
and CUDA peak allocated/reserved bytes. CUDA measurements synchronize only at
explicit benchmark phase boundaries; normal training does not pay this cost.
After warmup, every timed repeat restores identical model, optimizer, scaler,
and RNG state, so realized topology and sampled work are stable across repeats.

Create a device-local baseline after the machine is idle:

```bash
pbg-benchmark --config configs/train/nlhe6.yaml --warmup 2 --repeats 7 \
  --output benchmarks/results/nlhe6.json
```

Enforce a later run with the default 20% tolerance:

```bash
pbg-benchmark --config configs/train/nlhe6.yaml --warmup 2 --repeats 7 \
  --baseline benchmarks/results/nlhe6.json
```

The gate refuses a different device, compute capability, CPU thread count, or
workload unless `--allow-environment-mismatch` is explicit. Throughput may not
fall below the tolerance; phase total, target bytes per decision, retained
bytes, and CUDA peak reserved bytes may not rise above it.

For CUDA, the runner requires an empty PyTorch allocator before setup, permits
at most 64 MiB of process-global framework/evaluator context after teardown,
records that residue, and preserves the pre-run free-memory level within 64
MiB of allocator storage and 512 MiB of driver-context accounting. Operators
must still run `nvidia-smi` before and after the command to
ensure the selected GPU is globally idle; the process cannot identify another
user's workload safely.

`workloads.yaml` is the minimum active NLHE coverage matrix. Compact-deck
coverage is exercised by the poker geometry and integration tests. Larger
NLHE rows use command-line size overrides so the scientific configs remain the
single source of game/model semantics.
