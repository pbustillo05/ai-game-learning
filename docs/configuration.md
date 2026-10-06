# Configuration

Normal NLHE runs use the closed `nlhe_run_v1` schema. A run file contains only
scientific choices, run control, recovery policy, and explicit performance
preferences. It never contains a capacity manifest, physical layout, device
memory claim, tensor shape, or workspace-byte estimate.

The configuration flow has three products:

1. The user-authored YAML is parsed and canonicalized. Unknown and
   mode-inapplicable fields are rejected.
2. The NLHE capacity compiler combines that specification with poker geometry,
   the selected mode, implementation workspace formulas, and memory observed
   from the selected CUDA device. It emits one immutable resolved execution
   manifest. The resolved mapping records `nlhe_capacity_v2`, its individual
   derivation authorities, the canonical source hash, and its own content hash.
3. Actual row counts, allocator high-water marks, pack utilization, free memory,
   and overflow recovery are runtime telemetry. They are never accepted as
   configuration input.

The smallest checked-in example is
[`configs/train/nlhe_hu_8bb_normal_smoke.yaml`](../configs/train/nlhe_hu_8bb_normal_smoke.yaml).
The `configs/experiments/` files use the same schema. Frozen profiler inputs are
different documents under `configs/profiles/` and declare
`kind: branching_profile_v1`; a selected scenario is first compiled into an
ordinary `nlhe_run_v1` specification.

## Conditional shapes

- `objective.value` is either `{kind: mse, coefficient: ...}` or a clipped
  object that also owns `epsilon`.
- `objective.entropy` is either constant or a linear-decay object that owns its
  iteration horizon.
- `schedule` contains `batch_size` and exactly one of `epochs` or `steps`.
- `objective.reference_kl` is absent when unused and owns its one refresh
  cadence when present. Reference caching is an execution policy.
- `completion: {samples: ...}` exists only for Rao–Blackwell modes.
- Branching configuration and branching loss terms exist only for branched
  modes. One `run.workload.tree_nodes` budget derives topology, leaf, edge, and
  target-arena capacities.
- `attention` is the scalar `sdpa`, or a `chunked_dense` object that owns
  `query_rows`.
- Best response uses one `run.frozen: {path, sha256}` object; self-play rejects
  it.

Architecture defaults are versioned by `nlhe_run_v1`. The compiler derives
lossless history length, input widths, private domains, action width, token and
token-pair bounds, sequential action rows, balanced best-response capacity,
forest edges, physical packs, and concurrent workspaces. Changing those
derivations requires a new capacity derivation version.

The shared model accepts one positive `scientific.model.layers` value; retired
`n_layers_pre`/`n_layers_post` keys and split checkpoint formats are rejected.
`run.execution.microbatch.learner_rows` accepts either one positive cap or an
exact family mapping, such as `{pre: 32, post_2: 8}` for heads-up. These are
physical packing limits and do not change the statistical schedule.

`run.execution.tensor_program` selects `mode: eager`, `compiled`, or `captured`
for the same learner tensor program, with bounded `cache_entries` and
`cache_bytes`. Precision is explicit: H100 supports native BF16; SM75 supports
FP32/FP16 and rejects BF16. Master parameters, gradients, optimizer state, loss
reductions and incidence accumulation remain FP32. The resolved manifest records
the precision policy, shape-derived memory admission and graph-cache allowance.
