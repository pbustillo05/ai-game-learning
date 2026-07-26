# CUDA range kernel databases

`generate.py` recreates the historical runtime associated with a checkpoint,
generates 1,000 policy trajectories on CUDA, samples 100 distinct decision
nodes uniformly without replacement, and saves:

- `ranges`: float32 `[100, P, 1326]`, using the same per-seat gauge-shifted
  range weights consumed by the policy;
- `is_folded`: bool `[100, P]`;
- trajectory, node, decision-row, street, and acting-player provenance;
- checkpoint hash, source revision, seeds, and sampling metadata.

Folded seats retain their actual fold-time reach rows. They are not replaced
with zeroes. The output `.pt` files are intended as compact CUDA-kernel
fixtures; matching `.json` files provide human-readable summaries.

The unversioned legacy runtime is generated in four consecutive batches of 250
trajectories to respect its historical half-free-VRAM guard. Streaming random
priorities select exactly the same uniform-without-replacement sample that a
single materialized union would. Its unused all-private-hand policy-log-prob
cache is replaced with shape-correct zero storage because modern Torch rejects
board-conflicting rows that the historical cache attempted to validate; action
sampling and range evolution are unchanged.

The legacy selected-hand sampler can also produce non-finite or zero action
mass under the current Torch runtime. Those rows alone use a uniform
distribution over legal actions. The exact number of fallbacks is stored as
`metadata.legacy_uniform_legal_action_fallbacks`; all nondegenerate rows use the
checkpoint policy unchanged.
