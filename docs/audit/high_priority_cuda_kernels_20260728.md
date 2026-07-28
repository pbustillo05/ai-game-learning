# High-priority CUDA kernels for queued poker branching

Date: 2026-07-28

Status: measurement-backed optimization audit. No implementation change is
authorized by this document.

## Executive conclusion

The current queued fixed-$\beta=3$ HU 32 bb rollout takes
**6.527 ms per retained decision**, compared with **8.575 ms** in the
2026-07-24 implementation. On the matched workload, this is:

- 2.048 ms less latency per retained decision;
- 23.9% less time per retained decision; and
- 153.2 rather than 116.6 retained decisions/s, or 1.314 times the throughput
  and a 31.4% throughput increase.

This comparison is for rollout only. There is no current matched measurement
of the older audit's complete sample + seal + completion + target wall time,
so the 3.0-times raw end-to-end ratio in that audit must not be reused as a
current speedup claim.

The rewrite removed most of the formerly dominant generic work. Relative to
the old legal fixed-$\beta=3$ median:

| Metric | 2026-07-24 | Current exact gate | Reduction |
|---|---:|---:|---:|
| Rollout per decision | 8.575 ms | 6.527 ms | 23.9% |
| Generic routing traffic | 51.35 GB | 88.48 MB | 99.83%, about 580 times |
| Host synchronizations | 2,494 | 66 | 97.35%, about 37.8 times |
| Child reservation + commit | 16.109 s | 1.115 s | 93.1%, about 14.4 times |
| Target propagation CUDA | 14.670 s | 0.00576 s | 99.96%, about 2,547 times |

The target row compares the old propagation time excluding completion with
the new propagation-kernel CUDA interval. The workload is matched, but the
implementation and timing boundary changed precisely because the old
Python-driven propagation was replaced by a CUDA target executor.

The highest-priority poker-specific custom kernel is now the
collision-conditioned posterior sampler. The selected-action policy
calculation is the next actionable operator, but it should first receive a
direct accelerator-native PyTorch formulation. Folded statistics are already
implemented as PyTorch CUDA tensor operations and are not exercised by HU;
they need a representative multiway profile before a custom kernel is
justified.

## Evidence and comparison discipline

The historical source is
`docs/audit/nlhe_hu_32bb_family_timing_attribution_20260724.md`. Its legal
branched workload used:

- HU, 32 bb, all 52 cards;
- fixed $\beta=3$;
- one distinguished root;
- family pack cap 84;
- 1,024 completion samples;
- one warmup and one measured iteration; and
- seeds 104830, 209559, and 314288 on RTX 2080 Ti GPUs.

The matched current source is the frozen Phase 8 CUDA gate in
`docs/queued/branched_rb_sampling.md`. It used the same game, controller,
checkpoint family, completion count, pack cap, seeds, warmup convention, and
GPU model. This is the authority for the exact 8.575-to-6.527 ms comparison.

Two fresh operation-attribution runs were also made against Git revision
`7e1a961f6b549f484ca70226319388f79ce7ef2f` with the same historical model
tensor dictionary:

- `runs/experiments/nlhe_hu_32bb_branched_fixed_operation_profile_20260728/profile_beta2p4.json`;
- `runs/experiments/nlhe_hu_32bb_branched_fixed_operation_profile_20260728/profile_beta2p4_seed209559.json`.

Those profiles used fixed $\beta=2.4$, seeds 104830 and 209559, no warmup, and
one RTX 2080 Ti each. Their median rollout was 6.555 ms/decision. That is
consistent with the matched gate, and is 23.6% below the old 8.575 ms value,
but it is not a replacement for the exact $\beta=3$ comparison because tree
geometry and warmup differ.

An unsafeguarded instrumented $\beta=3$ attempt exceeded the diagnostic's
40,000-node capacity. A second attempt with a manually activated tree
safeguard reached the asynchronous assertion
`postflop family disagrees with live-seat state`. Both were excluded. The
frozen exact $\beta=3$ gate remains valid; the assertion must be resolved
before this diagnostic is used for a new exact-$\beta=3$ operation breakdown.

CUDA event scopes were recorded asynchronously and resolved at the existing
phase boundary. Several scopes are nested. In particular, transition and
chance closure can contain posterior sampling and policy work, so percentages
below must not be summed.

## Current operation attribution

The table takes the median of the two per-seed rates and divides by the median
6.555 ms rollout rate. “Share” is a prioritization denominator, not an
additive decomposition.

| Scope | Median ms/decision | Rollout-equivalent share | Interpretation |
|---|---:|---:|---|
| Model inference | 5.151 | 78.58% | Largest overall ceiling; mostly postflop SDPA model work |
| Transition, nested | 1.009 | 15.39% | Contains several separately timed operations |
| Posterior profile sampler | 0.485 | 7.40% | Largest identified poker-specific kernel candidate |
| Selected-action log policy | 0.279 | 4.26% | Full policy composition followed by selected-action gather |
| Target setup wall | 0.118 | 1.80% | Outside rollout; host/setup work, not target CUDA |
| Folded/live range resolution | 0.067 | 1.03% | Small but measurable |
| Child reservation + commit | 0.066 | 1.00% | Former 16.1-second bottleneck is no longer primary |
| Chance card deal | 0.026 | 0.39% | Not a priority |
| Actor action sampling | 0.023 | 0.36% | Already operates on the actor's selected hand |
| Action-conditioned reach update | 0.0037 | 0.056% | Not a priority |
| Target propagation kernel | 0.00084 | 0.013% | Solved for this workload |

Model inference deserves the next kernel-level trace because even a 10%
improvement inside the model would reduce rollout by approximately 7.9%.
This audit does not preselect a new model kernel: the model scope includes
multiple projections, attention, feed-forward work, head composition, and
launch overhead. PyTorch profiler or Nsight evidence is required before
deciding whether the next model change belongs in SDPA use, projections,
token construction, policy heads, or launch fusion.

The two concrete poker operators have a combined perfect-removal ceiling of
about 11.7% of current rollout. A realistic first pass that removes 75% of
sampler time and 60% of selected-policy time would reduce rollout by about
8.1%, before secondary launch and memory effects.

## Current implementation status

There is no poker-specific custom CUDA extension for the posterior sampler,
policy operator, or folded statistics. All three are semantically implemented
and run on CUDA through PyTorch tensor operations:

| Proposed operation | Current owner | Current form |
|---|---|---|
| Collision-conditioned profile sampling | `pbg/games/poker/queued/rb_sampling.py` | PyTorch CUDA tensors over a dense attempt envelope |
| Policy composition and selected log policy | `pbg/games/poker/queued/policy.py` | PyTorch CUDA tensor composition |
| Folded statistics | `pbg/games/poker/queued/folded.py` | FP32 PyTorch CUDA finite differences, reductions, and `torch.linalg.eigh` |

The repository does have custom CUDA extensions for generic layered targets,
branching metadata, and alias-table construction under `pbg/core/csrc/`.
Those do not implement the three poker operations above.

The phrase “CUDA sampler” in the queued contract therefore means that the
sampler is accelerator-resident and CUDA-only. It does not currently mean one
fused custom CUDA kernel.

## Priority 0: fused posterior-profile sampler

### Why it should be custom CUDA

`CollisionConditionedProfileSampler.sample_children()` currently constructs
the complete tensors for an $[G,A,P]$ attempt envelope, where:

- $G$ is the fixed child-envelope row count;
- $A=256$ is the maximum number of attempts; and
- $P$ is the number of seats.

It constructs mesh grids and logical RNG keys, generates all categorical
targets, performs eleven tensor-level binary-search steps, materializes
selected hands and cards for every attempt, checks pairwise collisions, and
then selects the first success. In the two profiles it processed about
46.2 dense attempt rows for every executed tree edge. The scientific result
is only one profile plus attempt telemetry per fresh child.

This is the classic case for one fused kernel: the intermediate envelope is
much larger than the retained result, each row follows the same bounded
control flow, and the logical RNG makes physical execution order irrelevant.

### Recommended kernel boundary

The kernel should consume:

- FP32 seat range weights and board cards by parent;
- inherited profiles;
- child source, ordinal, active, root, path, and retry epoch;
- seed, iteration, geometry, and maximum attempts.

It should produce:

- one `int32[P]` profile per envelope child;
- one `int32` attempt count per child;
- one exhausted bit per child; and
- the device failure record/minimum failing logical address.

A source-segment workgroup should build or reuse each seat CDF once, then
evaluate logical attempts in small waves. It may stop physical work after the
first valid logical attempt because later attempts are not observable. It
must still select exactly the lowest successful logical attempt and preserve
the declared exhaustion result when no attempt succeeds.

The main goal is to avoid materializing the global $[G,A,P]$ grids, keys,
targets, selected hands, and card tensors. A full-deck implementation should
keep the 1,326-entry seat CDF search and pairwise two-card collision test
local to the workgroup as far as occupancy permits.

### Required correctness gates

The kernel must preserve:

- child zero inheritance with no posterior RNG consumption;
- no logical address for inactive rows;
- board masking before CDF construction;
- independent seat proposals conditioned only by collision rejection;
- exact first-success selection;
- logical subdivision and physical scheduling invariance;
- retry-epoch disjointness;
- attempt counts and terminal exhaustion publication; and
- CUDA-only execution with no host or CPU fallback.

The existing PyTorch implementation should remain the test oracle until the
custom path passes exact result and failure-record comparisons across HU and
three- through six-player legal geometries.

### Performance gate

Measure at least the three historical seeds on the exact fixed-$\beta=3$
checkpoint workload after the diagnostic assertion is resolved. A useful
first acceptance threshold is:

- at least 2 times faster in the sampler interval;
- at least 3% lower median rollout ms/decision; and
- no increase in host synchronizations, routing traffic, or peak memory.

The measured perfect-sampler ceiling is 7.4% on the current two-seed profile,
lower than the roughly 9.6% found in the old implementation.

## Priority 0: direct selected-action log policy

`PokerPolicyOperator.selected_log_pi()` currently calls the authoritative
full `probabilities()` composition and then gathers the one selected public
action for every private hand. The output needs the normalization over legal
events, but it does not need a retained $[B,H,E]$ probability tensor.

The first change should be a direct PyTorch formulation that:

- computes the selected fold/check-call/raise numerator directly;
- performs a streaming or recomputable legal-event log normalizer;
- returns only $[B,H]$ selected log probabilities; and
- remains equation-identical to the learner's authoritative composition.

If that formulation still launches too many small tensor kernels, fuse the
direct numerator and legal-event normalization into a custom rollout CUDA
operator. Keep the PyTorch formulation as the semantic and autograd oracle;
do not create a second policy law. The learner either needs a tested backward
implementation or must continue through the same authoritative PyTorch
equations.

The selected-action scope is 4.26% of rollout, while actor sampling is only
0.36%. Optimize `selected_log_pi` first; actor sampling is already restricted
to the sampled actor hand.

A sensible performance gate is at least 1.5 times faster for the selected
scope and at least 1.5% lower exact-work rollout time without changing action
probabilities, gradients, legal-event behavior, or MPS inference semantics.

## Priority 1: model kernel trace before model-kernel work

Model inference accounts for 78.58% of current rollout. It is therefore the
largest opportunity, but “write a model CUDA kernel” is not yet a sufficiently
specific optimization.

Run one warm profile with PyTorch profiler activities for CPU and CUDA,
recording shapes, memory, and stacks, then confirm the top kernels with Nsight
Systems or Nsight Compute. Attribute at least:

- preflop versus postflop model input construction;
- projections and feed-forward layers;
- SDPA;
- poker output heads;
- kernel launch gaps and synchronizations; and
- family-pack occupancy and tail batches.

Only then select a model kernel. The current evidence supports spending
profiling effort here, but not replacing already optimized SDPA on intuition.

## Priority 2: multiway folded-statistics measurement

HU folds are terminal, so neither fresh HU profile called
`folded_statistics()`. No HU timing can rank it.

The current implementation is already a reasonable correctness-first PyTorch
CUDA function. It computes zero-, one-, and two-card removal queries in
chunks, uses tensor reductions/einsums for collision terms, and delegates the
small symmetric eigendecomposition to `torch.linalg.eigh`.

Before custom work, profile representative 3-max and 6-max branched runs with
actual folded seats and report:

- calls and rows by street and number of folded seats;
- CUDA time per decision and per folded row;
- peak scratch memory;
- time in removal-query construction, collision terms, and eigendecomposition;
- launch count; and
- overlap with transition and range-resolution scopes.

If it is material, the likely custom boundary is the finite-difference query
and collision reduction, not the eigendecomposition. Retain
`torch.linalg.eigh` unless profiler evidence shows otherwise. If it remains
below 1% of rollout, leave it as PyTorch.

## Work that is no longer high priority

Do not spend immediate custom-kernel effort on:

- layered target propagation at 0.013% of rollout;
- action-conditioned reach update at 0.056%;
- chance-card dealing at 0.39%;
- actor action sampling at 0.36%;
- child reservation/commit at 1.00%; or
- generic routing solely because it dominated the old audit.

Routing, reservation/commit, synchronization, and target propagation should
remain regression gates. Their reductions are why the current bottleneck has
moved.

Transition still deserves finer attribution because its broad nested scope is
15.39%. The individually identified transition suboperations above are small.
Add lower-level ranges or NVTX scopes before proposing a monolithic transition
kernel.

## Reproduction

`tools/profile-queued-branching.py` is the reusable operation profiler. It
strictly loads only the model tensor dictionary from the supplied checkpoint
and creates current-format owners and runtime state.

After selecting an idle GPU, the measured diagnostic form is:

```bash
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=2 python tools/profile-queued-branching.py \
  --checkpoint \
  runs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723/benchmark_init_ckpt_0001110_dummy_heads_seed91301_v8.pt \
  --output \
  runs/experiments/nlhe_hu_32bb_branched_fixed_operation_profile_20260728/profile_beta2p4_seed104830.json \
  --seed 104830 \
  --beta 2.4 \
  --warmup 0
```

For optimization acceptance, use one warmup, three measured seeds, the exact
fixed-$\beta=3$ workload, and an adequate immutable capacity after the
diagnostic safeguard assertion is resolved. Record pre/post GPU state and
confirm that the selected GPU returns to its idle memory baseline.
