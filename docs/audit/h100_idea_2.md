# Poker Player Transformer Model and Learner Implementation

## Purpose

This document specifies the poker policy/value model and a practical implementation strategy for training many independent replicas efficiently on one NVIDIA H100.

The priorities are, in order:

1. Correct independent-training semantics.
2. A clean, modular model implementation.
3. Excellent performance for both `R=1` and `R>1`.
4. High aggregate learner throughput while the model is small.
5. Avoiding complex custom machinery that will become irrelevant if the model grows large enough to saturate the GPU with `R=1`.

The implementation should therefore have one canonical model definition and only a thin replica-vectorization layer around it. Replica parallelism is an execution strategy, not a different model architecture.

---

# 1. Workload

We train `R` independent models concurrently on one GPU.

Each replica has:

- the same architecture and hyperparameters;
- different initialized weights;
- different gradients;
- different optimizer state;
- its own rollout minibatch;
- no gradient reduction or parameter sharing with any other replica.

For replica `r`:

\[
(W_r, O_r, X_r)
\rightarrow
\text{forward}
\rightarrow
L_r
\rightarrow
\text{backward}
\rightarrow
(W'_r, O'_r).
\]

The replica dimension must never be included in loss normalization, gradient clipping, advantage normalization, or other run-local reductions.

The objective is to maximize aggregate learner throughput:

\[
\text{nodes/sec}
=
\frac{\text{independent rollout nodes trained}}
{\text{learner wall-clock time}},
\]

and:

\[
\text{tokens/sec}
=
\frac{\text{transformer tokens processed}}
{\text{learner wall-clock time}}.
\]

Approximately 60% of rollout nodes are preflop and 40% are postflop:

| Node type | Share | Sequence length |
|---|---:|---:|
| Preflop | 60% | 384 tokens |
| Postflop | 40% | 2688 tokens |

The weighted average sequence length per rollout node is therefore:

\[
0.6(384)+0.4(2688)=1305.6\text{ tokens}.
\]

Because full attention is quadratic in sequence length, postflop nodes dominate compute despite being only 40% of nodes.

---

# 2. Model Configuration

The baseline model is a pre-LayerNorm transformer with configurable hyperparameters.

Default configuration:

| Hyperparameter | Default |
|---|---:|
| Model width `d_model` | 256 |
| Attention heads | 4 |
| Head width | 64 |
| Transformer blocks | 4–6 |
| FFN expansion | 4× |
| FFN hidden width | 1024 |
| Precision | BF16 |
| Attention | Full bidirectional |
| Preflop sequence length | 384 |
| Postflop sequence length | 2688 |
| Players represented | 2 |

The implementation must accept other values for:

- `d_model`;
- number of heads;
- number of blocks;
- FFN expansion;
- sequence length;
- action-head dimensions;
- number of replicas;
- readout mode.

No kernel or model code should assume that `d_model=256` forever.

---

# 3. Simplifications for the Two-Player Model

The deployed setting has exactly two players.

There are no folded-player statistics or folded-player rows in the learner input. The following machinery is removed from the main model path:

- folded-player summaries;
- folded card-pressure features;
- folded unary or pairwise correction factors;
- folded-stat contractions;
- the incidence/blocker residual path;
- blocker-graph-specific projections and reductions.

This simplification is desirable for both model clarity and learner performance. If later experiments show that an explicit blocker/incidence feature materially improves learning quality, it can return as an optional module behind a configuration flag. It should not shape the core execution architecture.

The baseline model relies on:

- hand/class token features;
- public-state tokens;
- action-anchor tokens;
- learned full bidirectional attention;
- an additive key-side log-reach prior;
- ordinary transformer FFNs.

---

# 4. Token Sequences

## 4.1 Preflop

For the full-deck case, each player contributes 169 canonical hand-class rows.

With two players:

\[
2(169)=338\text{ hand-class tokens}.
\]

Public-state and action-anchor tokens bring the complete fixed sequence to:

\[
\boxed{384\text{ tokens}}.
\]

## 4.2 Postflop

Each player contributes all 1326 two-card hands.

With two players:

\[
2(1326)=2652\text{ hand tokens}.
\]

Public-state and action-anchor tokens bring the complete fixed sequence to:

\[
\boxed{2688\text{ tokens}}.
\]

Board-impossible hand rows remain in the fixed-shape sequence and are masked. This wastes some arithmetic but preserves only two static sequence shapes and makes batching, compilation, and CUDA Graph capture straightforward.

Compact board-dependent tokenization is not part of the initial implementation. It should be considered only if profiling shows that padding or impossible-hand computation is a dominant cost and the expected gain justifies the added complexity.

---

# 5. Transformer Trunk

For layer `l`:

\[
\widehat X^l
=
X^l+
\operatorname{MHA}^l(\operatorname{LN}(X^l)),
\]

\[
X^{l+1}
=
\widehat X^l+
\operatorname{FFN}^l(\operatorname{LN}(\widehat X^l)).
\]

The FFN is:

\[
\operatorname{FFN}(x)
=
W_2\operatorname{GELU}(W_1x+b_1)+b_2.
\]

The default FFN dimensions are:

\[
256\rightarrow1024\rightarrow256.
\]

## 5.1 Attention score

For head `m`, query token `i`, and key token `j`:

\[
s^{(m)}_{ij}
=
\frac{(W^Q_mx_i)^\top(W^K_mx_j)}
{\tau_m\sqrt{d_k}}
+
\ell_j\mathbf 1[j\text{ is a hand/class row}]
+
M_{ij},
\]

where:

- `tau_m` is a learned per-head content temperature;
- `ell_j` is the key-side log-reach prior;
- `M_ij` is `0` for valid positions and `-infinity` for masked positions.

The attention is fully bidirectional. Every valid token can attend to every other valid token.

The content temperature applies only to the dot-product term, not to the log-reach prior.

The implementation should keep the key-side prior fused into the attention score calculation. It should not materialize a dense `N × N` bias tensor.

---

# 6. Readout Modes

The trunk always processes the complete sequence. The readout can operate in one of two modes.

## 6.1 `true_hand_only`

Return policy and value outputs only for the actor's known true hand/class token.

Inputs include an integer row index identifying the actor's true hand token.

The implementation gathers that row after the transformer trunk and evaluates the policy/value readout only for the gathered state.

This is the preferred learner mode when the loss requires only the sampled trajectory hand.

## 6.2 `all_actor_hands`

Return policy and value outputs for every acting-player hand/class token:

- 169 rows preflop;
- 1326 rows postflop.

This mode is available when training objectives or downstream consumers require the entire actor range.

## 6.3 Performance consequence

Selecting one output row does **not** make the transformer trunk sparse. Full bidirectional attention requires all token states in every layer.

The savings are confined to:

- final readout projections;
- policy normalization;
- value prediction;
- loss construction;
- output activation storage.

For the default model, `true_hand_only` is expected to be only a few percent faster than `all_actor_hands`, not orders of magnitude faster.

---

# 7. Policy and Value Heads

The policy head predicts:

- passive/call/raise action type;
- the log-probability of a fixed sampled legal action;
- any compact parameters required by the continuous raise-size distribution.

The learner should compute the sampled action's log-probability directly. It should avoid materializing large tensors over every hand, every legal raise tick, and every mixture component unless the full distribution is explicitly required.

The exact sampled-action log-probability still includes the legal-action normalization term.

The value head predicts one scalar per requested hand row.

Public branching heads are not part of the baseline learner implementation. They may be added later as an independent optional head without changing the trunk or replica execution machinery.

---

# 8. Concrete Compute Estimate

Let:

- `N` be sequence length;
- `D=256`;
- `L` be the number of transformer blocks.

For one transformer block, approximate forward plus backward tensor-core work is:

\[
F_{\text{block}}
\approx
72ND^2+12N^2D.
\]

With `D=256`:

\[
F_{\text{block}}
\approx
4{,}718{,}592N+3{,}072N^2.
\]

This includes the major forward, activation-gradient, and weight-gradient GEMMs for:

- QKV projection;
- attention output projection;
- MLP up projection;
- MLP down projection;
- the two dense attention matrix multiplications and their backward work.

It excludes smaller costs such as:

- LayerNorm;
- GELU bookkeeping;
- softmax bookkeeping;
- residual operations;
- masks;
- the policy/value heads;
- the optimizer.

## 8.1 Approximate forward/loss/backward work per node

| Sequence | 4 blocks | 6 blocks |
|---:|---:|---:|
| 384 | 9.06 GFLOPs | 13.59 GFLOPs |
| 2688 | 139.52 GFLOPs | 209.28 GFLOPs |

A 2688-token node has 7× as many tokens as a 384-token node, but requires about 15.4× as much transformer training compute because attention is quadratic.

For the 60/40 node mix, postflop contributes approximately 91% of transformer FLOPs:

\[
\frac{0.4(209.28)}{0.6(13.59)+0.4(209.28)}\approx91.1\%
\]

for the six-block model. The percentage is essentially the same for four blocks.

This means optimization decisions should primarily be judged by their effect on the 2688-token path.

---

# 9. H100 Reference

Reference device: NVIDIA H100 SXM 80 GB.

| Property | Approximate value |
|---|---:|
| HBM capacity | 80 GB |
| HBM bandwidth | 3.35 TB/s |
| SMs | 132 |
| Tensor Cores | 528 |
| Published BF16 Tensor throughput with sparsity | 1,979 TFLOPS |
| Approximate dense BF16 Tensor peak | 989.5 TFLOPS |

The dense BF16 number is the relevant arithmetic ceiling because the model does not use 2:4 structured sparsity.

The raw compute-only ceilings are therefore:

| Sequence | Blocks | Theoretical nodes/sec | Theoretical tokens/sec |
|---:|---:|---:|---:|
| 384 | 4 | 109,200 | 41.9M |
| 384 | 6 | 72,800 | 28.0M |
| 2688 | 4 | 7,090 | 19.1M |
| 2688 | 6 | 4,730 | 12.7M |

These ceilings are not achievable application targets. They assume every operation runs at peak BF16 Tensor throughput and ignore memory traffic, normalization, softmax, masking, launch overhead, losses, and optimizer work.

---

# 10. Throughput We Can Reasonably Hope For

The following estimates assume:

- H100 SXM 80 GB;
- BF16 trunk computation;
- full bidirectional attention;
- a fused memory-efficient attention implementation;
- no incidence/blocker residual;
- a small policy/value head;
- enough replicas to fill the GPU when `R=1` is insufficient;
- batched or vectorized model execution;
- compiled execution or CUDA Graph replay if launch overhead is material.

They are planning ranges, not benchmark guarantees.

## 10.1 Forward + loss + backward, excluding optimizer

### Four blocks

| Node type | Reasonable range | Strong target |
|---|---:|---:|
| 384-token preflop | 38k–60k nodes/sec | ~50k |
| 2688-token postflop | 2.8k–4.3k nodes/sec | ~3.6k |
| 60/40 mixed workload | **6.4k–9.6k nodes/sec** | **~8.0k** |
| Mixed useful tokens/sec | **8.3M–12.6M** | **~10.4M** |

### Six blocks

| Node type | Reasonable range | Strong target |
|---|---:|---:|
| 384-token preflop | 25k–40k nodes/sec | ~32k |
| 2688-token postflop | 1.9k–2.8k nodes/sec | ~2.35k |
| 60/40 mixed workload | **4.3k–6.3k nodes/sec** | **~5.3k** |
| Mixed useful tokens/sec | **5.6M–8.3M** | **~6.9M** |

The mixed node rate is calculated from time, not by averaging rates:

\[
\text{mixed nodes/sec}
=
\frac{1}
{0.6/r_{384}+0.4/r_{2688}}.
\]

## 10.2 Including the optimizer

A full AdamW update is much more noticeable for 384-token nodes because optimizer traffic is mostly proportional to parameter count rather than sequence length.

For a well-vectorized or fused optimizer, a practical initial expectation is:

| Model | Mixed full learner updates/sec | Mixed tokens/sec |
|---|---:|---:|
| 4 blocks | **5.0k–7.5k** | **6.5M–9.8M** |
| 6 blocks | **3.4k–5.2k** | **4.4M–6.8M** |

These ranges assume optimizer state for the transformer and modest heads. Large embeddings or additional heads would reduce the rates.

## 10.3 Effect of readout mode

Expected whole-step effect relative to `true_hand_only`:

| Mode | 384 tokens | 2688 tokens |
|---|---:|---:|
| `true_hand_only` | Baseline | Baseline |
| `all_actor_hands` | Roughly 3–10% slower | Roughly 2–8% slower |

The exact difference depends on the width and implementation of the policy head, especially legal-action normalization.

## 10.4 Interpreting these estimates

For a six-block model with the 60/40 mix:

- below 3,000 mixed forward/loss/backward nodes/sec indicates substantial implementation headroom;
- 4,000–5,000 is credible for a good modular PyTorch implementation;
- around 5,300 is a strong target;
- above 6,000 is excellent and should not be assumed without measurement;
- the theoretical mixed ceiling is approximately 10,500 nodes/sec, but it is not a realistic application target.

For four blocks:

- around 8,000 mixed forward/loss/backward nodes/sec is a strong target;
- above 9,500 is excellent.

---

# 11. Modular Implementation Architecture

## 11.1 One canonical model

Implement one ordinary `torch.nn.Module` for a single model and a normal batch:

```python
class PokerTransformer(nn.Module):
    def forward(
        self,
        batch: PokerBatch,
        *,
        readout_mode: Literal["true_hand_only", "all_actor_hands"],
    ) -> ModelOutput:
        ...
```

This path must be efficient and idiomatic at `R=1`.

Do not maintain a separate hand-written replica model with duplicated transformer logic.

## 11.2 Thin functional replica wrapper

For `R>1`, stack model state and vectorize the canonical single-model function:

- `torch.func.stack_module_state` stacks parameters and buffers;
- `torch.func.functional_call` evaluates the same module with explicit state;
- `torch.func.vmap` maps the single-model function over replicas.

Conceptually:

```python
def forward_one(params, buffers, batch, readout_mode):
    return torch.func.functional_call(
        template_model,
        (params, buffers),
        (batch,),
        {"readout_mode": readout_mode},
    )

forward_many = torch.vmap(
    forward_one,
    in_dims=(0, 0, 0, None),
    randomness="different",
)
```

This preserves one source of truth for model behavior.

## 11.3 Optimize beneath the module boundary

The preferred optimization path is compiler-generated batching and fusion. Explicit `bmm` should be introduced only inside reusable primitive modules where it benefits both the replica path and remains easy to maintain.

For example, a `ReplicaLinear` primitive may select:

- ordinary `F.linear` when weights have no replica dimension;
- `torch.bmm` when weights have a leading replica dimension.

The transformer block, attention logic, readout, and loss should not be duplicated.

## 11.4 Keep `R` an execution parameter

The same training entry point should support:

```text
R = 1
R = 2
R = 4
...
```

At `R=1`, it should use the ordinary module path unless measurement shows that the functional path is equally fast.

At larger `R`, it should use the stacked functional path.

This small dispatch is acceptable because it selects an execution wrapper, not a different model implementation.

---

# 12. PyTorch Tools and Their Roles

## `nn.Module` and `F.linear`

The canonical model implementation. This is the preferred `R=1` path and should remain understandable, testable, and portable.

## `torch.func.functional_call`

Runs the same module with an explicitly supplied parameter/buffer dictionary. It enables stacked independent model states without rewriting the model.

## `torch.func.stack_module_state`

Stacks parameters and buffers from same-architecture models along a leading replica dimension.

## `torch.func.vmap`

Transforms a single-replica function into a replica-batched function. It is a program transformation, not a guarantee that every generated kernel is optimal.

Use it as the default modular replica implementation.

## `torch.bmm`

Computes equal-shape batched matrix multiplication:

\[
[R,M,K]@[R,K,N]\rightarrow[R,M,N].
\]

This is the natural low-level primitive for equal-shape independent weights. Use it where profiling confirms that the generic `vmap` result is leaving significant GEMM performance on the table.

## `torch.nn.functional.grouped_mm`

Computes a grouped set of GEMMs and can support jagged token counts. It is primarily useful when groups have different token counts or when its scheduler materially outperforms strided batched GEMM.

It is **not** part of the initial architecture because:

- the two learner shapes are already fixed;
- replicas within a bucket have equal matrix shapes;
- `bmm` is simpler;
- the API is more specialized and less mature;
- adopting it throughout the model would add maintenance burden.

Benchmark it only as a contained experiment behind the same linear primitive interface.

## `torch.compile`

Compile the canonical training step after the model and replica representation are correct.

Test:

```python
torch.compile(step, mode="default")
torch.compile(step, mode="reduce-overhead")
torch.compile(step, mode="max-autotune")
```

This is the preferred route to pointwise fusion, reduced dispatch overhead, matmul autotuning, and optional CUDA Graph use.

## CUDA Graphs

CUDA Graphs replay a fixed GPU schedule with very low CPU launch overhead.

Use them only if:

- Nsight Systems shows meaningful launch gaps after `torch.compile`;
- batches use one of the two fixed shapes;
- graph capture is straightforward;
- the gain is material.

Prefer compiler-managed graphs before implementing a large manual graph-management layer.

## Triton or CUTLASS

Custom kernels are a last resort. They are justified only when a specific dominant kernel has a large measured gap and the likely end-to-end gain exceeds roughly 10%.

The project should not build a custom transformer runtime for this model size.

---

# 13. Better ROI Order

## Tier 0 — Correctness and measurement

### 1. Build a high-quality `R=1` baseline

Use ordinary PyTorch modules, BF16 autocast, fused attention where possible, and a fused optimizer implementation.

Measure the 384-token and 2688-token paths independently.

**Expected ROI:** foundational.  
**Complexity:** low.

### 2. Add exact performance instrumentation

Record useful tokens/sec, nodes/sec, step latency, kernel time categories, peak memory, and compilation behavior.

**Expected ROI:** prevents wasted engineering.  
**Complexity:** low.

## Tier 1 — High ROI, low-to-moderate complexity

### 3. Verify the fused attention path

The 2688-token workload is dominated by attention. Ensure the learned content temperature, key-side log-reach bias, and masks stay on a memory-efficient fused attention implementation.

Avoid materializing the attention matrix or a dense bias tensor.

**Expected ROI:** potentially enormous.  
**Complexity:** moderate.

### 4. Use `torch.compile`

Compile the single-model and replica training steps. Compare `default`, `reduce-overhead`, and `max-autotune` modes.

**Expected ROI:** high.  
**Complexity:** low.

### 5. Add the functional `vmap` replica wrapper

Stack independent model state and vectorize the same canonical model implementation.

Sweep replica count separately for 384-token and 2688-token buckets.

**Expected ROI:** high while the model is too small at `R=1`.  
**Complexity:** moderate.

### 6. Use `true_hand_only` readout when permitted

Gather the requested actor row after the trunk and run only that readout.

**Expected ROI:** small but nearly free.  
**Complexity:** low.

### 7. Use a fused/foreach optimizer and vectorized replica-local clipping

Do not launch one optimizer per replica from Python. Keep gradient norms and updates independent across `R` while batching the underlying tensor operations.

**Expected ROI:** high for 384-token nodes; moderate for 2688-token nodes.  
**Complexity:** moderate.

## Tier 2 — Profile-directed, contained optimizations

### 8. Replace poorly batched linear operations with a reusable `bmm` primitive

Profile forward, activation-gradient, and weight-gradient GEMMs. Introduce explicit batched linear operations only where the `vmap`/compiler result is measurably worse.

Keep the change behind a small module or custom operator so the model definition remains unchanged.

**Expected ROI:** moderate.  
**Complexity:** moderate.

### 9. Add compiler-managed or minimal manual CUDA Graph capture

Only pursue this if launch gaps remain after compilation.

Use one graph for the 384 shape and one for the 2688 shape. Do not build a general graph scheduler.

**Expected ROI:** moderate for 384; usually smaller for 2688.  
**Complexity:** moderate.

### 10. Tune batch/replica bucketing and memory layout

Keep preflop and postflop in separate fixed-shape batches. Store stacked weights and optimizer state contiguously. Avoid unnecessary transposes or copies.

**Expected ROI:** moderate.  
**Complexity:** low-to-moderate.

## Tier 3 — Only after a compelling profile

### 11. Benchmark `grouped_mm`

Test it behind the same batched-linear interface. Keep it only if it improves full-step throughput, including backward, by at least 5%.

**Expected ROI:** uncertain.  
**Complexity:** moderate.

### 12. Compact board-impossible postflop rows

This reduces sequence length but introduces board-dependent shapes, gathers, scatters, and more graph variants.

Consider it only if attention remains the dominant cost and the measured theoretical savings are large enough to justify the added complexity.

**Expected ROI:** potentially meaningful.  
**Complexity:** high.

### 13. Custom Triton/CUTLASS kernels

Implement only for a dominant operation with at least roughly 10% credible end-to-end upside.

Do not implement a custom grouped transformer schedule, custom full backward stack, or bespoke optimizer runtime unless ordinary PyTorch has clearly failed.

**Expected ROI:** uncertain.  
**Complexity:** very high.

---

# 14. What Not to Do Initially

Do not:

- maintain separate `R=1` and replica transformer implementations;
- write one CUDA stream per replica;
- build dense block-diagonal weight matrices;
- manually schedule every transformer kernel;
- require `grouped_mm` throughout the model;
- write custom QKV, MLP, and weight-gradient kernels before profiling;
- introduce board-dependent ragged tokenization before the fixed-shape baseline is strong;
- retain incidence/blocker machinery solely because it might be theoretically expressive;
- optimize only forward microbenchmarks while ignoring backward and the optimizer.

The code should remain useful if the future model grows large enough that `R=1` saturates the GPU.

---

# 15. Metrics

## 15.1 Primary metrics

Track separately for 384-token, 2688-token, and weighted 60/40 traffic:

- forward/loss/backward nodes/sec;
- full optimizer-step nodes/sec;
- useful tokens/sec;
- aggregate replica updates/sec;
- per-replica updates/sec;
- mean, median, P95, and P99 step latency.

For mixed traffic:

\[
\text{mixed nodes/sec}
=
\frac{1}{0.6/r_{384}+0.4/r_{2688}}.
\]

Do not average the two node rates arithmetically.

## 15.2 GPU metrics

Track:

- achieved BF16 Tensor TFLOPS;
- SM active percentage;
- Tensor Core active percentage;
- HBM bandwidth utilization;
- L2 hit rate;
- achieved occupancy;
- GPU clocks, power, and throttling.

## 15.3 Time breakdown

Measure time spent in:

- QKV projection forward/backward;
- attention forward/backward;
- attention output projection;
- MLP forward/backward;
- LayerNorm and residual operations;
- readout and loss;
- gradient clipping;
- optimizer;
- input staging/copies;
- CPU launch gaps.

## 15.4 Runtime/compiler metrics

Track:

- kernel launches per step;
- median kernel duration;
- total inter-kernel idle time;
- graph breaks;
- recompilations;
- compilation/autotuning time, reported separately;
- number of captured graph variants;
- peak allocated and reserved memory.

## 15.5 Correctness metrics

Compare the replica path against `R` independent ordinary models:

- per-replica loss;
- gradients;
- gradient norm and clipping factor;
- updated weights;
- optimizer moments;
- multiple-step training trajectories.

Changing one replica's inputs must not affect any other replica's gradients or state.

---

# 16. When We Are Close to Optimal

The implementation is close to optimal when all of the following are true.

## 16.1 Replica count has plateaued

Increasing `R` produces less than about 2% additional throughput around a locally dense sweep.

Sweep non-powers-of-two near the plateau because GEMM tile scheduling can be non-monotonic.

The optimal `R` may differ substantially between 384-token and 2688-token batches.

## 16.2 The 2688-token attention path is fused and efficient

There is no materialized `N × N` attention-probability tensor, no dense expanded bias tensor, and no fallback to an obviously slower mathematical attention implementation.

Attention forward and backward should dominate postflop for understandable arithmetic reasons, not because of avoidable layout or backend problems.

## 16.3 CPU and launch gaps are negligible

Nsight Systems should show less than roughly 3–5% of learner time in unexplained GPU gaps or CPU submission stalls.

If this is already true, manual CUDA Graph work has little remaining value.

## 16.4 Dominant GEMMs are near their shape-specific limits

For each dominant GEMM, compare full-model throughput against an isolated benchmark with the same shape, stride, datatype, and transpose layout.

A small GEMM should be judged against its best practical shape-specific result, not only against the H100's global peak.

## 16.5 End-to-end utilization is strong

For this small mixed workload:

- below 25% estimated dense BF16 model FLOP utilization usually indicates substantial headroom;
- 35–50% is strong for a modular implementation;
- above 50% is excellent;
- a lower percentage can still be near-optimal if the profile proves that attention softmax, normalization, optimizer traffic, or another non-Tensor-Core component is limiting.

## 16.6 Remaining ideas fail the Amdahl test

If a component occupies fraction `f` of step time, eliminating it entirely can improve throughput by at most:

\[
\frac{1}{1-f}.
\]

Do not pursue a complex optimization unless its credible full-step improvement is at least:

- about 5% for a simple contained change;
- about 10% for a custom kernel;
- substantially more for a custom runtime or duplicated implementation.

---

# 17. Recommended Initial Build

The initial production-quality implementation should contain only:

1. One configurable `PokerTransformer` module.
2. One ordinary `R=1` training path.
3. One functional `vmap` replica wrapper.
4. Two fixed sequence buckets: 384 and 2688.
5. `true_hand_only` and `all_actor_hands` readout modes.
6. A fused attention implementation supporting the key-side log-reach prior.
7. `torch.compile` benchmarks and a selected production mode.
8. A fused or foreach optimizer with replica-local clipping.
9. Nsight and in-process performance instrumentation.

Only add explicit `bmm`, CUDA Graph management, `grouped_mm`, compact tokenization, or custom kernels when the measured profile supports the added complexity.

---

# 18. Bottom Line

The most important performance fact is that the 40% postflop traffic accounts for about 91% of transformer compute. The optimization strategy should therefore prioritize the fused 2688-token attention path rather than over-engineering the tiny preflop GEMMs.

For the default six-block model, a good target is approximately:

\[
\boxed{4.3\text{k}–6.3\text{k mixed forward/loss/backward nodes/sec}}
\]

or:

\[
\boxed{5.6\text{M}–8.3\text{M mixed tokens/sec}}.
\]

A strong central planning number is approximately:

\[
\boxed{5.3\text{k nodes/sec and }6.9\text{M tokens/sec}}
\]

before the optimizer, and approximately:

\[
\boxed{3.4\text{k}–5.2\text{k full AdamW updates/sec}}
\]

for a well-vectorized complete learner.

The implementation should pursue these numbers with ordinary modular PyTorch first. Replica batching is temporary leverage for a small model, not a reason to build a permanent custom transformer runtime.

---

# 19. Reference Documentation

- [NVIDIA H100 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch `torch.func.vmap`](https://docs.pytorch.org/docs/stable/generated/torch.func.vmap.html)
- [PyTorch `torch.func.functional_call`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html)
- [PyTorch `torch.nn.functional.grouped_mm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html)
