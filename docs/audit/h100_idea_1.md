# Saturating an H100 for Tiny Transformer Training Using Replica-Axis Parallelism

## Executive Summary

We need to train (R) completely independent transformer models on one H100.

Every replica has:

* The same architecture and tensor shapes.
* Different weights and optimizer state.
* A separate random-number stream.
* A separate rollout minibatch containing (B) sequences.
* No parameter sharing, gradient sharing, loss reduction, or other communication with another replica.

For replica (r), the learner performs:

[
(W_r, O_r, X_r)
\rightarrow
\text{forward}
\rightarrow
L_r
\rightarrow
\nabla W_r
\rightarrow
(W'_r, O'_r)
]

The objective is not to reduce the latency of one replica. The objective is to maximize aggregate learner throughput:

[
\text{useful tokens/sec}
========================

\frac{\sum_{r=1}^{R}\text{non-padding tokens processed by replica }r}
{\text{learner wall-clock time}}
]

A single model is too small to fill an H100 efficiently. The main strategy is therefore to expose the replica dimension directly to GPU kernels:

[
\text{parameters}: [R,\ldots]
]

[
\text{activations}: [R,B,S,D]
]

[
\text{optimizer state}: [R,\ldots]
]

This allows one batched or grouped operation to process all independent models without coupling their training semantics.

The expected high-ROI implementation is:

[
\boxed{
\text{stacked model state}
+
\text{batched GEMMs}
+
\text{flattened SDPA batch}
+
\text{vectorized optimizer}
+
\text{CUDA Graph replay}
}
]

Grouped or custom persistent GEMMs should be introduced only when profiling demonstrates that regular batched GEMMs leave material performance on the table.

---

# 1. Exact Problem Definition

Let:

* (R): number of independent training replicas.
* (B): rollout minibatch size per replica.
* (S): sequence length.
* (T=B S): token positions processed per replica per learner step.
* (D=256): model width.
* (H=4): attention heads.
* (D_h=64): dimension per head.
* (D_{\text{ff}}=1024): MLP hidden width.
* (L\in{4,5,6}): number of transformer blocks.

The total number of input token positions processed per learner step is:

[
N_{\text{tokens}}=RBS=RT
]

The models are independent:

[
W_r \ne W_j \qquad \text{for }r\ne j
]

and each replica receives a different rollout batch:

[
X_r \ne X_j
]

However, all replicas have identical tensor dimensions. That regularity is what makes replica-axis batching possible.

For a linear layer, the computation is:

[
X\in\mathbb{R}^{R\times T\times K}
]

[
W\in\mathbb{R}^{R\times K\times N}
]

[
Y_r=X_rW_r
]

This is mathematically a batch of (R) independent GEMMs. It is not a single shared-weight GEMM.

---

# 2. Concrete Model Numbers

## 2.1 Transformer-block parameter count

Assume each block contains:

* A combined QKV projection: (256\rightarrow768).
* An attention output projection: (256\rightarrow256).
* An MLP up projection: (256\rightarrow1024).
* An MLP down projection: (1024\rightarrow256).
* Biases on all four linear layers.
* Two LayerNorms, each with scale and bias.

### Matrix parameters per block

| Operation                |           Shape |  Parameters |
| ------------------------ | --------------: | ----------: |
| QKV projection           |  (256\times768) |     196,608 |
| Attention output         |  (256\times256) |      65,536 |
| MLP up                   | (256\times1024) |     262,144 |
| MLP down                 | (1024\times256) |     262,144 |
| **Total matrix weights** |                 | **786,432** |

### Other parameters per block

| Component                       |                Parameters |
| ------------------------------- | ------------------------: |
| Linear biases                   |  (768+256+1024+256=2,304) |
| Two LayerNorm scales and biases | (2\times2\times256=1,024) |
| **Other parameters**            |                 **3,328** |

Therefore:

[
\boxed{789{,}760\text{ parameters per block}}
]

The transformer core contains:

| Blocks | Transformer-core parameters |
| -----: | --------------------------: |
|      4 |                   3,159,040 |
|      5 |                   3,948,800 |
|      6 |                   4,738,560 |

These numbers exclude:

* Token or observation embeddings.
* Positional embeddings.
* Policy, value, vocabulary, or other output heads.
* Any auxiliary losses or learned task-specific state.

For an embedding table with vocabulary or observation cardinality (V), add:

[
256V
]

parameters per replica. An untied vocabulary output head adds another (256V).

## 2.2 Persistent training-state memory

A common BF16 AdamW configuration requires approximately:

| State                           | Bytes per parameter |
| ------------------------------- | ------------------: |
| BF16 parameter                  |                   2 |
| BF16 gradient                   |                   2 |
| FP32 first moment               |                   4 |
| FP32 second moment              |                   4 |
| **Total without master weight** |              **12** |
| Optional FP32 master weight     |              **+4** |

Ignoring allocator overhead and activations:

| Blocks | State per replica, 12 B/parameter | State per replica, 16 B/parameter | 256 replicas, 12 B/parameter | 256 replicas, 16 B/parameter |
| -----: | --------------------------------: | --------------------------------: | ---------------------------: | ---------------------------: |
|      4 |                           37.9 MB |                           50.5 MB |                       9.7 GB |                      12.9 GB |
|      6 |                           56.9 MB |                           75.8 MB |                      14.6 GB |                      19.4 GB |

The embedding or output head can substantially increase these numbers. Activations, CUDA Graph memory pools, temporary GEMM workspaces, and allocator fragmentation must also fit in memory.

## 2.3 Transformer-core training FLOPs

For one token in one transformer block, the four linear projections require the following forward-pass work:

[
2(256)(768)
+
2(256)(256)
+
2(256)(1024)
+
2(1024)(256)
]

# [

1{,}572{,}864\text{ FLOPs/token/block}
]

For a linear layer, forward, activation-gradient, and weight-gradient GEMMs together are approximately three times the forward GEMM work:

[
\text{linear training FLOPs}
\approx
4{,}718{,}592
\text{ FLOPs/token/block}
]

The two attention matrix-multiplication stages, (QK^\top) and (PV), contribute approximately:

[
1{,}024S
]

forward FLOPs per token per block, or approximately:

[
3{,}072S
]

training FLOPs per token per block.

A useful estimate for transformer-core training compute is therefore:

[
\boxed{
F_{\text{token}}
\approx
L\left(4{,}718{,}592+3{,}072S\right)
}
]

This excludes normalization, softmax, dropout, residual operations, the optimizer, embeddings, and task-specific heads.

### Approximate transformer-core training FLOPs per token

| Sequence length |     4 blocks |     6 blocks |
| --------------: | -----------: | -----------: |
|             128 | 20.45 MFLOPs | 30.67 MFLOPs |
|             256 | 22.02 MFLOPs | 33.03 MFLOPs |
|             512 | 25.17 MFLOPs | 37.75 MFLOPs |

Total estimated transformer-core work per learner step is:

[
F_{\text{step}}
\approx
RBSF_{\text{token}}
]

---

# 3. H100 Hardware Target

All hardware numbers below use the **H100 SXM 80 GB** as the reference SKU.

| H100 SXM property                              |        Value |
| ---------------------------------------------- | -----------: |
| Streaming multiprocessors                      |          132 |
| Tensor Cores                                   |          528 |
| HBM capacity                                   |        80 GB |
| L2 cache                                       |        50 MB |
| HBM bandwidth                                  |    3.35 TB/s |
| Maximum TDP                                    |  Up to 700 W |
| Published BF16 Tensor throughput with sparsity | 1,979 TFLOPS |
| Approximate dense BF16 Tensor throughput       | 989.5 TFLOPS |

NVIDIA’s published 1,979 BF16 Tensor TFLOPS figure includes structured sparsity. Because our model uses ordinary dense weights, the relevant theoretical peak is approximately half of that:

[
\boxed{P_{\text{BF16,dense}}\approx989.5\text{ TFLOPS}}
]

The H100 SXM has 132 SMs and 528 fourth-generation Tensor Cores. NVIDIA currently specifies 80 GB of memory and 3.35 TB/s of memory bandwidth for the SXM model.

## 3.1 Compute-only token-rate ceiling

Dividing dense BF16 peak throughput by the transformer-core FLOP estimate gives an intentionally optimistic roofline:

| Sequence length | 4-block theoretical ceiling | 6-block theoretical ceiling |
| --------------: | --------------------------: | --------------------------: |
|             128 |     48.4 million tokens/sec |     32.3 million tokens/sec |
|             256 |     44.9 million tokens/sec |     30.0 million tokens/sec |
|             512 |     39.3 million tokens/sec |     26.2 million tokens/sec |

These are not expected real-world results. They exclude:

* Kernel launch overhead.
* LayerNorm and pointwise operations.
* Softmax and attention bookkeeping.
* Optimizer work.
* Embeddings and output heads.
* Memory traffic.
* Padding.
* Imperfect tile utilization.
* Tensor Core pipeline bubbles.

They provide a useful upper bound for calculating model FLOP utilization.

For example, a six-block, (S=256) learner running at 15 million useful tokens/sec executes approximately:

[
15\times10^6
\times
33.03\times10^6
===============

495.5\times10^{12}\text{ FLOPs/sec}
]

Its estimated dense model FLOP utilization would be:

[
\frac{495.5}{989.5}\approx50.1%
]

---

# 4. Why One Replica Underutilizes the H100

The transformer’s inner dimensions are only 256, 768, and 1024. These are Tensor-Core-friendly multiples, but each individual GEMM may expose too few independent output tiles to occupy 132 SMs efficiently.

For one replica, the four principal forward GEMMs are:

```text
QKV:       [T, 256]  @ [256, 768]
Attn out:  [T, 256]  @ [256, 256]
MLP up:    [T, 256]  @ [256, 1024]
MLP down:  [T, 1024] @ [1024, 256]
```

where:

```text
T = B × S
```

Problems include:

1. **Insufficient grid size.** A small matrix may not create enough thread blocks to occupy all 132 SMs.

2. **Wave quantization.** Performance changes discontinuously depending on how the number of output tiles maps onto SM waves.

3. **Short kernel duration.** Kernel launch and scheduling overhead become large relative to useful computation.

4. **Many pointwise kernels.** Norms, residuals, activation functions, dropout, loss operations, and optimizer operations can create a large number of short launches.

5. **Low weight reuse across replicas.** Because every replica has different weights, batching improves scheduling but does not let replicas share a weight matrix.

Replica-axis vectorization addresses the first three issues by exposing (R) independent matrix multiplications to the GPU at once.

---

# 5. PyTorch Tooling: What Each Tool Actually Does

## 5.1 Python loop over modules

```python
for r in range(R):
    loss = models[r](batches[r])
    loss.backward()
    optimizers[r].step()
```

This is the semantic reference implementation.

Advantages:

* Simple.
* Easy to verify.
* Naturally maintains independent model state.

Disadvantages:

* Repeats Python and dispatcher overhead (R) times.
* Can launch hundreds or thousands of small kernels.
* Gives the GPU one small operation at a time.
* Makes optimizer overhead especially severe.

Use this as the correctness baseline, not as the final high-throughput implementation.

## 5.2 `torch.func.stack_module_state`

`stack_module_state` takes multiple same-class modules and creates parameter and buffer dictionaries with a leading model dimension:

```text
weight: [R, ...]
bias:   [R, ...]
```

The stacked parameters are independent optimizable leaf tensors. PyTorch specifically documents it as a way to prepare same-class modules for vectorization with `vmap`.

This is useful for constructing the first stacked implementation without manually rewriting the parameter initialization and state-loading logic.

It does not by itself execute the model or guarantee efficient GPU kernels.

## 5.3 `torch.func.functional_call`

`functional_call` executes a module using an explicitly supplied parameter and buffer dictionary rather than the parameters physically stored inside the module.

It is commonly combined with `stack_module_state` and `vmap`:

```python
def run_one(params, buffers, batch):
    return torch.func.functional_call(template_model, (params, buffers), batch)
```

This separates model structure from model state, which is necessary when the model state has a leading replica dimension.

## 5.4 `torch.func.vmap`

`vmap` transforms a function that processes one replica into a function that processes a batch of replicas:

```python
batched_run = torch.vmap(
    run_one,
    in_dims=(0, 0, 0),
    randomness="different",
)
```

Conceptually:

```python
vmap(f)(x)[r] == f(x[r])
```

PyTorch describes `vmap` as pushing the map operation into the PyTorch operations called by the function.

Important properties:

* It is a **program transformation**, not a specific GEMM algorithm.
* It is excellent for building a correctness-first vectorized implementation.
* It may turn supported operations into batched operations.
* It does not guarantee that every resulting kernel is optimal.
* Unsupported side effects, mutations, custom operators, or randomness can require code changes.
* Independent dropout and other stochastic behavior require an explicit independent-randomness policy.

Use `vmap` as the first vectorized baseline and as a semantic reference for more manual implementations.

## 5.5 `torch.matmul`

`torch.matmul` is a general matrix-multiplication interface. It handles vectors, matrices, batched tensors, and broadcasted batch dimensions.

Advantages:

* Convenient.
* Flexible broadcasting.

Disadvantages:

* Its flexibility can hide broadcasts or layout conversions.
* It communicates less clearly that the operation is an equal-shape strided batch.

It is acceptable for prototyping, but explicit `bmm` is generally clearer for this workload.

## 5.6 `torch.bmm`

`torch.bmm` performs a batch of equal-shaped matrix multiplications:

```text
[R, M, K] @ [R, K, N] -> [R, M, N]
```

PyTorch requires both operands to be three-dimensional and to contain the same number of matrices.

For this learner, it maps directly to:

```python
y = torch.bmm(x, weight)
```

or, when weights are stored as `[R, N, K]`:

```python
y = torch.bmm(x, weight.transpose(1, 2))
```

Advantages:

* Exact match for equal-shape replicas.
* Minimal per-group metadata.
* Predictable strides.
* Supports autograd.
* Likely to use an efficient strided-batched GEMM implementation.

Disadvantages:

* Every replica in one call must have the same matrix dimensions.
* A general library batched kernel may still perform poorly for unusually small matrices.
* Forward, activation-gradient, and weight-gradient paths may have different efficiency.

For fixed (B) and (S), this should be the default GEMM baseline.

## 5.7 `torch.nn.functional.grouped_mm`

`grouped_mm` executes a collection of independent GEMMs as one grouped operation.

It supports:

* A three-dimensional form in which dimension zero enumerates groups.
* A two-dimensional jagged form using offsets.
* Different token counts per group in its jagged form.

The current PyTorch API is designed around grouped or MoE-style matrix multiplication. It currently expects BF16 CUDA inputs on devices with compute capability SM80 or newer. Passing `out_dtype=torch.float32` makes the operation non-differentiable through that API.

Advantages:

* Can support jagged token counts.
* Gives one scheduler visibility over many GEMM problems.
* May use a persistent or grouped scheduling strategy.
* Can reduce launch count.

Disadvantages:

* More general than necessary for equal-shaped replicas.
* The API does not guarantee that the selected implementation is persistent.
* Group metadata and generality may make it slower than `bmm`.
* Forward, `dX`, and `dW` support and performance must all be tested.
* API behavior and maturity depend on the installed PyTorch version.

Use it as an A/B competitor to `bmm`, not as an assumed upgrade.

## 5.8 Scaled dot-product attention

After replica-specific QKV projections, attention tensors have shape:

```text
[R, B, H, S, Dh]
```

They can be reshaped to:

```text
[R × B, H, S, Dh]
```

and passed through one scaled dot-product-attention call:

```python
q = q.reshape(R * B, H, S, Dh)
k = k.reshape(R * B, H, S, Dh)
v = v.reshape(R * B, H, S, Dh)

out = torch.nn.functional.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=mask,
    dropout_p=dropout_p,
    is_causal=is_causal,
)
```

Attention does not mix different batch elements, so flattening (R\times B) preserves replica independence. PyTorch SDPA may select among fused and mathematical attention implementations depending on the inputs and backend constraints.

Backend selection and actual kernel efficiency must be verified by profiling.

## 5.9 `torch.compile`

`torch.compile` traces and compiles a function or module. It can:

* Fuse pointwise operations.
* Reduce Python and dispatcher overhead.
* Generate Triton kernels.
* Autotune matrix-multiplication configurations.
* Use CUDA Graphs where applicable.
* Pad matrix shapes for better hardware alignment.

Relevant modes include:

```python
torch.compile(train_step, mode="default")
torch.compile(train_step, mode="reduce-overhead")
torch.compile(train_step, mode="max-autotune")
torch.compile(train_step, mode="max-autotune-no-cudagraphs")
```

PyTorch documents `reduce-overhead` and `max-autotune` as supported modes. Compiler options include epilogue fusion, matmul autotuning, shape padding, and CUDA Graph support.

`torch.compile` is not a replacement for choosing the correct tensor representation. It works best after model state and operations have already been vectorized over (R).

## 5.10 CUDA Graphs

A CUDA Graph records a fixed GPU execution schedule and replays it with very low CPU launch overhead.

Instead of issuing hundreds of operations from Python:

```text
Python -> launch -> launch -> launch -> ...
```

the steady-state learner becomes approximately:

```text
copy rollout data into static buffers
graph.replay()
```

CUDA Graphs are most appropriate when shapes and control flow are static and the workload is at least partly CPU-launch-bound. PyTorch exposes raw CUDA Graph capture and recommends graphs for graph-safe workloads with static structure.

The captured graph should contain one vectorized learner step over all (R) replicas, not (R) separately captured model executions.

## 5.11 `foreach` and fused optimizers

PyTorch optimizers may offer:

* A scalar tensor-by-tensor implementation.
* A `foreach` implementation that operates over lists of tensors.
* A fused implementation that combines optimizer operations.

These can reduce launch count relative to a naive optimizer, but a custom stacked model representation may permit something even simpler:

```text
parameter: [R, ...]
gradient:  [R, ...]
adam_m:    [R, ...]
adam_v:    [R, ...]
```

AdamW is elementwise over these stacked tensors, so one kernel can update all replicas without coupling them.

Optimizer performance must be measured early rather than treated as an afterthought.

## 5.12 Triton and CUTLASS

Triton and CUTLASS allow custom kernels specialized for the exact matrix shapes:

```text
256 -> 768
256 -> 256
256 -> 1024
1024 -> 256
```

Possible customizations include:

* Persistent work scheduling.
* Hopper-specific tile shapes.
* TMA-based data movement.
* Fused bias or activation epilogues.
* Specialized weight-gradient reductions.
* Split-K or stream-K scheduling.

These tools offer the greatest control but also the highest engineering and maintenance cost.

They should be used only after identifying a specific kernel whose end-to-end impact justifies custom implementation.

---

# 6. Correct Vectorized Learner Layout

## 6.1 Parameters and optimizer state

Store state by parameter type with a leading replica dimension:

```text
qkv_weight:       [R, 768, 256]
qkv_bias:         [R, 768]

attn_out_weight:  [R, 256, 256]
attn_out_bias:    [R, 256]

mlp_up_weight:    [R, 1024, 256]
mlp_up_bias:      [R, 1024]

mlp_down_weight:  [R, 256, 1024]
mlp_down_bias:    [R, 256]

norm_weight:      [R, 256]
norm_bias:        [R, 256]

adam_m:           same shape as parameter
adam_v:           same shape as parameter
```

Avoid representing each replica as a separate collection of small Python parameter objects in the final implementation.

## 6.2 Activations

Use:

```text
[R, B, S, D]
```

and reshape to:

```text
[R, B × S, D]
```

for linear operations.

For the QKV projection:

```text
[R, B × S, 256] @ [R, 256, 768]
```

For attention:

```text
[R, B, H, S, Dh] -> [R × B, H, S, Dh]
```

## 6.3 Replica-local reductions

No reduction may accidentally combine replicas.

Correct loss reduction:

```python
# token_loss and mask: [R, B, S]

loss_per_replica = (
    (token_loss * mask).sum(dim=(1, 2))
    / mask.sum(dim=(1, 2)).clamp_min(1)
)

total_loss = loss_per_replica.sum()
```

Summing the independent scalar losses is safe because the parameters are disjoint.

Do not average across replicas unless the resulting (1/R) gradient scaling is explicitly compensated.

## 6.4 Gradient clipping

Compute one global gradient norm per replica:

[
n_r=
\sqrt{
\sum_p
\lVert g_{r,p}\rVert_2^2
}
]

The resulting clipping scale has shape:

```text
[R]
```

and must be broadcast independently onto each replica’s gradients.

A single norm over all replicas would change training semantics by coupling otherwise independent runs.

The same rule applies to:

* Advantage normalization.
* Return normalization.
* KL statistics.
* Adaptive coefficients.
* Dynamic loss scaling.
* Learning-rate schedules.
* Update masks.
* Early-stopping state.

---

# 7. ROI-Ordered Optimization Plan

## 1. Establish a correct loop baseline

Implement and retain a conventional (R)-model loop for validation.

Measure:

* One-step losses.
* Per-replica gradients.
* Updated weights.
* Optimizer moments.
* Gradient clipping.
* Several hundred training steps.

Success condition:

* The vectorized learner produces numerically equivalent independent training trajectories within the expected BF16 tolerance.

## 2. Stack model state and build a `vmap` baseline

Use:

* `stack_module_state`.
* `functional_call`.
* `vmap`.
* Independent randomness for stochastic operations.

Expected ROI:

* High semantic value.
* Moderate performance improvement.
* Low-to-moderate engineering cost.

This becomes the vectorized correctness baseline.

## 3. Replace principal linear layers with explicit `bmm`

Use explicit replica-batched matrix multiplications for:

* QKV forward, `dX`, and `dW`.
* Attention output forward, `dX`, and `dW`.
* MLP up forward, `dX`, and `dW`.
* MLP down forward, `dX`, and `dW`.
* Any large replica-specific output head.

Expected ROI:

* Very high.
* Moderate engineering cost.
* Likely the most important GPU-utilization improvement.

Measure each of forward, activation-gradient, and weight-gradient performance separately.

## 4. Flatten (R\times B) for SDPA

Execute one attention operation over all replica-batch elements.

Expected ROI:

* High.
* Low engineering cost.
* Uses established fused attention implementations.

Verify the actual selected attention backend and mask behavior.

## 5. Vectorize loss, clipping, and optimizer operations

Eliminate operations launched independently per replica.

Prioritize:

* Per-replica loss normalization.
* Per-replica gradient norms.
* Gradient scaling.
* AdamW moment updates.
* Parameter updates.
* Gradient clearing or overwriting.

Expected ROI:

* High when the current optimizer uses many small tensors.
* Moderate engineering cost.
* Often more valuable than custom forward GEMMs.

## 6. Sweep replica count

After establishing a vectorized implementation, sweep:

```text
R = 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256
```

Do not rely exclusively on powers of two. Once a plateau or regression appears, sample nearby values densely because GEMM tile and wave boundaries can create non-monotonic results.

Measure:

* Useful tokens/sec.
* Aggregate replica updates/sec.
* Per-replica updates/sec.
* Step latency.
* Peak memory.
* Estimated model FLOP utilization.

Stop increasing (R) when throughput plateaus, memory becomes limiting, or rollout ingress can no longer feed the learner.

## 7. Benchmark `torch.compile`

Test at least:

```text
eager
compile/default
compile/reduce-overhead
compile/max-autotune
compile/max-autotune-no-cudagraphs
```

Expected ROI:

* High potential.
* Low initial engineering cost.
* May automatically fuse pointwise operations and reduce launch overhead.

Record:

* Compilation time separately.
* Steady-state throughput.
* Graph breaks.
* Recompilations.
* Generated-kernel count.
* CUDA Graph usage.

## 8. Capture the complete learner step in a CUDA Graph

Capture:

```text
forward
loss
backward
per-replica clipping
optimizer
gradient reset
```

Use persistent input buffers and replay the graph after copying new rollout data into those buffers.

Expected ROI:

* High when the Nsight Systems timeline contains CPU submission gaps.
* Low when GPU kernels already execute continuously.
* Moderate engineering cost.

Use one graph per common static shape bucket, such as:

```text
(R, B, S=128)
(R, B, S=256)
(R, B, S=512)
```

## 9. Compare `bmm` with `grouped_mm`

Benchmark both implementations for:

* Forward GEMMs.
* Activation-gradient GEMMs.
* Weight-gradient GEMMs.
* Full learner steps.

Expected ROI:

* Medium.
* Low-to-moderate experimental cost.
* More likely to help when token counts differ or individual GEMMs are extremely small.

Do not adopt grouped GEMM based only on forward microbenchmarks.

## 10. Introduce sequence bucketing or jagged compaction

Pursue this only when padding wastes a material fraction of learner compute.

Options:

* Static sequence-length buckets.
* Compact valid tokens for linear layers.
* Jagged grouped GEMM.
* A variable-length attention implementation.

Expected ROI:

* High when padding efficiency is poor.
* Low when nearly all tokens are valid.
* Moderate-to-high engineering cost.

Compacting only the linear layers is insufficient if padded attention remains dominant.

## 11. Fuse memory-bound operations

Candidates include:

* Bias plus GELU.
* Bias plus residual.
* Dropout plus residual.
* LayerNorm or RMSNorm with adjacent pointwise operations.
* Gradient clipping and optimizer scaling.
* Optimizer moment and parameter updates.

LayerNorm is a reduction, not merely a matrix-multiplication epilogue. It should be optimized as a normalization pipeline rather than treated as a simple affine GEMM epilogue.

Expected ROI:

* Incremental to moderate.
* Most useful after launch overhead and GEMM scheduling have been addressed.

## 12. Implement custom Triton or CUTLASS kernels

Target only operations with demonstrated end-to-end headroom.

Likely candidates:

* Persistent equal-shape replica-batched GEMM.
* Small-(M) weight-gradient GEMM.
* Fused MLP-up plus activation.
* Fused MLP-down plus residual.
* Vectorized optimizer kernels.

Expected ROI:

* Potentially high for a confirmed bad kernel.
* High engineering and maintenance cost.

Proceed only when the Amdahl-limited end-to-end opportunity is at least approximately 5%.

---

# 8. Required Metrics

## 8.1 Primary throughput metrics

### Useful learner tokens/sec

[
\boxed{
\text{useful tokens/sec}
========================

\frac{\sum_r\text{valid tokens}_r}
{\text{learner time}}
}
]

Use valid, non-padding tokens.

### Physical tokens/sec

[
\text{physical tokens/sec}
==========================

\frac{RBS}{\text{learner time}}
]

This exposes whether an apparent speedup is actually caused by processing more padding.

### Padding efficiency

[
\boxed{
\text{padding efficiency}
=========================

\frac{\text{valid tokens}}
{RBS}
}
]

### Aggregate replica updates/sec

[
\boxed{
\text{aggregate updates/sec}
============================

\frac{R}{\text{step time}}
}
]

### Per-replica updates/sec

[
\boxed{
\text{per-replica updates/sec}
==============================

\frac{1}{\text{step time}}
}
]

Aggregate throughput can improve while each individual run becomes slower, so report both.

## 8.2 Step latency

Report:

* Mean step time.
* Median step time.
* P95 step time.
* P99 step time.
* Standard deviation.
* CUDA Graph replay time.
* Input-staging time separately.

Use CUDA events or profiler GPU timestamps for device-only learner time.

## 8.3 Estimated model FLOP utilization

Calculate:

[
\text{estimated FLOPs/sec}
==========================

\text{useful tokens/sec}
\times
F_{\text{token}}
]

Then:

[
\boxed{
\text{dense BF16 MFU}
=====================

\frac{\text{estimated FLOPs/sec}}
{989.5\times10^{12}}
}
]

Report this as an estimate because the transformer-core FLOP model excludes some operations and may not exactly match fused-kernel execution.

Useful diagnostic bands for this tiny full-training workload are:

| Dense BF16 MFU | Interpretation                                                      |
| -------------: | ------------------------------------------------------------------- |
|      Below 20% | Likely substantial launch, occupancy, padding, or non-GEMM overhead |
|         20–40% | Useful progress, but probably meaningful headroom remains           |
|         40–60% | Strong end-to-end utilization for a tiny training workload          |
|      Above 60% | Excellent; investigate carefully before pursuing major custom work  |

These are engineering heuristics, not hardware guarantees.

## 8.4 GPU metrics

Track:

* SM active percentage.
* Tensor Core active percentage.
* Achieved BF16 Tensor TFLOPS.
* Achieved occupancy.
* Active warps per SM.
* DRAM throughput in GB/s.
* DRAM throughput as a percentage of peak.
* L2 hit rate.
* GPU clock frequency.
* HBM clock frequency.
* Board power.
* Thermal or power throttling.

Use stable clocks and power settings when comparing experiments.

## 8.5 Kernel metrics

For every major kernel category, report:

| Category              | Required measurements             |
| --------------------- | --------------------------------- |
| QKV GEMM              | Forward, `dX`, `dW`, time, TFLOPS |
| Attention output GEMM | Forward, `dX`, `dW`, time, TFLOPS |
| MLP up GEMM           | Forward, `dX`, `dW`, time, TFLOPS |
| MLP down GEMM         | Forward, `dX`, `dW`, time, TFLOPS |
| SDPA                  | Forward and backward time         |
| Normalization         | Forward and backward time         |
| Pointwise             | GELU, residual, dropout, masking  |
| Loss                  | Forward and backward time         |
| Optimizer             | Clipping, moments, update, reset  |
| Data movement         | H2D and static-buffer copies      |

Also report:

* Number of kernel launches per learner step.
* Median kernel duration.
* Total launch-to-launch idle time.
* Percentage of step time spent in GPU gaps.
* Number of device synchronizations.
* Number and duration of memory copies.
* Number of CUDA Graph replays.
* Compiler-generated versus library kernels.

Nsight Systems is the primary tool for the launch timeline, CPU submission, gaps, synchronization, and data movement. Nsight Compute is the primary tool for detailed per-kernel compute, memory, occupancy, and pipeline analysis. NVIDIA notes that Nsight Compute metric collection can add substantial profiling overhead, so it should be used selectively on representative kernels rather than for final unprofiled throughput numbers.

## 8.6 Per-shape GEMM efficiency

For each important GEMM shape, create an isolated benchmark using the best available implementation.

Calculate:

[
\boxed{
\text{GEMM shape efficiency}
============================

\frac{\text{full-learner kernel TFLOPS}}
{\text{best isolated TFLOPS for the same shape and layout}}
}
]

This is more actionable than comparing every small GEMM directly against the full 989.5 TFLOPS hardware peak.

A small GEMM may be close to its practical shape-specific limit while still achieving a modest fraction of theoretical device peak.

## 8.7 Memory metrics

Track:

* Memory allocated for parameters.
* Memory allocated for gradients.
* Memory allocated for optimizer state.
* Peak activation memory.
* GEMM and SDPA workspace memory.
* CUDA Graph private-pool memory.
* PyTorch peak allocated memory.
* PyTorch peak reserved memory.
* Allocator fragmentation.
* Maximum feasible replica count.
* Memory consumed per additional replica.

## 8.8 Compiler metrics

For compiled implementations, track:

* Number of graph breaks.
* Number of recompilations.
* Compilation and autotuning time.
* Number of generated kernels.
* Number of fused operations.
* Shape guards.
* Number of captured CUDA Graphs.
* Steady-state execution time after warmup.

Do not include compilation or autotuning in steady-state tokens/sec.

## 8.9 Correctness metrics

For every optimization, compare against the loop reference:

* Maximum and mean loss difference per replica.
* Maximum and mean activation difference.
* Maximum and mean gradient difference.
* Maximum and mean updated-weight difference.
* Optimizer-moment differences.
* Per-replica gradient norms.
* Per-replica clipping decisions.
* Training curves over an extended run.

Check explicitly that changing one replica’s batch or loss does not change any other replica’s gradients or update.

---

# 9. How to Determine Whether We Are Near Optimal

The learner is close to optimal when all of the following are true.

## 9.1 Replica-count plateau

Increasing (R) produces less than approximately 2% additional useful tokens/sec for at least two successive increases, and a dense local sweep around the plateau finds no materially better value.

Do not declare a plateau based only on powers-of-two measurements.

## 9.2 Negligible launch gaps

Nsight Systems shows:

* Less than approximately 3% of learner GPU time in unexplained inter-kernel gaps.
* No long CPU submission stalls.
* No unnecessary device synchronizations.
* CUDA Graph replay or compiled execution keeping the device continuously supplied.

## 9.3 Dominant kernels are individually efficient

For kernels accounting for most learner time:

* GEMMs are within approximately 10% of the best isolated implementation for the exact shape and layout.
* Compute-bound kernels have high Tensor Core activity.
* Memory-bound kernels reach at least approximately 70% of the measured sustainable bandwidth available to that access pattern.
* Occupancy limitations are understood and are not fixable merely by increasing grid size.

## 9.4 GPU time is concentrated in unavoidable work

At least approximately 90% of device time is explained by expected major operations such as:

* GEMMs.
* SDPA.
* Normalization.
* Loss.
* Optimizer.

There should not be a long tail of tiny launch-bound pointwise kernels consuming a material fraction of the step.

## 9.5 Weight-gradient performance is acceptable

The `dW` GEMMs should be measured separately.

They are acceptable when either:

* Their shape-specific efficiency is close to the best isolated implementation, or
* Their total Amdahl-limited effect on learner throughput is below approximately 5%.

Comparing `dW` TFLOPS directly with forward TFLOPS is insufficient because their shapes and reduction behavior differ.

## 9.6 Padding is not hiding wasted compute

Either:

* Padding efficiency is high, or
* Sequence bucketing or variable-length execution has already been evaluated.

Always optimize useful tokens/sec rather than padded tokens/sec.

## 9.7 End-to-end MFU is strong

For this tiny full-training workload:

* Above approximately 40% dense BF16 MFU is strong.
* Above approximately 60% is excellent.

A lower MFU can still be near optimal if the remaining workload is fundamentally memory-bound, reduction-bound, or dominated by a task-specific head. That conclusion must be supported by per-kernel roofline measurements.

## 9.8 Remaining opportunities fail the Amdahl test

For an optimization targeting a component that occupies fraction (f) of total step time, even eliminating that component entirely can improve throughput by at most:

[
\text{maximum speedup}
======================

\frac{1}{1-f}
]

Examples:

| Current component share | Maximum gain if made free |
| ----------------------: | ------------------------: |
|                      2% |                      2.0% |
|                      5% |                      5.3% |
|                     10% |                     11.1% |
|                     20% |                     25.0% |

Stop pursuing custom kernels when their credible end-to-end opportunity is below approximately 5%, unless they provide another strategic benefit.

---

# 10. Profiling Decision Tree

## The timeline contains CPU launch gaps

Try:

1. `torch.compile(mode="reduce-overhead")`.
2. Full-step CUDA Graph capture.
3. Additional fusion.
4. Fewer independently represented parameter tensors.

## GEMMs are short and have low SM utilization

Try:

1. Increase (R).
2. Explicit `bmm`.
3. Compare `grouped_mm`.
4. Persistent batched or grouped Triton/CUTLASS kernels.

## Forward GEMMs are good but `dW` is poor

Try:

1. Verify gradient layouts and hidden transposes.
2. Benchmark grouped weight-gradient GEMM.
3. Test split-K or stream-K.
4. Implement a specialized persistent `dW` kernel.

## SDPA dominates

Try:

1. Confirm the selected fused backend.
2. Verify Q/K/V layout and contiguity.
3. Flatten (R\times B).
4. Reduce padding through bucketing.
5. Evaluate variable-length attention.

## Pointwise operations dominate

Try:

1. `torch.compile`.
2. Bias plus GELU fusion.
3. Residual plus dropout fusion.
4. Fused normalization pipelines.
5. Fused clipping and optimizer operations.

## Optimizer dominates

Try:

1. Fused or `foreach` optimizer baselines.
2. Stack optimizer state across (R).
3. Flatten compatible parameter tensors.
4. Fuse gradient scaling, moment updates, and parameter updates.

## Memory bandwidth is saturated

Increasing replica count or occupancy may not help.

Try:

1. Reduce intermediate reads and writes.
2. Fuse adjacent operations.
3. Avoid unnecessary copies and transposes.
4. Use lower-precision state only when training semantics permit it.

---

# 11. Explicit Non-Goals and Anti-Patterns

## Do not average gradients across replicas

These are independent training runs, not data-parallel workers training one shared model.

No all-reduce or cross-replica gradient averaging is required.

## Do not reduce losses or statistics across (R)

The replica dimension must remain independent for:

* Loss normalization.
* Advantage normalization.
* Gradient clipping.
* Adaptive coefficients.
* Optimizer state.
* Learning-rate schedules.

## Do not build a dense block-diagonal matrix

Constructing:

[
\operatorname{diag}(W_1,W_2,\ldots,W_R)
]

and applying one dense GEMM performs large amounts of multiplication by zero.

Use batched or grouped GEMM.

## Do not create one CUDA stream per replica

One stream per replica introduces unnecessary stream-management and dependency complexity.

The primary design should use grouped or batched operations that expose all replica work to one kernel scheduler.

A small bounded stream-pool baseline may be tested only when profiling demonstrates that individual kernels remain underoccupied and cannot be grouped effectively.

## Do not assume `vmap` implies optimal kernels

`vmap` establishes vectorized semantics. The generated execution must still be inspected.

## Do not assume `grouped_mm` is faster than `bmm`

Equal-shaped replicas are the ideal case for strided batched GEMM. Grouped GEMM must win an end-to-end benchmark before adoption.

## Do not optimize only the forward pass

Training performance includes:

* Forward.
* Activation gradients.
* Weight gradients.
* Loss.
* Gradient clipping.
* Optimizer.
* Gradient reset.

A faster forward microbenchmark can produce negligible learner improvement.

---

# 12. Recommended Final Architecture

The target steady-state learner should look conceptually like this:

```text
Copy R rollout batches into static GPU buffers
                    |
                    v
Replica-aware embedding or input projection
                    |
                    v
For each transformer block:
    batched QKV GEMM over R
                    |
    reshape [R,B,H,S,Dh] -> [R*B,H,S,Dh]
                    |
    one SDPA forward
                    |
    reshape back to replica layout
                    |
    batched attention-output GEMM over R
                    |
    fused replica-local residual/norm operations
                    |
    batched MLP-up GEMM over R
                    |
    GELU
                    |
    batched MLP-down GEMM over R
                    |
    fused replica-local residual/norm operations
                    |
                    v
Replica-specific task head
                    |
Replica-local losses and normalization
                    |
Vectorized backward over stacked state
                    |
Replica-local gradient norm and clipping
                    |
Vectorized/fused optimizer over stacked state
                    |
Graph replay completes
```

The full step should be compiled or captured as a small number of static CUDA Graphs keyed by common learner shapes.

---

# 13. Minimal Experiment Record

Every experiment should record:

```text
Software:
    PyTorch version
    CUDA version
    NVIDIA driver
    Triton/CUTLASS version
    compile mode
    SDPA backend

Hardware:
    exact H100 SKU
    GPU clocks
    HBM clocks
    power limit
    MIG status

Workload:
    R
    B
    S
    valid-token count
    padding efficiency
    number of layers
    head/output dimensions
    optimizer
    gradient clipping
    precision

Results:
    useful tokens/sec
    physical tokens/sec
    aggregate replica updates/sec
    per-replica updates/sec
    mean/p50/p95 step latency
    estimated dense BF16 MFU
    peak allocated memory
    peak reserved memory
    kernel launches/step
    GPU gap percentage
    GEMM time
    SDPA time
    pointwise/norm time
    loss time
    optimizer time
    H2D/staging time
```

Use the same warmup count, timed iteration count, clock configuration, and rollout-buffer location for every comparison.

---

# 14. Recommended Implementation Order

1. Preserve a loop-based correctness reference.
2. Stack parameters and buffers across (R).
3. Build a `functional_call` plus `vmap` baseline.
4. Implement replica-local reductions correctly.
5. Replace equal-shaped linear layers with `bmm`.
6. Flatten (R\times B) for SDPA.
7. Vectorize loss, clipping, and optimizer operations.
8. Sweep (R), including non-powers-of-two near plateaus.
9. Benchmark all `torch.compile` modes.
10. Capture the full learner step in CUDA Graphs.
11. Compare `bmm` against `grouped_mm` for forward, `dX`, and `dW`.
12. Address padding through bucketing or jagged execution when needed.
13. Fuse bandwidth- and launch-bound operations.
14. Implement Triton or CUTLASS kernels only for confirmed bottlenecks.
15. Stop when every remaining optimization has less than approximately 5% credible end-to-end value.

---

# 15. Bottom Line

The workload is a batch of independent learners:

[
\boxed{
R\text{ different parameter sets}
\times
B\text{ different rollout samples per set}
}
]

The replica dimension should be represented explicitly in model state and computation.

For equal-shaped rollout batches, the default optimized primitive should be:

[
\boxed{\text{strided batched GEMM with } \texttt{torch.bmm}}
]

The most likely high-ROI path is:

[
\boxed{
\texttt{stack_module_state}
\rightarrow
\texttt{vmap baseline}
\rightarrow
\texttt{bmm}
\rightarrow
\text{flattened SDPA}
\rightarrow
\text{vectorized optimizer}
\rightarrow
\texttt{torch.compile}
\rightarrow
\text{CUDA Graph}
}
]

`grouped_mm`, persistent Triton kernels, and CUTLASS kernels are important experiments, but they should be justified by measured deficiencies in the equal-shape batched baseline.

The implementation is close to optimal when useful tokens/sec has plateaued across (R), launch gaps are negligible, dominant kernels are close to their exact-shape isolated limits, dense BF16 MFU is strong, and every remaining optimization has less than approximately 5% credible end-to-end impact.