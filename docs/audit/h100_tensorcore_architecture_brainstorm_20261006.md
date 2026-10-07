# H100 model architecture candidates — 2026-10-06

This is a brainstorm requested by the user after the H100 learner profiles.
No architecture below has been implemented or benchmarked. No GPU jobs were
launched for this report. Protected theory and queued-design documents remain
unchanged. Implementation of a selected architecture needs a separately
reviewed specification; this report does not redefine existing contracts.

The target is faster learning at comparable quality and capacity. A high
Tensor activity percentage obtained by adding unnecessary matrix work would
not be progress. Measure complete learner/training time as well as Tensor
activity, memory and eventual learning quality.

## Evidence and invariants

Actual HU learner packets contain 176 preflop and 80 postflop states, despite
physical caps of 256/128. The tested model has six layers, width 128, four
heads of width 32, FFN hidden width 512, and class embedding width 32.
Profiled summed GPU kernel durations put GEMMs plus attention at 28.41% pre
and 46.50% post; a single learner program launches 3,466 / 3,981 kernels.
Full learner GPM Tensor activity is around 9–10%. A dense 8192-square BF16
GEMM on the same GPU reaches 92.9% Tensor activity and approximately 681
arithmetic TFLOP/s. These quantities have different denominators and should
not be equated. See `h100_bounded_final_handoff_20261006.md` for raw artifacts.

Reading `shared.py`, `preflop.py`, `postflop.py`, `incidence.py`,
`primitives.py`, `heads.py`, `pbg/core/tensor_ops.py`, and queued `model.py`
shows substantial repeated work around each attention call:

- Three normalization calls per block: incidence normalization, trunk
  pre-attention normalization, and post-attention normalization. Six blocks
  plus final normalization give 19 trunk/incidence normalization calls.
  Each current lane-aware norm also performs separate affine multiply/add.
- Five incidence projections, two packed-attention/output projections,
  and three common/static/output FFN projections per block: 60 linear calls
  across six blocks, before encoders/readout. This is a count of module calls,
  not CUDA kernels; backward, casts and bias additions launch further kernels.
- FP32 mass/incidence contractions, overlap products, residual additions,
  padding and feature concatenation repeat in every block. The side features
  are invariant across blocks, yet their padding and layer-specific static
  projections are performed inside the loop.

Every proposal preserves all full private input/output IDs, external legality
masking, policy mixture/value/public-head contracts, no selected dealt hand as
privileged model input, and independent replica parameters. Pre/post adapters
may differ, but the main trunk and output heads share weights. Retain raw and
normalized reaches, board/folded inputs, and the learned reach/key prior.
When using attention over original private rows, the reach prior is additive
after content temperature, never divided by that temperature. No queue
partition, street-specific compact bound, or reduction of the hand domain is
proposed. Existing exact incidence semantics remain the default baseline;
any change in their placement is explicitly an architecture ablation.

## Ranked architecture experiments

### 1. Fewer, wider shared blocks

Start with **3 layers, width192, 3 heads of width64, hidden768** against the
current 6×128 / hidden512. This uses existing configurable components and
keeps the same model structure and exact incidence pathway. It halves serial
normalization, overlap construction, residual and feature-assembly rounds,
while giving each projection and attention head more work.

Ignoring public/readout/static-adapter details, projection/FFN arithmetic
scales as $L d^2$: 3×192 costs approximately **1.125×** the baseline. Dense
attention score/value arithmetic scales as $L d$, approximately **0.75×**.
This makes it a more informative first comparison than simply doubling all
dimensions. The post static width changes with $d$, so the full arithmetic
and parameter budget must be counted from the actual instantiated model.

Also test **2×256, heads4, hidden1024** (approximately 1.33× projection and
0.67× attention arithmetic) and, if quality needs more depth, **3×256**
(approximately 2× projection and unchanged attention arithmetic). No speedup
is guaranteed: larger GEMMs may help while added work or reduced depth may
hurt. Fewer composition steps can reduce strategic expressivity and training
stability; retain a six-layer wide model as a quality control only if budget
permits. This is my first recommendation because it isolates the cost of
serial architecture with minimal architectural novelty.

### 2. Full-domain row encoder → latent trunk → full-domain row decoder

Keep every preflop class and postflop hand row. Encode all rows with their
current range/geometry features, then use **32/64/128 learned latent queries**
to cross-attend to all original private and public rows. The original row
keys retain their learned temperatures, additive log-reach priors, and masks.
Run a wider shared transformer/MLP trunk on these latents. Decode each full
actor private row by cross-attention to the latents, retaining a local row
skip and explicit exact incidence/card-summary features. Readout contracts
and queues remain the same. Latent keys represent summaries, not individual
hands; do not incorrectly attach original per-hand reach priors to them.

This is an **internal learned representation**, not queue splitting or hand
pruning. The sampled learner may select its output only after full-context
encoding, just as the current implementation does. RB and rollout still
produce every natural-domain actor output.

For illustration, ignoring public tokens, HU postflop has $N=2×1326=2652$
private tokens and $A=1326$ actor outputs. At width128, one full self-attention
score/value pair costs about $4N^2d=3.60$ GFLOPs per state per layer, counting
multiply and add separately. Six such layers cost 21.61 GFLOPs before
projections/softmax/backward. With 64 latent tokens, one read and one decode cost
$4d(NM+AM)≈0.130$ GFLOPs per state; latent self-attention adds $4LM^2d$.
These are analytic attention arithmetic estimates, **not measured speedups**;
a repeated row/latent read-update architecture costs more than a single read.

The benefit is much less expensive long-range context handling and fewer
per-hand bookkeeping rounds. The risk is a finite latent bottleneck: rare
hands, sharp blocker effects, and arbitrarily detailed opponent ranges may
be compressed poorly. Retain per-hand raw reach, private embedding, board
geometry, and exact card/identical-combo residuals in the row decoder; test
latent count and one-vs-three row/latent interaction rounds. This candidate
could achieve lower Tensor activity yet faster training because it removes
quadratic work; evaluate the resulting latency/quality, not activity alone.

### 3. Dense range-conditioned residual MLP with a full-row decoder

A simpler and more radical competitor: family-specific dense adapters embed
the complete two-player range tensors, masks, public state/history, and
folded statistics into a shared context. A shared residual MLP processes that
context. A wide batched per-hand decoder combines context, hand/class
embedding, local reaches, geometry and explicit exact card summaries to emit
the existing policy/value fields for every actor row. Learned weighted
pooling can preserve the reach prior via
$\alpha_{mj}=\operatorname{softmax}_j(a_m(x_j)/\tau_m+\ell_j)$;
the reach is still added after temperature. Keep raw unnormalized mass as a
separate feature, since normalized attention alone discards it.

This replaces many attention/incidence feedback rounds with a few wide dense
matrices. A direct first range projection from HU's 2×1326 range entries to
512 context features contains approximately 1.36M weights, before extra
channels/public features. A preflop 2×169-to-512 adapter has approximately
173k weights. These are illustrative single-channel sizes, not total model
budgets or recommendations to discard folded inputs.

Risks: a shared context bottleneck, loss of repeated hand-specific reasoning,
fixed-card-ID dependence, and weaker suit generalization. Preflop retains its
canonical class indexing; postflop stays in actual suit IDs with optional
suit-consistency augmentation already described in the model specification.
Use a learned attention/weighted-basis encoder rather than a raw flattening
as another ablation. Preserve exact incidence features computed once; moving
them from every layer to once is a representational change, not an exact
optimization. This is a valuable simple speed/quality baseline, but higher
risk than candidate1.

### 4. Shared wide feature block and consolidated output projections

The current readout uses separate type, raise and value two-layer feature
maps over the same actor/context input. A shared hidden bank can feed the
three output projections, with small task-specific adapters if needed. The
policy mixture nonlinearities and all-in/value outputs remain identical in
form. Similarly, a shared joint dense map can replace the separate query,
identical and own incidence branches, taking their concatenation as input.

Sharing the hidden bank changes parameter sharing/capacity; merging independent
projection weights into one bank can instead be algebraically equivalent.
Those two variants must not be conflated. This is likely a smaller gain than
changing six repeated trunk blocks, especially in the sampled learner where
only one actor row is read out. RB full-domain readout is the better target.

### 5. Fewer normalization sites and simpler overlap features

Architectural choices include one prenorm per block reused by its branches,
RMSNorm instead of LayerNorm, or a learned fused residual block consuming
raw card/equal overlap vectors instead of explicitly constructing their
squares/products. These reduce arithmetic bookkeeping and memory traffic.

They change the existing function class: incidence and trunk norms currently
have distinct affine owners; RMSNorm removes mean-centering; removing explicit
overlap products removes a supplied polynomial feature. Keep exact incidence
contractions and evaluate each change individually. Normalization alone is
only 5.6–7.6% of profiled kernel time, so changing the norm formula alone
cannot explain or remove the entire non-matrix bottleneck.

## Separate implementation improvements that do not require new architecture

Do these before judging whether the current model is intrinsically slow:

1. Fused norm affine/residual/cast and GEMM bias/GELU epilogues; keep FP32
   statistics/master state where useful but avoid unnecessary FP32 activation
   banks. The current generic `linear` uses BMM followed by a separate bias
   add, and `LayerNorm` adds separate affine kernels. Exact-operation fusion
   differs from changing normalization or accumulation precision.
2. Hoist side-feature padding outside the six-layer loop. Consider one packed
   static-projection bank for all six layers. With 80 HU postflop states,
   2652 private tokens and hidden512, six BF16 static outputs alone occupy
   approximately **1.30 GB**; do not trade fewer launches for excessive live
   activation residency. Profile actual backward lifetime and allocator use.
3. Combine independent incidence projections or the three readout input
   projections in one GEMM while retaining independent output slices. Packed
   parameter ownership/checkpoint and zero-copy lane semantics need explicit
   design; repeatedly concatenating master weights at runtime can erase gains.
4. Aligned working dimensions: the measured post static343→344 padding gave
   approximately 4% model-only improvement and selected Hopper GEMMs, with
   output/gradient oracles passing. It remains experimental. Such zero working
   padding changes neither trainable width nor private-domain cardinality.
5. Compile/fuse the complete eager non-matrix path before comparing structural
   changes. Investigate the outstanding asynchronous captured K=2 illegal
   access separately; a blocking diagnostic pass is not a correctness fix.

## Minimal future comparison

First run baseline6×128, 3×192, and 2×256 using HU P=2, R=13, matched
batch/packs/reference-cache configuration and the same statistical schedule.
Capture forward/backward and complete optimizer timing for both actual
176-row pre and 80-row post packet shapes, and full RB training iterations.
Record presentations/s, iteration latency, allocated/reserved memory,
source identity, kernel categories/counts, SM/Tensor/DRAM GPM activity, and
parameter/FLOP estimates. Do not mix compilation cold-start with warmed timing.

After finite gradients, masks, lane independence and public output contracts
pass on CUDA, use several seeds and equal wall-time **and** equal presentation
budgets to compare learning curves. A frozen-dataset imitation/value probe
can quickly expose capacity loss but cannot prove self-play quality. Use
the repository's held-out/evaluation methodology for long-run quality rather
than interpreting short finite loss as equivalence. Only then try a 64-latent
or dense-context competitor. These ablations are future work; they were not
added to the bounded final H100 series.
