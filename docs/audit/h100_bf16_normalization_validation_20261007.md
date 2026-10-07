# BF16 normalization output bank — 2026-10-07

This experiment follows the user's preference for practical BF16 speed over
immaterial numeric differences. It changes CUDA BF16 model normalization
working outputs/affine operations to BF16 while native LayerNorm statistics
accumulate in FP32. FP32 masters/gradients/optimizer, incidence/mass/reach
reductions and loss/target arithmetic remain FP32. FP32 and FP16 execution
behavior is unchanged. Queues, all private IDs, architecture and shared
pre/post weights are unchanged; protected theory/design files were not edited.

The baseline uses torch autocast's FP32 LayerNorm output policy and separate
FP32 affine multiply/add. The candidate disables autocast locally, feeds BF16
to native CUDA LayerNorm, and applies the existing lane-aware affine operations
in BF16. Native CUDA LayerNorm's Welford statistics use float accumulation:
[PyTorch2.12.1 source](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu).
This is a precision/output-storage optimization, not fewer normalization sites
or a different norm formula. It is separate from the343→344 alignment candidate.

## Output/gradient CUDA comparison

Four full six-layer R13 model checks cover pre/post, K=1/2 and every natural
actor output field. Same initial weights/inputs; compare current production
normalization against the candidate. They passed in2.43s, with one existing
strided-gradient warning. Relative L2 output differences peak at0.71975%
(type logits/pre); master-gradient differences peak at0.45800%, with cosine
similarity at least0.99998968. Zero-copy gradient addresses remain stable,
shapes agree and all candidate values/gradients are finite. The loss differs
about0.03% for preflop and negligibly for postflop in these initial-model probes.
No tolerance in the repository suite was changed. These are numerical/storage
checks; they do not establish long-run strategic learning quality.

Raw: `runs/experiments/nlhe_hu_32bb_normal_h100_bf16_norm_oracle/`.
The oracle ran against source1b69b5b before any production normalization edit.
Reproduce the original comparison from that revision with the archived wrapper
and test script; running its baseline against newer normalization would compare
the changed implementation against itself.

## Matched full training experiment

HU/P=2/R=13/32bb, source1b69b5b, Normal K=1, roots64, batch256/steps116,
physical learner caps256/128, cache1GiB, BF16/IEEE, eager execution. One warmup
and three unprofiled repetitions per arm, exclusively owned GPU. The candidate
is enabled only during learner program invocation; rollout/reference stay on
the original normalization so both arms consume the same sealed workload.

| Arm | Individual seconds | Median | Peak allocated bytes |
| --- | --- | ---: | ---: |
| Baseline | 30.44185 /30.39571 /30.37671 | 30.39571s | 20,408,236,544 |
| BF16 learner normalization | 29.87454 /29.18032 /29.15162 | 29.18032s | 16,457,662,464 |

Measured gain **1.04165×**; peak allocated memory falls **19.3577%**.
Every repetition has29,696 learner presentations,116 updates and169 fresh
player nodes. FIFO/store structures, per-family sigma/action counts, cohort
calls, reference-cache counts and logical work agree exactly. The structure
comparison excludes its embedded measured model times, which naturally differ;
all remaining structural fields agree. Source identity and GPU isolation passed.
The wrapper records runtime override SHA and17,632 candidate normalization
calls (8,816 each with FP32/BF16 incoming activations). It is explicitly recorded
because ordinary on-disk source identity does not describe a monkeypatch.

Raw pairs: `runs/experiments/nlhe_hu_32bb_normal_h100_norm_pair_`
`{baseline,candidate}_1b69b5b/`. Exact comparison checks:
validation `reports/bf16_norm_pair_comparison.json`.
Neither this comparison nor finite short loss establishes learning quality.
Do not multiply it into earlier RB speedups across different configurations.

## Integration gate

An equivalent CUDA-BF16-only change is being verified by the full CUDA suite.
The production variant applies uniformly whenever CUDA BF16 autocast is active,
including rollout/reference; the measured pair above applies only in the
learner. Therefore the pair does not establish the global variant's exact
full-training speed or unchanged rollout draws. FP32/FP16 behavior is retained.
The precision manifest adds normalization_statistics=float32.

The user's00:24 UTC request ends experimentation. No further benchmark is
being launched. Retain this change only if the full verification passes;
otherwise revert it and keep the experiment/patch. The closing outcome below
records the decision, source identity, test result and remaining limits.


## Closing outcome at00:29 UTC

The full suite on the production-shaped BF16 normalization candidate reached
75% plus additional tests with no reported assertion failures, but hit its
390s limit(exit124). It has **no complete test summary**, so the full gate is
unproven. The uncommitted production change was reverted. Production again
matches1b69b5b/8236a61 in pbg/tools/tests, which has the earlier572-test CUDA
pass. BF16 normalization is **experimental, not adopted**, despite four full
model output/gradient oracles and the matched4.2% training gain/19.4% memory
reduction. The patch and exact validation-source hashes remain in validation
`reports/bf16_norm_production_candidate.patch` and
`reports/bf16_norm_validation_identity.json`; the partial test log is
`reports/bf16_norm_full_cuda.log`. No repository tolerance was changed.

Both exact normalization wrapper versions are saved in validation
`harness_snapshots/`, and their SHA256 hashes match the oracle/training reports.
This preserves reproducibility despite later learner-only instrumentation.
No further GPU task was launched after the user's request to finish. The
selected GPU has no compute owner. Complete the broader suite and public
workflow validation before future adoption; multi-seed quality comparisons
remain necessary for architecture changes. Capture K=2 is still unresolved.
