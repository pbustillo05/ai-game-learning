# Immediate Quest H100 Implementation Plan

## Goal

Run the existing heads-up 32 BB Normal and RB experiments efficiently on H100s
without changing their model or statistical hyperparameters.

The main optimization is replica-batched training: one CUDA process executes
several independent models together using larger batched kernels. This reduces
kernel launches while preserving every run's statistical experiment.

## Changes to implement now

1. **H100 hardware profile**
   - Add a BF16, SM90, 80 GB Quest profile.
   - Keep the selected 2080 Ti logical training parameters unchanged.

2. **Slurm-safe launcher**
   - Remove hard-coded local venv, CUDA, evaluator, and `tmux` assumptions.
   - Respect Slurm's `CUDA_VISIBLE_DEVICES`.
   - Accept the environment, evaluator path, run config, and replica count `K`.
   - Launch one replica-batched process per assigned GPU.

3. **Replica-batched trainer**
   - Stack model/reference parameters and optimizer state with leading dimension
     `K` as a physical layout only; the `K` parameter sets remain disjoint.
   - Retain independent RNG, rollout, controller, metrics, and checkpoints.
   - Run only one mode per GPU: all replicas are Normal or all are RB.
   - Batch learner work first, then rollout inference, then RB completion.
   - Compute `g[k] = grad(loss[k], params[k])` with no loss or gradient
     reduction across replicas.
   - Apply gradient norms, clipping, AdamW state, and updates per replica.
   - Preserve per-replica sample weights, normalization, optimizer boundaries,
     logical counters, and objective mass.

4. **Replica-batched rollout**
   - Take stable FIFO prefixes independently from each replica.
   - Batch only matching `pre` or `post_A` families and retain replica IDs.
   - Padding must not create samples, weights, or objective mass.
   - Prototype with `torch.func.vmap`; keep it only if profiling confirms fewer
     kernels, otherwise implement explicit batched/grouped operations.

5. **Evaluator and environment**
   - Compile the poker evaluator with `TORCH_CUDA_ARCH_LIST=9.0`.
   - Verify the allocated node's driver before selecting CUDA 12.8 or CUDA 13.
   - Record source revisions, resolved configs, driver, CUDA/PyTorch versions,
     GPU UUID, and Slurm allocation.

6. **Job-boundary safety**
   - Handle an advance Slurm signal.
   - Checkpoint every replica independently before Quest's 48-hour limit.

## RunPod selection test

Use one H100 SXM for at most 90 minutes with the frozen 2080 Ti configuration:

1. Verify `K=1` against the existing trainer using identical inputs and seeds.
2. Learner microbenchmark `K = 1, 2, 4, 8`: 10 warm-up and 50 measured steps.
3. End-to-end test `K = 1, 2, 4, 8` separately for Normal and RB: one warm-up
   and three measured iterations per candidate.
4. Record aggregate decisions/second, kernels/decision, phase times, peak
   memory, and per-replica counters/objective mass.
5. Select the largest `K` that:
   - improves aggregate throughput by at least 10% over the previous candidate;
   - stays below 70 GiB peak memory;
   - measurably reduces kernel launches per unit of work;
   - preserves counters, objective mass, independent clipping, and finite values.

The optimization metric is total fresh decisions per physical H100-hour.
Select `K` independently for Normal and RB, then give both modes equal physical
GPU-hour budgets and matched training seeds on Quest.

## Not changing now

- Individual model size or architecture.
- Logical rollout width or statistical minibatch.
- Learning rate, optimizer schedule, or sample reuse.
- RB completion count.
- CUDA MPS, multiple training processes per GPU, FP8, or Transformer Engine.

Attention-backend and `torch.compile` tuning come only after replica batching.
