# Execution prompt

Implement and run the attached `variance_iteration_experiment_spec.md` in `/home/pablob/ai-game-learning` on `nova.iems.northwestern.edu`. **Deliver analyzed results and figures within 48 hours**, including implementation, validation, execution, and reporting.

Run these two experiments:

1. **Actor-gradient variance:** use Normal 10k and RB 10k checkpoints. Apply Sampled and RB-1024 to each identical checkpoint, with 384 public rollouts, 256 decision records, and 32 independent paired repetitions. Measure full actor gradients before clipping and optimizer updates.
2. **One-iteration optimization:** use the same checkpoints and estimators, with six independent paired rollout batches per checkpoint. Match starting weights, Adam state, data, minibatch schedules, and hyperparameters. Run up to 100 steps and evaluate at 0/10/20/38/100 against a common independent held-out objective. Step 38 is the primary endpoint. Preserve production target caching.

Spend the first three hours on focused correctness checks and a representative paired timing pilot. Lock the queue by hour 6 using measured runtime. If completion would exceed hour 40, cap optimization at 38 steps, then reduce gradient repetitions to 16 if necessary. Retain both checkpoints and six paired iteration repetitions. Choose the scope before inspecting method effects.

Follow the specification's common PPO reference objective and uncertainty procedure. Reuse verified code and compatible completed measurements. Follow repository instructions, use available GPUs and the supported CUDA environment, and preserve unrelated work. Treat `docs/theory/` and historical artifacts as read-only. Handle credentials through the existing secure connection.

Save initial verified measurements by hour 12, interim results and figures by hour 24, and finish measurements by hour 40. Reserve the final eight hours for analysis and delivery. Use durable, resumable jobs and enforce the 48-hour deadline. Report blockers promptly and deliver verified partial results within that deadline if resources limit completion.

Deliver `RESULTS.md`, gradient-variance ratios, held-out improvement curves versus steps and compute time, paired confidence intervals, raw data, configs/seeds/hashes, focused test outputs, reference-precision checks, and reproduction commands. Include actual sample sizes, compute use, failures, null results, and regressions.

Proceed autonomously through implementation, verification, execution, and analysis. Begin now.
