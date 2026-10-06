# Revised shared poker parameter experiment

Revision 4 corrects the posterior scheduling quantity and the analysis memory failure that stopped revision 3. It retains the loss and schedule search, the four workloads at approximately fixed reuse, common scientific parameters for Normal and RB, normalized rewards, and league-based selection. No winning hyperparameters have been established.

Profile: `configs/profiles/nlhe_hu_32bb_shared_loss_search_v4.yaml`.
Output: `runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v4_20261006`.
The failed revision 3 directory and its original results are preserved. Its generating source was archived as `source_before_v4.tar.gz` before code edits.

**The scheduling quantity follows the pre-terminal decision law.** Let $p_{i,p}(h)$ be player $p$'s combo posterior conditional on trajectory $i$'s pre-terminal history, after its last player action and before final chance completion. Let $d_{i,p}$ count that player's fresh trained decisions on the trajectory. The RB exposure increment is

$$\Delta C_{RB}=\sum_i\sum_p d_{i,p}\exp(H(p_{i,p})).$$

Normal increments by $\Delta C_N=\sum_i\sum_p d_{i,p}$. Thus the average breadth is weighted by the actual player decisions, matching the pooled decision sampling law. Units are effective player decisions; schedule thresholds use these units consistently in both modes. No learner reuse, completion sample count, sample ESS, arbitrary exponent, or unmeasured correlation discount enters this clock.

For heads-up play the observer derives exact marginals from both pre-terminal leaf ranges, including private-card collision conditioning. It does not use intermediate decision ranges or a finite Monte Carlo histogram. It retains combo resolution for preflop ancestors, even when the learner later reduces equivalent combos into hand classes. CUDA pointer doubling maps every trained decision to its unique descendant leaf; each decision is counted once. The observer reads leaves before the completion sampler executes. Production theory files are unchanged.

The previously reported breadth of approximately 1093 described intermediate decision posteriors. It does not measure this requested clock, and the associated schedule extrapolation is withdrawn for revision 4. Corrected posterior observations will be saved by every new probe and training iteration.

**The registered trials retain the central loss search.** All values are explicit in the YAML profile. Normal and RB share the same parameters for every setting and have equal weight in selecting a shared configuration. Estimator mode is never selected as a hyperparameter.

| Workload | Roots per GPU | Statistical batch per GPU | Learning steps | Global roots on three GPUs |
|---|---:|---:|---:|---:|
| Control | 512 | 128 | 24 | 1536 |
| More roots and updates | 1024 | 128 | 48 | 3072 |
| More roots and larger batches | 1024 | 256 | 24 | 3072 |
| Large rollout | 2048 | 256 | 48 | 6144 |

Each has $BS/N=6$. Realized reuse is global presentations divided by actual fresh player decisions, measured every iteration. Workload trials receive two hours per mode.

The control starts at entropy 0.0015625, reference KL 0.00625, value weight 0.016, value clip 0.00625, learning rate 0.00005, policy clip 0.2 and lambda 0.95. Entropy decays linearly to exactly zero at 250000 fresh global decisions; the control reference interval is 400000. These are registered baselines, not measured optima.

There are **21 loss/schedule settings**, each receiving two hours per mode:

| Axis | Alternatives to control |
|---|---|
| Initial entropy | 0; 0.00078125; 0.003125 |
| Entropy zero endpoint | 125000; 1000000 |
| Reference KL weight | 0.003125; 0.0125 |
| Reference refresh interval | 100000; fixed reference |
| Learning rate | 0.000025; 0.0001 |
| Critic weight | 0.004; 0.064 |
| Critic loss | Unclipped MSE |
| Critic clipping | 0.025 |
| Policy clipping | 0.1 |
| Return lambda | 0.9 |
| Pre-terminal posterior clock | Entropy only; reference only; both |

The two additional clock arms separate entropy scheduling from reference scheduling. All three use the corrected pre-terminal quantity. Schedules update at iteration boundaries; threshold overshoot and actual reference copies are logged explicitly. The selective response audit remains restricted to control versus both schedules using raw breadth, in both modes, with three response restarts and a common held-out attack panel. It is an achieved-response comparison, not a Nash exploitability certificate.

Reference caching remains 1 GiB. Physical learner packs are six for the large rollout and twelve otherwise, with rollout rows 48 and completion rows 512. Both modes share physical settings for each workload. RB uses 1024 completion samples. The profile records AdamW betas, epsilon, weight decay, gradient clip, precision, loss scale and global-mean distributed gradient reduction. No cooling pauses are inserted. Reward units remain one stack (32 bb). Initialization remains checkpoint 1110, since checkpoint 1100 was absent; the historical input adapter and value-head conversion remain active.

**Analysis memory is bounded, and full-size validation is mandatory.** The repaired analyzer retains only one Normal/RB group pair on CUDA. Paired covariance and bootstrap calculations accumulate small Gram matrices across chunks of 65536 parameters, without full float64 replicate matrices. Completed mean vectors move to host storage only for checkpoint serialization; statistical computation remains on CUDA.

Before any long experiment, the launcher requires:

1. All requested GPUs to have no compute jobs. It waits and records occupancy without stopping any process.
2. The campaign regression suite to pass with CUDA available.
3. Analysis of the actual 192 saved gradients from revision 3: 12 groups, 16 replicates each, six paired comparisons, and at most 5 GiB peak CUDA allocated memory. Original artifact hashes and generating source identity are validated separately from the repaired analyzer's identity.
4. A bounded integration run covering new rollouts, posterior observation, both modes, clocks, training, league/anchor evaluation, selective response probes, confirmation, transfer and resume.

A validation pass is bound to the source hash, profile hash, test hashes and GPU UUIDs. A one-GPU developer validation does not replace the launcher's three-GPU integration gate. Revision 4 regenerates its short 192-probe diagnostic so new telemetry uses the correct posterior definition; old probe outputs are never silently relabeled.

**Launch inside tmux.** Run from the server:

```bash
tmux new-session -d -s poker-v4 'bash /home/pablob/ai-game-learning/experiments/hyperparameter_search/run_shared_loss_search_v4.sh'
tmux set-option -t poker-v4 remain-on-exit on
tmux attach-session -t poker-v4
```

Detach with `Ctrl-b`, then `d`. SSH disconnection does not stop tmux. The command waits for all three GPUs to be idle before validation. It starts long training only if all gates succeed. There is no blind automatic retry of a deterministic failure. Rerun the same script after a resolved failure to resume completed validation/training/evaluation work. Do not run two launchers for the same output; a file lock prevents that.

All logs are files. `launcher.log` records stage starts and exits; `launcher_status.json` contains the process IDs, current stage, heartbeat, waiting GPUs, or explicit failure. Validation logs and its pass marker are under `validation/<identity>/`. `campaign.log` records the campaign supervisor output. Per-job process logs, metrics, checkpoints, snapshots and evaluation games remain under their stage directories. A `running` status with an old heartbeat requires checking its recorded PID; sudden server power loss cannot write a final failure record. Tmux does not survive a server reboot, but the run's saved state can be resumed.

```bash
cat /home/pablob/ai-game-learning/runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v4_20261006/launcher_status.json
tail -f /home/pablob/ai-game-learning/runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v4_20261006/launcher.log
```

**Budget and selection.** The registered maximum training allocation is 16 workload hours, 84 loss/schedule hours, 36 confirmation hours, 16 early-policy transfer hours and four wall hours of selective response training: **156 hours on three GPUs**, plus validation, diagnostic and evaluation time. Registered league and anchor evaluations account for 722432 games; paired response evaluations add 28672 simulated games. These are work budgets, not a promised finish date. Actual completion depends on measured evaluation throughput and GPU availability; the budget is roughly a week of training plus evaluation.

Screening promotes shared configurations using balanced randomized cross-play. Confirmation retains the original control, a workload-only change and a compatible workload-plus-loss combination, using two fresh training seeds. Two finalists transfer from the recovered early historical reference policy. Inspect common-anchor gains, both modes separately, confidence intervals and strategic cycles before interpreting a league ranking. The selected profile is provisional and saved to `configs/profiles/nlhe_hu_32bb_shared_loss_selected_v4.json` and the campaign's `recommended_profile.json`.

For a new conversation, read this document, the profile, `launcher_status.json`, validation pass records, `report.json` if present, stage league/anchor reports, `noise/analysis/result.json`, and `clock_audit/report.json`. A missing final report means the study has not finished. No claim of ideal parameters should be made from the engineering smoke or scalar gradient norms.

**Verification at handoff:** 78 regression tests passed on CUDA GPU 2, including the new posterior and chunked-statistics tests. One additional orchestration test passed for all 168 workload/loss/mode resolutions. Changed Python files pass Ruff and the shell launcher passes `bash -n`. A new external job occupied GPU 2 before full-size analysis could start. Full-size analysis and integration therefore remain pending; the supplied launcher runs both as mandatory gates. The developer validation-only waiter was stopped, its lock is free, and no long campaign has been launched. Evidence is saved in `developer_validation_20261006.json` in the v4 output directory.
