# Queued Phase 1 inventory

Date: 2026-07-26
Phase starting revision: `80a9e23`

## Accepted production roots

| Root | Product | Phase owner |
|---|---|---|
| `pbg.cli.train` / `pbg.runtime` / `pbg.config` | ordinary and distributed training | queued composition root, Phase 10 cutover |
| `pbg.cli.train_replicas` | replica training | queued replica composition, Phase 9 |
| `pbg.cli.runpod_replica_selection` | H100 replica selection harness | experiment/tool consumer, Phase 9 |
| `pbg.cli.benchmark` / `pbg.benchmark` | matched-work mode benchmark | queued benchmark consumer, Phase 10 |
| `pbg.cli.benchmark_replica_learner` | replica learner benchmark | queued benchmark consumer, Phase 9 |
| `pbg.cli.br_eval` / `pbg.eval.br_probe` | shared-response evaluation | ordinary best-response consumer, Phase 6 |
| `pbg.cli.export_inference` / `pbg.inference_artifact` | model-only export | queued persistence owner, Phase 10 |
| `pbg.cli.benchmark_inference` / `pbg.inference` | CUDA/MPS stateful inference | queued inference composition, Phase 10 |
| public imports from `pbg.games.poker` | NLHE rules/model/hooks | poker queued hooks, Phases 5–8 |

Console scripts in `pyproject.toml`, public runtime imports, dynamic poker
backend callbacks, checkpoint resume, preemption, and tests were included in
reachability. No accepted root imports the queued package before cutover, and
the static gate rejects roots that reach both engines.

## Active configuration fields

Every current field has a queued disposition. `rewrite` preserves behavior
under a closed new owner; `delete` removes a representation or stale policy,
not an unapproved capability.

| Current fields | Disposition and new owner |
|---|---|
| `device` | rewrite as accelerator execution profile; `cuda` for training, `cuda/mps` for supported inference |
| `game.n_players`, `stack_bb`, `rank_count` | rewrite unchanged in immutable NLHE geometry |
| `training.mode`, `rollout.branching_mode` | delete aggregate selectors after offline conversion; replace with the three `ModeSchema` axes |
| `sequential_roots_per_gpu_initial/minimum` | rewrite as `root_count` and retry minimum |
| `sequential_roots_per_gpu_maximum` | delete adaptive widening; fixed manifest capacity is authoritative |
| `rollout.beta`, `adaptive_kkt.*` | rewrite as fixed/KKT controller configuration under the rollout axis |
| `branching_min_nodes`, `branching_ideal_nodes`, `safe_tree_size` | rewrite as protected-prefix/refill and safeguard controls |
| `fixed_beta_scale_min/max`, `fixed_beta_ema/step`, `branching_memory_pressure_target`, `branching_emergency_backoff`, `branching_oom_retry_limit` | rewrite into bounded whole-attempt fixed/KKT backoff and controller state |
| `safe_tree_activation_gamma_deviance_max`, `safe_tree_activation_p95_underprediction_max`, `safe_tree_activation_consecutive_iters` | rewrite unchanged under safeguard activation state |
| `family_batch_rows`, `max_active_trajectories`, `queue_capacity_rows`, `execution_memory_bytes` | rewrite as explicit manifest capacities/budgets |
| `storage_page_rows` | delete with paging |
| `memory_calibration_window`, `memory_growth_fraction`, `memory_growth_delay`, `memory_prediction_z` | delete online capacity growth; retain prior observations only as offline calibration evidence |
| `completion.n_samples` | rewrite as call-level completion configuration, never a leaf column |
| `loss.name`, `lam`, `clip_eps`, `clip_eps_v`, `c_v`, `c_h`, `c_kl`, `c_t`, `c_c`, `epsilon_t`, `value_loss`, `utility_normalization` | rewrite under the sole learner/loss operation |
| `loss.entropy_schedule`, `entropy_decay_iters` | rewrite with independent decay horizon |
| `training.iters`, `training.gd_steps_per_iter`, `training.ppo_epochs`, `training.minibatch_size` | rewrite as iteration and exact statistical-presentation schedule |
| `training.lr`, `training.adam_betas`, `training.optimizer`, `training.grad_clip`, `training.amp` | rewrite as dense optimizer/precision configuration |
| `training.seed`, `training.reproducibility` | rewrite as logical RNG identity and reproducibility profile |
| `training.checkpoint_every`, `training.out_dir` | rewrite as persistence/cadence configuration, excluded from scientific identity where already approved |
| `training.kl_ref_interval`, `training.kl_ref_effective_hands_rollout` | rewrite as the two reference-refresh laws and persisted exposure state |
| `probe.target_checkpoint`, `probe.target_sha256`, `probe.initialization` | filesystem path is execution-only; hash and initialization law remain scientific identity |
| `hardware.name`, `hardware.scheduler`, `hardware.slurm`, `hardware.max_devices`, `hardware.worker_count`, `hardware.checkpoint_every` | rewrite as launch/checkpoint profile |
| `hardware.attention_backend`, `hardware.query_chunk_size`, `hardware.training_max_tokens`, `hardware.training_max_token_pairs`, `hardware.inference_max_tokens`, `hardware.inference_max_token_pairs` | rewrite as explicit implementation/capacity values; delete `auto` before cutover |
| `hardware.reserve_vram_gib`, `hardware.max_memory_fraction`, `hardware.min_vram_gib`, `hardware.max_vram_gib`, `hardware.expected_compute_capability` | rewrite as capacity admission and hardware identity |
| `hardware.precision` | rewrite as `fp32`, `fp16`, or `bf16` precision profile |

The current `model` mapping is closed by `PokerModelConfig`:

| Model fields | Disposition |
|---|---|
| `d_model`, `n_heads`, `n_layers_pre`, `n_layers_post`, `m_raise`, `s_min`, `class_embedding_dim`, `ffn_hidden`, `folder_beta_rank`, `folder_line_dim`, `eps` | preserve under authoritative pre/post model operation |
| `k_act` | derive as `3 + m_raise`; offline converter verifies then removes duplication |
| `max_players` | convert to validated geometry/profile maximum; active run uses exact $P$ |
| `use_branching_heads` | derive from rollout axis; offline converter verifies then removes nullable selector |
| `max_history_tokens` | derive from $SP+3P-S+1$ plus public separators; profile admits the resulting attention footprint |

## Scientific and execution state

Preserved scientific state includes model/reference weights; optimizer and
scaler; iteration and logical-work counters; explicit and ambient RNG state per
rank/lane; fixed/KKT/safeguard state; KL cadence and exposure; probe target
hash, initialization, and selected seats; loss coefficients; utility units;
and the strict/seeded/throughput profile.

Execution state includes immutable capacities, tails, failures, queue metadata,
pack limits, device/rank/lane assignment, launch streams, attention
implementation, checkpoint cadence, paths, and hardware identity. Execution
changes may not alter logical work, RNG addresses, fields, losses, or sampling.

## Capacity sources

| Capacity family | Configuration/calibration source |
|---|---|
| player/learner/action rows by partition | `(P,S,R)`, mode, `root_count`, beta bound, pilot high-water plus declared margin |
| leaf/topology/edge/folded rows | exact game bound, branching bound, root/refill law, pilot high-water |
| family/chance/completion packs | hardware profile and evaluator-v2 bounded workspace |
| target arenas and packets | exact realized topology maxima admitted before seal |
| learner sigma/action packs | statistical batch request plus ragged action bound |
| model tokens/token pairs/query chunks | `MHA_model.md`, exact history bound, hardware attention profile |
| optimizer, collectives, evaluator/library workspaces | model parameter geometry, precision profile, world/replica count, versioned external workspace query |
| guard and total budget | hardware VRAM identity, reserve, maximum fraction |

No capacity is inferred by first write, raised after overflow, or selected by a
runtime engine switch.

## Persistence and artifact inventory

### Schemas

| Surface | Current identity | Disposition |
|---|---|---|
| training checkpoint | `pbg-checkpoint-v8` with exact required field set | one-time converter, extended by Phases 5–9; queued runtime accepts only queued schema |
| rollout/training storage | `pbg-rollout-storage-v2`, `pbg-training-storage-v2` | delete after representation rewrite; never persist into queued runtime |
| controller state | sequential-memory-v2, fixed-beta-v1, adaptive-kkt-v2, cost-safeguard-v1 | semantic state converted into new sole controller owner |
| inference artifact | `pbg-inference-v1` | one-time model/config converter in Phase 10 |
| evaluator | contract v2 and its versioned RNG/deck/folder identities | exact consumer check; no compatibility version |
| metrics/run JSON | gap-free metrics, run/config/hardware identity | offline conversion retaining exact completed prefix and source hashes |

### Selected artifacts

- Every current-schema checkpoint in:
  `runs/nlhe_6max_10bb_full_branched/`,
  `runs/nlhe_hu_32bb_full_sequential_rb/`,
  `runs/nlhe_hu_32bb_24h_normal_20260722/`, and
  `runs/nlhe_hu_32bb_24h_rb_20260722/`.
- Their `run.json`, `hardware*.json`, `status.json`, and gap-free metrics.
- The tracked range database pairs
  `nlhe_6max_10bb_full_branched.{json,pt}` and
  `nlhe_hu_32bb_full_sequential_rb.{json,pt}`.

Converters record source hash and schema, reject unknown fields, write
atomically, and are never imported by runtime loading. Checked-in experiment
configs and benchmark JSON remain evidence only. Unselected/unknown legacy
artifacts have delete disposition unless separately approved.

## Preservation baseline mapping

| Preservation family | Planned owner | Regression gate |
|---|---|---|
| update schedule, exact tails, shuffled sequential reuse | learner sampler and iteration composer, Phase 5 | exact presentation count and tail fixtures |
| PPO/value/entropy/KL/T/C loss surface | sole learner operation, Phases 3/5 | FP64 reference and CUDA gradient agreement |
| KL refresh/caching and cost-aware allocation | learner/reference owner, Phases 5/9 | resume, cached/uncached identity, distributed normalization |
| safeguard statistics/state | branching controller, Phase 7 | deviance/p95/streak/reset/resume fixtures |
| NLHE geometry, actions, history | poker hooks, Phase 5 | boundary matrix for all legal `(P,S,R)` |
| pre/post model, symmetry, ragged live seats, folded inputs | poker model operation, Phases 5/6 | model reference, equivariance, masking, capacity |
| FP32/FP16/BF16 and attention implementations | learner/model plus hardware profile, Phases 5/9 | precision and subdivision tolerance |
| utility conversion | target/learner boundary and converter, Phase 5 | exact affine conversion fixture |
| probe identity, initialization, seat laws, evaluation | ordinary best-response schema/consumer, Phase 6 | hash, balanced pairing, cutoff, paired statistics |
| reproducibility profiles | logical RNG and run identity, Phase 4 | subdivision/schedule/lane invariance |
| strict configuration | queued public config, Phase 5 | dotted-path unknown/inapplicable failures |
| atomic checkpoints/resume/preemption | persistence/iteration owner, Phases 5/9 | exact state, drift, metrics trim, signal boundary |
| inference artifact/session | queued inference composition, Phase 10 | strict identity, CUDA/MPS legal action/range behavior |
| performance gates and diagnostics | benchmark/observability consumers, Phase 10 | matched identity, declared metrics, no warmup inclusion |
| distributed common initialization and diagnostics | distributed/replica composition, Phase 9 | rank/lane parity and global/per-rank accounting |

Every row in `features_to_preserve.md` is covered by one of these families.
Removal or semantic changes beyond the explicitly approved queued contracts
remain prohibited.
