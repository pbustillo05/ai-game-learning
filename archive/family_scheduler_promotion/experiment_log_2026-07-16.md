# FIFO experiment log — 2026-07-16

This is the supporting evidence for
[`fifo_training_readiness_2026-07-16.md`](fifo_training_readiness_2026-07-16.md).
Keep the readiness file short; put commands, raw artifacts, and detailed
interpretation here.

## Artifact conventions

- Benchmark JSON under `benchmarks/results/` is the raw source of timing,
  topology, allocation, synchronization, checkpoint, and environment data.
- Comparisons require the same checkpoint identity, logical seed grid, mode,
  completion configuration, and hardware profile.
- RTX 2080 Ti measurements are the acceptance evidence. H100 is a later
  calibration/deployment activity, not a release gate.
- Record a failed or interrupted CUDA attempt when it teaches a safety limit;
  do not treat it as a performance sample.

## Completed evidence

| Date / scope | Result | Evidence |
|---|---|---|
| Shallow HU, P=2/S=8/R=13 | FIFO normal and sequential-RB required calibration; fixed q=64 did not pass every paired 1.20 upper bound. | Readiness audit history and `benchmarks/results/queued_phase2_*`. |
| Deep HU, P=2/S=32/R=13 | q=128 improved sequential-RB at 256 starts; q=64 remained the safe 1,024-start choice under the local reserve. | `configs/benchmarks/nlhe_hu_deep_fifo.yaml` and benchmark artifacts. |
| Trained 6-max normal, checkpoint 5,750 | Actual-cost greedy packing changed the trained 1,536-start workload from 2.40x slower to a 3-seed FIFO/wavefront rollout ratio of 1.059, 95% paired interval [1.042, 1.112]. | `nlhe6_10bb_checkpoint_actual_cost_q128_vs_wavefront_1536_3seed_gpu1_2026-07-16.json`. |
| Actual-cost device-index follow-up | Device stable permutation plus span views removed physical-pack index uploads without changing output order, logical work, model calls, scheduler reads, or rollout reservation. | `tests/test_p13a_gpu_rollout.py::test_poker_actual_cost_keeps_physical_pack_indices_on_device`. |
| Trained 6-max sequential-RB, profiled, 512 starts | 3,582 decisions; 287 model calls; 78 declared control reads; finite reduced gradients; zero undeclared syncs; 849,346,560 B peak rollout reservation. | `nlhe6_10bb_checkpoint_fifo_device_indices_profiled_q128_512_1seed_gpu1_2026-07-16.json`. |
| Trained 6-max normal, profiled, 1,536 starts | Same 10,783 decisions / 12,388 nodes as the matched predecessor. H2D transfers fell 858→2; rollout time/decision 0.001957→0.001927 s (one matching cold seed). | `nlhe6_10bb_checkpoint_fifo_device_indices_normal_q128_1536_1seed_gpu1_2026-07-16.json`. |
| Trained 6-max sequential-RB, 1,536 starts | q=128/admission=1,536 passed three paired seeds: rollout ratio 1.016, interval [0.983, 1.098]; iteration ratio 1.009, [0.988, 1.078]. Peak FIFO reservation was 6,922,698,752 B; gradients/completion were valid and FIFO had zero undeclared syncs. | [Comparison JSON](../../benchmarks/results/nlhe6_10bb_checkpoint_sequential_rb_q128_vs_wavefront_1536_3seed_gpu1_2026-07-16.json) and its two source artifacts. |
| Trained 6-max fixed-beta branched RB | Branched FIFO now dispatches logical family blocks through the common actual-cost backend splitter; singleton blocks skip impossible planning work. Five fixed-root samples had aggregate rollout ratio 1.077, bootstrap [0.972, 1.198], and iteration ratio 0.994, [0.920, 1.080]. | [Topology-aware comparison JSON](../../benchmarks/results/nlhe6_10bb_checkpoint_branched_rb_fixed_root_wall_time_5seed_gpu1_2026-07-16.json), FIFO/wavefront sources, and the retained standard failed report. |
| Two-rank all-mode soak/resume | Normal, sequential RB, and branched RB each ran iterations 0--19, resumed checkpoint 20, and completed 20--24. Peaks were 226,492,416 B, 452,984,832 B, and 92,274,688 B; execution reservation series were flat (normal varied only 10,680 B), all gradients were finite, and invalid completion mass/undeclared FIFO syncs were zero. | `runs/acceptance/nlhe_hu_fifo_*_two_rank_soak/{metrics.jsonl,ckpt_0000020.pt,ckpt_0000025.pt,hardware_resume_0000020.json}`. |

Single-device CUDA runs used idle GPU 1; the soaks used idle GPUs 1 and 2. They
returned to 9 MiB and 19 MiB idle memory respectively. Focused CPU
actual-cost/config regressions and two CUDA packing tests passed.

## Safety finding

An unprofiled 1,536-start attempt was stopped cleanly at 9.7 GiB used on an
11 GiB RTX 2080 Ti, below the required 2 GiB reserve. The cause was an
unbounded dynamic `torch.compile` attention shape after omitting the local
hardware profile. Every benchmark or training launch must therefore select a
calibrated local hardware profile; unprofiled capacity results are inadmissible.

## Branched comparison judgment

The standard comparison artifact remains failed because it normalizes each
executor by its own realized decisions despite `pathwise_identical=false` and
exactly one root/tree. The five FIFO trees realized 6/7/7/7/9 decisions, while
wavefront realized 7/7/14/10/7. The apparent worst 2.04x normalized sample was
0.276 s FIFO versus 0.271 s wavefront in absolute rollout time. A paired
bootstrap of aggregate fixed-root wall time, the actual scientific work unit,
put both rollout and complete iteration below the 1.20 upper limit. This is the
dashboard's narrow written exception; it does not waive correctness, memory,
completion, or synchronization gates.

## Still missing

- Only the post-promotion first larger production run and its resume validation;
  this is not a local promotion gate.

## Raw-artifact shortlist

- `nlhe6_10bb_checkpoint_fifo_actual_cost_q128_1536_3seed_gpu1_2026-07-16.json`;
- `nlhe6_10bb_checkpoint_wavefront_1536_3seed_gpu1_2026-07-16.json`;
- `nlhe6_10bb_checkpoint_actual_cost_q128_vs_wavefront_1536_3seed_gpu1_2026-07-16.json`;
- `nlhe6_10bb_checkpoint_fifo_device_indices_profiled_q128_1536_1seed_gpu1_2026-07-16.json`; and
- `nlhe6_10bb_checkpoint_fifo_device_indices_normal_q128_1536_1seed_gpu1_2026-07-16.json`.

When adding an experiment, give its fixed scientific workload, hardware profile,
warmup/repeat protocol, decision/node counts, peak allocated/reserved bytes,
and a link to the JSON. Put only the conclusion and any changed release status
in the readiness file.
