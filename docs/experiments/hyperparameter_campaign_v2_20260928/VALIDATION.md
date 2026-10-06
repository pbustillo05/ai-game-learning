# V2 execution and validation record

The capacity method follows
[the July execution-capacity audit](../../audit/nlhe_hu_32bb_beta24_execution_capacity_audit_20260728.md):
separate retained storage, phase-specific workspace, PyTorch allocated versus
reserved peaks, and physical packs from logical learning batches. Its old
static `LayerWork` defect is already addressed by current
`GeneratedOwners.bind_layer_work`: each output uses the realized layer-plan
extent. The July branching report did not execute learner forward/backward;
its learner pack recommendation is not evidence of training safety here.

The retained initial probes are under
`runs/experiments/nlhe_hu_32bb_rb_league_v2_validation/`.
`capacity_results.json` aggregates actual per-rank metrics. Times include the
worker's charged iteration/checkpoint overhead and exclude initialization.
These are one-iteration capacity/throughput measurements from the fixed
initializer, not policy-strength measurements or a long-run tail guarantee.

| Probe | GPUs | N/rank | B/rank | Steps | Learner pack | Cache budget | Seconds | Max allocated GiB | Max reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `largest_rb` | 3 | 1024 | 256 | 48 | 12 | 256 MiB | 196.14 | 5.559 | 6.928 |
| `bridge_normal` | 3 | 2048 | 256 | 48 | 12 | 256 MiB | 247.47 | 4.133 | 6.639 |
| `bridge_rb_pack6` | 3 | 2048 | 128 | 96 | 6 | 256 MiB | 413.63 | 7.895 | 8.307 |
| `bridge_rb_pack6_cache1g` | 3 | 2048 | 128 | 96 | 6 | 1 GiB | 285.17 | 6.988 | 8.283 |
| `bridge_normal_pack6_cache1g` | 1 | 2048 | 256 | 48 | 6 | 1 GiB | 234.20 | 3.893 | 6.133 |
| `bridge_cold_rb` | 1 | 2048 | 256 | 48 | 6 | 1 GiB | 204.15 | 6.914 | 7.777 |
| `bridge_cold_normal` | 1 | 2048 | 256 | 48 | 6 | 1 GiB | 206.68 | 3.893 | 6.133 |

All successful rows applied every requested optimizer step with finite
gradients; distributed rows finished with identical hashes on all three ranks.
The last three probes ran concurrently on separate GPUs to test each local
worker's capacity. They do not measure three-GPU throughput. The cold probes
load the recovered historical reference initializer used by the final stage.

The original RB bridge at learner pack 12 and cache budget 256 MiB failed
inside the uncached reference forward (`learner.py:1902`). Its diagnostic log
is `bridge_rb.log`. It is not an admitted execution setting.

Current reference admission is all-or-none across input partitions. A next
partition requires `resident_bytes + build_bytes + payload_bytes <= budget`,
where the caller charges `build_bytes = payload_bytes`. A nominal cache budget
of 256 MiB can therefore reject a payload much smaller than 256 MiB. Rejection
causes a live reference forward alongside the learner's autograd forward on
every physical batch. Learner pack 6 bounds both forwards; logical N/B/steps,
accumulated loss weights and optimizer transitions remain unchanged.

A 1 GiB cache budget plus pack 6 reduced the matched large-RB iteration time
by 31.1% relative to pack 6 with 256 MiB, while lowering peak allocation.
This is one matched iteration, not a universal throughput factor. The profile
uses the larger cache in all arms and the six-row pack in both large-root arms,
identically for Normal and RB. The smaller factorial cells retain pack 12.

Rollout pack 48 retains the July audit's practical memory/throughput compromise.
Completion pack 512 is retained here: a sequential rank has only 512–2048 leaves,
and the July 8192-row result came from a much larger branching workload. No
claim is made that its branching optimum transfers to sequential training.
Thermal flags, power and utilization are recorded; there are no cooling pauses
or fan/power-limit changes.

## Pipeline checks

The final suite passed **91/91 tests with CUDA enabled**:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/test_hyperparameter_*.py
```

Ruff passed on the v2 launcher, league, raw evaluator, gradient diagnostic and
corresponding tests. The launch shell script passed `bash -n`.

The complete reduced campaign at
`runs/experiments/nlhe_hu_32bb_rb_league_v2_smoke` passed all 114 job statuses:
98 main commands plus 16 common-response-panel commands. It exercised:

- four actual CUDA gradient probes and CUDA variance/mean/bootstrap analysis;
- three-GPU Normal/RB workload and loss screens;
- balanced raw leagues, zero-sum matrix checks, shared-score and mode-gap
  confidence reporting, and the fixed checkpoint-1110 anchor panel;
- all twelve selective BR restarts and the sixteen-cell held-out common
  response panel;
- candidate deduplication, confirmation, and early-reference-policy transfer;
- entropy reaching exactly zero and raw-posterior reference refresh;
- completed resume with the command counts unchanged: no duplicate launches.

Smoke training uses reduced workloads and each match uses four hands. It
validates execution and artifact semantics, not strength, variance calibration,
or statistical power. It exports no recommended training profile.

The full campaign is prepared at
`runs/experiments/nlhe_hu_32bb_rb_shared_league_v2_20260928`, with all twenty
initial workload jobs, 192 planned gradient probes, and 136 nominal
training-plus-BR hours. It has **not been launched**. Full results and a quality
winner remain pending.

Frozen source SHA-256:
`34ab8614f148cf5e636bea2902df13cd717e0bc694daf16e292c7ce2bc61b662`.

The machine-readable record is
`runs/experiments/nlhe_hu_32bb_rb_league_v2_validation/validation_summary.json`.
The failed pack-12 bridge's generated checkpoint directory was removed; its
exact job and error log remain. Successful preflights, smoke results and their
checkpoints are retained. All three GPUs returned to their idle baseline after
validation.
