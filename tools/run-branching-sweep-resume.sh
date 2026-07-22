#!/usr/bin/env bash
set -euo pipefail

cd /home/pablob/ai-game-learning

log=runs/experiments/branching_hu32_ckpt814/resume_streets.log
exec > >(tee -a "$log") 2>&1

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.

checkpoint=runs/nlhe_hu_32bb_full_sequential_rb_from_ckpt_0000020_optimized/ckpt_0000814.pt
experiment=runs/experiments/branching_hu32_ckpt814
common=(
  --config configs/train/nlhe_hu_32bb_rb.yaml
  --checkpoint "$checkpoint"
  --hardware-profile configs/hardware/local_3x2080ti.yaml
  --device cuda:0
  --repeats 64
  --node-budget 70000
  --stop-cutoff 50000
  --hard-memory-bytes 9500000000
  --seed 91301
)

./tools/poker-python tools/branching-sweep.py \
  "${common[@]}" \
  --beta 2.25 \
  --start-repeat 39 \
  --output "$experiment/fixed_beta_stop50k_b225_resume39_streets.json"

./tools/poker-python tools/branching-sweep.py \
  "${common[@]}" \
  --beta 2.5 \
  --beta 3.0 \
  --beta 3.5 \
  --beta 4.0 \
  --output "$experiment/fixed_beta_stop50k_b250_400_coarse64_streets.json"
