#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 PHYSICAL_GPU {screening|validation|test} TARGET_ID ITERATION..." >&2
  exit 2
fi

physical_gpu="$1"
stage="$2"
target_id="$3"
shift 3

case "$physical_gpu" in
  0|1|2) ;;
  *) echo "physical GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac

case "$target_id" in
  normal_i10000)
    response_run="runs/experiments/nlhe_hu_32bb_rb_brprobe_normal_i10000_300iter_20260913"
    ;;
  rb_i10000)
    response_run="runs/experiments/nlhe_hu_32bb_rb_brprobe_rb_i10000_300iter_20260831"
    ;;
  rb_i7700)
    response_run="runs/experiments/nlhe_hu_32bb_rb_brprobe_rb_i7700_300iter_20260913"
    ;;
  rb_i6600)
    response_run="runs/experiments/nlhe_hu_32bb_rb_brprobe_rb_i6600_300iter_20260913"
    ;;
  rb_i4400)
    response_run="runs/experiments/nlhe_hu_32bb_rb_brprobe_rb_i4400_300iter_20260820"
    ;;
  *) echo "unknown target id: $target_id" >&2; exit 2 ;;
esac

case "$stage" in
  screening)
    hands=1000
    namespace="main-screening-v2"
    seed=2718
    ;;
  validation)
    hands=5000
    namespace="main-validation-v2"
    seed=314159
    ;;
  test)
    hands=20000
    namespace="main-test-v2"
    seed=1729
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

if [[ "$stage" == "screening" && "$*" != "50 100 150 200 250 300" ]]; then
  echo "screening requires the precommitted common six-checkpoint grid" >&2
  exit 2
fi
if [[ "$stage" == "validation" && "$#" -ne 3 ]]; then
  echo "validation requires exactly three promoted checkpoints" >&2
  exit 2
fi
if [[ "$stage" == "test" && "$#" -ne 1 ]]; then
  echo "test requires exactly one frozen selected checkpoint" >&2
  exit 2
fi

source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$physical_gpu"

output_root="runs/experiments/nlhe_hu_32bb_rb_br_comparison_protocol_v2_20260916"
for iteration in "$@"; do
  printf -v checkpoint "%04d" "$iteration"
  output="$output_root/${stage}_${target_id}_ckpt${checkpoint}_${hands}traj_v2.json"
  python -m pbg.cli.br_eval \
    --probe "$response_run/probe_inference_ckpt${checkpoint}.pt" \
    --target "$response_run/target_inference.pt" \
    --hands "$hands" \
    --split-namespace "$namespace" \
    --seed "$seed" \
    --device cuda:0 \
    --output "$output"
done
