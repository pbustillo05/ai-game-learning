#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash tools/launch-nlhe-hu32-24h-ab.sh --normal-gpu GPU --rb-gpu GPU" >&2
}

normal_gpu=""
rb_gpu=""
while (($#)); do
  case "$1" in
    --normal-gpu) normal_gpu=$2; shift 2 ;;
    --rb-gpu) rb_gpu=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "$normal_gpu" =~ ^[0-9]+$ || ! "$rb_gpu" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi
if [[ "$normal_gpu" == "$rb_gpu" ]]; then
  echo "normal and RB require distinct GPUs" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

check_idle() {
  local gpu=$1 used utilization
  IFS=, read -r used utilization < <(
    nvidia-smi --id="$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used=${used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if ((used > 100 || utilization > 5)); then
    echo "GPU $gpu is not idle: ${used} MiB used, ${utilization}% utilization" >&2
    exit 1
  fi
}

check_clean_output() {
  local run_dir=$1
  if [[ -e "$run_dir/run.json" || -e "$run_dir/metrics.jsonl" ]]; then
    echo "output directory already contains a run: $run_dir" >&2
    exit 1
  fi
  if compgen -G "$run_dir/ckpt_*.pt" >/dev/null; then
    echo "output directory already contains checkpoints: $run_dir" >&2
    exit 1
  fi
}

normal_dir="runs/nlhe_hu_32bb_24h_normal_20260722"
rb_dir="runs/nlhe_hu_32bb_24h_rb_20260722"
normal_session="pbg-hu32-normal-24h-20260722"
rb_session="pbg-hu32-rb-24h-20260722"

check_idle "$normal_gpu"
check_idle "$rb_gpu"
check_clean_output "$normal_dir"
check_clean_output "$rb_dir"
if tmux has-session -t "$normal_session" 2>/dev/null; then
  echo "tmux session already exists: $normal_session" >&2
  exit 1
fi
if tmux has-session -t "$rb_session" 2>/dev/null; then
  echo "tmux session already exists: $rb_session" >&2
  exit 1
fi

bash tools/run-training.sh \
  --config configs/experiments/nlhe_hu_32bb_normal_24h_20260722.yaml \
  --hardware-profile configs/hardware/local_2080ti_10gb_hard.yaml \
  --run-dir "$normal_dir" \
  --gpus "$normal_gpu" \
  --nproc-per-node 1 \
  --session "$normal_session" \
  --tmux

bash tools/run-training.sh \
  --config configs/experiments/nlhe_hu_32bb_rb_24h_20260722.yaml \
  --hardware-profile configs/hardware/local_2080ti_10gb_hard.yaml \
  --run-dir "$rb_dir" \
  --gpus "$rb_gpu" \
  --nproc-per-node 1 \
  --session "$rb_session" \
  --tmux

echo "Both runs launched."
echo "Normal: tmux attach -t $normal_session"
echo "RB:     tmux attach -t $rb_session"
