#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash tools/launch-nlhe-hu32-brprobes-sequential.sh --gpu GPU [--session NAME]" >&2
}

gpu=""
session="pbg-hu32-brprobes-700-20260723"
tmux_child=0
while (($#)); do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --session) session=$2; shift 2 ;;
    --_tmux-child) tmux_child=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

normal_config="configs/experiments/nlhe_hu_32bb_rb_brprobe_normal_700iter_20260723.yaml"
rb_config="configs/experiments/nlhe_hu_32bb_rb_brprobe_target_rb_700iter_20260723.yaml"
profile="configs/hardware/local_2080ti_10gb_hard.yaml"
normal_dir="runs/experiments/nlhe_hu_32bb_rb_brprobe_normal_700iter_20260723"
rb_dir="runs/experiments/nlhe_hu_32bb_rb_brprobe_target_rb_700iter_20260723"

check_idle() {
  local selected_gpu=$1 used utilization
  IFS=, read -r used utilization < <(
    nvidia-smi --id="$selected_gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used=${used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if ((used > 100 || utilization > 5)); then
    echo "GPU $selected_gpu is not idle: ${used} MiB used, ${utilization}% utilization" >&2
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

if ((!tmux_child)); then
  check_idle "$gpu"
  check_clean_output "$normal_dir"
  check_clean_output "$rb_dir"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" \
    bash "$repo_root/tools/launch-nlhe-hu32-brprobes-sequential.sh" \
      --gpu "$gpu" --session "$session" --_tmux-child
  echo "started sequential BR-probe session: $session"
  echo "normal status: tools/run-status.sh $normal_dir"
  echo "RB status:     tools/run-status.sh $rb_dir"
  exit 0
fi

export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="/home/pablob/.venvs/poker/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

check_idle "$gpu"
bash tools/run-training.sh \
  --config "$normal_config" \
  --hardware-profile "$profile" \
  --run-dir "$normal_dir" \
  --gpus "$gpu" \
  --nproc-per-node 1 \
  --session "$session"

check_idle "$gpu"
bash tools/run-training.sh \
  --config "$rb_config" \
  --hardware-profile "$profile" \
  --run-dir "$rb_dir" \
  --gpus "$gpu" \
  --nproc-per-node 1 \
  --session "$session"
