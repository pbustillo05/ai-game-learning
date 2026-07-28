#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/run-training.sh --config CONFIG --gpus GPU_IDS [--nproc-per-node N]
    [--resume CHECKPOINT] [--tmux]

The launcher writes stdout.log, stderr.log, combined.log, status.json,
gpu_start.csv, and gpu_end.csv under RUN_DIR. With --tmux it starts a detached
session and returns immediately.
EOF
}

config=""
run_dir=""
gpus=""
nproc_per_node=2
resume=""
use_tmux=0
tmux_child=0
session=""

while (($#)); do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --nproc-per-node) nproc_per_node=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    --tmux) use_tmux=1; shift ;;
    --session) session=$2; shift 2 ;;
    --_tmux-child) tmux_child=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$config" || -z "$gpus" ]]; then
  usage >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=$(cd "$repo_root" && realpath "$config")
run_dir=$(/home/pablob/.venvs/poker/bin/python - "$config" <<'PY'
import sys
from pbg.config import load_config
print(load_config(sys.argv[1]).execution.output_dir)
PY
)
run_dir=$(cd "$repo_root" && mkdir -p "$run_dir" && cd "$run_dir" && pwd)
launcher_name=$(basename "$0")
if [[ -z "$session" ]]; then
  session="pbg-$(basename "$run_dir" | tr -cs '[:alnum:]_.-' '-')"
fi

if ((use_tmux && !tmux_child)); then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; run without --tmux or install tmux" >&2
    exit 2
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
  args=(--config "$config" --gpus "$gpus" --nproc-per-node "$nproc_per_node" --session "$session" --_tmux-child)
  [[ -n "$resume" ]] && args+=(--resume "$resume")
  tmux new-session -d -s "$session" "$repo_root/tools/$launcher_name" "${args[@]}"
  echo "started tmux session: $session"
  echo "inspect with: tmux attach -t $session"
  echo "status with:  tools/run-status.sh $run_dir"
  exit 0
fi

attempt_id=$(date +%Y%m%dT%H%M%S)
stdout_log="$run_dir/stdout_${attempt_id}.log"
stderr_log="$run_dir/stderr_${attempt_id}.log"
combined_log="$run_dir/combined_${attempt_id}.log"
command_log="$run_dir/command_${attempt_id}.txt"
telemetry_log="$run_dir/gpu_telemetry_${attempt_id}.csv"
status_json="$run_dir/status.json"
child_pid=""
monitor_pid=""
termination_signal=""

forward_signal() {
  termination_signal=$1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -"$1" "$child_pid" 2>/dev/null || true
  fi
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

write_status() {
  local state=$1 exit_code=${2:-} now checkpoint metrics
  now=$(date --iso-8601=seconds)
  checkpoint=$(
    find "$run_dir" -maxdepth 1 \
      \( -name 'checkpoint_????????.pt' -o -name 'failure_checkpoint_????????.pt' \) \
      -printf '%T@ %f\n' |
      sort -n |
      tail -n 1 |
      cut -d' ' -f2-
  )
  metrics=$(wc -l < "$run_dir/metrics.jsonl" 2>/dev/null || echo 0)
  jq -n \
    --arg state "$state" --arg started "$started_at" --arg updated "$now" \
    --arg ended "${ended_at:-}" --arg host "$(hostname)" \
    --arg pid "${child_pid:-}" --arg session "$session" \
    --arg signal "$termination_signal" --arg nproc "$nproc_per_node" \
    --arg stdout "$(basename "$stdout_log")" --arg stderr "$(basename "$stderr_log")" \
    --arg combined "$(basename "$combined_log")" --arg command "$(basename "$command_log")" \
    --arg telemetry "$(basename "$telemetry_log")" \
    --arg config "$config" \
    --arg gpus "$gpus" --arg resume "$resume" --arg checkpoint "$checkpoint" \
    --arg metrics "$metrics" --arg exit_code "$exit_code" \
    '{state:$state,started_at:$started,updated_at:$updated,ended_at:$ended,
      host:$host,pid:$pid,tmux_session:$session,termination_signal:$signal,
      nproc_per_node:($nproc|tonumber),config:$config,
      gpus:$gpus,resume:$resume,
      stdout_log:$stdout,stderr_log:$stderr,combined_log:$combined,command_log:$command,
      telemetry_log:$telemetry,
      latest_checkpoint:$checkpoint,metrics_lines:($metrics|tonumber),
      exit_code:(if $exit_code == "" then null else ($exit_code|tonumber) end)}' \
    > "$status_json"
}

started_at=$(date --iso-8601=seconds)
write_status running
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits \
  | awk -v ts="$started_at" 'BEGIN { print "timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw" } { print ts "," $0 }' \
  > "$telemetry_log" 2> "$run_dir/gpu_telemetry_start.err" || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  > "$run_dir/gpu_start.csv" 2> "$run_dir/gpu_start.err" || true

monitor_status() {
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 30
    write_status running
    now=$(date --iso-8601=seconds)
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits \
      | awk -v ts="$now" '{ print ts "," $0 }' >> "$telemetry_log" 2>> "$run_dir/gpu_telemetry.err" || true
  done
}

export CUDA_VISIBLE_DEVICES="$gpus"
export CUDA_HOME="${CUDA_HOME:-/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13}"
export PATH="/home/pablob/.venvs/poker/bin:$PATH"
export PYTHONPATH="$repo_root:/home/pablob/poker-gpu-evaluator${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cmd=(/home/pablob/.venvs/poker/bin/torchrun --standalone --nproc-per-node="$nproc_per_node" -m pbg.cli.train
  --config "$config")
[[ -n "$resume" ]] && cmd+=(--resume "$resume")
printf 'start: %s\ncommand:' "$started_at" > "$command_log"
printf ' %q' "${cmd[@]}" >> "$command_log"
printf '\n' >> "$command_log"

"${cmd[@]}" > >(tee -a "$stdout_log" "$combined_log") \
  2> >(tee -a "$stderr_log" "$combined_log" >&2) &
child_pid=$!
monitor_status &
monitor_pid=$!
wait "$child_pid"
exit_code=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

ended_at=$(date --iso-8601=seconds)
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  > "$run_dir/gpu_end.csv" 2> "$run_dir/gpu_end.err" || true
now=$(date --iso-8601=seconds)
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits \
  | awk -v ts="$now" '{ print ts "," $0 }' >> "$telemetry_log" 2>> "$run_dir/gpu_telemetry.err" || true
write_status finished "$exit_code"
exit "$exit_code"
