#!/usr/bin/env bash
set -uo pipefail

if (($# != 1)); then
  echo "Usage: tools/run-status.sh RUN_DIR" >&2
  exit 2
fi

run_dir=$1
if [[ -f "$run_dir/status.json" ]]; then
  jq . "$run_dir/status.json"
else
  echo "No status.json found in $run_dir" >&2
fi

if [[ -f "$run_dir/metrics.jsonl" ]]; then
  echo "--- latest metric ---"
  tail -n 1 "$run_dir/metrics.jsonl" | jq '{iter,optimizer_steps,loss,sequential_starts_per_gpu_current,sequential_starts_per_gpu_next,iteration_peak_reserved_bytes}'
fi

echo "--- GPU state ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

session=$(jq -r '.tmux_session // empty' "$run_dir/status.json" 2>/dev/null || true)
if [[ -n "$session" ]] && command -v tmux >/dev/null 2>&1; then
  echo "--- tmux ---"
  tmux has-session -t "$session" 2>/dev/null && echo "active: $session" || echo "not active: $session"
fi

echo "--- recent stderr ---"
stderr_log=$(jq -r '.stderr_log // empty' "$run_dir/status.json" 2>/dev/null || true)
tail -n 20 "$run_dir/$stderr_log" 2>/dev/null || true

echo "--- recent GPU telemetry ---"
telemetry_log=$(jq -r '.telemetry_log // empty' "$run_dir/status.json" 2>/dev/null || true)
tail -n 5 "$run_dir/$telemetry_log" 2>/dev/null || true
