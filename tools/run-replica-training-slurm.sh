#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 --config FILE --hardware-profile FILE --replicas K --run-root DIR" \
    "  --python PYTHON --evaluator DIR [--env-file FILE] [--resume-root DIR]"
}

config=""
hardware_profile=""
replicas=""
run_root=""
python=""
evaluator=""
env_file=""
resume_root=""

while (($#)); do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --hardware-profile) hardware_profile=$2; shift 2 ;;
    --replicas) replicas=$2; shift 2 ;;
    --run-root) run_root=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --evaluator) evaluator=$2; shift 2 ;;
    --env-file) env_file=$2; shift 2 ;;
    --resume-root) resume_root=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$config" || -z "$hardware_profile" || -z "$replicas" ||
      -z "$run_root" || -z "$python" || -z "$evaluator" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$replicas" =~ ^[1-9][0-9]*$ ]]; then
  printf 'replicas must be a positive integer\n' >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'CUDA_VISIBLE_DEVICES is empty; launch this inside a GPU allocation\n' >&2
  exit 2
fi
if [[ ! -x "$python" ]]; then
  printf 'python executable is not executable: %s\n' "$python" >&2
  exit 2
fi
if [[ ! -d "$evaluator" ]]; then
  printf 'evaluator directory does not exist: %s\n' "$evaluator" >&2
  exit 2
fi
if [[ -n "$env_file" ]]; then
  if [[ ! -f "$env_file" ]]; then
    printf 'environment file does not exist: %s\n' "$env_file" >&2
    exit 2
  fi
  source "$env_file"
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$run_root"
run_root=$(cd "$run_root" && pwd)
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")
hardware_profile=$(cd "$(dirname "$hardware_profile")" && pwd)/$(basename "$hardware_profile")
evaluator=$(cd "$evaluator" && pwd)

export PYTHONPATH="$repo_root:$evaluator${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0

IFS=',' read -r -a assigned_gpus <<< "$CUDA_VISIBLE_DEVICES"
if ((${#assigned_gpus[@]} == 0)); then
  printf 'no Slurm-visible GPUs were parsed\n' >&2
  exit 2
fi

pids=()
forward_signal() {
  local signal=$1
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -"$signal" "$pid" 2>/dev/null || true
    fi
  done
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

metadata="$run_root/allocation.json"
"$python" - "$metadata" "$config" "$hardware_profile" "$replicas" <<'PY'
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict

import torch

from pbg.config import load_config

target, config, profile, replicas = sys.argv[1:]
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
except Exception as exc:
    gpu = [f"nvidia-smi failed: {exc}"]
def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None

resolved = asdict(load_config(config, hardware_profile=profile))
payload = {
    "config": config,
    "hardware_profile": profile,
    "replicas_per_gpu": int(replicas),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "slurm": {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
    },
    "host": platform.node(),
    "python": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": gpu,
    "source_revision": revision(os.environ["PYTHONPATH"].split(":")[0]),
    "evaluator_revision": revision(os.environ["PYTHONPATH"].split(":")[1]),
    "resolved_config": resolved,
}
with open(target, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

for logical_index in "${!assigned_gpus[@]}"; do
  gpu=${assigned_gpus[$logical_index]}
  gpu_root="$run_root/gpu_$(printf '%03d' "$logical_index")"
  mkdir -p "$gpu_root"
  cmd=(
    "$python" -m pbg.cli.train_replicas
    --config "$config"
    --hardware-profile "$hardware_profile"
    --replicas "$replicas"
    --out-root "$gpu_root"
  )
  if [[ -n "$resume_root" ]]; then
    cmd+=(--resume-root "$resume_root/gpu_$(printf '%03d' "$logical_index")")
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "${cmd[@]}"
  ) >"$gpu_root/stdout.log" 2>"$gpu_root/stderr.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
    forward_signal TERM
  fi
done
exit "$status"
