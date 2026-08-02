#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 --config FILE --replicas K --python PYTHON --evaluator DIR [--resume CHECKPOINT]"
}

config=""
replicas=""
python=""
evaluator=""
resume=""
while (($#)); do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --replicas) replicas=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --evaluator) evaluator=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$config" || -z "$replicas" || -z "$python" || -z "$evaluator" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'CUDA_VISIBLE_DEVICES is empty; launch inside a GPU allocation\n' >&2
  exit 2
fi
if [[ ! "$replicas" =~ ^[1-9][0-9]*$ ]] || [[ ! -x "$python" ]] || [[ ! -d "$evaluator" ]]; then
  printf 'replicas, Python, or evaluator path is invalid\n' >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root:$(cd "$evaluator" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PBG_EVALUATOR_ROOT="$(cd "$evaluator" && pwd)"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SLURM_TMPDIR:-/tmp}/pbg-torch-extensions-sm90}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
IFS=',' read -r -a assigned_gpus <<< "$CUDA_VISIBLE_DEVICES"
nproc=${#assigned_gpus[@]}

"$python" - "$config" "$replicas" <<'PY'
import sys
from pbg.config import load_config
cfg = load_config(sys.argv[1])
if cfg.execution.manifest.n_replica != int(sys.argv[2]):
    raise SystemExit("replicas must match root.execution.manifest.n_replica")
PY

cmd=("$python" -m torch.distributed.run --standalone --nproc-per-node "$nproc"
  -m pbg.cli.train --config "$config")
[[ -n "$resume" ]] && cmd+=(--resume "$resume")
exec "${cmd[@]}"
