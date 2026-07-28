#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 --config FILE --report FILE --replicas K --python PYTHON --evaluator DIR --deadline-minutes N [--resume CHECKPOINT]"
}

config=""
report=""
replicas=""
python=""
evaluator=""
deadline_minutes=""
resume=""
while (($#)); do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    --replicas) replicas=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --evaluator) evaluator=$2; shift 2 ;;
    --deadline-minutes) deadline_minutes=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$config" || -z "$report" || -z "$replicas" || -z "$python" ||
      -z "$evaluator" || -z "$deadline_minutes" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || ! -x "$python" || ! -d "$evaluator" ]]; then
  printf 'CUDA visibility, Python, or evaluator path is invalid\n' >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root:$(cd "$evaluator" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PBG_EVALUATOR_ROOT="$(cd "$evaluator" && pwd)"
export PYTHONUNBUFFERED=1
cmd=("$python" -m pbg.cli.runpod_replica_selection
  --config "$config" --report "$report" --replicas "$replicas"
  --deadline-minutes "$deadline_minutes")
[[ -n "$resume" ]] && cmd+=(--resume "$resume")
exec "${cmd[@]}"
