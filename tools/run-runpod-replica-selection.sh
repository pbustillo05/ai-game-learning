#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 --normal-config FILE --rb-config FILE --hardware-profile FILE" \
    "  --output-root DIR --python PYTHON --evaluator DIR [--deadline-minutes 90]"
}

normal_config=""
rb_config=""
hardware_profile=""
output_root=""
python=""
evaluator=""
deadline_minutes=90

while (($#)); do
  case "$1" in
    --normal-config) normal_config=$2; shift 2 ;;
    --rb-config) rb_config=$2; shift 2 ;;
    --hardware-profile) hardware_profile=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --evaluator) evaluator=$2; shift 2 ;;
    --deadline-minutes) deadline_minutes=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$normal_config" || -z "$rb_config" || -z "$hardware_profile" ||
      -z "$output_root" || -z "$python" || -z "$evaluator" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'CUDA_VISIBLE_DEVICES must select the RunPod H100\n' >&2
  exit 2
fi
if [[ ! -x "$python" || ! -d "$evaluator" ]]; then
  printf 'python or evaluator path is invalid\n' >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root:$(cd "$evaluator" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PBG_EVALUATOR_ROOT="$(cd "$evaluator" && pwd)"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0

exec "$python" -m pbg.cli.runpod_replica_selection \
  --normal-config "$normal_config" \
  --rb-config "$rb_config" \
  --hardware-profile "$hardware_profile" \
  --evaluator-root "$PBG_EVALUATOR_ROOT" \
  --output-root "$output_root" \
  --replicas 1 2 4 8 \
  --deadline-minutes "$deadline_minutes" \
  --learner-warmup 10 \
  --learner-steps 50 \
  --minimum-throughput-gain 0.10 \
  --peak-limit-gib 70
