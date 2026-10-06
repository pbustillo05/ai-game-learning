#!/usr/bin/env bash
# CUDA validation gates, durable status/logs, then the shared three-GPU study.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
exec python -m experiments.hyperparameter_search.launch_v4 "$@"
