#!/usr/bin/env bash
# Resume the prepared diagnosis-led study on the three available GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
exec python -m experiments.hyperparameter_search.league_campaign \
  --profile configs/profiles/nlhe_hu_32bb_shared_loss_search_v3.yaml \
  --output runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v3_20260928 \
  --gpus 0,1,2 --resume "$@"
