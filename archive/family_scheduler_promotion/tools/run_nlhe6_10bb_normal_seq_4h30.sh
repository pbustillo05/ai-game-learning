#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES=1
export PATH=/home/pablob/.venvs/poker/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13/lib

cd /home/pablob/ai-game-learning || exit 1

run_timed_training() {
  local config="$1"
  /usr/bin/timeout --signal=INT --kill-after=120s 270m \
    /home/pablob/.venvs/poker/bin/python -m pbg.cli.train \
    --config "$config" --hardware-profile local_3x2080ti
  local status=$?
  case "$status" in
    0|124|130) return 0 ;;
    *) return "$status" ;;
  esac
}

nvidia-smi
run_timed_training configs/train/nlhe6_10bb_normal_streamed_4h30.yaml || exit $?
nvidia-smi
run_timed_training configs/train/nlhe6_10bb_sequential_rb_streamed_4h30.yaml || exit $?
nvidia-smi
