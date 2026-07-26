#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source /home/pablob/.venvs/poker/bin/activate
export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

artifact_root="runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724"
mkdir -p "$artifact_root"

rb_config="configs/train/nlhe_hu_32bb_rb.yaml"
branch_config="configs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723.yaml"
hardware="configs/hardware/local_3x2080ti.yaml"
rb_checkpoint="runs/experiments/nlhe_hu_32bb_rb_storage_baseline_20260723/benchmark_init_ckpt_0001110_v8.pt"
branch_checkpoint="runs/experiments/nlhe_hu_32bb_branched_fixed_storage_baseline_20260723/benchmark_init_ckpt_0001110_dummy_heads_seed91301_v8.pt"

run_rb() {
  local gpu="$1"
  local base_seed="$2"
  local gpu_dir="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m pbg.cli.benchmark \
    --config "$rb_config" \
    --hardware-profile "$hardware" \
    --checkpoint "$rb_checkpoint" \
    --mode rao_blackwell \
    --rollout-only \
    --warmup 1 \
    --repeats 1 \
    --seed "$base_seed" \
    --roots-per-gpu 3072 \
    --completion-samples 1024 \
    --gd-steps 1 \
    --output "$gpu_dir/sequential_rb.json" \
    >"$gpu_dir/sequential_rb.stdout.log" \
    2>"$gpu_dir/sequential_rb.stderr.log"
}

run_branch() {
  local gpu="$1"
  local base_seed="$2"
  local gpu_dir="$3"
  local variant="$4"
  local extra=()
  if [[ "$variant" == "reuse_private_profile" ]]; then
    extra+=(--reuse-root-private-profile)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m pbg.cli.benchmark \
    --config "$branch_config" \
    --hardware-profile "$hardware" \
    --checkpoint "$branch_checkpoint" \
    --mode branched_rao_blackwell \
    --rollout-only \
    --warmup 1 \
    --repeats 1 \
    --seed "$base_seed" \
    --branching-upper-cutoff-nodes 10000 \
    --completion-samples 1024 \
    --gd-steps 1 \
    "${extra[@]}" \
    --output "$gpu_dir/branched_${variant}.json" \
    >"$gpu_dir/branched_${variant}.stdout.log" \
    2>"$gpu_dir/branched_${variant}.stderr.log"
}

run_gpu() {
  local gpu="$1"
  local repeat_seed="$2"
  local order="$3"
  local base_seed=$((repeat_seed - 104729))
  local gpu_dir="$artifact_root/gpu${gpu}_seed${repeat_seed}"
  mkdir -p "$gpu_dir"

  nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    --id="$gpu" \
    --loop=1 >"$gpu_dir/gpu_telemetry.csv" &
  local telemetry_pid=$!
  trap 'kill "$telemetry_pid" 2>/dev/null || true' RETURN

  IFS=',' read -r -a variants <<<"$order"
  for variant in "${variants[@]}"; do
    case "$variant" in
      sequential_rb)
        run_rb "$gpu" "$base_seed" "$gpu_dir"
        ;;
      baseline)
        run_branch "$gpu" "$base_seed" "$gpu_dir" "baseline"
        ;;
      reuse_private_profile)
        run_branch "$gpu" "$base_seed" "$gpu_dir" "reuse_private_profile"
        ;;
      *)
        echo "unknown diagnostic variant: $variant" >&2
        return 2
        ;;
    esac
    nvidia-smi \
      --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used \
      --format=csv,noheader,nounits \
      --id="$gpu" >>"$gpu_dir/job_boundaries.csv"
    sleep 30
  done

  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - RETURN
}

git rev-parse HEAD >"$artifact_root/git_head.txt"
git status --short >"$artifact_root/git_status.txt"
git diff --binary >"$artifact_root/working_tree.patch"
nvidia-smi -q >"$artifact_root/nvidia_smi_before.txt"

run_gpu 0 104830 "sequential_rb,baseline,reuse_private_profile" &
pid0=$!
run_gpu 1 209559 "reuse_private_profile,baseline,sequential_rb" &
pid1=$!
run_gpu 2 314288 "baseline,reuse_private_profile,sequential_rb" &
pid2=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
wait "$pid2" || status=$?

nvidia-smi -q >"$artifact_root/nvidia_smi_after.txt"
exit "$status"
