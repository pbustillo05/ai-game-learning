#!/usr/bin/env bash
# Source after activating a clean venv populated by requirements-h100.lock.
# NVIDIA's runtime wheel ships a versioned library; torch extension builds use
# -lcudart and therefore also require its unversioned linker name.
set -euo pipefail

: "${VIRTUAL_ENV:?activate the pinned poker venv first}"
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13"
if [[ ! -x "$CUDA_HOME/bin/nvcc" || ! -f "$CUDA_HOME/lib/libcudart.so.13" ]]; then
  printf 'missing pinned CUDA 13 toolkit or runtime in %s\n' "$CUDA_HOME" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
  ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
