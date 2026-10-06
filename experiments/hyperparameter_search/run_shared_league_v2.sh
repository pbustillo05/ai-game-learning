#!/usr/bin/env bash
# Compatibility entry point: use the current diagnosis-led scientific study.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_shared_loss_search_v3.sh" "$@"
