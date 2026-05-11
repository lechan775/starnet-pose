#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_PRESET=${1:-coco_full}
if [[ $# -gt 0 ]]; then
    shift
fi

bash "${SCRIPT_DIR}/run_all.sh" "${TRAIN_PRESET}" "$@"
bash "${SCRIPT_DIR}/run_pending_experiments.sh" "$@"
