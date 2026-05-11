#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_PRESET=${1:-}

if [[ -n "${TRAIN_PRESET}" ]]; then
    shift
fi

bash "${SCRIPT_DIR}/run_suite.sh" occlusion_sweep "${TRAIN_PRESET}" "$@"
