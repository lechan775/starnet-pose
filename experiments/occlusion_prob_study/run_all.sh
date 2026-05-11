#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_PRESET=${1:-}

if [[ -n "${TRAIN_PRESET}" ]]; then
    shift
fi

if [[ -z "${TRAIN_PRESET}" ]]; then
    TRAIN_PRESET=$("${PYTHON_BIN}" "${SCRIPT_DIR}/generate_configs.py" --print-default-train-preset)
fi

if [[ -n "${SUITES:-}" ]]; then
    read -r -a SUITE_LIST <<< "${SUITES}"
else
    mapfile -t SUITE_LIST < <("${PYTHON_BIN}" "${SCRIPT_DIR}/generate_configs.py" --print-default-suites)
fi

for suite_name in "${SUITE_LIST[@]}"; do
    bash "${SCRIPT_DIR}/run_suite.sh" "${suite_name}" "${TRAIN_PRESET}" "$@"
done
