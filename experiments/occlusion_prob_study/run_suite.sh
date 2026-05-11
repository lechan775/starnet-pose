#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SUITE_NAME=${1:-}
TRAIN_PRESET=${2:-}

if [[ -z "${SUITE_NAME}" ]]; then
    echo "Usage: bash experiments/occlusion_prob_study/run_suite.sh <suite_name> [train_preset] [extra train args...]"
    exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -z "${TRAIN_PRESET}" ]]; then
    TRAIN_PRESET=$("${PYTHON_BIN}" "${SCRIPT_DIR}/generate_configs.py" --print-default-train-preset)
fi

shift || true
if [[ $# -gt 0 ]]; then
    shift || true
fi
EXTRA_ARGS=("$@")

DEFAULT_GPUS=$("${PYTHON_BIN}" "${SCRIPT_DIR}/generate_configs.py" --print-default-gpus)
GPUS=${GPUS:-${DEFAULT_GPUS}}
PORT=${PORT:-29500}
GENERATED_DIR=${GENERATED_DIR:-"${SCRIPT_DIR}/generated_configs"}
WORK_DIR_ROOT=${WORK_DIR_ROOT:-"${PROJECT_ROOT}/work_dirs/occlusion_prob_study/${TRAIN_PRESET}/${SUITE_NAME}"}

mkdir -p "${GENERATED_DIR}"
mkdir -p "${WORK_DIR_ROOT}"

mapfile -t CONFIGS < <(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_configs.py" \
        --suite "${SUITE_NAME}" \
        --train-preset "${TRAIN_PRESET}" \
        --output-dir "${GENERATED_DIR}" \
        --print-paths
)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No generated configs found for suite=${SUITE_NAME} train_preset=${TRAIN_PRESET}"
    exit 1
fi

echo "Suite: ${SUITE_NAME}"
echo "Train preset: ${TRAIN_PRESET}"
echo "Generated configs: ${#CONFIGS[@]}"
echo "GPUs: ${GPUS}"
echo "Work root: ${WORK_DIR_ROOT}"

for config_path in "${CONFIGS[@]}"; do
    experiment_name=$(basename "${config_path}" .py)
    work_dir="${WORK_DIR_ROOT}/${experiment_name}"

    if [[ "${SKIP_COMPLETED:-0}" == "1" && -f "${work_dir}/last_checkpoint" ]]; then
        echo "Skipping completed experiment: ${experiment_name}"
        continue
    fi

    echo "Running: ${experiment_name}"

    if [[ "${GPUS}" -gt 1 ]]; then
        PORT=${PORT} bash "${PROJECT_ROOT}/tools/dist_train.sh" \
            "${config_path}" \
            "${GPUS}" \
            --work-dir "${work_dir}" \
            "${EXTRA_ARGS[@]}"
    else
        PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
            "${config_path}" \
            --work-dir "${work_dir}" \
            "${EXTRA_ARGS[@]}"
    fi
done
