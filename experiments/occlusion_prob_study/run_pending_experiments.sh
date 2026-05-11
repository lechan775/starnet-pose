#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
GPUS=${GPUS:-1}
PORT=${PORT:-29500}
DEVICE=${DEVICE:-cuda:0}
WORK_ROOT=${WORK_ROOT:-"${PROJECT_ROOT}/work_dirs/paper_pending"}
RESULT_ROOT=${RESULT_ROOT:-"${PROJECT_ROOT}/work_dirs/paper_pending_results"}
REQUIRE_ALL=${REQUIRE_ALL:-1}
EXTRA_ARGS=("$@")

mkdir -p "${WORK_ROOT}" "${RESULT_ROOT}"

function handle_missing() {
    local message=$1
    if [[ "${REQUIRE_ALL}" == "1" ]]; then
        echo "${message}"
        exit 1
    fi
    echo "Skip: ${message}"
    return 1
}

function require_file_or_skip() {
    local path=$1
    local label=$2
    if [[ -z "${path}" ]]; then
        handle_missing "Missing ${label}"
        return 1
    fi
    if [[ ! -f "${path}" ]]; then
        handle_missing "Not found ${label}: ${path}"
        return 1
    fi
    return 0
}

function run_train() {
    local experiment_name=$1
    local config_path=$2
    shift 2
    local work_dir="${WORK_ROOT}/${experiment_name}"
    mkdir -p "${work_dir}"
    if [[ "${GPUS}" -gt 1 ]]; then
        PORT=${PORT} bash "${PROJECT_ROOT}/tools/dist_train.sh" \
            "${config_path}" "${GPUS}" --work-dir "${work_dir}" "$@" "${EXTRA_ARGS[@]}"
    else
        PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
            "${config_path}" --work-dir "${work_dir}" "$@" "${EXTRA_ARGS[@]}"
    fi
}

function run_test() {
    local experiment_name=$1
    local config_path=$2
    local checkpoint_path=$3
    shift 3
    local work_dir="${RESULT_ROOT}/${experiment_name}"
    mkdir -p "${work_dir}"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/test.py" \
        "${config_path}" "${checkpoint_path}" \
        --work-dir "${work_dir}" \
        --out "${work_dir}/metrics.json" "$@" "${EXTRA_ARGS[@]}"
}

function run_flops() {
    local experiment_name=$1
    local config_path=$2
    shift 2
    local out_file="${RESULT_ROOT}/${experiment_name}_flops.txt"
    mkdir -p "${RESULT_ROOT}"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/analysis_tools/get_flops.py" \
        "${config_path}" --device "${DEVICE}" "$@" | tee "${out_file}"
}

function run_latency() {
    local experiment_name=$1
    local config_path=$2
    local checkpoint_path=$3
    local out_file="${RESULT_ROOT}/${experiment_name}_latency.txt"
    mkdir -p "${RESULT_ROOT}"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmark_latency.py" \
        "${config_path}" --checkpoint "${checkpoint_path}" --device "${DEVICE}" | tee "${out_file}"
}

VITPOSE_B_CONFIG=${VITPOSE_B_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"}
VITPOSE_L_CONFIG=${VITPOSE_L_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"}
VITPOSE_H_CONFIG=${VITPOSE_H_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"}
VITPOSE_B_CKPT=${VITPOSE_B_CKPT:-}
VITPOSE_L_CKPT=${VITPOSE_L_CKPT:-}
VITPOSE_H_CKPT=${VITPOSE_H_CKPT:-}

STARNET_CA_CONFIG=${STARNET_CA_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py"}
STARNET_CA_CKPT=${STARNET_CA_CKPT:-}
STARNET_SE_CONFIG=${STARNET_SE_CONFIG:-}
STARNET_SE_CKPT=${STARNET_SE_CKPT:-}
STARNET_CBAM_CONFIG=${STARNET_CBAM_CONFIG:-}
STARNET_CBAM_CKPT=${STARNET_CBAM_CKPT:-}

OCHUMAN_VAL_ANN=${OCHUMAN_VAL_ANN:-}
OCHUMAN_TEST_ANN=${OCHUMAN_TEST_ANN:-}
OCHUMAN_IMG_DIR=${OCHUMAN_IMG_DIR:-}

IMAGENET_PRETRAIN_CONFIG=${IMAGENET_PRETRAIN_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnet-s3_8xb256-420e_coco-256x192.py"}
MIM_PRETRAIN_CONFIG=${MIM_PRETRAIN_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnet-s3_8xb256-420e_coco-256x192.py"}
MIM_PRETRAIN_CKPT=${MIM_PRETRAIN_CKPT:-}

MULTIDATA_CONFIG=${MULTIDATA_CONFIG:-}
PARTIAL_FINETUNE_CONFIG=${PARTIAL_FINETUNE_CONFIG:-}

DISTILL_HOMO_CONFIG=${DISTILL_HOMO_CONFIG:-}
DISTILL_HETERO_CONFIG=${DISTILL_HETERO_CONFIG:-}
DISTILL_TOKEN_CONFIG=${DISTILL_TOKEN_CONFIG:-}

require_file_or_skip "${VITPOSE_B_CONFIG}" "VITPOSE_B_CONFIG" || true
require_file_or_skip "${VITPOSE_L_CONFIG}" "VITPOSE_L_CONFIG" || true
require_file_or_skip "${VITPOSE_H_CONFIG}" "VITPOSE_H_CONFIG" || true
if [[ -f "${VITPOSE_B_CONFIG}" ]]; then
    run_flops "vitpose_b" "${VITPOSE_B_CONFIG}"
fi
if [[ -f "${VITPOSE_L_CONFIG}" ]]; then
    run_flops "vitpose_l" "${VITPOSE_L_CONFIG}"
fi
if [[ -f "${VITPOSE_H_CONFIG}" ]]; then
    run_flops "vitpose_h" "${VITPOSE_H_CONFIG}"
fi
if require_file_or_skip "${VITPOSE_B_CKPT}" "VITPOSE_B_CKPT"; then
    run_test "vitpose_b_coco_val" "${VITPOSE_B_CONFIG}" "${VITPOSE_B_CKPT}"
fi
if require_file_or_skip "${VITPOSE_L_CKPT}" "VITPOSE_L_CKPT"; then
    run_test "vitpose_l_coco_val" "${VITPOSE_L_CONFIG}" "${VITPOSE_L_CKPT}"
fi
if require_file_or_skip "${VITPOSE_H_CKPT}" "VITPOSE_H_CKPT"; then
    run_test "vitpose_h_coco_val" "${VITPOSE_H_CONFIG}" "${VITPOSE_H_CKPT}"
fi

if require_file_or_skip "${STARNET_CA_CONFIG}" "STARNET_CA_CONFIG"; then
    run_flops "starnet_ca" "${STARNET_CA_CONFIG}"
fi
if require_file_or_skip "${STARNET_SE_CONFIG}" "STARNET_SE_CONFIG"; then
    run_flops "starnet_se" "${STARNET_SE_CONFIG}"
fi
if require_file_or_skip "${STARNET_CBAM_CONFIG}" "STARNET_CBAM_CONFIG"; then
    run_flops "starnet_cbam" "${STARNET_CBAM_CONFIG}"
fi
if require_file_or_skip "${STARNET_CA_CKPT}" "STARNET_CA_CKPT"; then
    run_test "attn_ca_coco_val" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}"
fi
if require_file_or_skip "${STARNET_SE_CKPT}" "STARNET_SE_CKPT"; then
    run_test "attn_se_coco_val" "${STARNET_SE_CONFIG}" "${STARNET_SE_CKPT}"
fi
if require_file_or_skip "${STARNET_CBAM_CKPT}" "STARNET_CBAM_CKPT"; then
    run_test "attn_cbam_coco_val" "${STARNET_CBAM_CONFIG}" "${STARNET_CBAM_CKPT}"
fi

if require_file_or_skip "${STARNET_CA_CKPT}" "STARNET_CA_CKPT"; then
    if [[ -n "${OCHUMAN_VAL_ANN}" && -n "${OCHUMAN_IMG_DIR}" ]]; then
        run_test "starnetca_ochuman_val" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}" \
            --cfg-options \
            val_dataloader.dataset.type=OCHumanDataset \
            val_dataloader.dataset.ann_file="${OCHUMAN_VAL_ANN}" \
            val_dataloader.dataset.data_prefix.img="${OCHUMAN_IMG_DIR}/" \
            val_dataloader.dataset.test_mode=True \
            test_dataloader.dataset.type=OCHumanDataset \
            test_dataloader.dataset.ann_file="${OCHUMAN_VAL_ANN}" \
            test_dataloader.dataset.data_prefix.img="${OCHUMAN_IMG_DIR}/" \
            test_dataloader.dataset.test_mode=True \
            val_evaluator.ann_file="${OCHUMAN_VAL_ANN}" \
            test_evaluator.ann_file="${OCHUMAN_VAL_ANN}"
    else
        handle_missing "Missing OCHUMAN_VAL_ANN or OCHUMAN_IMG_DIR"
    fi
    if [[ -n "${OCHUMAN_TEST_ANN}" && -n "${OCHUMAN_IMG_DIR}" ]]; then
        run_test "starnetca_ochuman_test" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}" \
            --cfg-options \
            val_dataloader.dataset.type=OCHumanDataset \
            val_dataloader.dataset.ann_file="${OCHUMAN_TEST_ANN}" \
            val_dataloader.dataset.data_prefix.img="${OCHUMAN_IMG_DIR}/" \
            val_dataloader.dataset.test_mode=True \
            test_dataloader.dataset.type=OCHumanDataset \
            test_dataloader.dataset.ann_file="${OCHUMAN_TEST_ANN}" \
            test_dataloader.dataset.data_prefix.img="${OCHUMAN_IMG_DIR}/" \
            test_dataloader.dataset.test_mode=True \
            val_evaluator.ann_file="${OCHUMAN_TEST_ANN}" \
            test_evaluator.ann_file="${OCHUMAN_TEST_ANN}"
    else
        handle_missing "Missing OCHUMAN_TEST_ANN or OCHUMAN_IMG_DIR"
    fi
fi

if require_file_or_skip "${IMAGENET_PRETRAIN_CONFIG}" "IMAGENET_PRETRAIN_CONFIG"; then
    run_train "pretrain_imagenet" "${IMAGENET_PRETRAIN_CONFIG}"
fi
if require_file_or_skip "${MIM_PRETRAIN_CONFIG}" "MIM_PRETRAIN_CONFIG"; then
    if require_file_or_skip "${MIM_PRETRAIN_CKPT}" "MIM_PRETRAIN_CKPT"; then
        run_train "pretrain_mim" "${MIM_PRETRAIN_CONFIG}" \
            --cfg-options model.backbone.pretrained="${MIM_PRETRAIN_CKPT}"
    fi
fi

if require_file_or_skip "${MULTIDATA_CONFIG}" "MULTIDATA_CONFIG"; then
    run_train "joint_coco_aic_mpii" "${MULTIDATA_CONFIG}"
fi

if require_file_or_skip "${PARTIAL_FINETUNE_CONFIG}" "PARTIAL_FINETUNE_CONFIG"; then
    run_train "partial_finetune_frozen_mhsa_ffn" "${PARTIAL_FINETUNE_CONFIG}"
fi

if require_file_or_skip "${DISTILL_HOMO_CONFIG}" "DISTILL_HOMO_CONFIG"; then
    run_train "distill_homogeneous" "${DISTILL_HOMO_CONFIG}"
fi
if require_file_or_skip "${DISTILL_HETERO_CONFIG}" "DISTILL_HETERO_CONFIG"; then
    run_train "distill_heterogeneous" "${DISTILL_HETERO_CONFIG}"
fi
if require_file_or_skip "${DISTILL_TOKEN_CONFIG}" "DISTILL_TOKEN_CONFIG"; then
    run_train "distill_token" "${DISTILL_TOKEN_CONFIG}"
fi

if require_file_or_skip "${STARNET_CA_CKPT}" "STARNET_CA_CKPT"; then
    run_latency "starnet_ca_latency" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}"
fi
