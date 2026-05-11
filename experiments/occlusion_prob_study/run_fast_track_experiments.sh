#!/usr/bin/env bash

set -euo pipefail

# ==============================================================================
# 精简版待补充实验脚本 (针对二区/三区期刊优化)
# 
# 砍掉了：多数据集联合、蒸馏、MIM预训练等耗时且非必要的实验。
# 保留了：ViTPose测试(出Pareto图)、OCHuman测试(证明遮挡鲁棒性)、FPS/FLOPs测速、
#        以及注意力消融(建议在mini数据集上运行)。
# ==============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
GPUS=${GPUS:-1}
PORT=${PORT:-29500}
DEVICE=${DEVICE:-cuda:0}

# 结果保存目录
WORK_ROOT=${WORK_ROOT:-"${PROJECT_ROOT}/work_dirs/paper_fast_track"}
RESULT_ROOT=${RESULT_ROOT:-"${PROJECT_ROOT}/work_dirs/paper_fast_track_results"}

# REQUIRE_ALL=0 允许跳过未配置的项
REQUIRE_ALL=${REQUIRE_ALL:-0}
EXTRA_ARGS=("$@")

mkdir -p "${WORK_ROOT}" "${RESULT_ROOT}"

# ======================= 辅助函数 =======================

function handle_missing() {
    local message=$1
    if [[ "${REQUIRE_ALL}" == "1" ]]; then
        echo "Error: ${message}"
        exit 1
    fi
    echo "Skip: ${message}"
    return 1
}

function require_file_or_skip() {
    local path=$1
    local label=$2
    if [[ -z "${path}" ]]; then
        handle_missing "Missing variable: ${label}"
        return 1
    fi
    if [[ ! -f "${path}" ]]; then
        handle_missing "File not found for ${label}: ${path}"
        return 1
    fi
    return 0
}

function require_pose_checkpoint_or_skip() {
    local path=$1
    local label=$2
    require_file_or_skip "${path}" "${label}" || return 1
    local filename
    filename=$(basename "${path}")
    if [[ "${filename}" == "starnet_s3.pth" ]]; then
        handle_missing "${label} points to ${path}; this is the StarNet backbone initialization checkpoint, not a trained pose checkpoint. Use best_coco_AP_epoch_*.pth or epoch_*.pth from pose training."
        return 1
    fi
    return 0
}

function run_test() {
    local experiment_name=$1
    local config_path=$2
    local checkpoint_path=$3
    shift 3
    local work_dir="${RESULT_ROOT}/${experiment_name}"
    mkdir -p "${work_dir}"
    echo ">>> Running Test: ${experiment_name} <<<"
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
    echo ">>> Calculating FLOPs/Params: ${experiment_name} <<<"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/analysis_tools/get_flops.py" \
        "${config_path}" --device "${DEVICE}" "$@" > "${out_file}" 2>&1
    cat "${out_file}"
}

function run_latency() {
    local experiment_name=$1
    local config_path=$2
    local checkpoint_path=$3
    local out_file="${RESULT_ROOT}/${experiment_name}_latency.txt"
    mkdir -p "${RESULT_ROOT}"
    echo ">>> Benchmarking Latency/FPS: ${experiment_name} <<<"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/benchmark_latency.py" \
        "${config_path}" --checkpoint "${checkpoint_path}" --device "${DEVICE}" > "${out_file}" 2>&1
    cat "${out_file}"
}

function run_train() {
    local experiment_name=$1
    local config_path=$2
    shift 2
    local work_dir="${WORK_ROOT}/${experiment_name}"
    mkdir -p "${work_dir}"
    echo ">>> Running Train: ${experiment_name} <<<"
    if [[ "${GPUS}" -gt 1 ]]; then
        PORT=${PORT} bash "${PROJECT_ROOT}/tools/dist_train.sh" \
            "${config_path}" "${GPUS}" --work-dir "${work_dir}" "$@" "${EXTRA_ARGS[@]}"
    else
        PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
            "${config_path}" --work-dir "${work_dir}" "$@" "${EXTRA_ARGS[@]}"
    fi
}

# ======================= 环境变量配置 =======================

# 1. 超大模型扩展 (ViTPose 测试)
VITPOSE_B_CONFIG=${VITPOSE_B_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"}
VITPOSE_L_CONFIG=${VITPOSE_L_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"}
VITPOSE_H_CONFIG=${VITPOSE_H_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"}
# 需提供权重路径才能进行 test
VITPOSE_B_CKPT=${VITPOSE_B_CKPT:-}
VITPOSE_L_CKPT=${VITPOSE_L_CKPT:-}
VITPOSE_H_CKPT=${VITPOSE_H_CKPT:-}

# 2. 本文核心模型测试 (用于 OCHuman和测速)
STARNET_CA_CONFIG=${STARNET_CA_CONFIG:-"${PROJECT_ROOT}/configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py"}
STARNET_CA_CKPT=${STARNET_CA_CKPT:-} # 必须显式设置为训练好的 pose 权重，例如 best_coco_AP_epoch_*.pth

# 3. 极端遮挡数据集 OCHuman 路径
OCHUMAN_VAL_ANN=${OCHUMAN_VAL_ANN:-}
OCHUMAN_TEST_ANN=${OCHUMAN_TEST_ANN:-}
OCHUMAN_IMG_DIR=${OCHUMAN_IMG_DIR:-}

# 4. 注意力消融 (强烈建议用生成好的 coco_mini 配置进行训练)
STARNET_SE_MINI_CONFIG=${STARNET_SE_MINI_CONFIG:-}
STARNET_CBAM_MINI_CONFIG=${STARNET_CBAM_MINI_CONFIG:-}


echo "======================================================="
echo "        MMPose 快速实验通道 (Fast Track) 开始          "
echo "======================================================="

# ======================= 实验 1: FLOPs & 参数量评估 (极快) =======================
echo -e "\n[1/4] 评估 FLOPs 与 参数量..."

if require_file_or_skip "${VITPOSE_B_CONFIG}" "VITPOSE_B_CONFIG"; then
    run_flops "vitpose_b" "${VITPOSE_B_CONFIG}"
fi
if require_file_or_skip "${VITPOSE_L_CONFIG}" "VITPOSE_L_CONFIG"; then
    run_flops "vitpose_l" "${VITPOSE_L_CONFIG}"
fi
if require_file_or_skip "${VITPOSE_H_CONFIG}" "VITPOSE_H_CONFIG"; then
    run_flops "vitpose_h" "${VITPOSE_H_CONFIG}"
fi
if require_file_or_skip "${STARNET_CA_CONFIG}" "STARNET_CA_CONFIG"; then
    run_flops "starnet_ca" "${STARNET_CA_CONFIG}"
fi

# ======================= 实验 2: 推理速度评估 (极快) =======================
echo -e "\n[2/4] 评估推理速度 (FPS / Latency)..."

if require_pose_checkpoint_or_skip "${STARNET_CA_CKPT}" "STARNET_CA_CKPT"; then
    run_latency "starnet_ca_speed" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}"
fi

# ======================= 实验 3: 现有模型在各验证集上的测试 (快) =======================
echo -e "\n[3/4] 运行预训练模型测试 (COCO & OCHuman)..."

# ViTPose 在 COCO 上的复测
if require_file_or_skip "${VITPOSE_B_CKPT}" "VITPOSE_B_CKPT"; then
    run_test "vitpose_b_coco_val" "${VITPOSE_B_CONFIG}" "${VITPOSE_B_CKPT}"
fi
if require_file_or_skip "${VITPOSE_L_CKPT}" "VITPOSE_L_CKPT"; then
    run_test "vitpose_l_coco_val" "${VITPOSE_L_CONFIG}" "${VITPOSE_L_CKPT}"
fi
if require_file_or_skip "${VITPOSE_H_CKPT}" "VITPOSE_H_CKPT"; then
    run_test "vitpose_h_coco_val" "${VITPOSE_H_CONFIG}" "${VITPOSE_H_CKPT}"
fi

# 你的模型在 COCO 与 OCHuman 上的测试 (核心卖点)
if require_pose_checkpoint_or_skip "${STARNET_CA_CKPT}" "STARNET_CA_CKPT"; then
    run_test "starnetca_coco_val" "${STARNET_CA_CONFIG}" "${STARNET_CA_CKPT}"

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
        echo "Skip: Missing OCHUMAN_VAL_ANN or OCHUMAN_IMG_DIR for validation."
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
        echo "Skip: Missing OCHUMAN_TEST_ANN or OCHUMAN_IMG_DIR for testing."
    fi
fi

# ======================= 实验 4: 注意力机制消融训练 (较耗时, 建议用 Mini 数据集) =======================
echo -e "\n[4/4] 运行注意力消融训练 (仅在配置了对应路径时启动)..."

if require_file_or_skip "${STARNET_SE_MINI_CONFIG}" "STARNET_SE_MINI_CONFIG"; then
    run_train "starnet_se_ablation" "${STARNET_SE_MINI_CONFIG}"
fi
if require_file_or_skip "${STARNET_CBAM_MINI_CONFIG}" "STARNET_CBAM_MINI_CONFIG"; then
    run_train "starnet_cbam_ablation" "${STARNET_CBAM_MINI_CONFIG}"
fi

echo -e "\n======================================================="
echo "                     所有配置的任务已完成                    "
echo "======================================================="
