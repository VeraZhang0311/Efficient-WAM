#!/bin/bash
set -euo pipefail

STAGE="${STAGE:-stage1}"
PROJECT_NAME="${PROJECT_NAME:-${STAGE}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero0.json}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
NODE_RANK="${NODE_RANK:-0}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RESUME_FROM="${RESUME_FROM:-}"
ACTION_INIT_CHECKPOINT="${ACTION_INIT_CHECKPOINT:-}"
EFFICIENT_WAM_INIT_CHECKPOINT="${EFFICIENT_WAM_INIT_CHECKPOINT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS:-}"
PIN_MEMORY="${PIN_MEMORY:-}"

if [[ "${STAGE}" == "stage1" ]]; then
    TRAIN_ENTRY="train/stage1.py"
    CONFIG_FILE="${CONFIG_FILE:-configs/robotwin/stage1_video_distill.yaml}"
    RUN_NAME="${RUN_NAME:-efficient_wam_stage1}"
elif [[ "${STAGE}" == "stage2" ]]; then
    TRAIN_ENTRY="train/stage2.py"
    CONFIG_FILE="${CONFIG_FILE:-configs/robotwin/stage2_action.yaml}"
    RUN_NAME="${RUN_NAME:-efficient_wam_stage2}"
elif [[ "${STAGE}" == "stage3" ]]; then
    TRAIN_ENTRY="train/stage3.py"
    CONFIG_FILE="${CONFIG_FILE:-configs/robotwin/stage3_joint.yaml}"
    RUN_NAME="${RUN_NAME:-efficient_wam_stage3}"
else
    echo "Unsupported STAGE=${STAGE}. Expected stage1, stage2, or stage3."
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    EXTRA_ARGS+=(--resume_from "${RESUME_FROM}")
fi
if [[ -n "${ACTION_INIT_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--action_init_checkpoint "${ACTION_INIT_CHECKPOINT}")
fi
if [[ -n "${EFFICIENT_WAM_INIT_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--efficient_wam_init_checkpoint "${EFFICIENT_WAM_INIT_CHECKPOINT}")
fi
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    EXTRA_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    EXTRA_ARGS+=(--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${PER_DEVICE_BATCH_SIZE}" ]]; then
    EXTRA_ARGS+=(--per_device_batch_size "${PER_DEVICE_BATCH_SIZE}")
fi
if [[ -n "${NUM_WORKERS_OVERRIDE}" ]]; then
    EXTRA_ARGS+=(--num_workers "${NUM_WORKERS_OVERRIDE}")
fi
if [[ -n "${PIN_MEMORY}" ]]; then
    EXTRA_ARGS+=(--pin_memory "${PIN_MEMORY}")
fi

echo "Launching EfficientWAM ${STAGE} training"
echo "  Config: ${CONFIG_FILE}"
echo "  DeepSpeed: ${DEEPSPEED_CONFIG}"
echo "  Project name: ${PROJECT_NAME}"
echo "  Run name: ${RUN_NAME}"
echo "  Nodes: ${NUM_NODES}"
echo "  GPUs per node: ${NUM_GPUS}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
    --nnodes="${NUM_NODES}" \
    --nproc_per_node="${NUM_GPUS}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --config "${CONFIG_FILE}" \
    --project_name "${PROJECT_NAME}" \
    --run_name "${RUN_NAME}" \
    --report_to "${REPORT_TO}" \
    "${EXTRA_ARGS[@]}"
