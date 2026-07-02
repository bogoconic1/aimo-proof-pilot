#!/usr/bin/env bash
set -euo pipefail

# Native VERL OPD 4xH200 smoke.
# Layout:
#   GPU 0-1: actor training through VERL automodel FP8
#   GPU 2-3: frozen teacher vLLM in FP8
#
# Run inside the Modal/Singularity training image, or send through operator_client:
#   python submissions-instructions/scripts/operator_client.py send --file submissions-instructions/operator_commands/verl_opd_4xh200_fp8_testcsv_smoke.sh

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-verl_opd_4x_fp8_testcsv_${STAMP}}"

MODEL_PATH="${MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-v33-s150/opd-32b-v33-s150}"
TOKENIZER_PATH_WAS_SET="${TOKENIZER_PATH+x}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy}"
if [ ! -f "${MODEL_PATH}/config.json" ] && [ -f "${TEACHER_MODEL_PATH}/config.json" ]; then
  echo "Policy MODEL_PATH=${MODEL_PATH} is missing; using TEACHER_MODEL_PATH for smoke startup."
  MODEL_PATH="${TEACHER_MODEL_PATH}"
  if [ -z "${TOKENIZER_PATH_WAS_SET}" ]; then
    TOKENIZER_PATH="${MODEL_PATH}"
  fi
fi
DATASET_PATH="${DATASET_PATH:-/workspace/submissions-instructions/test.csv}"
if [ ! -f "${DATASET_PATH}" ] && [ -f /tmp/submissions-instructions-runtime/test.csv ]; then
  DATASET_PATH="/tmp/submissions-instructions-runtime/test.csv"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/olmo_train_assets/output/verl_opd_4x_fp8}"
LOG_ROOT="${LOG_ROOT:-/vol/olmo_train_assets/logs/verl_opd_4x_fp8}"
CACHE_ROOT="${CACHE_ROOT:-/vol/olmo_train_assets/cache/verl_opd_4x_fp8}"
RUNTIME_DEPS_DIR="${RUNTIME_DEPS_DIR:-/tmp/olmo-train-runtime-deps-verl-opd-fp8}"

export OLMO_RUN_DIR_NAME="${OLMO_RUN_DIR_NAME:-${RUN_NAME}}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-olmo3_sink.verl_bootstrap}"

exec /usr/bin/python /app/train.py \
  --fetch-update \
  --submissions-ref "${SUBMISSIONS_REF:-main}" \
  --verl-ref "${VERL_REF:-main}" \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir "${RUNTIME_DEPS_DIR}" \
  --backend verl_opd \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --opd_teacher_model_path "${TEACHER_MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${OUTPUT_ROOT}" \
  --logdir "${LOG_ROOT}" \
  --cache_dir "${CACHE_ROOT}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:--1}" \
  --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-1}" \
  --verl_train_nodes 1 \
  --verl_train_gpus_per_node "${VERL_TRAIN_GPUS_PER_NODE:-2}" \
  --verl_teacher_nodes 1 \
  --verl_teacher_gpus_per_node "${VERL_TEACHER_GPUS_PER_NODE:-2}" \
  --verl_train_batch_size "${VERL_TRAIN_BATCH_SIZE:-2}" \
  --verl_val_batch_size "${VERL_VAL_BATCH_SIZE:-2}" \
  --verl_rollout_n "${VERL_ROLLOUT_N:-1}" \
  --verl_max_prompt_length "${VERL_MAX_PROMPT_LENGTH:-1024}" \
  --verl_max_response_length "${VERL_MAX_RESPONSE_LENGTH:-512}" \
  --verl_ppo_mini_batch_size "${VERL_PPO_MINI_BATCH_SIZE:-2}" \
  --verl_ppo_micro_batch_size_per_gpu "${VERL_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  --verl_training_fp8 true \
  --verl_actor_compile "${VERL_ACTOR_COMPILE:-false}" \
  --verl_actor_param_offload "${VERL_ACTOR_PARAM_OFFLOAD:-true}" \
  --verl_actor_optimizer_offload "${VERL_ACTOR_OPTIMIZER_OFFLOAD:-true}" \
  --verl_automodel_distributed_strategy "${VERL_AUTOMODEL_DISTRIBUTED_STRATEGY:-fsdp2}" \
  --verl_automodel_tp_size "${VERL_AUTOMODEL_TP_SIZE:-1}" \
  --verl_automodel_cp_size "${VERL_AUTOMODEL_CP_SIZE:-1}" \
  --verl_automodel_activation_checkpointing "${VERL_AUTOMODEL_ACTIVATION_CHECKPOINTING:-true}" \
  --verl_automodel_model_dtype "${VERL_AUTOMODEL_MODEL_DTYPE:-bf16}" \
  --verl_automodel_attn_implementation "${VERL_AUTOMODEL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --verl_automodel_backend_attn "${VERL_AUTOMODEL_BACKEND_ATTN:-sdpa}" \
  --verl_automodel_backend_linear "${VERL_AUTOMODEL_BACKEND_LINEAR:-te}" \
  --verl_automodel_backend_rms_norm "${VERL_AUTOMODEL_BACKEND_RMS_NORM:-torch_fp32}" \
  --verl_rollout_tp "${VERL_ROLLOUT_TP:-2}" \
  --verl_teacher_tp "${VERL_TEACHER_TP:-2}" \
  --verl_teacher_dp "${VERL_TEACHER_DP:-1}" \
  --verl_rollout_quantization "${VERL_ROLLOUT_QUANTIZATION:-fp8}" \
  --verl_teacher_quantization "${VERL_TEACHER_QUANTIZATION:-fp8}" \
  --verl_rollout_gpu_memory_utilization "${VERL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}" \
  --verl_teacher_gpu_memory_utilization "${VERL_TEACHER_GPU_MEMORY_UTILIZATION:-0.40}" \
  --verl_rollout_max_num_seqs "${VERL_ROLLOUT_MAX_NUM_SEQS:-4}" \
  --verl_teacher_max_num_seqs "${VERL_TEACHER_MAX_NUM_SEQS:-4}" \
  --verl_rollout_max_num_batched_tokens "${VERL_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}" \
  --verl_teacher_max_num_batched_tokens "${VERL_TEACHER_MAX_NUM_BATCHED_TOKENS:-2048}" \
  --verl_rollout_temperature "${VERL_ROLLOUT_TEMPERATURE:-0.7}" \
  --verl_rollout_top_p "${VERL_ROLLOUT_TOP_P:-0.95}" \
  --opd_loss_mode "${OPD_LOSS_MODE:-k1}" \
  --opd_topk "${OPD_TOPK:-32}" \
  --opd_use_task_rewards false \
  --opd_use_policy_gradient true \
  --olmo3_sink true \
  --olmo3_sink_attn_implementation "${OLMO3_SINK_ATTN_IMPLEMENTATION:-flash_attention_3}" \
  --wandb_mode "${WANDB_MODE:-online}"
