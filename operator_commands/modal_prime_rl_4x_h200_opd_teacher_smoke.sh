#!/usr/bin/env bash
set -euo pipefail

# Manual Modal 4xH200 Prime-RL OPD smoke.
# Layout:
#   GPU 0: live policy vLLM rollout
#   GPU 1-2: trainer
#   GPU 3: frozen OPD teacher vLLM
#
# Run inside the Modal training image:
#   bash /workspace/submissions-instructions/operator_commands/modal_prime_rl_4x_h200_opd_teacher_smoke.sh

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-prime_rl_4x_opd_teacher_smoke_${STAMP}}"

MODEL_PATH="${MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-v33-s150/opd-32b-v33-s150}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/olmo_train_assets/output/prime_rl_opd_4x}"
LOG_ROOT="${LOG_ROOT:-/vol/olmo_train_assets/logs/prime_rl_opd_4x}"

export OLMO_RUN_DIR_NAME="${OLMO_RUN_DIR_NAME:-${RUN_NAME}}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export PRIME_RL_OLMO3_SINK="${PRIME_RL_OLMO3_SINK:-1}"

exec /usr/bin/python /app/train.py \
  --fetch-update \
  --submissions-ref "${SUBMISSIONS_REF:-main}" \
  --prime-rl-ref "${PRIME_RL_REF:-main}" \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir "/tmp/olmo-train-runtime-deps-${RUN_NAME}" \
  --backend prime_rl \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --output_path "${OUTPUT_ROOT}" \
  --logdir "${LOG_ROOT}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-1}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-8192}" \
  --rollout_max_completion_tokens "${ROLLOUT_MAX_COMPLETION_TOKENS:-1024}" \
  --prime_algorithm opd \
  --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
  --prime_opd_start_teacher true \
  --prime_opd_teacher_gpu_ids "${PRIME_OPD_TEACHER_GPU_IDS:-3}" \
  --prime_opd_teacher_port "${PRIME_OPD_TEACHER_PORT:-8001}" \
  --prime_opd_teacher_vllm_tensor_parallel_size "${PRIME_OPD_TEACHER_VLLM_TP:-1}" \
  --prime_opd_teacher_vllm_data_parallel_size "${PRIME_OPD_TEACHER_VLLM_DP:-1}" \
  --prime_opd_teacher_vllm_max_model_len "${PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN:-8192}" \
  --prime_opd_teacher_vllm_dtype bfloat16 \
  --prime_opd_teacher_vllm_quantization "${PRIME_OPD_TEACHER_VLLM_QUANTIZATION:-fp8}" \
  --prime_opd_teacher_vllm_gpu_memory_utilization "${PRIME_OPD_TEACHER_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --prime_opd_teacher_vllm_max_num_seqs "${PRIME_OPD_TEACHER_VLLM_MAX_NUM_SEQS:-8}" \
  --prime_opd_teacher_vllm_max_num_batched_tokens "${PRIME_OPD_TEACHER_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --prime_batch_size "${PRIME_BATCH_SIZE:-2}" \
  --prime_group_size "${PRIME_GROUP_SIZE:-2}" \
  --prime_max_inflight_rollouts "${PRIME_MAX_INFLIGHT_ROLLOUTS:-4}" \
  --prime_train_gpus "${PRIME_TRAIN_GPUS:-2}" \
  --prime_infer_gpus "${PRIME_INFER_GPUS:-1}" \
  --prime_gpus_per_node "${PRIME_GPUS_PER_NODE:-4}" \
  --prime_trainer_model_impl auto \
  --prime_trainer_attn flash_attention_3 \
  --prime_trainer_fsdp_cpu_offload false \
  --prime_trainer_optim_cpu_offload true \
  --prime_trainer_fp8 false \
  --prime_vllm_tensor_parallel_size "${PRIME_VLLM_TP:-1}" \
  --prime_vllm_data_parallel_size "${PRIME_VLLM_DP:-1}" \
  --prime_vllm_max_model_len "${PRIME_VLLM_MAX_MODEL_LEN:-8192}" \
  --prime_vllm_dtype bfloat16 \
  --prime_vllm_quantization "${PRIME_VLLM_QUANTIZATION:-fp8}" \
  --prime_vllm_gpu_memory_utilization "${PRIME_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --prime_vllm_max_num_seqs "${PRIME_VLLM_MAX_NUM_SEQS:-8}" \
  --prime_vllm_max_num_batched_tokens "${PRIME_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true \
  --prime_temperature "${PRIME_TEMPERATURE:-0.7}" \
  --prime_top_p "${PRIME_TOP_P:-0.95}" \
  --wandb_mode "${WANDB_MODE:-disabled}"
