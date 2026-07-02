#!/usr/bin/env bash
set -euo pipefail

# Manual Modal 4xH200 Prime-RL run for OLMo3Sink proof RL.
# Run inside the Modal training image:
#   bash /workspace/submissions-instructions/operator_commands/modal_prime_rl_4x_h200_olmo3sink_seq81920_seqs16_bt16384.sh
#
# The defaults match the verified launch with one vLLM rollout GPU and three
# trainer GPUs. Override paths or run shape through environment variables.

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-prime_rl_4x_olmo3sink_seq81920_seqs16_bt16384_${STAMP}}"

MODEL_PATH="${MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-v33-s150/opd-32b-v33-s150}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/olmo_train_assets/output/prime_rl_4x}"
LOG_ROOT="${LOG_ROOT:-/vol/olmo_train_assets/logs/prime_rl_4x}"

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
  --dataset_path "${DATASET_PATH}" \
  --prime_proof_dataset_path "${DATASET_PATH}" \
  --output_path "${OUTPUT_ROOT}" \
  --logdir "${LOG_ROOT}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-1}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-81920}" \
  --rollout_max_completion_tokens "${ROLLOUT_MAX_COMPLETION_TOKENS:-65000}" \
  --prime_batch_size "${PRIME_BATCH_SIZE:-16}" \
  --prime_group_size "${PRIME_GROUP_SIZE:-8}" \
  --prime_max_inflight_rollouts "${PRIME_MAX_INFLIGHT_ROLLOUTS:-16}" \
  --prime_disable_zero_advantage_filter true \
  --prime_train_gpus "${PRIME_TRAIN_GPUS:-3}" \
  --prime_infer_gpus "${PRIME_INFER_GPUS:-1}" \
  --prime_gpus_per_node "${PRIME_GPUS_PER_NODE:-4}" \
  --prime_trainer_model_impl auto \
  --prime_trainer_attn flash_attention_3 \
  --prime_trainer_fsdp_cpu_offload false \
  --prime_trainer_optim_cpu_offload true \
  --prime_trainer_fp8 false \
  --prime_vllm_tensor_parallel_size "${PRIME_VLLM_TP:-1}" \
  --prime_vllm_data_parallel_size "${PRIME_VLLM_DP:-1}" \
  --prime_vllm_max_model_len "${PRIME_VLLM_MAX_MODEL_LEN:-81920}" \
  --prime_vllm_dtype bfloat16 \
  --prime_vllm_quantization fp8 \
  --prime_vllm_gpu_memory_utilization "${PRIME_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --prime_vllm_max_num_seqs "${PRIME_VLLM_MAX_NUM_SEQS:-16}" \
  --prime_vllm_max_num_batched_tokens "${PRIME_VLLM_MAX_NUM_BATCHED_TOKENS:-16384}" \
  --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true \
  --prime_temperature "${PRIME_TEMPERATURE:-0.7}" \
  --prime_top_p "${PRIME_TOP_P:-0.95}" \
  --prime_proof_judge_backend api \
  --prime_proof_judge_base_url "${PRIME_PROOF_JUDGE_BASE_URL:-http://localhost:8000/v1}" \
  --prime_proof_judge_api_key "${PRIME_PROOF_JUDGE_API_KEY:-dummy}" \
  --prime_proof_judge_model "${PRIME_PROOF_JUDGE_MODEL:-${MODEL_PATH}}" \
  --prime_proof_judge_temperature "${PRIME_PROOF_JUDGE_TEMPERATURE:-0.7}" \
  --prime_proof_judge_top_p "${PRIME_PROOF_JUDGE_TOP_P:-0.95}" \
  --prime_proof_judge_max_tokens "${PRIME_PROOF_JUDGE_MAX_TOKENS:-40000}" \
  --prime_proof_judge_max_context_length "${PRIME_PROOF_JUDGE_MAX_CONTEXT_LENGTH:-40000}" \
  --prime_proof_require_format false \
  --prime_proof_partial_format_score "${PRIME_PROOF_PARTIAL_FORMAT_SCORE:-0.7}" \
  --prime_proof_enable_meta_verification true \
  --prime_proof_refine_rounds "${PRIME_PROOF_REFINE_ROUNDS:-1}" \
  --prime_proof_refine_review_n "${PRIME_PROOF_REFINE_REVIEW_N:-1}" \
  --prime_proof_max_examples "${PRIME_PROOF_MAX_EXAMPLES:-128}" \
  --wandb_mode "${WANDB_MODE:-disabled}"
