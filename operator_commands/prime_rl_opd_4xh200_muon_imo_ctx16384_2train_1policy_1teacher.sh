#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${OLMO_RUN_DIR_NAME:-prime_rl_opd_muon_imo_ctx16384_2train_1policy_1teacher_$(date -u +%Y%m%d_%H%M%S)}"
export OLMO_RUN_DIR_NAME="${RUN_NAME}"

MODEL_PATH="${PRIME_OPD_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy}"
TEACHER_MODEL_PATH="${PRIME_OPD_TEACHER_MODEL_PATH:-${MODEL_PATH}}"
DATASET_PATH="${PRIME_OPD_DATASET_PATH:-/workspace/submissions-instructions/imo_data_1959_2024.csv}"
CTX_LEN="${PRIME_OPD_CTX_LEN:-16384}"
COMPLETION_TOKENS="${PRIME_OPD_COMPLETION_TOKENS:-12288}"
BATCHED_TOKENS="${PRIME_OPD_BATCHED_TOKENS:-16384}"
TEACHER_GPU_MEMORY_UTILIZATION="${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
TEACHER_MAX_NUM_SEQS="${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-4}"
POLICY_GPU_MEMORY_UTILIZATION="${PRIME_OPD_POLICY_GPU_MEMORY_UTILIZATION:-0.95}"
POLICY_MAX_NUM_SEQS="${PRIME_OPD_POLICY_MAX_NUM_SEQS:-8}"

/usr/bin/python /app/train.py \
  --fetch-update \
  --submissions-ref main \
  --prime-rl-ref main \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir "/tmp/olmo-train-runtime-deps-prime-rl-opd-${RUN_NAME}" \
  --backend prime_rl \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path /vol/olmo_train_assets/output/prime_rl_opd_4x_manual \
  --logdir /vol/olmo_train_assets/logs/prime_rl_opd_4x_manual \
  --max_train_steps "${MAX_TRAIN_STEPS:-1}" \
  --max_seq_length "${CTX_LEN}" \
  --rollout_max_completion_tokens "${COMPLETION_TOKENS}" \
  --optimizer muon \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --prime_algorithm opd \
  --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
  --prime_opd_start_teacher true \
  --prime_opd_teacher_gpu_ids 3 \
  --prime_opd_teacher_port 8001 \
  --prime_opd_teacher_vllm_tensor_parallel_size 1 \
  --prime_opd_teacher_vllm_data_parallel_size 1 \
  --prime_opd_teacher_vllm_max_model_len "${CTX_LEN}" \
  --prime_opd_teacher_vllm_dtype bfloat16 \
  --prime_opd_teacher_vllm_enforce_eager false \
  --prime_opd_teacher_vllm_quantization fp8 \
  --prime_opd_teacher_vllm_gpu_memory_utilization "${TEACHER_GPU_MEMORY_UTILIZATION}" \
  --prime_opd_teacher_vllm_max_num_seqs "${TEACHER_MAX_NUM_SEQS}" \
  --prime_opd_teacher_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
  --prime_env_id deepseek-math-v2-env \
  --prime_env_name proof_math \
  --prime_proof_dataset_path "${DATASET_PATH}" \
  --prime_proof_problem_column auto \
  --prime_proof_solution_column auto \
  --prime_proof_judge_backend none \
  --prime_batch_size 2 \
  --prime_group_size 2 \
  --prime_max_inflight_rollouts 2 \
  --prime_train_gpus 2 \
  --prime_infer_gpus 1 \
  --prime_gpus_per_node 4 \
  --prime_trainer_model_impl custom \
  --prime_trainer_attn flash_attention_3 \
  --prime_trainer_context_parallel_size 2 \
  --prime_trainer_cp_style ulysses \
  --prime_trainer_fsdp_cpu_offload false \
  --prime_trainer_optim_cpu_offload true \
  --prime_trainer_fp8 true \
  --prime_vllm_tensor_parallel_size 1 \
  --prime_vllm_data_parallel_size 1 \
  --prime_vllm_max_model_len "${CTX_LEN}" \
  --prime_vllm_dtype bfloat16 \
  --prime_vllm_enforce_eager false \
  --prime_vllm_quantization fp8 \
  --prime_vllm_gpu_memory_utilization "${POLICY_GPU_MEMORY_UTILIZATION}" \
  --prime_vllm_max_num_seqs "${POLICY_MAX_NUM_SEQS}" \
  --prime_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
  --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true \
  --prime_temperature 0.7 \
  --prime_top_p 0.95 \
  --wandb_mode disabled
