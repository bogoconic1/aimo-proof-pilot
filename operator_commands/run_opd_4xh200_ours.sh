#!/bin/bash
# Run Nguyen's reference single-node OPD experiment on OUR baked image (chankhavu/aimo-opd-sft:v2).
# 4x H200 on ONE node: GPU0=policy rollout, GPU1-2=trainer (FSDP+CP), GPU3=teacher.
# Differences from operator_commands/prime_rl_opd_4xh200_muon_...sh (Nguyen's):
#   * runs OUR venv (/app/.venv) + the git-pulled train.py, not /usr/bin/python /app/train.py
#   * --no-fetch-update --no-ensure-runtime-training-deps -> uses our BAKED, worker-bug-free prime-rl
#     instead of re-fetching+installing nguyen599/prime-rl@main (the buggy 6ee9a5dc)
# Run inside a relay shell:  bash run_opd_4xh200_ours.sh   (or launch detached: see bottom)
set -euo pipefail

# ============ FILL THESE IN (your staged assets) ============
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to your local 32B checkpoint dir}"   # e.g. /workspace/opd-32b-deploy
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your proof CSV}"             # e.g. /workspace/imo_data_1959_2024.csv
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/opd_out}"                                 # checkpoints+logs (use a MOUNTED volume)
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-rl}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ============ 1. pull the training code (train.py + proof_opd_env) + set PYTHONPATH ============
eval "$(opd-env-sync)"
REPO=/tmp/imochallenge/opd-env/aimo-proof-pilot
# astralbench.csv (the verifiable set) ships inside proof_opd_env; override VERIFIABLE_DATASET_PATH if you want
VERIFIABLE_DATASET_PATH="${VERIFIABLE_DATASET_PATH:-$REPO/src/proof_opd_env/astralbench.csv}"

# ============ 2. sanity checks ============
command -v nvidia-smi >/dev/null && nvidia-smi -L | sed 's/^/[gpu] /'
[ -e "$MODEL_PATH" ]              || { echo "MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
[ -e "$DATASET_PATH" ]           || { echo "DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
[ -e "$VERIFIABLE_DATASET_PATH" ]|| { echo "VERIFIABLE not found: $VERIFIABLE_DATASET_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ============ 3. run (baked prime-rl; NO runtime fetch/install) ============
exec /app/.venv/bin/python "$REPO/src/train.py" \
  --no-fetch-update --no-ensure-runtime-training-deps \
  --backend prime_rl \
  --model_path "$MODEL_PATH" --tokenizer_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_DIR/out" --logdir "$OUTPUT_DIR/logs" \
  --max_train_steps "${MAX_TRAIN_STEPS:-30}" \
  --max_seq_length "${CTX_LEN:-20480}" \
  --rollout_max_completion_tokens "${COMPLETION_TOKENS:-20480}" \
  --optimizer muon --learning_rate 1e-6 --weight_decay 0.0 --max_grad_norm 1.0 \
  --prime_algorithm opd \
  --prime_opd_teacher_model "${TEACHER_MODEL_PATH:-$MODEL_PATH}" \
  --prime_opd_start_teacher true --prime_opd_teacher_gpu_ids 3 --prime_opd_teacher_port 8001 \
  --prime_opd_teacher_vllm_tensor_parallel_size 1 --prime_opd_teacher_vllm_data_parallel_size 1 \
  --prime_opd_teacher_vllm_max_model_len "${VLLM_CTX_LEN:-40960}" \
  --prime_opd_teacher_vllm_dtype bfloat16 --prime_opd_teacher_vllm_enforce_eager false \
  --prime_opd_teacher_vllm_quantization fp8 \
  --prime_opd_teacher_vllm_gpu_memory_utilization "${TEACHER_GPU_MEM:-0.85}" \
  --prime_opd_teacher_vllm_max_num_seqs 8 --prime_opd_teacher_vllm_max_num_batched_tokens 16384 \
  --prime_env_id proof-opd-env --prime_env_name proof_math \
  --prime_proof_dataset_path "$DATASET_PATH" \
  --prime_proof_verifiable_dataset_path "$VERIFIABLE_DATASET_PATH" \
  --prime_proof_verifiable_fraction 0.20 --prime_proof_verifiable_answer_column auto \
  --prime_proof_mix_seed 34521 --prime_proof_problem_column auto --prime_proof_solution_column auto \
  --prime_proof_judge_backend none --prime_proof_max_examples "${PROOF_MAX_EXAMPLES:-20}" \
  --prime_batch_size 2 --prime_group_size 2 --prime_max_inflight_rollouts 8 \
  --prime_train_gpus 2 --prime_infer_gpus 1 --prime_gpus_per_node 4 \
  --prime_trainer_model_impl custom --prime_trainer_attn olmo3_sink_fa3 \
  --prime_trainer_context_parallel_size 2 --prime_trainer_cp_style ulysses \
  --prime_trainer_fsdp_cpu_offload false --prime_trainer_optim_cpu_offload false \
  --prime_trainer_fp8 true \
  --prime_vllm_tensor_parallel_size 1 --prime_vllm_data_parallel_size 1 \
  --prime_vllm_max_model_len "${VLLM_CTX_LEN:-40960}" --prime_vllm_dtype bfloat16 \
  --prime_vllm_enforce_eager false --prime_vllm_quantization fp8 \
  --prime_vllm_gpu_memory_utilization "${POLICY_GPU_MEM:-0.95}" --prime_vllm_max_num_seqs 16 \
  --prime_vllm_max_num_batched_tokens 16384 --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true --prime_temperature 0.7 --prime_top_p 0.95 \
  --with_tracking --wandb_mode "$WANDB_MODE" --wandb_project "$WANDB_PROJECT"
