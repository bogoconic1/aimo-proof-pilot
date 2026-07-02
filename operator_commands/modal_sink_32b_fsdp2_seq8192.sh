#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/workspace/submissions-instructions/src:/opt/OLMo-core/src:/opt/open-instruct:${PYTHONPATH:-}"
export OPEN_INSTRUCT_DIR="${OPEN_INSTRUCT_DIR:-/opt/open-instruct}"
export OLMO_CORE_DIR="${OLMO_CORE_DIR:-/opt/OLMo-core}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/vol/olmo_train_assets/cache/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/vol/olmo_train_assets/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/vol/olmo_train_assets/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/vol/olmo_train_assets/cache/triton}"

MODEL_PATH="${MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-v33-s150/opd-32b-v33-s150}"
DATASET_PATH="${DATASET_PATH:-/vol/olmo_train_assets/train_phase2.parquet}"
ASSET_ROOT="${ASSET_ROOT:-/vol/olmo_train_assets}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
TP="${TP:-1}"
CP="${CP:-1}"
PP="${PP:-1}"
MASTER_PORT="${MASTER_PORT:-29631}"
RUN_NAME="${RUN_NAME:-modal_sink_32b_fsdp${NUM_GPUS}_seq${MAX_SEQ_LENGTH}_steps${MAX_TRAIN_STEPS}}"
TRAIN_ENTRY="${TRAIN_ENTRY:-/workspace/submissions-instructions/src/train.py}"
SUBMISSIONS_REF="${SUBMISSIONS_REF:-main}"
RUNTIME_FETCH_STATE_DIR="${RUNTIME_FETCH_STATE_DIR:-/tmp/train-runtime-fetch-${RUN_NAME}}"
DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-true}"
OPTIMIZER="${OPTIMIZER:-te_fused_adamw}"
ADAMW_8BIT_BLOCK_SIZE="${ADAMW_8BIT_BLOCK_SIZE:-64}"
FORCE_COMPILE_MODEL="${FORCE_COMPILE_MODEL:-true}"
ACTIVATION_CHECKPOINTING_MODE="${ACTIVATION_CHECKPOINTING_MODE:-selected_ops}"
TENSOR_PARALLEL_ASYNC="${TENSOR_PARALLEL_ASYNC:-true}"

if [ "${TP}" = "1" ] && [ "${TENSOR_PARALLEL_ASYNC}" = "true" ]; then
  echo "TENSOR_PARALLEL_ASYNC=true requires TP>1 in train_engine.py; using false for TP=1 FSDP/HSDP smoke."
  TENSOR_PARALLEL_ASYNC="false"
fi

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}"

exec python "${TRAIN_ENTRY}" \
  --fetch-update \
  --runtime-fetch-state-dir "${RUNTIME_FETCH_STATE_DIR}" \
  --submissions-ref "${SUBMISSIONS_REF}" \
  --backend olmo_core_sft \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --chat_template_model "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${ASSET_ROOT}/debug/${RUN_NAME}/outputs" \
  --logdir "${ASSET_ROOT}/debug/${RUN_NAME}/logs" \
  --cache_dir "${ASSET_ROOT}/cache/${RUN_NAME}/runtime" \
  --olmo_core_checkpoint_cache "${ASSET_ROOT}/cache/${RUN_NAME}/olmo_core_checkpoint" \
  --olmo_core_dataset_cache "${ASSET_ROOT}/cache/${RUN_NAME}/olmo_core_dataset" \
  --num_gpus "${NUM_GPUS}" \
  --num_nodes 1 \
  --world_size_mode nodes \
  --master_port "${MASTER_PORT}" \
  --model_arch olmo3_32b \
  --tensor_parallel_degree "${TP}" \
  --tensor_parallel_async "${TENSOR_PARALLEL_ASYNC}" \
  --context_parallel_degree 1 \
  --pipeline_parallel_degree "${PP}" \
  --pipeline_schedule 1F1B \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --per_device_batch_size 1 \
  --rank_microbatch_size_sequences 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --attn_implementation flash_3 \
  --attention_sink true \
  --attention_sink_init_value 0.0 \
  --optimizer "${OPTIMIZER}" \
  --optimizer_state_dtype "auto" \
  --lm_loss_implementation fused_linear \
  --activation_memory_budget 0 \
  --activation_checkpointing_mode "${ACTIVATION_CHECKPOINTING_MODE}" \
  --compile_model true \
  --force_compile_model true \
  --float8 false \
  --checkpointing_steps 500 \
  --ephemeral_save_interval 0 \
  --checkpoint_keep_last 1 \
  --logging_steps 1 \
  --dataset_messages_mode auto \
  --dataset_transform_profile olmo \
  --dataset_backend auto \
  --dataset_num_proc 8 \
  --dataset_map_batch_size 512 \
  --data_loader_num_workers 4 \
  --data_loader_prefetch_factor 2 \
  --adamw_8bit_block_size "${ADAMW_8BIT_BLOCK_SIZE}" \
  --offline false \
  --wandb_mode online
