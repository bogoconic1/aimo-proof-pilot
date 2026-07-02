#!/usr/bin/env bash
set -euo pipefail

HOST="$(hostname)"
case "${HOST}" in
  hnode096) NODE_RANK=0 ;;
  hnode597) NODE_RANK=1 ;;
  hnode601) NODE_RANK=2 ;;
  *)
    NODE_RANK="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-0}}}"
    ;;
esac

MASTER_HOST="${MASTER_ADDR:-hnode096}"
PORT="${MASTER_PORT:-29612}"
RUN_NAME="rlcsd_verl_async_cispo_32b_1train_2rollout_imo4x8_100steps_fp8_phase3ckpt9"
MODEL_HF="${MODEL_HF:-/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf}"
DATASET="${DATASET:-/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_ibn1,mlx5_ibn2,mlx5_ibn3,mlx5_ibn4,mlx5_ibn5,mlx5_ibn6,mlx5_ibn7,mlx5_ibn8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "CISPO/verl async run host=${HOST} node_rank=${NODE_RANK} master=${MASTER_HOST}:${PORT}"
echo "dataset=${DATASET}"
echo "model=${MODEL_HF}"

python /app/train.py \
  --fetch-update \
  --submissions-ref main \
  --open-instruct-ref main \
  --olmo-core-ref main \
  --rlcsd-ref main \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir "/tmp/olmo-train-runtime-deps-${RUN_NAME}" \
  --backend verl_rlcsd \
  --rlcsd_method cispo \
  --rlcsd_async_rollout true \
  --model_path "${MODEL_HF}" \
  --dataset_path "${DATASET}" \
  --output_path "/tmp/olmo3_rl/outputs/${RUN_NAME}" \
  --logdir "/tmp/olmo3_rl/logs/${RUN_NAME}" \
  --cache_dir "/tmp/olmo3_rl/cache/${RUN_NAME}" \
  --num_gpus 8 \
  --num_nodes 3 \
  --world_size_mode nodes \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_HOST}" \
  --master_port "${PORT}" \
  --per_device_batch_size 4 \
  --max_train_steps 100 \
  --learning_rate 3e-7 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --checkpointing_steps 10 \
  --hf_checkpoint_upload true \
  --hf_checkpoint_repo nguyen599/olmo3-ckpt-phase3 \
  --hf_checkpoint_path_prefix checkpoints \
  --hf_checkpoint_upload_workers 8 \
  --hf_checkpoint_upload_report_interval_seconds 300 \
  --rlcsd_train_nodes 1 \
  --rlcsd_train_gpus_per_node 8 \
  --rlcsd_rollout_nodes 2 \
  --rlcsd_rollout_gpus_per_node 8 \
  --rlcsd_rollout_tensor_parallel_size 4 \
  --rlcsd_rollout_data_parallel_size 1 \
  --rlcsd_group_size 8 \
  --rlcsd_vllm_gpu_memory_utilization 0.80 \
  --rlcsd_vllm_quantization fp8 \
  --rlcsd_max_prompt_length 5536 \
  --rlcsd_max_response_length 60000 \
  --rlcsd_ppo_mini_batch_size 4 \
  --rlcsd_ppo_micro_batch_size_per_gpu 1 \
  --rlcsd_proof_reward_weight 0.76 \
  --rlcsd_self_eval_reward_weight 0.24 \
  --rlcsd_partial_format_score 0.7 \
  --rlcsd_clip_ratio_low 10 \
  --rlcsd_clip_ratio_high 0.2 \
  --rlcsd_async_trigger_parameter_sync_step 1 \
  --rlcsd_async_require_batches 1 \
  --rlcsd_async_partial_rollout true \
  --offline false \
  --with_tracking \
  --wandb_mode online \
  --wandb_project olmo3-32b-verl \
  --hf_log_upload false
