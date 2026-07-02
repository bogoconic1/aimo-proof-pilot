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
PORT="${MASTER_PORT:-29596}"
RUN_NAME="rlcsd_verl_async_32b_1train_2rollout_smoke"
MODEL_HF="${MODEL_HF:-/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf}"
DATASET="${DATASET:-/tmp/submissions-instructions-runtime/proofbench_v2.csv}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_ibn1,mlx5_ibn2,mlx5_ibn3,mlx5_ibn4,mlx5_ibn5,mlx5_ibn6,mlx5_ibn7,mlx5_ibn8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=1

echo "RLCSd/verl async smoke host=${HOST} node_rank=${NODE_RANK} master=${MASTER_HOST}:${PORT}"

python /app/train.py \
  --fetch-update \
  --submissions-ref main \
  --backend verl_rlcsd \
  --rlcsd_method grpo \
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
  --per_device_batch_size 1 \
  --max_train_steps 2 \
  --learning_rate 5e-7 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --checkpointing_steps 2 \
  --rlcsd_train_nodes 1 \
  --rlcsd_train_gpus_per_node 8 \
  --rlcsd_rollout_nodes 2 \
  --rlcsd_rollout_gpus_per_node 8 \
  --rlcsd_rollout_tensor_parallel_size 4 \
  --rlcsd_rollout_data_parallel_size 1 \
  --rlcsd_group_size 8 \
  --rlcsd_vllm_gpu_memory_utilization 0.80 \
  --rlcsd_max_prompt_length 5536 \
  --rlcsd_max_response_length 60000 \
  --rlcsd_ppo_mini_batch_size 1 \
  --rlcsd_ppo_micro_batch_size_per_gpu 1 \
  --rlcsd_proof_reward_weight 0.76 \
  --rlcsd_self_eval_reward_weight 0.24 \
  --rlcsd_partial_format_score 0.7 \
  --rlcsd_async_trigger_parameter_sync_step 1 \
  --rlcsd_async_require_batches 1 \
  --rlcsd_async_partial_rollout true \
  --offline false \
  --with_tracking \
  --wandb_mode online \
  --wandb_project olmo3-32b-rlcsd \
  --hf_log_upload false
