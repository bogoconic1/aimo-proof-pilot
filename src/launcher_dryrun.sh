#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GLOBAL_RANK:=0}"
: "${WORLD_SIZE:=1}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

export GLOBAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
unset RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME LOCAL_WORLD_SIZE

exec python "${SCRIPT_DIR}/train.py" \
  --no-fetch-update \
  --model_path "${MODEL_PATH:-/tmp/model}" \
  --dataset_path "${DATASET_PATH:-/tmp/data.jsonl}" \
  --output_path "${OUTPUT_PATH:-/tmp/olmo-train-out}" \
  --logdir "${LOGDIR:-/tmp/olmo-train-logs}" \
  --backend "${BACKEND:-olmo_core_sft}" \
  --model_arch "${MODEL_ARCH:-olmo3_32b}" \
  --num_gpus "${NUM_GPUS:-8}" \
  --num_nodes "${NUM_NODES:-0}" \
  --world_size_mode "${WORLD_SIZE_MODE:-nodes}" \
  --tensor_parallel_degree "${TP:-8}" \
  --context_parallel_degree "${CP:-1}" \
  --pipeline_parallel_degree "${PP:-3}" \
  --dry_run_launch \
  "$@"
