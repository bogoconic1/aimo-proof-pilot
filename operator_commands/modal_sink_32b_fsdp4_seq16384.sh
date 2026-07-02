#!/usr/bin/env bash
set -euo pipefail

# Manual Modal smoke for OLMo3 sink 32B on 4xH200 with FSDP2.
# Run inside the Modal training image:
#   bash /workspace/submissions-instructions/operator_commands/modal_sink_32b_fsdp4_seq16384.sh
#
# Override DATASET_PATH to use the full dataset. The default smoke dataset keeps
# the run cheap and checks conversion, attention sinks, FSDP2, compile, and the
# TE fused optimizer path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/modal_sink_32b_fsdp2_seq8192.sh}"

export DATASET_PATH="${DATASET_PATH:-/vol/olmo_train_assets/train_phase2_smoke_128.parquet}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10}"
export RUN_NAME="${RUN_NAME:-modal_sink_32b_fsdp4_seq16384_fullac_te_trainpy_smoke1}"

export NUM_GPUS="${NUM_GPUS:-4}"
export TP="${TP:-1}"
export CP="${CP:-1}"
export PP="${PP:-1}"

export ACTIVATION_CHECKPOINTING_MODE="${ACTIVATION_CHECKPOINTING_MODE:-full}"
export FORCE_COMPILE_MODEL="${FORCE_COMPILE_MODEL:-true}"
export TENSOR_PARALLEL_ASYNC="${TENSOR_PARALLEL_ASYNC:-false}"

export OPTIMIZER="${OPTIMIZER:-te_fused_adamw}"

exec bash "${BASE_SCRIPT}"
