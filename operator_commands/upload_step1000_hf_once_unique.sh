#!/usr/bin/env bash
set -euo pipefail

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
echo "upload_step1000_hf_once_unique node=${NODE_LABEL} run_dir=${OLMO_RUN_DIR_NAME:-none}"
if [ "${NODE_LABEL}" != "0" ]; then
  echo "skip step1000-hf upload on node${NODE_LABEL}"
  exit 0
fi

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

WORK_DIR="/tmp/submissions-instructions-hf-watcher-${OLMO_RUN_DIR_NAME:-step1000-once}"
echo "using work dir ${WORK_DIR}"
rm -rf "${WORK_DIR}"
git clone --depth 1 https://github.com/nguyen599/submissions-instructions.git "${WORK_DIR}"
git -C "${WORK_DIR}" rev-parse HEAD

echo "starting step1000-hf upload"
python -u "${WORK_DIR}/src/hf_checkpoint_watcher.py" \
  --scan-root /tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536 \
  --repo nguyen599/olmo3-ckpt-phase2 \
  --path-prefix checkpoints \
  --run-name phase2_32b_tp8_pp3_seq65536 \
  --folder-glob 'step1000-hf' \
  --interval-seconds 200 \
  --stability-seconds 10 \
  --heartbeat-seconds 60 \
  --workers 20 \
  --min-safetensors 2 \
  --once \
  --force \
  --log-file /tmp/olmo3_phase2/logs/hf_checkpoint_watcher_step1000_unique.log
echo "step1000-hf upload command finished"
