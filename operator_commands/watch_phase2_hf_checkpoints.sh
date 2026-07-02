NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
if [ "${NODE_LABEL}" != "0" ]; then
  echo "skip HF checkpoint watcher on node${NODE_LABEL}"
  exit 0
fi

set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

WORK_DIR="/tmp/submissions-instructions-hf-watcher-${OLMO_RUN_DIR_NAME:-phase2-watch}"
echo "starting phase2 HF checkpoint watcher node=${NODE_LABEL} work_dir=${WORK_DIR}"
rm -rf "${WORK_DIR}"
git clone --depth 1 https://github.com/nguyen599/submissions-instructions.git "${WORK_DIR}"
git -C "${WORK_DIR}" rev-parse HEAD

python -u "${WORK_DIR}/src/hf_checkpoint_watcher.py" \
  --scan-root /tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536 \
  --repo nguyen599/olmo3-ckpt-phase2 \
  --path-prefix checkpoints \
  --run-name phase2_32b_tp8_pp3_seq65536 \
  --folder-glob 'step*-hf' \
  --interval-seconds 200 \
  --stability-seconds 30 \
  --workers 20 \
  --min-safetensors 2 \
  --log-file /tmp/olmo3_phase2/logs/hf_checkpoint_watcher.log
