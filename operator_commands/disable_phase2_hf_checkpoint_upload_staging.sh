#!/usr/bin/env bash
set -euo pipefail

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
if [ "${NODE_LABEL}" != "0" ]; then
  echo "skip staging disable on node${NODE_LABEL}"
  exit 0
fi

RUN_DIR="/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536"
STAGING="${RUN_DIR}/.hf_large_upload_staging"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "node=${NODE_LABEL} host=$(hostname)"
echo "disabling HF checkpoint upload staging: ${STAGING}"

if [ -d "${STAGING}" ]; then
  mv "${STAGING}" "${STAGING}.disabled_${STAMP}"
  echo "moved ${STAGING} -> ${STAGING}.disabled_${STAMP}"
fi

if [ ! -e "${STAGING}" ]; then
  printf 'HF checkpoint upload disabled externally at %s for command ec4e62\n' "$(date -u +%FT%TZ)" > "${STAGING}"
  echo "created blocker file ${STAGING}"
fi

ls -ld "${STAGING}" "${STAGING}".disabled_* 2>/dev/null || true

echo "trainer/operator process summary:"
ps -eo pid,ppid,pgid,stat,etime,cmd \
  | grep -E 'operator_mode|cmd_ec4e62|torchrun' \
  | grep -v grep \
  | head -n 40
