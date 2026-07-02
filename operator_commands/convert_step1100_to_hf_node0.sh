#!/usr/bin/env bash
set -euo pipefail

HOST="$(hostname)"
NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
echo "CONVERT_STEP1100_HF host=${HOST} node=${NODE_LABEL}"

if [[ "${NODE_LABEL}" != "0" ]]; then
  echo "Skipping conversion on node ${NODE_LABEL}; node0 owns HF conversion."
  exit 0
fi

export HF_HOME=/tmp/olmo3_rl/hf_home
export XDG_CACHE_HOME=/tmp/olmo3_rl/xdg_cache
export TRANSFORMERS_CACHE=/tmp/olmo3_rl/hf_home/transformers
export HF_HUB_CACHE=/tmp/olmo3_rl/hf_home/hub

BASE=/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536
SRC="${BASE}/step1100"
OUT="${BASE}/.hf_converted_checkpoints/step1100-hf"
TMP="${OUT}.tmp.$$"
TOKENIZER=/tmp/olmo3_phase2/model/allenai-Olmo-3.1-32B-Think-ckpt-2k
CONVERTER=/tmp/OLMo-core-runtime/src/examples/huggingface/convert_checkpoint_to_hf.py

if [[ ! -d "${SRC}/model_and_optim" ]]; then
  echo "Missing native OLMo-core checkpoint: ${SRC}/model_and_optim" >&2
  exit 2
fi

if [[ ! -f "${CONVERTER}" ]]; then
  echo "Missing converter: ${CONVERTER}" >&2
  exit 3
fi

mkdir -p "$(dirname "${OUT}")" "${HF_HOME}" "${XDG_CACHE_HOME}"

if find "${OUT}" -maxdepth 1 \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \) -print -quit 2>/dev/null | grep -q .; then
  echo "HF checkpoint already exists: ${OUT}"
  find "${OUT}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
  exit 0
fi

echo "Disk before conversion:"
df -h /tmp "${BASE}" || true
du -sh "${SRC}" || true

rm -rf "${TMP}" "${OUT}.tmp."*
mkdir -p "${TMP}"

START_SECONDS="$(date +%s)"
python "${CONVERTER}" \
  --checkpoint-input-path "${SRC}" \
  --huggingface-output-dir "${TMP}" \
  --tokenizer "${TOKENIZER}" \
  --max-sequence-length 65536 \
  --device cpu \
  --skip-validation
END_SECONDS="$(date +%s)"
echo "Conversion wall_seconds=$((END_SECONDS - START_SECONDS))"

if ! find "${TMP}" -maxdepth 1 \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \) -print -quit | grep -q .; then
  echo "Conversion completed but no HF safetensors/index files were found in ${TMP}" >&2
  find "${TMP}" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort | tail -50
  exit 4
fi

rm -rf "${OUT}"
mv "${TMP}" "${OUT}"

echo "HF conversion ready: ${OUT}"
du -sh "${OUT}" || true
find "${OUT}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
