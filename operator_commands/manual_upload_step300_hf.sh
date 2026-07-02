#!/usr/bin/env bash
set -euo pipefail

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
if [ "${NODE_LABEL}" != "0" ]; then
  echo "skip node${NODE_LABEL}"
  exit 0
fi

python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

folder = Path(
    "/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/"
    "phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step300-hf"
)
repo = "nguyen599/olmo3-ckpt-phase2"
path_in_repo = Path("checkpoints/phase2_32b_tp8_pp3_seq65536/step300-hf")
staging = Path("/tmp/olmo3_phase2/manual_hf_upload_staging/step300_hf")

print(f"manual_step300_hf_upload folder_exists={folder.is_dir()} folder={folder}")
if not folder.is_dir():
    raise SystemExit(2)

if staging.exists():
    shutil.rmtree(staging)
stage_root = staging / path_in_repo
file_count = 0
for src in folder.rglob("*"):
    if not src.is_file():
        continue
    rel = src.relative_to(folder)
    dst = stage_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        os.link(src, dst)
    file_count += 1

print(f"staged={staging} files={file_count}")
api = HfApi()
api.upload_large_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path=str(staging),
    private=True,
    num_workers=20,
    print_report=False,
)
print(f"uploaded https://huggingface.co/datasets/{repo}/tree/main/{path_in_repo.as_posix()}")
shutil.rmtree(staging, ignore_errors=True)
PY
