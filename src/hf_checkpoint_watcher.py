#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from train_logging import (
    hf_log_token,
    hf_transfer_heartbeat,
    quiet_hf_transfer,
    retry_hf_operation,
    suppress_noisy_third_party_loggers,
)
from train_utils import sanitize_slug_part


DEFAULT_REPO_ID = "nguyen599/olmo3-ckpt-phase2"
DEFAULT_INTERVAL_SECONDS = 20 * 60
DEFAULT_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXCLUDED_SCAN_DIRS = {
    ".git",
    ".hf_checkpoint_watcher_state",
    ".hf_large_upload_staging",
    ".hf_upload_watcher_staging",
    ".locks",
    "__pycache__",
    "hf_dataset_cache",
    "model_and_optim",
    "multinode_prepare_markers",
}


@dataclass(frozen=True)
class Candidate:
    folder: Path
    path_in_repo: str


@dataclass(frozen=True)
class FolderInspection:
    ok: bool
    reason: str
    safetensors: tuple[str, ...]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Periodically scan converted HF checkpoint folders and upload complete "
            "step*-hf directories to a Hugging Face dataset repo."
        )
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        required=True,
        help="Directory to scan. Repeat to scan multiple output roots.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help="HF dataset repo to upload into.")
    parser.add_argument("--path-prefix", default="checkpoints", help="Repo prefix before run_name/stepXXX-hf.")
    parser.add_argument(
        "--run-name",
        default="auto",
        help=(
            "Run folder name inside the repo. Use 'auto' to infer it from the parent of "
            ".hf_converted_checkpoints."
        ),
    )
    parser.add_argument("--folder-glob", default="step*-hf", help="Candidate folder-name glob.")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--stability-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--min-safetensors", type=int, default=2)
    parser.add_argument(
        "--required-file",
        action="append",
        dest="required_files",
        default=None,
        help="Required file inside each HF folder. Repeat to override defaults.",
    )
    parser.add_argument("--state-dir", default=None, help="Local upload state directory.")
    parser.add_argument("--primary-only", type=parse_bool, default=True)
    parser.add_argument("--check-remote", type=parse_bool, default=True)
    parser.add_argument("--force", action="store_true", help="Upload even when local/remote state says done.")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Only report candidates; do not upload.")
    parser.add_argument("--log-file", default=None, help="Optional file to append watcher logs.")
    return parser.parse_args(argv)


def configure_logging(log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(funcName)s() %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    suppress_noisy_third_party_loggers()


def node_rank_label() -> str | None:
    for name in ("GROUP_RANK", "GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value not in {None, ""}:
            return value
    if os.environ.get("LOCAL_RANK") in {None, ""}:
        value = os.environ.get("RANK")
        if value not in {None, ""}:
            return value
    return None


def primary_process() -> bool:
    label = node_rank_label()
    return label in {None, "", "0"}


def state_dir_for(args: argparse.Namespace) -> Path:
    if args.state_dir:
        return Path(args.state_dir).expanduser().resolve()
    first_root = Path(args.scan_root[0]).expanduser().resolve()
    return first_root / ".hf_checkpoint_watcher_state"


def remote_path_for(folder: Path, scan_root: Path, args: argparse.Namespace) -> str:
    if args.run_name != "auto":
        run_name = sanitize_slug_part(args.run_name)
    else:
        run_name = ""
        for parent in folder.parents:
            if parent.name == ".hf_converted_checkpoints":
                run_name = sanitize_slug_part(parent.parent.name)
                break
        if not run_name:
            try:
                relative_parts = folder.relative_to(scan_root).parts
                if len(relative_parts) > 1:
                    run_name = sanitize_slug_part(relative_parts[-2])
            except ValueError:
                pass
        if not run_name:
            run_name = sanitize_slug_part(scan_root.name)

    parts = [part.strip("/") for part in (args.path_prefix, run_name, folder.name) if part and part.strip("/")]
    return "/".join(parts)


def iter_candidate_folders(scan_root: Path, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not scan_root.is_dir():
        logging.warning("Scan root does not exist or is not a directory: %s", scan_root)
        return candidates

    for dirpath, dirnames, _filenames in os.walk(scan_root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in EXCLUDED_SCAN_DIRS and not dirname.endswith(".tmp")
        ]
        folder = Path(dirpath)
        if not fnmatch.fnmatch(folder.name, args.folder_glob):
            continue
        candidates.append(Candidate(folder=folder, path_in_repo=remote_path_for(folder, scan_root, args)))
    return sorted(candidates, key=lambda candidate: candidate.folder.as_posix())


def inspect_hf_folder(folder: Path, required_files: tuple[str, ...], min_safetensors: int) -> FolderInspection:
    if not folder.is_dir():
        return FolderInspection(False, "not a directory", ())

    missing = [name for name in required_files if not (folder / name).is_file()]
    if missing:
        return FolderInspection(False, f"missing required files: {', '.join(missing)}", ())

    empty_required = [name for name in required_files if (folder / name).stat().st_size <= 0]
    if empty_required:
        return FolderInspection(False, f"required files are empty: {', '.join(empty_required)}", ())

    safetensors = tuple(sorted(path.name for path in folder.glob("*.safetensors") if path.is_file()))
    if len(safetensors) < min_safetensors:
        return FolderInspection(
            False,
            f"found {len(safetensors)} safetensors files, need at least {min_safetensors}",
            safetensors,
        )

    index_path = folder / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return FolderInspection(False, f"could not parse model.safetensors.index.json: {exc}", safetensors)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return FolderInspection(False, "model.safetensors.index.json has no weight_map", safetensors)

    expected_shards = sorted({str(value) for value in weight_map.values()})
    missing_shards = [name for name in expected_shards if not (folder / name).is_file()]
    if missing_shards:
        return FolderInspection(False, f"missing shards from index: {', '.join(missing_shards[:8])}", safetensors)

    empty_shards = [name for name in expected_shards if (folder / name).stat().st_size <= 0]
    if empty_shards:
        return FolderInspection(False, f"empty shards from index: {', '.join(empty_shards[:8])}", safetensors)

    return FolderInspection(True, f"complete with {len(safetensors)} safetensors files", safetensors)


def folder_snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[path.relative_to(folder).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def folder_is_stable(folder: Path, wait_seconds: float) -> bool:
    if wait_seconds <= 0:
        return True
    before = folder_snapshot(folder)
    time.sleep(wait_seconds)
    after = folder_snapshot(folder)
    return before == after


def remote_folder_complete(
    repo_files: set[str],
    path_in_repo: str,
    required_files: tuple[str, ...],
    min_safetensors: int,
) -> bool:
    prefix = path_in_repo.rstrip("/") + "/"
    names = {file_path[len(prefix) :] for file_path in repo_files if file_path.startswith(prefix)}
    if not names:
        return False
    if any(required not in names for required in required_files):
        return False
    safetensors = [name for name in names if "/" not in name and name.endswith(".safetensors")]
    return len(safetensors) >= min_safetensors


def state_key(path_in_repo: str) -> str:
    return hashlib.sha256(path_in_repo.encode("utf-8")).hexdigest()[:16]


def state_file_for(state_dir: Path, path_in_repo: str) -> Path:
    return state_dir / f"{state_key(path_in_repo)}.json"


def read_uploaded_state(state_dir: Path, path_in_repo: str) -> bool:
    state_path = state_file_for(state_dir, path_in_repo)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return state.get("status") == "uploaded" and state.get("path_in_repo") == path_in_repo


def write_uploaded_state(state_dir: Path, candidate: Candidate, safetensors: tuple[str, ...]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "uploaded",
        "folder": str(candidate.folder),
        "path_in_repo": candidate.path_in_repo,
        "safetensors": list(safetensors),
        "uploaded_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_file_for(state_dir, candidate.path_in_repo).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_upload_staging(folder: Path, path_in_repo: str) -> Path:
    source = folder.expanduser().resolve()
    stage_key = hashlib.sha256(f"{source}:{path_in_repo}".encode("utf-8")).hexdigest()[:16]
    stage_root = source.parent / ".hf_upload_watcher_staging" / stage_key
    staged_folder = stage_root / Path(path_in_repo)
    shutil.rmtree(staged_folder, ignore_errors=True)
    staged_folder.mkdir(parents=True, exist_ok=True)

    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source)
        staged_path = staged_folder / relative_path
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_path.is_symlink() or staged_path.exists():
            try:
                if staged_path.resolve() == source_path.resolve():
                    continue
            except FileNotFoundError:
                pass
            staged_path.unlink()
        try:
            staged_path.symlink_to(source_path)
        except OSError:
            os.link(source_path, staged_path)
    return stage_root


def upload_candidate(args: argparse.Namespace, candidate: Candidate, token: str | None) -> bool:
    from huggingface_hub import HfApi

    stage_root = prepare_upload_staging(candidate.folder, candidate.path_in_repo)

    def upload_once():
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True, token=token)
        logging.info(
            "Uploading HF checkpoint folder to %s/%s: %s (staging=%s workers=%s)",
            args.repo,
            candidate.path_in_repo,
            candidate.folder,
            stage_root,
            args.workers or "auto",
        )
        with quiet_hf_transfer(), hf_transfer_heartbeat(
            f"HF watcher upload to {args.repo}/{candidate.path_in_repo}",
            args.heartbeat_seconds,
        ):
            return api.upload_large_folder(
                repo_id=args.repo,
                repo_type="dataset",
                folder_path=str(stage_root),
                private=True,
                ignore_patterns=["**/__pycache__/**", "**/.nfs*"],
                num_workers=args.workers or None,
                print_report=False,
            )

    try:
        retry_hf_operation(f"HF watcher upload to {args.repo}/{candidate.path_in_repo}", upload_once)
    except Exception:
        logging.exception(
            "Upload failed for %s. Staging left for resumable retry: %s",
            candidate.folder,
            stage_root,
        )
        return False

    logging.info(
        "Uploaded HF checkpoint folder: https://huggingface.co/datasets/%s/tree/main/%s",
        args.repo,
        candidate.path_in_repo,
    )
    shutil.rmtree(stage_root, ignore_errors=True)
    return True


def list_remote_files(args: argparse.Namespace, token: str | None) -> set[str]:
    if not args.check_remote:
        return set()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logging.warning("Cannot check remote repo; huggingface_hub is not installed.")
        return set()
    try:
        files = retry_hf_operation(
            f"HF watcher list repo {args.repo}",
            lambda: HfApi(token=token).list_repo_files(repo_id=args.repo, repo_type="dataset", token=token),
        )
    except Exception:
        logging.exception("Could not list remote HF checkpoint repo; uploads will rely on local state.")
        return set()
    return set(files)


def scan_once(args: argparse.Namespace, state_dir: Path, token: str | None) -> int:
    required_files = tuple(args.required_files or DEFAULT_REQUIRED_FILES)
    roots = [Path(value).expanduser().resolve() for value in args.scan_root]
    candidates: list[Candidate] = []
    for root in roots:
        candidates.extend(iter_candidate_folders(root, args))

    logging.info("Watcher scan found %d candidate folders.", len(candidates))
    remote_files = list_remote_files(args, token) if candidates else set()
    uploaded_count = 0

    for candidate in candidates:
        inspection = inspect_hf_folder(candidate.folder, required_files, args.min_safetensors)
        if not inspection.ok:
            logging.info("Skipping %s: %s", candidate.folder, inspection.reason)
            continue

        if not args.force:
            if remote_files and remote_folder_complete(
                remote_files,
                candidate.path_in_repo,
                required_files,
                args.min_safetensors,
            ):
                logging.info("Skipping %s; remote folder is already complete.", candidate.path_in_repo)
                write_uploaded_state(state_dir, candidate, inspection.safetensors)
                continue
            if read_uploaded_state(state_dir, candidate.path_in_repo):
                logging.info("Skipping %s; local watcher state says uploaded.", candidate.path_in_repo)
                continue

        logging.info("Candidate ready: %s -> %s (%s)", candidate.folder, candidate.path_in_repo, inspection.reason)
        if not folder_is_stable(candidate.folder, args.stability_seconds):
            logging.info("Skipping %s; files changed during stability check.", candidate.folder)
            continue

        if args.dry_run:
            logging.info("Dry-run would upload %s -> %s", candidate.folder, candidate.path_in_repo)
            continue

        if upload_candidate(args, candidate, token):
            write_uploaded_state(state_dir, candidate, inspection.safetensors)
            uploaded_count += 1
    return uploaded_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    if args.primary_only and not primary_process():
        logging.info("HF checkpoint watcher exiting on non-primary node rank=%s.", node_rank_label())
        return 0

    state_dir = state_dir_for(args)
    state_dir.mkdir(parents=True, exist_ok=True)
    token = hf_log_token()
    if not token and not args.dry_run:
        logging.warning("No HF token found; upload may fail for private repos.")

    logging.info(
        "HF checkpoint watcher starting: repo=%s scan_roots=%s folder_glob=%s interval=%ss "
        "state_dir=%s primary_only=%s",
        args.repo,
        ", ".join(args.scan_root),
        args.folder_glob,
        args.interval_seconds,
        state_dir,
        args.primary_only,
    )

    while True:
        try:
            uploaded_count = scan_once(args, state_dir, token)
            logging.info("Watcher scan complete; uploaded=%d.", uploaded_count)
        except KeyboardInterrupt:
            raise
        except Exception:
            logging.exception("Watcher scan failed.")
        if args.once:
            break
        time.sleep(max(1.0, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
