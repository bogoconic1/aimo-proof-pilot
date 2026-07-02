from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any


def add_cuda_memory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cuda_memory_history",
        default="false",
        choices=("false", "oom", "true"),
        help=(
            "Record CUDA allocator history without inserting profiler calls into model forward. "
            "'oom' dumps memory diagnostics only on CUDA OOM/errors; 'true' also dumps on success."
        ),
    )
    parser.add_argument(
        "--cuda_memory_history_max_entries",
        type=int,
        default=200000,
        help="Maximum CUDA allocator history entries to retain per rank.",
    )
    parser.add_argument(
        "--cuda_memory_history_top_allocations",
        type=int,
        default=50,
        help="Number of largest active CUDA allocation blocks to write to JSON.",
    )
    parser.add_argument(
        "--cuda_memory_history_dump_pickle",
        default="false",
        choices=("true", "false"),
        help=(
            "Also write the raw torch CUDA memory snapshot pickle. This can be large; "
            "HF log upload ignores these pickle files by default."
        ),
    )


def is_cuda_oom_exception(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = "".join(traceback.format_exception_only(type(exc), exc)).lower()
    return "cuda out of memory" in text or "torch.outofmemoryerror" in text


def _rank_token() -> str:
    rank = os.environ.get("RANK", os.environ.get("GLOBAL_RANK", "none"))
    local_rank = os.environ.get("LOCAL_RANK", "none")
    node_rank = os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", os.environ.get("GLOBAL_RANK", "none")))
    return f"node{node_rank}_rank{rank}_local{local_rank}_pid{os.getpid()}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _block_frames(block: dict[str, Any]) -> list[dict[str, Any]]:
    frames = block.get("frames")
    if isinstance(frames, list):
        return frames
    history = block.get("history")
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, dict) and isinstance(entry.get("frames"), list):
                return entry["frames"]
    return []


def _largest_active_allocations(snapshot: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for segment in snapshot.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        for block in segment.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue
            state = str(block.get("state", ""))
            if "active" not in state:
                continue
            blocks.append(
                {
                    "size": int(block.get("size") or 0),
                    "requested_size": int(block.get("requested_size") or block.get("size") or 0),
                    "state": state,
                    "address": block.get("address"),
                    "segment_address": segment.get("address"),
                    "segment_type": segment.get("segment_type"),
                    "stream": segment.get("stream"),
                    "frames": _block_frames(block)[-12:],
                }
            )
    blocks.sort(key=lambda item: int(item["size"]), reverse=True)
    return blocks[: max(limit, 0)]


class CudaMemoryHistoryRecorder:
    def __init__(self, args: argparse.Namespace, logdir: Path):
        self.mode = getattr(args, "cuda_memory_history", "false")
        self.max_entries = int(getattr(args, "cuda_memory_history_max_entries", 200000) or 0)
        self.top_allocations = int(getattr(args, "cuda_memory_history_top_allocations", 50) or 0)
        self.dump_pickle = getattr(args, "cuda_memory_history_dump_pickle", "false") == "true"
        self.logdir = Path(logdir).expanduser().resolve()
        self.enabled = False
        self._torch = None

    def start(self) -> None:
        if self.mode == "false":
            return
        if self.max_entries <= 0:
            logging.warning("CUDA memory history disabled because max_entries=%d.", self.max_entries)
            return
        try:
            import torch

            self._torch = torch
            if not torch.cuda.is_available():
                logging.warning("CUDA memory history requested, but CUDA is not available.")
                return
            record = getattr(torch.cuda.memory, "_record_memory_history", None)
            if record is None:
                logging.warning("CUDA memory history requested, but torch lacks _record_memory_history.")
                return
            try:
                record(enabled="all", context="all", stacks="all", max_entries=self.max_entries)
            except TypeError:
                try:
                    record(enabled=True, max_entries=self.max_entries)
                except TypeError:
                    record(True)
            self.enabled = True
            logging.warning(
                "Started CUDA allocator memory history: mode=%s max_entries=%d dump_pickle=%s.",
                self.mode,
                self.max_entries,
                self.dump_pickle,
            )
        except Exception:
            logging.exception("Failed to start CUDA memory history.")

    def should_dump_for_exception(self, exc: BaseException) -> bool:
        if not self.enabled:
            return False
        if self.mode == "true":
            return True
        return self.mode == "oom" and is_cuda_oom_exception(exc)

    def dump(self, reason: str) -> None:
        if not self.enabled or self._torch is None:
            return
        torch = self._torch
        token = _rank_token()
        self.logdir.mkdir(parents=True, exist_ok=True)
        prefix = self.logdir / f"cuda_memory_{reason}_{token}"
        try:
            if torch.cuda.is_available():
                summary_path = prefix.with_suffix(".summary.txt")
                summary_path.write_text(torch.cuda.memory_summary(abbreviated=False), encoding="utf-8")
                stats_path = prefix.with_suffix(".stats.json")
                stats_path.write_text(
                    json.dumps(_json_safe(torch.cuda.memory_stats()), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except Exception:
            logging.exception("Failed to write CUDA memory summary for reason=%s.", reason)

        snapshot: dict[str, Any] | None = None
        try:
            snapshot_fn = getattr(torch.cuda.memory, "_snapshot", None)
            if snapshot_fn is not None:
                snapshot = snapshot_fn()
                top_path = prefix.with_suffix(".top_allocations.json")
                payload = {
                    "reason": reason,
                    "rank_token": token,
                    "top_allocations": _largest_active_allocations(snapshot, self.top_allocations),
                }
                top_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
        except Exception:
            logging.exception("Failed to write CUDA memory top-allocation snapshot for reason=%s.", reason)

        if self.dump_pickle:
            try:
                pickle_path = self.logdir / f"cuda_memory_{reason}_{token}.memory_snapshot.pickle"
                dump_snapshot = getattr(torch.cuda.memory, "_dump_snapshot", None)
                if dump_snapshot is not None:
                    dump_snapshot(str(pickle_path))
                elif snapshot is not None:
                    with pickle_path.open("wb") as f:
                        pickle.dump(snapshot, f)
            except Exception:
                logging.exception("Failed to write raw CUDA memory snapshot pickle for reason=%s.", reason)

        logging.warning("CUDA memory diagnostics written for reason=%s prefix=%s", reason, prefix)

    def dump_success_if_requested(self) -> None:
        if self.enabled and self.mode == "true":
            self.dump("success")

    def stop(self) -> None:
        if not self.enabled or self._torch is None:
            return
        torch = self._torch
        with suppress(Exception):
            record = getattr(torch.cuda.memory, "_record_memory_history", None)
            if record is not None:
                record(enabled=None)
        self.enabled = False
