#!/usr/bin/env python3
"""Prepare a local chat calibration JSONL for NVIDIA Model Optimizer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def assistant_content_with_reasoning(message: dict[str, Any]) -> str | None:
    content = nonempty_text(message.get("content"))
    reasoning = nonempty_text(message.get("reasoning_content")) or nonempty_text(message.get("reasoning"))
    if reasoning is None:
        return content
    if content is not None and "<think>" in content:
        return content
    reasoning_block = f"<think>{reasoning}</think>"
    if content is None:
        return reasoning_block
    return f"{reasoning_block}\n{content}"


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: cleaned for k, v in value.items() if (cleaned := clean_value(v)) is not None}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


def normalize_messages(messages: Any, system_prompt: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"message must be an object, got {type(message).__name__}")
        msg = clean_value(message)
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = assistant_content_with_reasoning(msg)
            if content is not None:
                msg["content"] = content
        cleaned.append(msg)

    if not cleaned or cleaned[0].get("role") != "system":
        cleaned.insert(0, {"role": "system", "content": system_prompt})

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet file with a messages column.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--limit", type=int, default=2048, help="Maximum rows to export.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep source order instead of sampling.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(input_path, columns=[args.messages_column])
    total = df.height
    indices = list(range(total))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    exported = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx in indices:
            messages = df.row(idx, named=True)[args.messages_column]
            try:
                normalized = normalize_messages(messages, args.system_prompt)
            except Exception:
                skipped += 1
                continue
            f.write(json.dumps({"messages": normalized}, ensure_ascii=False) + "\n")
            exported += 1

    print(
        f"Wrote {exported} calibration rows to {output_path} "
        f"from {input_path} ({skipped} skipped, {total} source rows)."
    )


if __name__ == "__main__":
    main()
