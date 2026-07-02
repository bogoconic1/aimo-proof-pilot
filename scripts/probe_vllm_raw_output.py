#!/usr/bin/env python3
"""Probe a running vLLM OpenAI server and save raw completion output.

This uses `/v1/completions` with a tokenizer-rendered chat prompt so the output
text is the raw continuation. That makes reasoning tags/content visible instead
of relying on chat-completion reasoning parsers.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def load_problem(csv_path: Path, row_index: int) -> dict[str, str]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} out of range for {len(rows)} rows")
    row = rows[row_index]
    problem = row.get("problem") or row.get("question")
    if not problem:
        raise ValueError(f"Row {row_index} has no problem/question column: {row}")
    return {"id": row.get("id", str(row_index)), "problem": problem}


def build_prompt(tokenizer_path: str, problem: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. Show your reasoning and put the final answer in "
                "\\boxed{}.\n\n"
                f"{problem}"
            ),
        }
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"]


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}:\n{body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--csv", default="/tmp/submissions-instructions-runtime/test.csv")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out-dir", default="/tmp/vllm_raw_probe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_path = args.tokenizer or args.model
    row = load_problem(Path(args.csv), args.row_index)
    prompt = build_prompt(tokenizer_path, row["problem"])

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"row{args.row_index}_{row['id']}_{int(time.time())}"
    (out_dir / f"{stem}.prompt.txt").write_text(prompt)
    (out_dir / f"{stem}.payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    response = post_json(f"{args.base_url.rstrip('/')}/completions", payload, args.timeout)
    raw_text = response.get("choices", [{}])[0].get("text", "")
    (out_dir / f"{stem}.response.json").write_text(json.dumps(response, indent=2, ensure_ascii=False))
    (out_dir / f"{stem}.raw_text.txt").write_text(raw_text)

    print(f"problem_id={row['id']}")
    print(f"prompt_chars={len(prompt)} raw_text_chars={len(raw_text)}")
    print(f"response_json={out_dir / f'{stem}.response.json'}")
    print(f"raw_text={out_dir / f'{stem}.raw_text.txt'}")
    print("raw_text_preview:")
    print(raw_text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
