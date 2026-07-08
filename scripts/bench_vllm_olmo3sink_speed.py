#!/usr/bin/env python3
"""Benchmark OLMo3Sink generation throughput with vLLM offline inference.

Default workload:
  - 1 GPU
  - 8 concurrent prompts
  - about 1,024 prompt tokens per request
  - 131,072 total requested output tokens (16,384 per request)

If you need 128k output tokens *per request*, pass
`--max-tokens-per-request 128000 --max-model-len 131072`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--vllm-extra-json must decode to a JSON object")
    return value


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def add_src_to_path(src_dir: Path) -> None:
    src_text = str(src_dir.resolve())
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        if src_text not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = src_text + os.pathsep + existing
    else:
        os.environ["PYTHONPATH"] = src_text


def register_olmo3sink(src_dir: Path, skip: bool) -> None:
    if skip:
        return
    add_src_to_path(src_dir)
    from olmo3_sink import register_olmo3_sink

    register_olmo3_sink()

    try:
        from vllm import ModelRegistry
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("vLLM is not importable; install vllm before running this benchmark") from exc

    ModelRegistry.register_model(
        "Olmo3SinkForCausalLM",
        "olmo3_sink.vllm_adapter:Olmo3SinkForCausalLM",
    )


def tokenizer_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer: Any, filler_repeats: int, request_id: int) -> str:
    problem = (
        "Prove a difficult olympiad-style inequality. "
        "Write a rigorous proof, check edge cases, and put the final answer in boxed form. "
        f"Benchmark request id {request_id:04d}.\n\n"
    )
    filler = (
        "We need a complete mathematical derivation with clear definitions, "
        "careful estimates, and no skipped algebraic steps. "
    )
    content = problem + filler * filler_repeats
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def build_target_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")

    low = 0
    high = 1
    while tokenizer_len(tokenizer, render_prompt(tokenizer, high, request_id)) < target_tokens:
        high *= 2

    best = render_prompt(tokenizer, low, request_id)
    best_len = tokenizer_len(tokenizer, best)
    while low <= high:
        mid = (low + high) // 2
        candidate = render_prompt(tokenizer, mid, request_id)
        length = tokenizer_len(tokenizer, candidate)
        if length <= target_tokens:
            best = candidate
            best_len = length
            low = mid + 1
        else:
            high = mid - 1

    # Add a tiny amount of neutral padding when the binary-search granularity
    # leaves us noticeably below the requested token count.
    pad = " Therefore"
    while best_len < target_tokens:
        candidate = best + pad
        length = tokenizer_len(tokenizer, candidate)
        if length > target_tokens:
            break
        best = candidate
        best_len = length

    return best, best_len


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable: {exc}"


def sampling_params_kwargs(args: argparse.Namespace, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
    }
    signature = inspect.signature(SamplingParams)
    if "ignore_eos" in signature.parameters:
        kwargs["ignore_eos"] = args.ignore_eos
    if args.force_max_tokens and "min_tokens" in signature.parameters:
        kwargs["min_tokens"] = max_tokens
    return kwargs


def init_llm(args: argparse.Namespace, max_model_len: int, max_num_batched_tokens: int):
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_mode": args.tokenizer_mode,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs or args.batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": args.disable_log_stats,
    }
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.disable_custom_all_reduce is not None:
        kwargs["disable_custom_all_reduce"] = args.disable_custom_all_reduce
    kwargs.update(parse_json_object(args.vllm_extra_json))

    print("vllm_engine_kwargs=" + json.dumps(kwargs, default=str, sort_keys=True))
    start = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - start
    print(f"engine_load_seconds={load_seconds:.3f}")
    return llm, load_seconds, kwargs


def run_generate(llm: Any, prompts: list[str], sampling_params: Any) -> tuple[list[Any], float]:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    seconds = time.perf_counter() - start
    return outputs, seconds


def summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
    output_lens = [len(item.outputs[0].token_ids) for item in outputs]
    finish_reasons = Counter(str(item.outputs[0].finish_reason) for item in outputs)
    return {
        "num_outputs": len(outputs),
        "output_tokens_total": sum(output_lens),
        "output_tokens_min": min(output_lens) if output_lens else 0,
        "output_tokens_max": max(output_lens) if output_lens else 0,
        "output_tokens_mean": statistics.mean(output_lens) if output_lens else 0.0,
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }


def parse_args() -> argparse.Namespace:
    default_src = repo_root_from_script() / "src"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="OLMo3Sink HF checkpoint path or repo id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--olmo3sink-src", default=str(default_src), help="Path containing the olmo3_sink package.")
    parser.add_argument("--skip-olmo3sink-register", action="store_true", help="Skip local OLMo3Sink/vLLM registration.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--total-output-tokens", type=int, default=131072)
    parser.add_argument("--max-tokens-per-request", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--enforce-eager", type=str_to_bool, default=False)
    parser.add_argument("--disable-custom-all-reduce", type=str_to_bool, default=None)
    parser.add_argument("--trust-remote-code", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ignore-eos", type=str_to_bool, default=True)
    parser.add_argument("--force-max-tokens", type=str_to_bool, default=True)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable-log-stats", type=str_to_bool, default=True)
    parser.add_argument("--vllm-extra-json", default=None, help="Extra JSON object merged into vLLM LLM kwargs.")
    parser.add_argument("--out-json", default=None, help="Optional path for benchmark metrics JSON.")
    parser.add_argument("--print-preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    register_olmo3sink(Path(args.olmo3sink_src), args.skip_olmo3sink_register)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    prompts: list[str] = []
    prompt_lens: list[int] = []
    for idx in range(args.batch_size):
        prompt, length = build_target_prompt(tokenizer, args.prompt_tokens, idx)
        prompts.append(prompt)
        prompt_lens.append(length)

    max_tokens = args.max_tokens_per_request
    if max_tokens is None:
        max_tokens = round_up(args.total_output_tokens, args.batch_size) // args.batch_size
    max_model_len = args.max_model_len or round_up(max(prompt_lens) + max_tokens + 256, 1024)
    max_num_batched_tokens = args.max_num_batched_tokens or round_up(max(prompt_lens) * args.batch_size, 1024)

    print("benchmark_config=" + json.dumps(
        {
            "batch_size": args.batch_size,
            "prompt_tokens_min": min(prompt_lens),
            "prompt_tokens_max": max(prompt_lens),
            "requested_max_tokens_per_request": max_tokens,
            "requested_output_tokens_total": max_tokens * args.batch_size,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "kv_cache_dtype": args.kv_cache_dtype,
            "force_max_tokens": args.force_max_tokens,
            "ignore_eos": args.ignore_eos,
        },
        sort_keys=True,
    ))
    print("nvidia_smi_before=\n" + nvidia_smi_snapshot())

    llm, load_seconds, llm_kwargs = init_llm(args, max_model_len, max_num_batched_tokens)

    if args.warmup_tokens > 0:
        warmup_kwargs = sampling_params_kwargs(args, args.warmup_tokens)
        warmup_kwargs["ignore_eos"] = True
        if "min_tokens" in inspect.signature(SamplingParams).parameters:
            warmup_kwargs["min_tokens"] = args.warmup_tokens
        print("running_warmup=true")
        _, warmup_seconds = run_generate(llm, [prompts[0]], SamplingParams(**warmup_kwargs))
        print(f"warmup_seconds={warmup_seconds:.3f}")

    sampling_params = SamplingParams(**sampling_params_kwargs(args, max_tokens))
    round_metrics: list[dict[str, Any]] = []
    for round_idx in range(args.rounds):
        outputs, seconds = run_generate(llm, prompts, sampling_params)
        output_summary = summarize_outputs(outputs)
        total_output_tokens = int(output_summary["output_tokens_total"])
        total_prompt_tokens = sum(prompt_lens)
        metrics = {
            "round": round_idx,
            "seconds": seconds,
            "prompt_tokens_total": total_prompt_tokens,
            "output_tokens_total": total_output_tokens,
            "total_tokens": total_prompt_tokens + total_output_tokens,
            "decode_tokens_per_second": total_output_tokens / seconds if seconds > 0 else 0.0,
            "decode_tokens_per_second_per_request": (
                total_output_tokens / seconds / args.batch_size if seconds > 0 else 0.0
            ),
            "end_to_end_tokens_per_second": (
                (total_prompt_tokens + total_output_tokens) / seconds if seconds > 0 else 0.0
            ),
            **output_summary,
        }
        round_metrics.append(metrics)
        print("round_metrics=" + json.dumps(metrics, sort_keys=True))
        preview = outputs[0].outputs[0].text[: args.print_preview_chars]
        if preview:
            print(f"round_{round_idx}_first_output_preview={preview!r}")

    print("nvidia_smi_after=\n" + nvidia_smi_snapshot())
    final = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "load_seconds": load_seconds,
        "prompt_tokens": prompt_lens,
        "max_tokens_per_request": max_tokens,
        "llm_kwargs": llm_kwargs,
        "rounds": round_metrics,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final, indent=2, sort_keys=True))
        print(f"metrics_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
