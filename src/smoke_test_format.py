"""One-shot format smoke test for the proof pipeline against a served vLLM endpoint.

Sends the exact generation prompt the pipeline uses, prints the raw completion, and
reports whether parse_generation_response / require_valid_candidate_response would
accept it. Optionally chains a verifier call to check the score-first parse path too.

Run AFTER `vllm serve` is healthy (or against any OpenAI-compatible endpoint):

    python smoke_test_format.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model proof-model \
        --problem "Prove that there are infinitely many primes."

Defaults match run.py (DEFAULT_SERVED_MODEL_NAME / DEFAULT_API_KEY / port 8000).
"""

from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI

# Reuse the pipeline's real prompt builders and parsers so this test stays faithful.
from run import (
    build_deepseek_proof_generation_prompt,
    build_deepseek_proof_verification_prompt,
    parse_generation_response,
    parse_verifier_response,
)

DEFAULT_PROBLEM = (
    "Let $n$ be a positive integer. Prove that the sum of the first $n$ positive odd "
    "integers equals $n^2$."
)


def _call(client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    started = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "latency_s": time.time() - started,
    }


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="proof-model", help="served_model_name from run.py")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--proof-max-new-tokens", type=int, default=60000)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--skip-verify", action="store_true", help="Only test the generation stage.")
    parser.add_argument("--show-raw", action="store_true", help="Print the full raw completion text.")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=1)

    # ---- Stage 1: generation -------------------------------------------------
    _banner("STAGE 1 — proof_generation")
    gen_prompt = build_deepseek_proof_generation_prompt(args.problem)
    gen = _call(client, args.model, gen_prompt, args.proof_max_new_tokens, args.temperature, args.top_p)
    print(
        f"finish_reason={gen['finish_reason']} completion_tokens={gen['completion_tokens']} "
        f"latency={gen['latency_s']:.1f}s chars={len(gen['text'])}"
    )
    if args.show_raw:
        print("\n--- RAW COMPLETION ---\n" + gen["text"] + "\n--- END RAW ---")

    parsed = parse_generation_response(gen["text"])
    is_valid = bool(
        parsed["has_solution_section"]
        and parsed["proof"]
        and parsed["has_self_evaluation_section"]
        and parsed["self_evaluation"]
        and parsed["self_score"] is not None
    )
    print("\nparse_generation_response:")
    print(f"  has_solution_section       = {parsed['has_solution_section']}")
    print(f"  has_self_evaluation_section= {parsed['has_self_evaluation_section']}")
    print(f"  self_score                 = {parsed['self_score']}")
    print(f"  proof chars                = {len(parsed['proof'])}")
    print(f"  self_evaluation chars      = {len(parsed['self_evaluation'])}")
    print(f"  -> require_valid_candidate_response would {'ACCEPT' if is_valid else 'DROP'} this candidate")
    if parsed["proof"]:
        preview = parsed["proof"][:300].replace("\n", " ")
        print(f"  proof preview: {preview}...")

    if gen["finish_reason"] == "length":
        print(
            "\n  ⚠ finish_reason=length: the model hit max_tokens. The boxed self-score is emitted "
            "last, so truncation here is the #1 cause of dropped candidates. Raise the proof budget "
            "or shorten thinking."
        )
    if not is_valid:
        print(
            "\n  ✗ Generation output would be DROPPED. Check that the model emits a closing </think>, "
            "then '## Solution', '## Self Evaluation', and a final \\boxed{0|0.5|1}."
        )
        return 1

    # ---- Stage 2: verification (score-first parse path) ----------------------
    if args.skip_verify:
        print("\nGeneration format OK. Skipping verifier stage (--skip-verify).")
        return 0

    _banner("STAGE 2 — proof_verify (score-first parse path)")
    ver_prompt = build_deepseek_proof_verification_prompt(args.problem, parsed["proof"])
    ver = _call(client, args.model, ver_prompt, args.verifier_max_new_tokens, args.temperature, args.top_p)
    print(
        f"finish_reason={ver['finish_reason']} completion_tokens={ver['completion_tokens']} "
        f"latency={ver['latency_s']:.1f}s chars={len(ver['text'])}"
    )
    if args.show_raw:
        print("\n--- RAW VERIFIER ---\n" + ver["text"] + "\n--- END RAW ---")

    ver_parsed = parse_verifier_response(ver["text"])
    print("\nparse_verifier_response:")
    print(f"  score = {ver_parsed['score']}")
    if ver_parsed["score"] is None:
        print(
            "  ⚠ No boxed score parsed. With a 'think' model the score-recovery path (temp=0, "
            "continue_final_message) would fire at runtime, but verify the model closes </think> and "
            "emits the boxed score within the verifier budget."
        )
        if ver["finish_reason"] == "length":
            print("  ⚠ finish_reason=length on the verifier — raise --verifier-max-new-tokens.")
    else:
        print("  -> verifier score parses cleanly.")

    print("\nDone. If both stages show ACCEPT / clean score, the format is pipeline-ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
