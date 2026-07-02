from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
PROOF_DATA_SOURCE = "proof_math"
PROOF_PROMPT_VERSION = "run_py_deepseek_generation_v1"
DEFAULT_SYSTEM_PROMPT = ""
EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""
DEFAULT_USER_TEMPLATE = """Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
__EVALUATION_RUBRIC__
```

In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.
- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation // Your evaluation of your own solution above should start with this exact same markdown title

Here is my evaluation of the solution: // Your analysis should start with this exact same phrase
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above

---

Here is your task input:

## Problem
{problem}""".replace("__EVALUATION_RUBRIC__", EVALUATION_RUBRIC)


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import polars as pl

            return pl.read_parquet(path).to_dicts()
        except ImportError:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
    if suffix in {".json", ".jsonl"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                if isinstance(payload, list):
                    return [dict(row) for row in payload]
                if isinstance(payload, dict):
                    for key in ("data", "train", "rows"):
                        value = payload.get(key)
                        if isinstance(value, list):
                            return [dict(row) for row in value]
                raise ValueError(f"Unsupported JSON dataset shape: {path}")
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
    if suffix == ".csv":
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported verl/RLCSD dataset suffix {suffix!r}: {path}")


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl

        pl.DataFrame(records).write_parquet(path)
        return
    except ImportError:
        import pandas as pd

        pd.DataFrame(records).to_parquet(path, index=False)


def _messages_from_problem(problem: str, system_prompt: str, user_template: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_template.format(problem=problem)})
    return messages


def _text_from_messages(messages: Any) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return "\n".join(part for part in parts if part).strip()


def _case_insensitive_get(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    lowered = {str(key).strip().lower(): key for key in row}
    for name in names:
        key = lowered.get(str(name).strip().lower())
        if key is not None:
            return row.get(key)
    return None


def prepare_verl_proof_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    problem_column: str = "problem",
    solution_column: str = "solution",
    data_source: str = PROOF_DATA_SOURCE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    max_rows: int = 0,
) -> Path:
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    records = _load_records(source)
    if max_rows and max_rows > 0:
        records = records[:max_rows]

    prepared: list[dict[str, Any]] = []
    skipped = 0
    missing_reference_solution = 0
    for index, row in enumerate(records):
        problem_value = _case_insensitive_get(
            row,
            problem_column,
            "problem",
            "question",
            "Problem",
            "Question",
            "prompt",
        )
        solution_value = _case_insensitive_get(
            row,
            solution_column,
            "solution",
            "ground_truth",
            "answer",
            "Solution",
            "Short Answer",
        )
        problem = str(problem_value or _text_from_messages(_case_insensitive_get(row, "messages")) or "")
        solution = str(solution_value or "")
        problem = problem.strip()
        solution = solution.strip()
        if not solution:
            missing_reference_solution += 1
        if not problem:
            skipped += 1
            continue
        prompt = _messages_from_problem(problem, system_prompt, user_template)
        prepared.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "proof",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "index": index,
                    "problem": problem,
                    "solution": solution,
                    "has_reference_solution": bool(solution),
                    "source_path": str(source),
                },
            }
        )
    if not prepared:
        raise ValueError(f"No usable proof rows found in {source}; skipped={skipped}")
    _write_parquet(prepared, target)
    LOGGER.info(
        "Prepared verl/RLCSD proof dataset: source=%s target=%s rows=%d skipped=%d missing_reference_solution=%d",
        source,
        target,
        len(prepared),
        skipped,
        missing_reference_solution,
    )
    return target


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _deepseekmath_v2_verifier() -> Any:
    from open_instruct.ground_truth_utils import DeepSeekMathV2Verifier, DeepSeekMathV2VerifierConfig

    config = DeepSeekMathV2VerifierConfig(
        llm_judge_model=os.environ.get("VERL_RLCSD_JUDGE_MODEL", "deepseek/deepseek-v4-pro"),
        llm_judge_base_url=os.environ.get("VERL_RLCSD_JUDGE_BASE_URL") or "https://openrouter.ai/api/v1",
        llm_judge_api_key_env=os.environ.get("VERL_RLCSD_JUDGE_API_KEY_ENV", "OPENROUTER_API_KEY"),
        llm_judge_api_key=os.environ.get("VERL_RLCSD_JUDGE_API_KEY"),
        deepseekmath_v2_judge_backend=os.environ.get("VERL_RLCSD_JUDGE_BACKEND", "api"),
        deepseekmath_v2_max_tokens=_env_int("VERL_RLCSD_JUDGE_MAX_TOKENS", 40000),
        deepseekmath_v2_max_context_length=_env_int("VERL_RLCSD_JUDGE_MAX_CONTEXT_LENGTH", 40000),
        deepseekmath_v2_context_margin_tokens=_env_int("VERL_RLCSD_JUDGE_CONTEXT_MARGIN_TOKENS", 256),
        deepseekmath_v2_min_completion_tokens=_env_int("VERL_RLCSD_JUDGE_MIN_COMPLETION_TOKENS", 2048),
        deepseekmath_v2_temperature=_env_float("VERL_RLCSD_JUDGE_TEMPERATURE", 1.0),
        deepseekmath_v2_top_p=_env_float("VERL_RLCSD_JUDGE_TOP_P", 0.95),
        deepseekmath_v2_timeout=_env_int("VERL_RLCSD_JUDGE_TIMEOUT", 1800),
        deepseekmath_v2_proof_weight=_env_float("VERL_RLCSD_PROOF_REWARD_WEIGHT", 0.76),
        deepseekmath_v2_self_eval_weight=_env_float("VERL_RLCSD_SELF_EVAL_REWARD_WEIGHT", 0.24),
        deepseekmath_v2_partial_format_score=_env_float("VERL_RLCSD_PARTIAL_FORMAT_SCORE", 0.7),
        deepseekmath_v2_enable_meta_verification=_env_bool("VERL_RLCSD_ENABLE_META_VERIFICATION", True),
        deepseekmath_v2_require_format=_env_bool("VERL_RLCSD_REQUIRE_FORMAT", True),
    )
    if config.deepseekmath_v2_api_key is None and config.llm_judge_api_key is None:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            config.llm_judge_api_key = openrouter_key
    return DeepSeekMathV2Verifier(config)


def _extract_reward_value(result: Any) -> tuple[float, dict[str, Any]]:
    if isinstance(result, dict):
        score = result.get("score", result.get("reward", 0.0))
        return float(score), dict(result)
    if isinstance(result, tuple):
        score = float(result[0])
        metadata = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        return score, dict(metadata)
    if hasattr(result, "score"):
        metadata: dict[str, Any] = {}
        cost = getattr(result, "cost", None)
        if cost is not None:
            metadata["judge_cost"] = cost
        reasoning = getattr(result, "reasoning", None)
        if reasoning:
            try:
                parsed_reasoning = json.loads(reasoning)
            except (TypeError, json.JSONDecodeError):
                metadata["reasoning_text"] = str(reasoning)[:1000]
            else:
                if isinstance(parsed_reasoning, dict):
                    metadata.update(parsed_reasoning)
        return float(getattr(result, "score", 0.0)), metadata
    return float(result), {}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl custom reward for proof RL.

    The score is kept in [0, 1]. RLCSD can still use thresholded positive and
    negative bins for teacher hints, but the actor reward remains normalized.
    """

    del kwargs
    extra_info = dict(extra_info or {})
    src = str(data_source or "").lower()
    if src != PROOF_DATA_SOURCE:
        LOGGER.warning("Unexpected RLCSD data_source=%s; falling back to zero reward.", data_source)
        return {"score": 0.0, "acc": 0.0, "formatted": 0.0, "reward_valid": 0.0}

    from open_instruct.ground_truth_utils import DeepSeekMathV2Verifier

    parsed = DeepSeekMathV2Verifier.parse_prediction(solution_str)
    fatal_errors = set(parsed.format_errors) & set(DeepSeekMathV2Verifier.FATAL_FORMAT_ERRORS)
    if fatal_errors:
        return {
            "score": 0.0,
            "acc": 0.0,
            "formatted": 0.0,
            "reward_valid": 0.0,
            "format_errors": ",".join(parsed.format_errors),
        }

    label_payload = {
        "problem": extra_info.get("problem", ""),
        "solution": ground_truth or extra_info.get("solution", ""),
    }
    query = str(extra_info.get("problem", ""))
    result = _deepseekmath_v2_verifier()(
        query=query,
        prediction=solution_str,
        label=label_payload,
        tokenized_prediction=[],
    )
    score, metadata = _extract_reward_value(result)
    score = max(0.0, min(1.0, score))
    format_score = metadata.get("format_score", 1.0 if parsed.format_ok else 0.0)
    try:
        formatted = max(0.0, min(1.0, float(format_score)))
    except (TypeError, ValueError):
        formatted = 1.0 if parsed.format_ok else 0.0
    output = {
        "score": score,
        "acc": 1.0 if score >= 0.75 else 0.0,
        "formatted": formatted,
        "reward_valid": 1.0,
        "proof_solution_chars": len(parsed.solution),
        "proof_self_eval_chars": len(parsed.self_evaluation),
        "proof_self_score": -1.0 if parsed.self_score is None else float(parsed.self_score),
    }
    for key, value in metadata.items():
        safe_key = re.sub(r"[^A-Za-z0-9_./-]+", "_", str(key))
        if isinstance(value, (int, float, str, bool)) or value is None:
            output[safe_key] = value
    return output
