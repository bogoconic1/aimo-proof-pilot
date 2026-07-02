#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf


LOGGER = logging.getLogger(__name__)

EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""


def build_deepseek_proof_generation_prompt(question: str) -> str:
    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
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
{question}"""


def build_deepseek_proof_refinement_prompt(question: str, proof: str, proof_analyses: list[str]) -> str:
    analyses = "\n\n".join(f"### Evaluation {idx + 1}\n{analysis}" for idx, analysis in enumerate(proof_analyses))
    return f"""{build_deepseek_proof_generation_prompt(question)}

## Candidate Solution(s) to Refine
Here are some solution sample(s) along with their correctness evaluation(s). You should provide a better solution by solving issues mentioned in the evaluation(s), or by re-using promising ideas mentioned in the solution sample(s), or by doing both.

### Candidate Solution
{proof}

{analyses}

## Final Instruction
Your final response should follow the format above, including a `## Solution` section followed by a `## Self Evaluation` section"""


def _parse_bool_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _clip_for_log(text: str) -> tuple[str, bool]:
    max_chars_text = os.environ.get("DEEPSEEK_MATH_V2_PROMPT_LOG_MAX_CHARS", "0").strip()
    try:
        max_chars = int(max_chars_text or "0")
    except ValueError:
        max_chars = 0
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + f"\n\n...[clipped {len(text) - max_chars} chars from middle by DEEPSEEK_MATH_V2_PROMPT_LOG_MAX_CHARS]...\n\n"
        + text[-tail:],
        True,
    )


def _log_llm_input(stage: str, payload: Any, *, context: dict[str, Any] | None = None) -> None:
    if not _parse_bool_env("DEEPSEEK_MATH_V2_LOG_LLM_INPUTS", True):
        return
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    logged_text, clipped = _clip_for_log(text)
    context = {key: value for key, value in (context or {}).items() if value not in {None, ""}}
    LOGGER.info(
        "DeepSeekMath-V2 LLM input stage=%s chars=%d clipped=%s context=%s\n%s",
        stage,
        len(text),
        clipped,
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        logged_text,
    )


def _messages_to_log_payload(messages: Any) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        payload.append({"role": str(role or ""), "content": _message_content_to_text(content)})
    return payload


def _state_log_context(state: Any | None, **extra: Any) -> dict[str, Any]:
    context = dict(extra)
    if isinstance(state, dict):
        context.setdefault("trajectory_id", state.get("trajectory_id"))
        input_payload = state.get("input")
        if isinstance(input_payload, dict):
            context.setdefault("task_id", input_payload.get("task_id"))
            context.setdefault("source_index", input_payload.get("source_index"))
            context.setdefault("dataset", input_payload.get("dataset"))
    return context


def _json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _string_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            text = str(row[key]).strip()
            if text:
                return text
    return ""


def _extract_problem_from_messages(value: Any) -> str:
    messages = _json_loads_maybe(value)
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or part.get("content") or "")
                for part in content
                if isinstance(part, dict)
            )
        text = str(content or "").strip()
        match = re.search(r"(?ims)^##[ \t]+Problem[ \t]*\n(?P<problem>.*)$", text)
        return match.group("problem").strip() if match else text
    return ""


def _read_dataset_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"DeepSeekMath-V2 proof dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "train", "rows", "examples"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"JSON dataset must contain a list of rows: {path}")
        return [dict(row) for row in data]

    import pandas as pd

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported DeepSeekMath-V2 proof dataset extension: {suffix}")
    return frame.to_dict(orient="records")


def _resolve_column(row: dict[str, Any], requested: str, candidates: list[str]) -> str | None:
    if requested and requested != "auto":
        return requested if requested in row else None
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        if candidate in row:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalize_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    problem_column: str = "auto",
    solution_column: str = "auto",
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        problem_key = _resolve_column(row, problem_column, ["problem", "question", "Problem", "Question"])
        solution_key = _resolve_column(row, solution_column, ["solution", "answer", "Solution", "Answer"])
        problem = str(row.get(problem_key) or "").strip() if problem_key else ""
        if not problem and "messages" in row:
            problem = _extract_problem_from_messages(row.get("messages"))
        if not problem:
            continue
        solution = str(row.get(solution_key) or "").strip() if solution_key else ""
        ground_truth = {"problem": problem}
        if solution:
            ground_truth["solution"] = solution
        task_id = _string_value(row, ["task_id", "id", "example_id", "problem_id", "question_uuid"]) or str(index)
        normalized.append(
            {
                "question": build_deepseek_proof_generation_prompt(problem),
                "problem": problem,
                "solution": solution,
                "answer": json.dumps(ground_truth, ensure_ascii=False),
                "dataset": "proof_math",
                "task_id": task_id,
                "source_index": index,
            }
        )
        if max_examples is not None and max_examples > 0 and len(normalized) >= max_examples:
            break
    if not normalized:
        raise ValueError("DeepSeekMath-V2 proof dataset produced zero usable rows.")
    return normalized


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or getattr(part, "content", "") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return str(completion or "")
    assistant_texts: list[str] = []
    all_texts: list[str] = []
    for message in completion:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        text = _message_content_to_text(content).strip()
        if not text:
            continue
        all_texts.append(text)
        if role == "assistant":
            assistant_texts.append(text)
    return "\n\n".join(assistant_texts or all_texts)


class DeepSeekMathV2PromptLoggingMixin:
    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        messages = await super().get_prompt_messages(state)  # type: ignore[misc]
        if state.get("final_env_response") is not None:
            return messages
        round_idx = len(state.get("trajectory") or [])
        stage = "proof_generation" if round_idx == 0 else f"proof_refinement_round_{round_idx}"
        _log_llm_input(stage, _messages_to_log_payload(messages), context=_state_log_context(state, round=round_idx))
        return messages


class DeepSeekMathV2SingleTurnEnv(DeepSeekMathV2PromptLoggingMixin, vf.SingleTurnEnv):
    pass


def _parse_answer(answer: Any) -> dict[str, Any]:
    parsed = _json_loads_maybe(answer)
    return parsed if isinstance(parsed, dict) else {}


class DeepSeekMathV2ProofRubric(vf.Rubric):
    def __init__(
        self,
        *,
        judge_backend: str = "api",
        llm_judge_model: str = "deepseek/deepseek-v4-pro",
        llm_judge_base_url: str | None = "https://openrouter.ai/api/v1",
        llm_judge_api_key: str | None = None,
        llm_judge_api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 40000,
        max_context_length: int = 40000,
        context_margin_tokens: int = 256,
        min_completion_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        timeout: int = 1800,
        extra_body_json: str | None = None,
        proof_weight: float = 0.76,
        self_eval_weight: float = 0.24,
        partial_format_score: float = 0.7,
        enable_meta_verification: bool = True,
        require_format: bool = True,
    ) -> None:
        super().__init__()
        self.judge_backend = judge_backend
        self.llm_judge_model = llm_judge_model
        self.llm_judge_base_url = llm_judge_base_url
        self.llm_judge_api_key = llm_judge_api_key
        self.llm_judge_api_key_env = llm_judge_api_key_env
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.context_margin_tokens = context_margin_tokens
        self.min_completion_tokens = min_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.extra_body_json = extra_body_json
        self.proof_weight = proof_weight
        self.self_eval_weight = self_eval_weight
        self.partial_format_score = partial_format_score
        self.enable_meta_verification = enable_meta_verification
        self.require_format = require_format
        self._verifier = None

        self.add_reward_func(self.deepseekmath_v2_reward)
        self.add_metric(self.deepseekmath_v2_format_score)
        self.add_metric(self.deepseekmath_v2_proof_score)
        self.add_metric(self.deepseekmath_v2_self_score)
        self.add_metric(self.deepseekmath_v2_self_eval_score)
        self.add_metric(self.deepseekmath_v2_score_alignment)
        self.add_metric(self.deepseekmath_v2_final_round_reward)
        self.add_metric(self.deepseekmath_v2_best_round_reward)
        self.add_metric(self.deepseekmath_v2_refine_rounds_used)
        self.add_metric(self.deepseekmath_v2_selected_round_index)

    def _build_verifier(self) -> Any:
        from open_instruct.ground_truth_utils import DeepSeekMathV2Verifier, DeepSeekMathV2VerifierConfig

        config = DeepSeekMathV2VerifierConfig(
            llm_judge_model=self.llm_judge_model,
            llm_judge_base_url=self.llm_judge_base_url,
            llm_judge_api_key_env=self.llm_judge_api_key_env,
            llm_judge_api_key=self.llm_judge_api_key,
            deepseekmath_v2_judge_backend="api",
            deepseekmath_v2_max_tokens=self.max_tokens,
            deepseekmath_v2_max_context_length=self.max_context_length,
            deepseekmath_v2_context_margin_tokens=self.context_margin_tokens,
            deepseekmath_v2_min_completion_tokens=self.min_completion_tokens,
            deepseekmath_v2_temperature=self.temperature,
            deepseekmath_v2_top_p=self.top_p,
            deepseekmath_v2_timeout=self.timeout,
            deepseekmath_v2_extra_body_json=self.extra_body_json,
            deepseekmath_v2_proof_weight=self.proof_weight,
            deepseekmath_v2_self_eval_weight=self.self_eval_weight,
            deepseekmath_v2_partial_format_score=self.partial_format_score,
            deepseekmath_v2_enable_meta_verification=self.enable_meta_verification,
            deepseekmath_v2_require_format=self.require_format,
        )
        return DeepSeekMathV2Verifier(config)

    @property
    def verifier(self) -> Any:
        if self._verifier is None:
            self._verifier = self._build_verifier()
        return self._verifier

    def _empty_solution_payload(self, prediction: str, raw_problem: str) -> dict[str, Any] | None:
        from open_instruct.ground_truth_utils import DeepSeekMathV2Verifier

        parsed = DeepSeekMathV2Verifier.parse_prediction(prediction)
        if parsed.solution.strip():
            return None
        return {
            "format_ok": False,
            "format_score": 0.0,
            "format_errors": parsed.format_errors,
            "fatal_format_errors": ["empty_solution"],
            "proof_score": 0.0,
            "self_score": parsed.self_score,
            "self_eval_score": 0.0,
            "score_alignment": 0.0,
            "reward": 0.0,
            "judge_backend": self.judge_backend,
            "judge_skipped": True,
            "judge_skip_reason": "empty_solution",
            "prediction_chars": len(prediction),
            "problem_chars": len(raw_problem),
        }

    async def _score(
        self,
        *,
        completion: Any,
        answer: Any,
        problem: str = "",
        question: str = "",
        state: Any | None = None,
        log_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state is not None and isinstance(state.get("deepseekmath_v2_reward_payload"), dict):
            return state["deepseekmath_v2_reward_payload"]

        prediction = completion_to_text(completion)
        answer_payload = _parse_answer(answer)
        raw_problem = problem or str(answer_payload.get("problem") or answer_payload.get("question") or "")
        if not raw_problem and question:
            match = re.search(r"(?ims)^##[ \t]+Problem[ \t]*\n(?P<problem>.*)$", question)
            raw_problem = match.group("problem").strip() if match else question

        empty_solution_payload = self._empty_solution_payload(prediction, raw_problem)
        if empty_solution_payload is not None:
            payload = empty_solution_payload
        elif self.judge_backend == "none":
            payload = self._score_without_judge(prediction)
        else:
            payload = await self._score_with_judge_details(
                prediction=prediction,
                raw_problem=raw_problem,
                answer_payload=answer_payload,
                log_context=log_context or _state_log_context(state),
            )

        payload.setdefault("prediction_chars", len(prediction))
        payload.setdefault("problem_chars", len(raw_problem))
        if state is not None:
            state["deepseekmath_v2_reward_payload"] = payload
        LOGGER.info("DeepSeekMath-V2 Prime-RL reward payload: %s", json.dumps(payload, ensure_ascii=False)[:4000])
        return payload

    async def _score_with_judge_details(
        self,
        *,
        prediction: str,
        raw_problem: str,
        answer_payload: dict[str, Any],
        log_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        verifier = self.verifier
        parsed = verifier.parse_prediction(prediction)
        fatal_format_errors = [error for error in parsed.format_errors if error in verifier.FATAL_FORMAT_ERRORS]
        format_score = verifier._format_score(parsed, fatal_format_errors)
        if self.require_format and fatal_format_errors:
            return {
                "format_ok": False,
                "format_score": format_score,
                "format_errors": parsed.format_errors,
                "fatal_format_errors": fatal_format_errors,
                "proof": parsed.solution,
                "solution": parsed.solution,
                "proof_score": 0.0,
                "self_score": parsed.self_score,
                "self_eval_score": 0.0,
                "score_alignment": 0.0,
                "reward": 0.0,
                "judge_backend": self.judge_backend,
                "judge_skipped": True,
                "judge_skip_reason": "fatal_format_error",
            }

        forwarded_self_evaluation, self_eval_clipped = verifier._clip_middle_text(
            parsed.self_evaluation,
            verifier.MAX_FORWARDED_SELF_EVALUATION_CHARS,
        )
        values = {
            "question": raw_problem,
            "proof": parsed.solution,
            "solution": parsed.solution,
            "proof_analysis": forwarded_self_evaluation,
            "proof analysis": forwarded_self_evaluation,
            "self_evaluation": forwarded_self_evaluation,
            "prediction": prediction,
            "label": {"problem": raw_problem, "solution": answer_payload.get("solution", "")},
        }
        total_cost = 0.0
        proof_prompt = verifier._render_template(verifier.proof_prompt_template, values)
        _log_llm_input("proof_verifier", proof_prompt, context=log_context)
        proof_content, proof_cost = await verifier._call_judge(
            proof_prompt,
            verifier.config.proof_model(),
            "proof",
        )
        total_cost += proof_cost
        proof_score = verifier._extract_score(proof_content)
        if proof_score is None:
            return {
                "format_ok": parsed.format_ok,
                "format_score": format_score,
                "format_errors": parsed.format_errors,
                "fatal_format_errors": fatal_format_errors,
                "proof": parsed.solution,
                "solution": parsed.solution,
                "error": "missing_or_invalid_proof_judge_score",
                "proof_judge_response": proof_content[:4000],
                "proof_judge_response_tail": proof_content[-4000:],
                "proof_score": 0.0,
                "self_score": parsed.self_score,
                "self_eval_score": 0.0,
                "score_alignment": 0.0,
                "reward": 0.0,
                "judge_cost": total_cost,
            }

        meta_content = ""
        if self.enable_meta_verification and parsed.self_evaluation:
            meta_prompt = verifier._render_template(verifier.meta_prompt_template, values)
            _log_llm_input("meta_verifier", meta_prompt, context=log_context)
            meta_content, meta_cost = await verifier._call_judge(
                meta_prompt,
                verifier.config.meta_model(),
                "meta",
            )
            total_cost += meta_cost
            self_eval_score = verifier._extract_score(meta_content)
            if self_eval_score is None:
                self_eval_score = 0.0
        else:
            self_eval_score = 0.0
        self_score = parsed.self_score if parsed.self_score is not None else 0.0
        score_alignment = max(0.0, 1.0 - abs(self_score - proof_score))
        base_reward = self.proof_weight * proof_score + self.self_eval_weight * score_alignment * self_eval_score
        reward = max(0.0, min(1.0, format_score * base_reward))
        return {
            "format_ok": parsed.format_ok,
            "format_score": format_score,
            "format_errors": parsed.format_errors,
            "fatal_format_errors": fatal_format_errors,
            "proof": parsed.solution,
            "solution": parsed.solution,
            "self_evaluation": parsed.self_evaluation,
            "proof_score": proof_score,
            "self_score": parsed.self_score,
            "self_eval_score": self_eval_score,
            "score_alignment": score_alignment,
            "base_reward": base_reward,
            "reward": reward,
            "proof_judge_response": proof_content,
            "meta_judge_response": meta_content,
            "meta_verification_enabled": self.enable_meta_verification,
            "self_eval_clipped": self_eval_clipped,
            "judge_cost": total_cost,
            "judge_backend": self.judge_backend,
            "nonfatal_format_errors_allowed": bool(parsed.format_errors),
        }

    def _score_without_judge(self, prediction: str) -> dict[str, Any]:
        from open_instruct.ground_truth_utils import DeepSeekMathV2Verifier

        parsed = DeepSeekMathV2Verifier.parse_prediction(prediction)
        fatal_errors = [
            error for error in parsed.format_errors if error in DeepSeekMathV2Verifier.FATAL_FORMAT_ERRORS
        ]
        if fatal_errors:
            format_score = 0.0
        elif parsed.format_ok:
            format_score = 1.0
        else:
            format_score = max(0.0, min(1.0, self.partial_format_score))
        self_score = parsed.self_score if parsed.self_score is not None else 0.0
        reward = max(0.0, min(1.0, format_score * self_score))
        return {
            "format_ok": parsed.format_ok,
            "format_score": format_score,
            "format_errors": parsed.format_errors,
            "fatal_format_errors": fatal_errors,
            "proof": parsed.solution,
            "solution": parsed.solution,
            "self_evaluation": parsed.self_evaluation,
            "proof_score": None,
            "self_score": parsed.self_score,
            "self_eval_score": None,
            "score_alignment": None,
            "reward": reward,
            "judge_backend": "none",
        }

    async def deepseekmath_v2_reward(
        self,
        completion: Any,
        answer: Any = None,
        problem: str = "",
        question: str = "",
        state: Any | None = None,
        **_: Any,
    ) -> float:
        payload = await self._score(
            completion=completion,
            answer=answer,
            problem=problem,
            question=question,
            state=state,
        )
        return float(payload.get("reward", 0.0) or 0.0)

    def _metric(self, state: Any, key: str, default: float = 0.0) -> float:
        payload = state.get("deepseekmath_v2_reward_payload") if state is not None else None
        if not isinstance(payload, dict):
            return default
        value = payload.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def deepseekmath_v2_format_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "format_score")

    async def deepseekmath_v2_proof_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "proof_score")

    async def deepseekmath_v2_self_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "self_score")

    async def deepseekmath_v2_self_eval_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "self_eval_score")

    async def deepseekmath_v2_score_alignment(self, state: Any, **_: Any) -> float:
        return self._metric(state, "score_alignment")

    async def deepseekmath_v2_final_round_reward(self, state: Any, **_: Any) -> float:
        return self._metric(state, "final_round_reward")

    async def deepseekmath_v2_best_round_reward(self, state: Any, **_: Any) -> float:
        return self._metric(state, "best_round_reward")

    async def deepseekmath_v2_refine_rounds_used(self, state: Any, **_: Any) -> float:
        return self._metric(state, "refine_rounds_used")

    async def deepseekmath_v2_selected_round_index(self, state: Any, **_: Any) -> float:
        return self._metric(state, "selected_round_index")


class DeepSeekMathV2RefinementEnv(DeepSeekMathV2PromptLoggingMixin, vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        max_refine_rounds: int = 3,
        refine_review_n: int = 1,
        refine_reward_mode: str = "selected",
        early_stop_reward: float = 0.95,
        **kwargs: Any,
    ) -> None:
        proof_rubric = kwargs.get("rubric")
        # Need one extra loop after the last model response so env_response can
        # score that response and set final_env_response before max_turns stops.
        super().__init__(max_turns=max(2, int(max_refine_rounds) + 2), **kwargs)
        if not isinstance(proof_rubric, DeepSeekMathV2ProofRubric):
            raise TypeError("DeepSeekMathV2RefinementEnv requires DeepSeekMathV2ProofRubric as rubric")
        self.proof_rubric = proof_rubric
        self.max_refine_rounds = max(0, int(max_refine_rounds))
        self.refine_review_n = max(1, int(refine_review_n))
        mode = str(refine_reward_mode or "selected").strip().lower()
        if mode not in {"selected", "best", "final"}:
            raise ValueError("refine_reward_mode must be one of: selected, best, final")
        self.refine_reward_mode = mode
        self.early_stop_reward = max(0.0, min(1.0, float(early_stop_reward)))

    async def setup_state(self, state: vf.State) -> None:
        state["deepseekmath_v2_rounds"] = []
        state["deepseekmath_v2_best_round_index"] = None
        state["deepseekmath_v2_reward_payload"] = None

    def _input_text(self, state: vf.State, key: str) -> str:
        value = state.get("input", {}).get(key)
        return str(value or "").strip()

    def _problem(self, state: vf.State) -> str:
        return self._input_text(state, "problem") or self._input_text(state, "question")

    def _answer(self, state: vf.State) -> Any:
        return state.get("input", {}).get("answer", "")

    def _question_prompt(self, state: vf.State) -> str:
        prompt = state.get("prompt") or []
        return completion_to_text(prompt)

    @staticmethod
    def _score_value(payload: dict[str, Any]) -> float:
        try:
            return float(payload.get("reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _proof_analysis(payload: dict[str, Any]) -> str:
        analysis = payload.get("proof_judge_response") or payload.get("proof_analysis")
        if not analysis:
            analysis = payload.get("reasoning")
        if isinstance(analysis, dict):
            analysis = json.dumps(analysis, ensure_ascii=False)
        analysis_text = str(analysis or "").strip()
        if not analysis_text:
            analysis_text = (
                "No verifier analysis was available. Re-check the candidate solution carefully, "
                "fix any gaps, and provide a complete solution plus faithful self evaluation."
            )
        score = payload.get("proof_score")
        reward = payload.get("reward")
        return f"Verifier score: {score}; reward: {reward}\n\n{analysis_text}"

    def _select_payload(self, state: vf.State) -> dict[str, Any] | None:
        rounds = state.get("deepseekmath_v2_rounds") or []
        if not rounds:
            return None
        if self.refine_reward_mode == "final":
            return dict(rounds[-1].get("payload") or {})
        best_idx = state.get("deepseekmath_v2_best_round_index")
        if best_idx is None:
            best_idx = max(
                range(len(rounds)),
                key=lambda idx: self._score_value(rounds[idx].get("payload") or {}),
            )
        return dict(rounds[int(best_idx)].get("payload") or {})

    async def _score_latest_turn(self, messages: vf.Messages, state: vf.State) -> dict[str, Any]:
        prediction = completion_to_text([messages[-1]]) if messages else ""
        round_idx = len(state.get("deepseekmath_v2_rounds") or [])
        payload = await self.proof_rubric._score(
            completion=prediction,
            answer=self._answer(state),
            problem=self._problem(state),
            question=self._question_prompt(state),
            state=None,
            log_context=_state_log_context(state, round=round_idx),
        )
        payload = dict(payload)
        payload["round_index"] = round_idx
        rounds = state.setdefault("deepseekmath_v2_rounds", [])
        rounds.append(
            {
                "round_index": round_idx,
                "prediction": prediction,
                "payload": payload,
            }
        )
        best_idx = state.get("deepseekmath_v2_best_round_index")
        if best_idx is None or self._score_value(payload) >= self._score_value(rounds[int(best_idx)]["payload"]):
            state["deepseekmath_v2_best_round_index"] = round_idx
        selected_payload = self._select_payload(state) or payload
        selected_payload["selected_round_index"] = (
            round_idx if self.refine_reward_mode == "final" else state.get("deepseekmath_v2_best_round_index", round_idx)
        )
        selected_payload["refine_rounds_used"] = max(0, len(rounds) - 1)
        selected_payload["final_round_reward"] = self._score_value(payload)
        selected_payload["best_round_reward"] = self._score_value(rounds[int(state["deepseekmath_v2_best_round_index"])]["payload"])
        state["deepseekmath_v2_reward_payload"] = selected_payload
        LOGGER.info(
            "DeepSeekMath-V2 refine round scored: round=%d reward=%.4f selected_round=%s best_reward=%.4f",
            round_idx,
            self._score_value(payload),
            selected_payload.get("selected_round_index"),
            selected_payload.get("best_round_reward"),
        )
        return payload

    async def env_response(self, messages: vf.Messages, state: vf.State, **_: Any) -> vf.Messages:
        payload = await self._score_latest_turn(messages, state)
        round_idx = int(payload.get("round_index", 0))
        reward = self._score_value(payload)
        if round_idx >= self.max_refine_rounds or reward >= self.early_stop_reward:
            state["final_env_response"] = []
            return []

        rounds = state.get("deepseekmath_v2_rounds") or []
        latest_payload = rounds[-1].get("payload") if rounds else {}
        latest_prediction = str(latest_payload.get("solution") or rounds[-1].get("prediction") or "")
        analyses = [self._proof_analysis(payload)]
        refinement_prompt = build_deepseek_proof_refinement_prompt(
            self._problem(state),
            latest_prediction,
            analyses[: self.refine_review_n],
        )
        _log_llm_input(
            f"proof_refinement_instruction_round_{round_idx + 1}",
            [{"role": "user", "content": refinement_prompt}],
            context=_state_log_context(state, round=round_idx + 1),
        )
        return [vf.UserMessage(content=refinement_prompt)]


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def load_environment(
    dataset_path: str,
    problem_column: str = "auto",
    solution_column: str = "auto",
    max_examples: int | None = None,
    judge_backend: str = "api",
    llm_judge_model: str = "deepseek/deepseek-v4-flash",
    llm_judge_base_url: str | None = "https://openrouter.ai/api/v1",
    llm_judge_api_key: str | None = None,
    llm_judge_api_key_env: str = "OPENROUTER_API_KEY",
    max_tokens: int = 40000,
    max_context_length: int = 40000,
    context_margin_tokens: int = 256,
    min_completion_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    timeout: int = 1800,
    extra_body_json: str | None = None,
    proof_weight: float = 0.76,
    self_eval_weight: float = 0.24,
    partial_format_score: float = 0.7,
    enable_meta_verification: bool | str = True,
    require_format: bool | str = True,
    refine_rounds: int = 0,
    refine_review_n: int = 1,
    refine_reward_mode: str = "selected",
    refine_early_stop_reward: float = 0.95,
    **_: Any,
) -> vf.Environment:
    rows = _normalize_dataset_rows(
        _read_dataset_rows(dataset_path),
        problem_column=problem_column,
        solution_column=solution_column,
        max_examples=max_examples,
    )
    LOGGER.info("Loaded DeepSeekMath-V2 proof env dataset: path=%s rows=%d", dataset_path, len(rows))
    dataset = Dataset.from_list(rows)
    rubric = DeepSeekMathV2ProofRubric(
        judge_backend=judge_backend,
        llm_judge_model=llm_judge_model,
        llm_judge_base_url=llm_judge_base_url,
        llm_judge_api_key=llm_judge_api_key,
        llm_judge_api_key_env=llm_judge_api_key_env,
        max_tokens=int(max_tokens),
        max_context_length=int(max_context_length),
        context_margin_tokens=int(context_margin_tokens),
        min_completion_tokens=int(min_completion_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        timeout=int(timeout),
        extra_body_json=extra_body_json,
        proof_weight=float(proof_weight),
        self_eval_weight=float(self_eval_weight),
        partial_format_score=float(partial_format_score),
        enable_meta_verification=_parse_bool(enable_meta_verification, True),
        require_format=_parse_bool(require_format, True),
    )
    if int(refine_rounds) > 0:
        return DeepSeekMathV2RefinementEnv(
            dataset=dataset,
            eval_dataset=dataset,
            rubric=rubric,
            message_type="chat",
            max_refine_rounds=int(refine_rounds),
            refine_review_n=int(refine_review_n),
            refine_reward_mode=refine_reward_mode,
            early_stop_reward=float(refine_early_stop_reward),
        )
    return DeepSeekMathV2SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=rubric,
        message_type="chat",
    )
