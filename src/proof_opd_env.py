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

MAX_FORWARDED_EVALUATION_CHARS = 12_000
MAX_FORWARDED_META_ANALYSIS_CHARS = 8_000
HEADER_SUFFIX_PATTERN = r"(?:\s*//[^\n]*)?\s*$"
BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}")


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


def build_deepseek_proof_verification_prompt(question: str, proof: str) -> str:
    return f"""## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{question}

## Solution
{proof}"""


def build_deepseek_meta_verification_prompt(question: str, proof: str, proof_analysis: str) -> str:
    proof_analysis, _ = clip_middle_text(proof_analysis, MAX_FORWARDED_EVALUATION_CHARS)
    return f"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
{EVALUATION_RUBRIC}
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.
2. You need to analyze the content of the "solution evaluation" from three aspects: Step Restatement, Defect Analysis, Expression Analysis, and Score Analysis.
3. The most important part is Defect Analysis: check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable.

You should rate the "solution evaluation" with:
- 1 if the evaluation's defect analysis and final score are reasonable.
- 0.5 if the evaluation is generally useful but has minor issues.
- 0 if the evaluation is misleading, ignores major issues, fabricates defects, or gives an unreasonable final score.

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""


def build_deepseek_proof_refinement_prompt(question: str, proof: str, proof_analyses: list[str]) -> str:
    analyses = "\n\n".join(f"### Evaluation {idx + 1}\n{analysis}" for idx, analysis in enumerate(proof_analyses))
    return f"""{build_deepseek_proof_generation_prompt(question)}

## Candidate Solution(s) to Refine
Here is a solution sample along with correctness evaluation(s). Provide a better solution by solving issues mentioned in the evaluations, reusing promising ideas from the solution, or both.

### Candidate Solution
{proof}

{analyses}

## Final Instruction
Your final response must follow the format above, including a `## Solution` section followed by a `## Self Evaluation` section."""


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return default if not text else text not in {"0", "false", "no", "off"}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def strip_reasoning_blocks(text: str) -> str:
    visible = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    visible = re.sub(r"(?is)^.*?</think>", "", visible)
    return visible.strip()


def has_closed_thinking(text: str) -> bool:
    return "</think>" in str(text or "").lower()


def coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.strip("$` .,;:")
    if text in {"1", "1.0", "correct"}:
        return 1.0
    if text in {"0.5", ".5", "1/2", "half", "partial"}:
        return 0.5
    if text in {"0", "0.0", "incorrect"}:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    if number in {0.0, 0.5, 1.0}:
        return number
    return None


def extract_boxed_score(text: str) -> float | None:
    scores = [coerce_score(match.group(1)) for match in BOXED_PATTERN.finditer(text or "")]
    scores = [score for score in scores if score is not None]
    if scores:
        return scores[-1]
    fallback = re.findall(r"(?i)\b(?:score|rating)[^0-9]{0,40}(0\.5|1/2|1(?:\.0)?|0(?:\.0)?)\b", text or "")
    if fallback:
        return coerce_score(fallback[-1])
    return None


def clip_middle_text(text: str, max_chars: int) -> tuple[str, bool]:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n\n...[clipped {len(text) - max_chars} chars]...\n\n" + text[-tail:], True


def header_matches(text: str, header: str) -> list[re.Match[str]]:
    header_pattern = re.escape(header.strip()).replace(r"\ ", r"[ \t]+")
    return list(re.finditer(rf"(?im)^[ \t]*{header_pattern}{HEADER_SUFFIX_PATTERN}", text or ""))


def parse_generation_response(text: str) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    closed_thinking = has_closed_thinking(text)
    solution_headers = header_matches(visible, "## Solution")
    evaluation_headers = header_matches(visible, "## Self Evaluation")
    if not solution_headers:
        proof = ""
        self_evaluation = ""
        has_solution = False
        has_self_eval = False
    else:
        solution_header = solution_headers[-1]
        following_eval = next((m for m in evaluation_headers if m.start() > solution_header.end()), None)
        has_solution = True
        has_self_eval = following_eval is not None
        if following_eval is None:
            proof = visible[solution_header.end() :].strip()
            self_evaluation = ""
        else:
            proof = visible[solution_header.end() : following_eval.start()].strip()
            self_evaluation = visible[following_eval.end() :].strip()
    self_score = extract_boxed_score(self_evaluation)
    return {
        "raw_chars": len(text or ""),
        "closed_thinking": closed_thinking,
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": has_solution,
        "has_self_evaluation_section": has_self_eval,
        "format_ok": bool(has_solution and proof and has_self_eval and self_score is not None),
    }


def extract_marked_section(text: str, markers: tuple[str, ...], max_chars: int) -> tuple[str, bool]:
    visible = strip_reasoning_blocks(text)
    lower = visible.lower()
    start = -1
    marker_len = 0
    for marker in markers:
        idx = lower.rfind(marker.lower())
        if idx > start:
            start = idx
            marker_len = len(marker)
    section = visible[start + marker_len :].strip() if start >= 0 else visible.strip()
    score_idx = section.lower().rfind("based on")
    if score_idx > 0:
        section = section[:score_idx].strip()
    return clip_middle_text(section, max_chars)


def parse_verifier_response(text: str) -> dict[str, Any]:
    evaluation, clipped = extract_marked_section(
        text,
        ("Here is my evaluation of the solution:", "Here is my evaluation"),
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    return {
        "evaluation": evaluation,
        "score": extract_boxed_score(text),
        "evaluation_clipped": clipped,
        "raw_chars": len(text or ""),
        "closed_thinking": has_closed_thinking(text),
    }


def verifier_invalid_reason(verifier: dict[str, Any]) -> str:
    if verifier.get("score") is None:
        return "missing_or_invalid_boxed_score"
    if not str(verifier.get("evaluation") or "").strip():
        return "empty_verifier_evaluation"
    return ""


def is_valid_verifier_response(verifier: dict[str, Any]) -> bool:
    return verifier_invalid_reason(verifier) == ""


def parse_meta_verifier_response(text: str) -> dict[str, Any]:
    analysis, clipped = extract_marked_section(
        text,
        ('Here is my analysis of the "solution evaluation":', "Here is my analysis"),
        MAX_FORWARDED_META_ANALYSIS_CHARS,
    )
    return {
        "analysis": analysis,
        "score": extract_boxed_score(text),
        "analysis_clipped": clipped,
        "raw_chars": len(text or ""),
        "closed_thinking": has_closed_thinking(text),
    }


def trajectory_step_text(step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    return completion_to_text(step.get("completion"))


def trajectory_step_finish_reason(step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    response = step.get("response")
    message = getattr(response, "message", None)
    finish_reason = getattr(message, "finish_reason", None)
    if finish_reason:
        return str(finish_reason)
    if isinstance(response, dict):
        for key in ("finish_reason", "finishReasons", "stop_reason"):
            if response.get(key):
                return str(response[key])
        response_message = response.get("message")
        if isinstance(response_message, dict):
            for key in ("finish_reason", "finishReasons", "stop_reason"):
                if response_message.get(key):
                    return str(response_message[key])
    return ""


def trajectory_step_is_truncated(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if bool(step.get("is_truncated")):
        return True
    tokens = step.get("tokens")
    if isinstance(tokens, dict) and bool(tokens.get("is_truncated")):
        return True
    response = step.get("response")
    message = getattr(response, "message", None)
    if bool(getattr(message, "is_truncated", False)):
        return True
    finish_reason = trajectory_step_finish_reason(step).lower()
    return finish_reason in {"length", "max_tokens", "token_limit"}


def json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
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
        text = message_content_to_text(content).strip()
        if not text:
            continue
        all_texts.append(text)
        if role == "assistant":
            assistant_texts.append(text)
    return "\n\n".join(assistant_texts or all_texts)


def log_llm_input(stage: str, prompt: str, *, state: Any | None = None) -> None:
    if not parse_bool(os.environ.get("PROOF_OPD_LOG_LLM_INPUTS"), True):
        return
    max_chars = int(os.environ.get("PROOF_OPD_LOG_LLM_INPUT_MAX_CHARS", "0") or "0")
    shown, clipped = clip_middle_text(prompt, max_chars) if max_chars > 0 else (prompt, False)
    input_payload = state.get("input", {}) if isinstance(state, dict) else {}
    LOGGER.info(
        "Proof-OPD LLM input stage=%s task_id=%s source_index=%s chars=%d clipped=%s\n%s",
        stage,
        input_payload.get("task_id"),
        input_payload.get("source_index"),
        len(prompt),
        clipped,
        shown,
    )


def read_dataset_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Proof-OPD dataset not found: {path}")
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
        raise ValueError(f"Unsupported Proof-OPD dataset extension: {suffix}")
    return frame.to_dict(orient="records")


def resolve_column(row: dict[str, Any], requested: str, candidates: list[str]) -> str | None:
    if requested and requested != "auto":
        return requested if requested in row else None
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        if candidate in row:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def extract_problem_from_messages(value: Any) -> str:
    messages = json_loads_maybe(value)
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = message_content_to_text(message.get("content")).strip()
        match = re.search(r"(?ims)^##[ \t]+Problem[ \t]*\n(?P<problem>.*)$", text)
        return match.group("problem").strip() if match else text
    return ""


def normalize_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    problem_column: str,
    solution_column: str,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        problem_key = resolve_column(row, problem_column, ["problem", "question", "Problem", "Question"])
        solution_key = resolve_column(row, solution_column, ["solution", "answer", "Solution", "Answer"])
        problem = str(row.get(problem_key) or "").strip() if problem_key else ""
        if not problem and "messages" in row:
            problem = extract_problem_from_messages(row.get("messages"))
        if not problem:
            continue
        solution = str(row.get(solution_key) or "").strip() if solution_key else ""
        task_id = str(row.get("task_id") or row.get("id") or row.get("problem_id") or index)
        answer = {"problem": problem}
        if solution:
            answer["solution"] = solution
        normalized.append(
            {
                "question": build_deepseek_proof_generation_prompt(problem),
                "problem": problem,
                "solution": solution,
                "answer": json.dumps(answer, ensure_ascii=False),
                "dataset": "proof_math",
                "task_id": task_id,
                "source_index": index,
                "info": {
                    "stage": "proof_generation",
                    "task_id": task_id,
                    "source_index": index,
                },
            }
        )
        if max_examples is not None and max_examples > 0 and len(normalized) >= max_examples:
            break
    if not normalized:
        raise ValueError("Proof-OPD dataset produced zero usable rows.")
    return normalized


class ProofOPDRubric(vf.Rubric):
    def __init__(self) -> None:
        super().__init__()
        self.add_reward_func(self.proof_opd_reward)
        self.add_metric(self.proof_opd_format_score)
        self.add_metric(self.proof_opd_proof_score)
        self.add_metric(self.proof_opd_meta_score)
        self.add_metric(self.proof_opd_round_index)

    async def proof_opd_reward(self, state: Any, **_: Any) -> float:
        payload = state.get("proof_opd_reward_payload") if isinstance(state, dict) else None
        return float((payload or {}).get("reward", 0.0) or 0.0)

    async def proof_opd_format_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "format_score")

    async def proof_opd_proof_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "proof_score")

    async def proof_opd_meta_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "meta_score")

    async def proof_opd_round_index(self, state: Any, **_: Any) -> float:
        return self._metric(state, "selected_round_index", -1.0)

    @staticmethod
    def _metric(state: Any, key: str, default: float = 0.0) -> float:
        payload = state.get("proof_opd_reward_payload") if isinstance(state, dict) else None
        value = (payload or {}).get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class ProofOPDEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        refine_rounds: int = 1,
        enable_meta_verification: bool = True,
        partial_format_score: float = 0.7,
        refine_early_stop_reward: float = 0.95,
        require_closed_think: bool | str = True,
        **kwargs: Any,
    ) -> None:
        self.refine_rounds = max(0, int(refine_rounds))
        self.enable_meta_verification = bool(enable_meta_verification)
        self.partial_format_score = clamp01(float(partial_format_score))
        self.refine_early_stop_reward = clamp01(float(refine_early_stop_reward))
        self.require_closed_think = parse_bool(require_closed_think, True)
        turns_per_round = 3 if self.enable_meta_verification else 2
        super().__init__(max_turns=turns_per_round * (self.refine_rounds + 1) + 1, **kwargs)

    async def setup_state(self, state: vf.State) -> None:
        state["proof_opd_stage"] = "proof"
        state["proof_opd_current_round"] = 0
        state["proof_opd_rounds"] = []
        state["proof_opd_stage_records"] = []
        state["proof_opd_reward_payload"] = None

    def _input_value(self, state: vf.State, key: str) -> str:
        value = state.get("input", {}).get(key)
        return str(value or "").strip()

    def _problem(self, state: vf.State) -> str:
        return self._input_value(state, "problem") or self._input_value(state, "question")

    def _format_score(self, parsed: dict[str, Any]) -> float:
        if not parsed.get("has_solution_section") or not str(parsed.get("proof") or "").strip():
            return 0.0
        if parsed.get("format_ok"):
            return 1.0
        return self.partial_format_score

    def _generation_invalid_reason(self, parsed: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not parsed.get("closed_thinking"):
            return "missing_closed_think"
        if not parsed.get("has_solution_section") or not str(parsed.get("proof") or "").strip():
            return "missing_solution_section_or_empty_proof"
        return ""

    def _verifier_invalid_reason(self, verifier: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not verifier.get("closed_thinking"):
            return "missing_closed_think"
        return verifier_invalid_reason(verifier)

    def _meta_invalid_reason(self, meta: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not meta.get("closed_thinking"):
            return "missing_closed_think"
        if meta.get("score") is None:
            return "missing_or_invalid_boxed_score"
        if not str(meta.get("analysis") or "").strip():
            return "empty_meta_analysis"
        return ""

    def _invalid_generation_payload(
        self,
        parsed: dict[str, Any],
        round_idx: int,
        reason: str,
        *,
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> dict[str, Any]:
        return {
            "round_index": round_idx,
            "selected_round_index": round_idx,
            "reward": 0.0,
            "format_score": 0.0,
            "format_ok": False,
            "proof_score": 0.0,
            "meta_score": 0.0 if self.enable_meta_verification else 1.0,
            "self_score": parsed.get("self_score"),
            "proof_chars": len(str(parsed.get("proof") or "")),
            "closed_thinking": parsed.get("closed_thinking", False),
            "is_truncated": is_truncated,
            "finish_reason": finish_reason,
            "reason": reason,
        }

    def _record_stage(
        self,
        state: vf.State,
        *,
        stage: str,
        parsed: dict[str, Any],
        invalid_reason: str = "",
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> None:
        state.setdefault("proof_opd_stage_records", []).append(
            {
                "stage": stage,
                "round_index": int(state.get("proof_opd_current_round", 0)),
                "raw_chars": int(parsed.get("raw_chars") or 0),
                "closed_thinking": bool(parsed.get("closed_thinking")),
                "is_truncated": bool(is_truncated),
                "finish_reason": finish_reason,
                "invalid_reason": invalid_reason,
            }
        )

    def _last_step_status(self, state: vf.State) -> tuple[str, bool, str]:
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return "", False, ""
        step = trajectory[-1]
        return trajectory_step_text(step), trajectory_step_is_truncated(step), trajectory_step_finish_reason(step)

    def _stop(self, state: vf.State) -> vf.Messages:
        state["final_env_response"] = []
        return []

    def _finalize_round(self, state: vf.State) -> dict[str, Any]:
        round_idx = int(state.get("proof_opd_current_round", 0))
        generation = dict(state.get("proof_opd_generation") or {})
        verifier = dict(state.get("proof_opd_verifier") or {})
        meta = dict(state.get("proof_opd_meta") or {})
        format_score = self._format_score(generation)
        proof_score = verifier.get("score")
        meta_score = meta.get("score") if self.enable_meta_verification else 1.0
        proof_score_value = 0.0 if proof_score is None else float(proof_score)
        meta_score_value = 0.0 if meta_score is None else float(meta_score)
        reward = clamp01(format_score * proof_score_value * meta_score_value)
        payload = {
            "round_index": round_idx,
            "reward": reward,
            "format_score": format_score,
            "format_ok": generation.get("format_ok", False),
            "proof_score": proof_score,
            "meta_score": meta_score,
            "self_score": generation.get("self_score"),
            "proof_chars": len(str(generation.get("proof") or "")),
            "self_evaluation_chars": len(str(generation.get("self_evaluation") or "")),
            "verifier_evaluation_chars": len(str(verifier.get("evaluation") or "")),
            "meta_analysis_chars": len(str(meta.get("analysis") or "")),
            "proof": generation.get("proof", ""),
            "verifier_evaluation": verifier.get("evaluation", ""),
            "verifier_invalid_reason": verifier.get("invalid_reason", ""),
            "meta_analysis": meta.get("analysis", ""),
            "meta_invalid_reason": meta.get("invalid_reason", ""),
            "stage_records": list(state.get("proof_opd_stage_records") or []),
        }
        rounds = state.setdefault("proof_opd_rounds", [])
        rounds.append(payload)
        best_idx = max(range(len(rounds)), key=lambda idx: float(rounds[idx].get("reward", 0.0) or 0.0))
        selected = dict(rounds[best_idx])
        selected["selected_round_index"] = best_idx
        selected["final_round_reward"] = reward
        selected["best_round_reward"] = float(rounds[best_idx].get("reward", 0.0) or 0.0)
        selected["refine_rounds_used"] = max(0, len(rounds) - 1)
        state["proof_opd_reward_payload"] = selected
        LOGGER.info("Proof-OPD round scored: %s", json.dumps(selected, ensure_ascii=False)[:4000])
        return selected

    def _should_refine(self, state: vf.State, payload: dict[str, Any]) -> bool:
        rounds = state.get("proof_opd_rounds") or []
        if len(rounds) > self.refine_rounds:
            return False
        return float(payload.get("reward", 0.0) or 0.0) < self.refine_early_stop_reward

    async def env_response(self, messages: vf.Messages, state: vf.State, **_: Any) -> vf.Messages:
        return []

    async def _advance_after_completion(self, state: vf.State) -> vf.Messages:
        stage = str(state.get("proof_opd_stage") or "proof")
        text, is_truncated, finish_reason = self._last_step_status(state)
        problem = self._problem(state)
        round_idx = int(state.get("proof_opd_current_round", 0))

        if stage in {"proof", "refine"}:
            parsed = parse_generation_response(text)
            state["proof_opd_generation"] = parsed
            invalid_reason = self._generation_invalid_reason(parsed, is_truncated)
            self._record_stage(
                state,
                stage=stage,
                parsed=parsed,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            if invalid_reason:
                payload = self._invalid_generation_payload(
                    parsed,
                    round_idx,
                    invalid_reason,
                    is_truncated=is_truncated,
                    finish_reason=finish_reason,
                )
                payload["stage_records"] = list(state.get("proof_opd_stage_records") or [])
                state.setdefault("proof_opd_rounds", []).append(payload)
                state["proof_opd_reward_payload"] = payload
                LOGGER.info("Proof-OPD invalid generation: %s", json.dumps(payload, ensure_ascii=False))
                return self._stop(state)
            prompt = build_deepseek_proof_verification_prompt(problem, parsed["proof"])
            log_llm_input("verifier", prompt, state=state)
            state["proof_opd_stage"] = "verifier"
            return [vf.UserMessage(content=prompt)]

        if stage == "verifier":
            verifier = parse_verifier_response(text)
            state["proof_opd_verifier"] = verifier
            invalid_reason = self._verifier_invalid_reason(verifier, is_truncated)
            self._record_stage(
                state,
                stage="verifier",
                parsed=verifier,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            if invalid_reason:
                verifier["invalid_reason"] = invalid_reason
                LOGGER.info(
                    "Proof-OPD skipping meta verifier: invalid verifier output "
                    "reason=%s score=%s evaluation_chars=%d raw_chars=%d truncated=%s finish_reason=%s",
                    invalid_reason,
                    verifier.get("score"),
                    len(str(verifier.get("evaluation") or "")),
                    int(verifier.get("raw_chars") or 0),
                    is_truncated,
                    finish_reason,
                )
            if self.enable_meta_verification and not invalid_reason:
                proof = str((state.get("proof_opd_generation") or {}).get("proof") or "")
                prompt = build_deepseek_meta_verification_prompt(problem, proof, verifier["evaluation"])
                log_llm_input("meta_verifier", prompt, state=state)
                state["proof_opd_stage"] = "meta"
                return [vf.UserMessage(content=prompt)]
            payload = self._finalize_round(state)
            if self._should_refine(state, payload):
                return self._next_refinement_prompt(state, payload)
            return self._stop(state)

        if stage == "meta":
            meta = parse_meta_verifier_response(text)
            invalid_reason = self._meta_invalid_reason(meta, is_truncated)
            if invalid_reason:
                meta["invalid_reason"] = invalid_reason
                LOGGER.info(
                    "Proof-OPD meta verifier invalid: reason=%s score=%s analysis_chars=%d raw_chars=%d "
                    "truncated=%s finish_reason=%s",
                    invalid_reason,
                    meta.get("score"),
                    len(str(meta.get("analysis") or "")),
                    int(meta.get("raw_chars") or 0),
                    is_truncated,
                    finish_reason,
                )
            self._record_stage(
                state,
                stage="meta",
                parsed=meta,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            state["proof_opd_meta"] = meta
            payload = self._finalize_round(state)
            if self._should_refine(state, payload):
                return self._next_refinement_prompt(state, payload)
            return self._stop(state)

        return self._stop(state)

    def _next_refinement_prompt(self, state: vf.State, payload: dict[str, Any]) -> vf.Messages:
        state["proof_opd_current_round"] = int(state.get("proof_opd_current_round", 0)) + 1
        state["proof_opd_stage"] = "refine"
        state["proof_opd_generation"] = None
        state["proof_opd_verifier"] = None
        state["proof_opd_meta"] = None
        analyses = [str(payload.get("verifier_evaluation") or "No verifier analysis was available.")]
        if payload.get("meta_analysis"):
            analyses.append(str(payload["meta_analysis"]))
        prompt = build_deepseek_proof_refinement_prompt(
            self._problem(state),
            str(payload.get("proof") or ""),
            analyses,
        )
        log_llm_input("refinement", prompt, state=state)
        return [vf.UserMessage(content=prompt)]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state.get("trajectory") or []) == 0:
            state["proof_opd_stage"] = "proof"
            log_llm_input("proof_generation", completion_to_text(state["prompt"]), state=state)
            return state["prompt"]
        return await self._advance_after_completion(state)


def load_environment(
    dataset_path: str,
    problem_column: str = "auto",
    solution_column: str = "auto",
    max_examples: int | None = None,
    enable_meta_verification: bool | str = True,
    partial_format_score: float = 0.7,
    require_closed_think: bool | str = True,
    refine_rounds: int = 1,
    refine_early_stop_reward: float = 0.95,
    **_: Any,
) -> vf.Environment:
    rows = normalize_dataset_rows(
        read_dataset_rows(dataset_path),
        problem_column=problem_column,
        solution_column=solution_column,
        max_examples=max_examples,
    )
    LOGGER.info("Loaded Proof-OPD dataset: path=%s rows=%d", dataset_path, len(rows))
    dataset = Dataset.from_list(rows)
    return ProofOPDEnv(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=ProofOPDRubric(),
        message_type="chat",
        refine_rounds=int(refine_rounds),
        enable_meta_verification=parse_bool(enable_meta_verification, True),
        partial_format_score=float(partial_format_score),
        require_closed_think=parse_bool(require_closed_think, True),
        refine_early_stop_reward=float(refine_early_stop_reward),
    )
