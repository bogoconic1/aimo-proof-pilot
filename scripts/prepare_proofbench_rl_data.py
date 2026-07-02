#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""


def build_deepseekmath_v2_generation_prompt(problem: str) -> str:
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
{problem}"""


def find_column(columns: list[str], requested: str, aliases: tuple[str, ...] = ()) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in (requested, *aliases):
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    expected = ", ".join(repr(candidate) for candidate in (requested, *aliases))
    raise ValueError(f"Missing required column {expected}; found columns={columns}")


def build_messages(problem: str, prompt_mode: str) -> list[dict[str, str]]:
    content = problem.strip()
    if prompt_mode == "deepseekmath_v2":
        content = build_deepseekmath_v2_generation_prompt(content)
    return [{"role": "user", "content": content}]


def build_ground_truth(problem: str, solution: str, ground_truth_format: str) -> str:
    if ground_truth_format == "solution":
        return solution
    return json.dumps(
        {"problem": problem, "solution": solution},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare proofbench_v2.csv for Open-Instruct GRPO. The script reads only "
            "the problem and solution source columns, normalizes them, and optionally "
            "adds the derived RL fields expected by rlvr_tokenize_v1."
        )
    )
    parser.add_argument(
        "--input",
        default="submissions-instructions/proofbench_v2.csv",
        help="Input CSV. Defaults to the repository proofbench_v2.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <input-stem>_grpo.parquet.",
    )
    parser.add_argument("--problem_column", default="problem", help="Problem column name, case-insensitive.")
    parser.add_argument("--solution_column", default="solution", help="Solution column name, case-insensitive.")
    parser.add_argument(
        "--schema",
        choices=("grpo", "raw"),
        default="grpo",
        help="raw writes only problem/solution; grpo also writes messages/ground_truth/dataset.",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=("deepseekmath_v2", "raw_problem"),
        default="deepseekmath_v2",
        help="How to build the GRPO user message. deepseekmath_v2 uses the paper appendix generation prompt.",
    )
    parser.add_argument(
        "--ground_truth_format",
        choices=("json", "solution"),
        default="json",
        help="json stores both problem and solution in ground_truth; solution stores only the reference solution text.",
    )
    parser.add_argument(
        "--dataset_source",
        default="proof_math",
        help="Verifier source stored in the GRPO dataset column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke tests. 0 keeps all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_grpo.parquet")

    df = pd.read_csv(input_path)
    problem_column = find_column(list(df.columns), args.problem_column, aliases=("question",))
    solution_column = find_column(list(df.columns), args.solution_column)
    df = df[[problem_column, solution_column]].rename(
        columns={problem_column: "problem", solution_column: "solution"}
    )
    df["problem"] = df["problem"].fillna("").astype(str).str.strip()
    df["solution"] = df["solution"].fillna("").astype(str).str.strip()
    df = df[(df["problem"] != "") & (df["solution"] != "")].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    if args.schema == "grpo":
        df["messages"] = df["problem"].map(lambda problem: build_messages(problem, args.prompt_mode))
        df["ground_truth"] = [
            build_ground_truth(problem, solution, args.ground_truth_format)
            for problem, solution in zip(df["problem"], df["solution"], strict=True)
        ]
        df["dataset"] = args.dataset_source

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = df.to_dict(orient="records")
    if output_path.suffix == ".jsonl":
        write_jsonl(rows, output_path)
    elif output_path.suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output suffix must be .parquet, .pq, .jsonl, or .csv")

    print(
        f"Wrote {len(df)} rows to {output_path} "
        f"using source columns {problem_column!r}, {solution_column!r}; "
        f"schema={args.schema} prompt_mode={args.prompt_mode}"
    )
    print("columns", list(df.columns))


if __name__ == "__main__":
    main()
