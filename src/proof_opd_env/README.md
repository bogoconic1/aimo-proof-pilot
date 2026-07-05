# proof-opd-env

## Overview

- **Environment ID**: `proof_opd_env`
- **Purpose**: evaluate or train proof-oriented OPD traces with a DeepSeekMath-V2-style workflow.
- **Task type**: multi-turn math proof generation, verification, meta-verification, and optional refinement.
- **Primary file**: `proof_opd_env.py`, copied from `aimo-proof-pilot/src/proof_opd_env.py` for standalone `vf-eval` use.

The environment asks the model to solve a proof/math problem, then re-prompts the same model as a verifier and meta-verifier. It is intended for checking whether the OPD environment logic, trace formatting, and metrics behave correctly before running Prime-RL.

## Workflow

Each rollout follows this stage order:

1. **Proof generation**: solve the problem and produce `## Solution` plus `## Self Evaluation`.
2. **Verifier**: evaluate the generated proof and output a score in `\boxed{0}`, `\boxed{0.5}`, or `\boxed{1}`.
3. **Meta-verifier**: evaluate whether the verifier analysis is reasonable.
4. **Refinement**: optionally generate a refined proof from selected verifier critiques, then repeat verification.

By default the env runs `num_verifiers=4`. Reward is:

```text
format_score * average(verifier_score_i * meta_score_i)
```

Invalid verifier output contributes `0`. Invalid meta output for a valid verifier contributes `0.5`.

## Datasets

Supported input formats are `.csv`, `.tsv`, and `.parquet`.

Proof datasets should contain one of:

- `problem`
- `question`
- `messages` as fallback; the first user message is used as the problem text

Verifiable datasets should additionally contain one of:

- `answer`
- `final_answer`
- `gold_answer`

Bundled local examples:

- `test.csv`: tiny smoke-test data.
- `aime_2026.csv`: verifiable math data.
- `astralbench.csv`: verifiable boxed-answer data.
- `proofbench_v3.csv`: proof-style data.

## Quickstart

Run from this folder:

```bash
cd aimo-proof-pilot/proof_opd_env
```

Long-context AIME eval:

```bash
vf-eval proof_opd_env -n 1 -r 1 -k "OPENROUTER_API_KEY" -b "https://openrouter.ai/api/v1" -m "openai/gpt-oss-120b" -a '{"dataset_path": "aime_2026.csv", "verifiable_dataset_path": "aime_2026.csv", "verifiable_fraction": 1}' -s -c 30 --max-retries 5 -t 64000
```

Set the key before running:

```bash
export OPENROUTER_API_KEY="..."
```

## Environment Arguments

Pass these through `vf-eval -a '{...}'`.

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_path` | str | required | Main proof dataset path. |
| `problem_column` | str | `auto` | Problem column override. |
| `solution_column` | str | `auto` | Optional solution column override. |
| `max_examples` | int | `null` | Limit loaded examples before mixing. |
| `verifiable_dataset_path` | str | `null` | Optional answerable dataset used only to add boxed-answer prompts to training. |
| `verifiable_fraction` | float | `0.2` | Fraction of final train rows drawn from the answerable dataset. |
| `verifiable_answer_column` | str | `auto` | Answer column used to filter usable train rows and to score eval rows. |
| `mix_seed` | int | `34521` | Shuffle seed for mixed datasets. |
| `enable_meta_verification` | bool | `true` | Run meta-verifier after valid verifier output. |
| `num_verifiers` | int | `4` | Number of verifier samples per proof. |
| `partial_format_score` | float | `0.7` | Format score for parseable proof missing full self-eval format. |
| `require_closed_think` | bool | `true` | Treat output without `</think>` as invalid. |
| `refine_rounds` | int | `1` | Maximum refinement rounds. |
| `refine_review_n` | int | `2` | Number of verifier critiques used for refinement. |
| `refine_early_stop_reward` | float | `0.95` | Skip refinement if reward is at least this value. |

Example:

```bash
vf-eval proof_opd_env \
  -n 20 -r 1 -m "openai/gpt-oss-120b" \
  -k "OPENROUTER_API_KEY" -b "https://openrouter.ai/api/v1" \
  -a '{"dataset_path": "proofbench_v3.csv", "verifiable_dataset_path": "astralbench.csv", "verifiable_fraction": 0.3, "num_verifiers": 4, "refine_rounds": 1}' \
  -s -c 20 --max-retries 5 -t 64000
```

## Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Main Proof-OPD reward in `[0, 1]`. |
| `proof_opd_format_score` | Format/parseability score for the proof generation stage. |
| `proof_opd_proof_score` | Average verifier score. |
| `proof_opd_meta_score` | Average effective meta-verifier score. |
| `proof_opd_round_index` | Selected/best round index. |

Boxed-answer accuracy is intentionally not emitted as a train metric. Mixed answerable train rows are logged as ordinary proof tasks; track accuracy with a separate verifiable eval dataset instead.

## Saved Completion Trace

This env overrides the default `MultiTurnEnv` completion renderer. Saved JSONL rows now include the full stage trace in `completion`, not only the last assistant message.

Expected order:

```text
assistant proof generation
user verifier prompt
assistant verifier output
user meta-verifier prompt
assistant meta-verifier output
user refinement prompt
assistant refined proof
...
```

The initial proof prompt remains in the top-level `prompt` field. This keeps `completion` useful for debugging full Proof-OPD rollouts while avoiding duplicate initial prompt text.

## Debugging

Useful environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `PROOF_OPD_LOG_LLM_INPUTS` | `true` | Log prompts sent to each stage. |
| `PROOF_OPD_LOG_LLM_INPUT_MAX_CHARS` | `0` | Clip logged prompts when positive. |
| `PROOF_OPD_MAX_FORWARDED_EVALUATION_CHARS` | `0` | Clip verifier text passed to meta-verifier when positive. |
| `PROOF_OPD_MAX_FORWARDED_META_ANALYSIS_CHARS` | `0` | Clip meta text passed onward when positive. |

For local syntax checks:

```bash
python3.11 -m py_compile proof_opd_env.py
```
