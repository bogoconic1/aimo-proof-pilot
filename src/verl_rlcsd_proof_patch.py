from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def _as_float_rewards(batch: Any) -> list[float]:
    reward_tensor = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch.keys() else batch.batch["rm_scores"]
    if reward_tensor.dim() > 1:
        return reward_tensor.sum(dim=-1).detach().cpu().tolist()
    return reward_tensor.detach().cpu().tolist()


def _reward_extra(batch: Any, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None or index >= len(values):
        return default
    value = values[index]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _format_proof_context(problem: str, response: str, reward: float, extra: dict[str, Any]) -> str:
    lines = [
        "## Problem",
        problem.strip(),
        "",
        "## Candidate Proof",
        response.strip(),
        "",
        "## Verifier Signal",
        f"normalized_reward: {reward:.4f}",
    ]
    for key in (
        "proof_self_score",
        "formatted",
        "proof_solution_chars",
        "proof_self_eval_chars",
        "deepseekmath_v2_reward",
        "deepseekmath_v2_correct_rate",
    ):
        if key in extra and extra[key] is not None:
            lines.append(f"{key}: {extra[key]}")
    return "\n".join(lines).strip()


def _build_teacher_context(self: Any, problem: str, response: str, reward: float, extra: dict[str, Any]) -> dict[str, str]:
    sd = __import__("src.self_distill_main", fromlist=["_teacher_chat_template_kwargs"])
    privileged_text = _format_proof_context(problem, response, reward, extra)
    messages = [
        {
            "role": "system",
            "content": (
                "You are training a proof model with contrastive feedback. "
                "Use the provided candidate proof and verifier signal as context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{privileged_text}\n\n"
                "Continue from this context and produce a rigorous proof response in the required format."
            ),
        },
    ]
    prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **sd._teacher_chat_template_kwargs(self),
    )
    return {"prompt": prompt, "text": privileged_text, "answer": f"{reward:.4f}"}


def _sample_other(indices: list[int], current_idx: int) -> int | None:
    candidates = [idx for idx in indices if idx != current_idx]
    if not candidates:
        return None
    return random.choice(candidates)


def _compute_proof_rlcsd_teacher_inputs(self: Any, batch: Any) -> Any | None:
    sd = __import__("src.self_distill_main", fromlist=[
        "RLCSD_DEFAULT_K_MAX",
        "RLCSD_LOSS_MODES",
        "_build_teacher_batch_from_prompts",
        "_build_teacher_multi_batch_from_prompt_groups",
        "_cfg_get",
        "_compute_prefilter_metric_overrides",
    ])
    loss_mode = sd._cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in sd.RLCSD_LOSS_MODES:
        raise ValueError(f"proof RLCSD teacher inputs called with unsupported loss_mode={loss_mode}")

    self._data_metric_overrides = sd._compute_prefilter_metric_overrides(self, batch)
    batch_size = batch.batch.batch_size[0]
    if batch_size == 0:
        batch.meta_info["skip_actor_update"] = True
        self._aligned_teacher_rollout_details = {}
        self._rlcsd_step_metrics = {"rlcsd/skip_actor_update": 1.0}
        return None

    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        raise ValueError("proof rlcsd requires uid in non_tensor_batch to recover rollout groups.")

    problems, _ground_truths, _solutions = self._extract_problems_answers_solutions(batch)
    responses = batch.batch["responses"]
    responses_text = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    rewards = _as_float_rewards(batch)

    positive_threshold = float(sd._custom_cfg_get(self, "rlcsd_positive_threshold", 0.75) or 0.75)
    negative_threshold = float(sd._custom_cfg_get(self, "rlcsd_negative_threshold", 0.25) or 0.25)
    groups: dict[str, list[int]] = {}
    for idx, uid in enumerate(uid_values.tolist()):
        groups.setdefault(str(uid), []).append(idx)

    valid_mask = np.zeros(batch_size, dtype=bool)
    correct_prompt_by_idx: dict[int, str] = {}
    wrong_prompt_by_idx: dict[int, str] = {}
    correct_text_by_idx: dict[int, str] = {}
    wrong_text_by_idx: dict[int, str] = {}
    correct_answer_by_idx: dict[int, str] = {}
    wrong_answer_by_idx: dict[int, str] = {}
    wrong_prompt_group_by_idx: dict[int, list[str]] = {}
    wrong_text_group_by_idx: dict[int, list[str]] = {}
    wrong_answer_group_by_idx: dict[int, list[str]] = {}
    wrong_pool_sizes: list[int] = []
    all_same_outcome_group_count = 0
    middle_sample_count = 0
    valid_group_count = 0
    aborted_group_count = 0

    for group_indices in groups.values():
        positives = [idx for idx in group_indices if rewards[idx] >= positive_threshold]
        negatives = [idx for idx in group_indices if rewards[idx] <= negative_threshold]
        middle_sample_count += len(group_indices) - len(positives) - len(negatives)
        if not positives or not negatives:
            all_same_outcome_group_count += 1
            aborted_group_count += 1
            continue

        problem = problems[group_indices[0]]
        positive_contexts = {
            idx: _build_teacher_context(
                self,
                problem,
                responses_text[idx],
                rewards[idx],
                {
                    key: _reward_extra(batch, key, idx)
                    for key in ("proof_self_score", "formatted", "proof_solution_chars", "proof_self_eval_chars")
                },
            )
            for idx in positives
        }
        negative_contexts = {
            idx: _build_teacher_context(
                self,
                problem,
                responses_text[idx],
                rewards[idx],
                {
                    key: _reward_extra(batch, key, idx)
                    for key in ("proof_self_score", "formatted", "proof_solution_chars", "proof_self_eval_chars")
                },
            )
            for idx in negatives
        }

        group_valid_count = 0
        for idx in positives + negatives:
            if idx in positive_contexts:
                positive_idx = _sample_other(positives, idx)
                if positive_idx is None:
                    positive_idx = idx
                wrong_idx = random.choice(negatives)
                wrong_pool = list(negatives)
            else:
                positive_idx = random.choice(positives)
                wrong_idx = _sample_other(negatives, idx)
                if wrong_idx is None:
                    wrong_idx = idx
                wrong_pool = [candidate for candidate in negatives if candidate != idx] or list(negatives)

            correct_context = positive_contexts[positive_idx]
            wrong_context = negative_contexts[wrong_idx]
            valid_mask[idx] = True
            correct_prompt_by_idx[idx] = correct_context["prompt"]
            wrong_prompt_by_idx[idx] = wrong_context["prompt"]
            correct_text_by_idx[idx] = correct_context["text"]
            wrong_text_by_idx[idx] = wrong_context["text"]
            correct_answer_by_idx[idx] = correct_context["answer"]
            wrong_answer_by_idx[idx] = wrong_context["answer"]

            k_max = int(sd._cfg_get(self, "rlcsd_k_max", sd.RLCSD_DEFAULT_K_MAX) or sd.RLCSD_DEFAULT_K_MAX)
            rest = [candidate for candidate in wrong_pool if candidate != wrong_idx]
            random.shuffle(rest)
            chosen = [wrong_idx] + rest[: max(k_max - 1, 0)]
            wrong_prompt_group_by_idx[idx] = [negative_contexts[candidate]["prompt"] for candidate in chosen]
            wrong_text_group_by_idx[idx] = [negative_contexts[candidate]["text"] for candidate in chosen]
            wrong_answer_group_by_idx[idx] = [negative_contexts[candidate]["answer"] for candidate in chosen]
            wrong_pool_sizes.append(len(chosen))
            group_valid_count += 1
        if group_valid_count:
            valid_group_count += 1
        else:
            aborted_group_count += 1

    valid_sample_count = int(valid_mask.sum())
    step_metrics = {
        "rlcsd/proof_positive_threshold": positive_threshold,
        "rlcsd/proof_negative_threshold": negative_threshold,
        "rlcsd/proof_middle_sample_count": float(middle_sample_count),
        "rlcsd/aborted_ratio": float(aborted_group_count / max(len(groups), 1)),
        "rlcsd/valid_group_ratio": float(valid_group_count / max(len(groups), 1)),
        "rlcsd/valid_group_count": float(valid_group_count),
        "rlcsd/valid_sample_count": float(valid_sample_count),
        "rlcsd/group_count": float(len(groups)),
        "rlcsd/all_same_outcome_group_count": float(all_same_outcome_group_count),
        "rlcsd/skipped_sample_count": float(batch_size - valid_sample_count),
        "rlcsd/actor_pad_count": 0.0,
        "rlcsd/actor_padded_batch_size": float(valid_sample_count),
        "rlcsd/wrong_pool_size_mean": float(np.mean(wrong_pool_sizes)) if wrong_pool_sizes else 0.0,
    }

    if not valid_mask.any():
        filtered_batch = batch.select_idxs(valid_mask)
        batch.batch = filtered_batch.batch
        batch.non_tensor_batch = filtered_batch.non_tensor_batch
        batch.meta_info["skip_actor_update"] = True
        step_metrics["rlcsd/skip_actor_update"] = 1.0
        self._aligned_teacher_rollout_details = {
            "teacher_correct_prompts": [],
            "teacher_wrong_prompts": [],
            "privileged_texts_correct": [],
            "privileged_texts_wrong": [],
            "correct_answers": [],
            "wrong_answers": [],
        }
        self._rlcsd_step_metrics = step_metrics
        return None

    kept_indices = [idx for idx in range(batch_size) if valid_mask[idx]]
    filtered_batch = batch.select_idxs(valid_mask)
    batch.batch = filtered_batch.batch
    batch.non_tensor_batch = filtered_batch.non_tensor_batch
    teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
    teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
    privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
    privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
    correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
    wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
    teacher_wrong_prompt_groups = [wrong_prompt_group_by_idx[idx] for idx in kept_indices]
    privileged_text_groups_wrong = [wrong_text_group_by_idx[idx] for idx in kept_indices]
    wrong_answer_groups = [wrong_answer_group_by_idx[idx] for idx in kept_indices]

    self._aligned_teacher_rollout_details = {
        "teacher_correct_prompts": teacher_correct_prompts,
        "teacher_wrong_prompts": teacher_wrong_prompts,
        "privileged_texts_correct": privileged_texts_correct,
        "privileged_texts_wrong": privileged_texts_wrong,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "teacher_correct_prompt_groups": [],
        "teacher_wrong_prompt_groups": teacher_wrong_prompt_groups,
        "privileged_text_groups_correct": [],
        "privileged_text_groups_wrong": privileged_text_groups_wrong,
        "correct_answer_groups": [],
        "wrong_answer_groups": wrong_answer_groups,
        "effective_modes": ["proof_reward_bins"] * len(teacher_correct_prompts),
    }
    self._rlcsd_step_metrics = step_metrics
    teacher_correct_pair = sd._build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_correct_prompts,
        responses=batch.batch["responses"],
        prefix="teacher_correct",
    )
    teacher_wrong_multi = sd._build_teacher_multi_batch_from_prompt_groups(
        self,
        teacher_prompt_groups=teacher_wrong_prompt_groups,
        responses=batch.batch["responses"],
        prefix="teacher_wrong_multi",
    )
    return teacher_correct_pair.union(teacher_wrong_multi)


def apply() -> None:
    sd = __import__("src.self_distill_main", fromlist=["RayPPOTrainer"])
    sd.RayPPOTrainer._compute_rlcsd_teacher_inputs = _compute_proof_rlcsd_teacher_inputs
    LOGGER.info("Applied proof-specific RLCSD teacher patch.")
