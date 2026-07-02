#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "on"}


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S,%f")[:-3]
    print(f"{timestamp} train_engine_verl {message}", flush=True)


def make_run_dir(base: str | None, fallback: str, run_dir_name: str | None) -> Path:
    root = Path(base or fallback).expanduser()
    if run_dir_name:
        return root / run_dir_name
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return root / f"verl_opd_{stamp}_pid{os.getpid()}"


def append_hydra_arg(args_list: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    else:
        rendered = str(value)
    args_list.append(f"{key}={rendered}")


def append_hydra_plus_arg(args_list: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    else:
        rendered = str(value)
    args_list.append(f"+{key}={rendered}")


def parse_extra_overrides(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_logged_subprocess(command: list[str], env: dict[str, str], cwd: Path | None = None) -> int:
    log("Running command: " + command_to_text(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def prepend_env_path(env: dict[str, str], key: str, *paths: Path | str) -> None:
    parts = [str(path) for path in paths if str(path)]
    existing = env.get(key)
    if existing:
        parts.extend(existing.split(os.pathsep))
    env[key] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def enable_olmo3_sink_external_module(env: dict[str, str]) -> None:
    module = "olmo3_sink.verl_bootstrap"
    modules = [part.strip() for part in env.get("VERL_USE_EXTERNAL_MODULES", "").split(",") if part.strip()]
    if module not in modules:
        modules.insert(0, module)
    env["VERL_USE_EXTERNAL_MODULES"] = ",".join(modules)


def resolve_verl_package_dir(args: argparse.Namespace, env: dict[str, str]) -> Path:
    candidates = []
    if args.verl_dir:
        candidates.append(Path(args.verl_dir).expanduser())
    if env.get("VERL_DIR"):
        candidates.append(Path(env["VERL_DIR"]).expanduser())
    for candidate in candidates:
        package_dir = candidate / "verl"
        if (package_dir / "trainer" / "config").is_dir():
            return package_dir.resolve()

    import verl

    package_dir = Path(verl.__file__).resolve().parent
    if not (package_dir / "trainer" / "config").is_dir():
        raise FileNotFoundError(f"Could not locate VERL trainer config under {package_dir}")
    return package_dir


def ensure_verl_automodel_ppo_configs(args: argparse.Namespace, env: dict[str, str]) -> None:
    """Add minimal PPO config aliases for the VERL automodel engine.

    VERL ships automodel engine/optimizer configs for SFT, but the PPO defaults
    currently do not include model_engine aliases such as automodel_actor. The
    aliases below keep the patch local to the runtime checkout and let Hydra
    compose actor/ref/critic configs with AutomodelEngineConfig.
    """

    package_dir = resolve_verl_package_dir(args, env)
    config_dir = package_dir / "trainer" / "config"
    write_text_if_changed(
        config_dir / "model_engine" / "automodel.yaml",
        """# @package _global_
model_engine: automodel
""",
    )
    write_text_if_changed(
        config_dir / "actor" / "automodel_actor.yaml",
        """defaults:
  - ../optim@optim: automodel
  - ../engine@engine: automodel
  - actor
  - _self_

_target_: verl.workers.config.ActorConfig
strategy: automodel
""",
    )
    write_text_if_changed(
        config_dir / "ref" / "automodel_ref.yaml",
        """defaults:
  - ref
  - ../optim@optim: automodel
  - ../engine@engine: automodel
  - _self_

_target_: verl.workers.config.ActorConfig
strategy: automodel

engine:
  forward_only: true
""",
    )
    write_text_if_changed(
        config_dir / "critic" / "automodel_critic.yaml",
        """defaults:
  - critic
  - ../optim@optim: automodel
  - ../engine@engine: automodel
  - _self_

_target_: verl.workers.config.CriticConfig
strategy: automodel
ppo_micro_batch_size_per_gpu: 1
""",
    )
    log(f"Ensured VERL automodel PPO config aliases under {config_dir}")


def prepare_verl_dataset(args: argparse.Namespace, cache_dir: Path) -> tuple[Path, Path]:
    if args.verl_train_file:
        train_file = Path(args.verl_train_file).expanduser().resolve()
        val_file = Path(args.verl_val_file).expanduser().resolve() if args.verl_val_file else train_file
        return train_file, val_file
    if not args.dataset_path:
        raise ValueError("--backend verl_opd requires --dataset_path or --verl_train_file.")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    target_dir = Path(args.verl_data_cache).expanduser().resolve() if args.verl_data_cache else cache_dir / "verl_opd_datasets"
    target = target_dir / f"{dataset_path.stem}_opd_proof.parquet"
    from verl_rlcsd_adapter import prepare_verl_proof_dataset

    prepared = prepare_verl_proof_dataset(
        dataset_path,
        target,
        max_rows=args.verl_max_rows,
    )
    val_file = Path(args.verl_val_file).expanduser().resolve() if args.verl_val_file else prepared
    return prepared, val_file


def build_hydra_args(args: argparse.Namespace, train_file: Path, val_file: Path, output_dir: Path) -> list[str]:
    max_model_len = args.verl_max_prompt_length + args.verl_max_response_length + 1
    rollout_max_tokens = args.verl_rollout_max_num_batched_tokens or max_model_len
    teacher_max_tokens = args.verl_teacher_max_num_batched_tokens or max_model_len
    ppo_mini_batch = args.verl_ppo_mini_batch_size or args.verl_train_batch_size
    ppo_token_budget = args.verl_actor_max_token_len_per_gpu or max_model_len
    warmup_steps = max(0, int(args.warmup_ratio * max(args.max_train_steps, 1)))

    hydra_args: list[str] = []
    append_hydra_arg(hydra_args, "algorithm.adv_estimator", "grpo")
    append_hydra_arg(hydra_args, "algorithm.use_kl_in_reward", False)
    append_hydra_arg(hydra_args, "critic.enable", False)
    append_hydra_arg(hydra_args, "reward.reward_model.enable", False)

    trainer_loggers = "['console']"
    if args.with_tracking and args.wandb_mode != "disabled":
        trainer_loggers = "['console','wandb']"
    append_hydra_arg(hydra_args, "trainer.logger", trainer_loggers)
    append_hydra_arg(hydra_args, "trainer.project_name", args.wandb_project)
    append_hydra_arg(hydra_args, "trainer.experiment_name", args.wandb_name or output_dir.name)
    append_hydra_arg(hydra_args, "trainer.default_local_dir", output_dir)
    append_hydra_arg(hydra_args, "trainer.n_gpus_per_node", args.verl_train_gpus_per_node)
    append_hydra_arg(hydra_args, "trainer.nnodes", args.verl_train_nodes)
    append_hydra_arg(hydra_args, "trainer.val_before_train", False)
    append_hydra_arg(hydra_args, "trainer.critic_warmup", 0)
    append_hydra_arg(hydra_args, "trainer.save_freq", args.checkpointing_steps)
    append_hydra_arg(hydra_args, "trainer.test_freq", args.verl_test_freq)
    append_hydra_arg(hydra_args, "trainer.total_epochs", 1)
    append_hydra_arg(hydra_args, "trainer.total_training_steps", args.max_train_steps)
    append_hydra_arg(hydra_args, "trainer.max_actor_ckpt_to_keep", args.checkpoint_keep_last)

    append_hydra_arg(hydra_args, "data.train_files", train_file)
    append_hydra_arg(hydra_args, "data.val_files", val_file)
    append_hydra_arg(hydra_args, "data.prompt_key", "prompt")
    append_hydra_arg(hydra_args, "data.train_batch_size", args.verl_train_batch_size)
    append_hydra_arg(hydra_args, "data.val_batch_size", args.verl_val_batch_size)
    append_hydra_arg(hydra_args, "data.max_prompt_length", args.verl_max_prompt_length)
    append_hydra_arg(hydra_args, "data.max_response_length", args.verl_max_response_length)
    append_hydra_arg(hydra_args, "data.filter_overlong_prompts", True)
    append_hydra_arg(hydra_args, "data.truncation", "error")
    append_hydra_arg(hydra_args, "data.shuffle", False)
    append_hydra_arg(hydra_args, "data.return_raw_chat", True)
    append_hydra_arg(hydra_args, "data.trust_remote_code", True)
    if args.verl_enable_thinking:
        append_hydra_plus_arg(hydra_args, "data.apply_chat_template_kwargs.enable_thinking", True)
        append_hydra_plus_arg(hydra_args, "data.val_apply_chat_template_kwargs.enable_thinking", True)

    append_hydra_arg(hydra_args, "actor_rollout_ref.nccl_timeout", 7200)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.path", args.model_path)
    if args.tokenizer_path:
        append_hydra_arg(hydra_args, "actor_rollout_ref.model.tokenizer_path", args.tokenizer_path)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.trust_remote_code", True)
    if args.olmo3_sink:
        append_hydra_arg(hydra_args, "actor_rollout_ref.model.external_lib", "olmo3_sink.verl_bootstrap")
        append_hydra_plus_arg(
            hydra_args,
            "actor_rollout_ref.model.override_config.attn_implementation",
            args.olmo3_sink_attn_implementation,
        )
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.use_remove_padding", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.enable_gradient_checkpointing", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.lora_rank", 0)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.lora.merge", False)

    if args.verl_training_fp8:
        hydra_args.append("model_engine=automodel")
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.strategy", "automodel")
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.distributed_strategy", args.verl_automodel_distributed_strategy)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.tp_size", args.verl_automodel_tp_size)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.pp_size", 1)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.cp_size", args.verl_automodel_cp_size)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.enable_fp8", True)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.enable_compile", args.verl_actor_compile)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.activation_checkpointing", args.verl_automodel_activation_checkpointing)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.model_dtype", args.verl_automodel_model_dtype)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.attn_implementation", args.verl_automodel_attn_implementation)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.param_offload", args.verl_actor_param_offload)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.optimizer_offload", args.verl_actor_optimizer_offload)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.backend_config.attn", args.verl_automodel_backend_attn)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.backend_config.linear", args.verl_automodel_backend_linear)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.backend_config.rms_norm", args.verl_automodel_backend_rms_norm)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.engine.backend_config.rope_fusion", args.verl_automodel_rope_fusion)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.optimizer", args.verl_automodel_optimizer)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.optimizer_impl", args.verl_automodel_optimizer_impl)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.master_weights", args.verl_automodel_optimizer_master_weights)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.store_param_remainders", args.verl_automodel_optimizer_store_param_remainders)
        if args.verl_automodel_optimizer_exp_avg_dtype:
            append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.exp_avg_dtype", args.verl_automodel_optimizer_exp_avg_dtype)
        if args.verl_automodel_optimizer_exp_avg_sq_dtype:
            append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.exp_avg_sq_dtype", args.verl_automodel_optimizer_exp_avg_sq_dtype)
    else:
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.strategy", args.verl_actor_strategy)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_torch_compile", args.verl_actor_compile)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.param_offload", args.verl_actor_param_offload)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.optimizer_offload", args.verl_actor_optimizer_offload)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.dtype", "bfloat16")

    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.lr", args.learning_rate)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.weight_decay", args.weight_decay)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.lr_warmup_steps", warmup_steps)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.grad_clip", args.max_grad_norm)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_mini_batch_size", ppo_mini_batch)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", args.verl_ppo_micro_batch_size_per_gpu)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_dynamic_bsz", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", ppo_token_budget)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_kl_loss", False)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.kl_loss_coef", 0.0)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.entropy_coeff", 0.0)

    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.name", "vllm")
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.tensor_model_parallel_size", args.verl_rollout_tp)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.gpu_memory_utilization", args.verl_rollout_gpu_memory_utilization)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.n", args.verl_rollout_n)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.temperature", args.verl_rollout_temperature)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.top_p", args.verl_rollout_top_p)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.top_k", args.verl_rollout_top_k)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.max_model_len", max_model_len)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.max_num_batched_tokens", rollout_max_tokens)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.max_num_seqs", args.verl_rollout_max_num_seqs)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", ppo_token_budget)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.enforce_eager", args.verl_rollout_enforce_eager)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.enable_prefix_caching", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.enable_chunked_prefill", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.load_format", "safetensors")
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.layered_summon", True)
    if args.verl_rollout_quantization:
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.quantization", args.verl_rollout_quantization)

    append_hydra_arg(hydra_args, "distillation.enabled", True)
    append_hydra_arg(hydra_args, "distillation.n_gpus_per_node", args.verl_teacher_gpus_per_node)
    append_hydra_arg(hydra_args, "distillation.nnodes", args.verl_teacher_nodes)
    append_hydra_arg(hydra_args, "distillation.teacher_key", "data_source")
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.model_path", args.opd_teacher_model_path)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.name", "vllm")
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size", args.verl_teacher_tp)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.data_parallel_size", args.verl_teacher_dp)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization", args.verl_teacher_gpu_memory_utilization)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.max_model_len", max_model_len)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens", teacher_max_tokens)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.max_num_seqs", args.verl_teacher_max_num_seqs)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.enforce_eager", args.verl_teacher_enforce_eager)
    append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.load_format", "safetensors")
    if args.verl_teacher_quantization:
        append_hydra_arg(hydra_args, "distillation.teacher_models.teacher_model.inference.quantization", args.verl_teacher_quantization)
    if args.opd_loss_mode == "forward_kl_topk":
        append_hydra_arg(
            hydra_args,
            "distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs",
            args.opd_topk,
        )

    append_hydra_arg(hydra_args, "distillation.distillation_loss.loss_mode", args.opd_loss_mode)
    append_hydra_arg(hydra_args, "distillation.distillation_loss.topk", args.opd_topk)
    append_hydra_arg(hydra_args, "distillation.distillation_loss.use_task_rewards", args.opd_use_task_rewards)
    append_hydra_arg(hydra_args, "distillation.distillation_loss.use_policy_gradient", args.opd_use_policy_gradient)
    append_hydra_arg(hydra_args, "distillation.distillation_loss.loss_max_clamp", args.opd_loss_max_clamp)
    append_hydra_arg(hydra_args, "distillation.distillation_loss.log_prob_min_clamp", args.opd_log_prob_min_clamp)
    hydra_args.extend(parse_extra_overrides(args.verl_extra_overrides))
    return hydra_args


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Native VERL OPD training entrypoint for OLMo3Sink smoke tests")
    parser.add_argument("--backend", default="verl_opd")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--verl_dir", "--verl-dir", default=None)
    parser.add_argument("--verl_train_file", "--verl-train-file", default=None)
    parser.add_argument("--verl_val_file", "--verl-val-file", default=None)
    parser.add_argument("--verl_data_cache", "--verl-data-cache", default=None)
    parser.add_argument("--verl_max_rows", "--verl-max-rows", type=int, default=0)
    parser.add_argument("--opd_teacher_model_path", "--opd-teacher-model-path", required=True)
    parser.add_argument("--opd_loss_mode", "--opd-loss-mode", default="k1")
    parser.add_argument("--opd_topk", "--opd-topk", type=int, default=32)
    parser.add_argument("--opd_use_task_rewards", "--opd-use-task-rewards", type=parse_bool, default=False)
    parser.add_argument("--opd_use_policy_gradient", "--opd-use-policy-gradient", type=parse_bool, default=True)
    parser.add_argument("--opd_loss_max_clamp", "--opd-loss-max-clamp", type=float, default=10.0)
    parser.add_argument("--opd_log_prob_min_clamp", "--opd-log-prob-min-clamp", type=float, default=-10.0)
    parser.add_argument("--max_train_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--checkpointing_steps", type=int, default=-1)
    parser.add_argument("--checkpoint_keep_last", type=int, default=1)
    parser.add_argument("--verl_train_nodes", type=int, default=1)
    parser.add_argument("--verl_train_gpus_per_node", type=int, default=2)
    parser.add_argument("--verl_teacher_nodes", type=int, default=1)
    parser.add_argument("--verl_teacher_gpus_per_node", type=int, default=2)
    parser.add_argument("--verl_teacher_tp", type=int, default=2)
    parser.add_argument("--verl_teacher_dp", type=int, default=1)
    parser.add_argument("--verl_rollout_tp", type=int, default=2)
    parser.add_argument("--verl_train_batch_size", type=int, default=2)
    parser.add_argument("--verl_val_batch_size", type=int, default=2)
    parser.add_argument("--verl_rollout_n", type=int, default=1)
    parser.add_argument("--verl_max_prompt_length", type=int, default=1024)
    parser.add_argument("--verl_max_response_length", type=int, default=512)
    parser.add_argument("--verl_actor_max_token_len_per_gpu", type=int, default=0)
    parser.add_argument("--verl_ppo_mini_batch_size", type=int, default=0)
    parser.add_argument("--verl_ppo_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--verl_actor_strategy", default="fsdp2", choices=("fsdp", "fsdp2"))
    parser.add_argument("--verl_actor_compile", type=parse_bool, default=False)
    parser.add_argument("--verl_actor_param_offload", type=parse_bool, default=True)
    parser.add_argument("--verl_actor_optimizer_offload", type=parse_bool, default=True)
    parser.add_argument("--verl_training_fp8", type=parse_bool, default=False)
    parser.add_argument("--verl_automodel_distributed_strategy", default="fsdp2", choices=("fsdp2", "megatron_fsdp", "ddp"))
    parser.add_argument("--verl_automodel_tp_size", type=int, default=1)
    parser.add_argument("--verl_automodel_cp_size", type=int, default=1)
    parser.add_argument("--verl_automodel_activation_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--verl_automodel_model_dtype", default="bf16")
    parser.add_argument("--verl_automodel_attn_implementation", default="flash_attention_2")
    parser.add_argument("--verl_automodel_backend_attn", default="sdpa")
    parser.add_argument("--verl_automodel_backend_linear", default="te")
    parser.add_argument("--verl_automodel_backend_rms_norm", default="torch_fp32")
    parser.add_argument("--verl_automodel_rope_fusion", type=parse_bool, default=True)
    parser.add_argument("--verl_automodel_optimizer", default="FusedAdam")
    parser.add_argument("--verl_automodel_optimizer_impl", default="transformer_engine.pytorch.optimizers.fused_adam")
    parser.add_argument("--verl_automodel_optimizer_master_weights", type=parse_bool, default=True)
    parser.add_argument("--verl_automodel_optimizer_store_param_remainders", type=parse_bool, default=True)
    parser.add_argument("--verl_automodel_optimizer_exp_avg_dtype", default="bf16")
    parser.add_argument("--verl_automodel_optimizer_exp_avg_sq_dtype", default="bf16")
    parser.add_argument("--verl_rollout_gpu_memory_utilization", type=float, default=0.40)
    parser.add_argument("--verl_teacher_gpu_memory_utilization", type=float, default=0.40)
    parser.add_argument("--verl_rollout_quantization", default="fp8")
    parser.add_argument("--verl_teacher_quantization", default="fp8")
    parser.add_argument("--verl_rollout_max_num_seqs", type=int, default=4)
    parser.add_argument("--verl_teacher_max_num_seqs", type=int, default=4)
    parser.add_argument("--verl_rollout_max_num_batched_tokens", type=int, default=0)
    parser.add_argument("--verl_teacher_max_num_batched_tokens", type=int, default=0)
    parser.add_argument("--verl_rollout_enforce_eager", type=parse_bool, default=False)
    parser.add_argument("--verl_teacher_enforce_eager", type=parse_bool, default=False)
    parser.add_argument("--verl_rollout_temperature", type=float, default=0.7)
    parser.add_argument("--verl_rollout_top_p", type=float, default=0.95)
    parser.add_argument("--verl_rollout_top_k", type=int, default=-1)
    parser.add_argument("--verl_enable_thinking", type=parse_bool, default=True)
    parser.add_argument("--verl_test_freq", type=int, default=-1)
    parser.add_argument("--verl_stop_ray_on_exit", type=parse_bool, default=True)
    parser.add_argument("--verl_extra_overrides", default="")
    parser.add_argument("--olmo3_sink", type=parse_bool, default=True)
    parser.add_argument("--olmo3_sink_attn_implementation", default="flash_attention_3")
    parser.add_argument("--dry_run_launch", action="store_true")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--wandb_project", default="olmo3-verl-opd")
    parser.add_argument("--wandb_name", default=None)
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def main(argv: list[str] | None = None) -> int:
    args, unknown = parse_args(sys.argv[1:] if argv is None else argv)
    if args.backend != "verl_opd":
        raise ValueError(f"train_engine_verl.py only supports --backend verl_opd, got {args.backend!r}")
    if unknown:
        log("Ignoring non-VERL args: " + " ".join(shlex.quote(item) for item in unknown))

    run_dir_name = os.environ.get("OLMO_RUN_DIR_NAME")
    output_dir = make_run_dir(args.output_path, "/tmp/olmo3_verl_opd/output", run_dir_name)
    log_dir = make_run_dir(args.logdir, "/tmp/olmo3_verl_opd/logs", run_dir_name)
    cache_dir = Path(args.cache_dir or (log_dir / "cache")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("RAY_DEDUP_LOGS", "1")
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env["WANDB_MODE"] = args.wandb_mode
    env["HF_HOME"] = env.get("HF_HOME", str(cache_dir / "hf_home"))
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", str(cache_dir / "hf_hub"))
    env["TRANSFORMERS_CACHE"] = env.get("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    verl_dir = Path(args.verl_dir or env.get("VERL_DIR", "")).expanduser()
    if verl_dir and str(verl_dir) != "." and verl_dir.exists():
        env["VERL_DIR"] = str(verl_dir)
        prepend_env_path(env, "PYTHONPATH", verl_dir)
    prepend_env_path(env, "PYTHONPATH", Path(__file__).resolve().parent, Path("/app"))
    if args.olmo3_sink:
        enable_olmo3_sink_external_module(env)
    if args.verl_training_fp8:
        ensure_verl_automodel_ppo_configs(args, env)

    train_file, val_file = prepare_verl_dataset(args, cache_dir)
    hydra_args = build_hydra_args(args, train_file, val_file, output_dir)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *hydra_args]
    (log_dir / "VERL_OPD_COMMAND.txt").write_text(command_to_text(command) + "\n", encoding="utf-8")
    (log_dir / "VERL_OPD_ENV.json").write_text(
        json.dumps(
            {
                "VERL_DIR": env.get("VERL_DIR", ""),
                "VERL_USE_EXTERNAL_MODULES": env.get("VERL_USE_EXTERNAL_MODULES", ""),
                "PYTHONPATH": env.get("PYTHONPATH", ""),
                "WANDB_MODE": env.get("WANDB_MODE", ""),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"VERL OPD output_dir={output_dir}")
    log(f"VERL OPD log_dir={log_dir}")
    log(f"VERL OPD train_file={train_file}")
    log(f"VERL OPD teacher_model={args.opd_teacher_model_path}")
    log(f"VERL OPD command file={log_dir / 'VERL_OPD_COMMAND.txt'}")
    if args.dry_run_launch:
        log("Dry-run launch requested; not starting VERL.")
        return 0

    start = time.monotonic()
    try:
        return_code = run_logged_subprocess(command, env)
        log(f"VERL OPD exit status: {return_code} duration_s={time.monotonic() - start:.1f}")
        return return_code
    finally:
        if args.verl_stop_ray_on_exit:
            subprocess.run(["ray", "stop", "--force"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    raise SystemExit(main())
