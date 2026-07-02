from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
import ray
from verl.trainer.main_ppo import TaskRunner, migrate_legacy_reward_impl, run_ppo
from verl.utils.device import auto_set_device

import verl_rlcsd_proof_patch


class ProofSelfDistillTaskRunner(TaskRunner):
    def run(self, config):
        from src.self_distill_main import _apply_patches as apply_sd_patches

        apply_sd_patches()
        verl_rlcsd_proof_patch.apply()
        return super().run(config)


def _config_dir() -> str:
    rlcsd_dir = Path(os.environ.get("RLCSD_DIR", "/tmp/RLCSD-runtime")).expanduser().resolve()
    config_dir = rlcsd_dir / "third_party" / "verl" / "verl" / "trainer" / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Could not find verl Hydra config dir: {config_dir}")
    return str(config_dir)


def _fully_async_config_dir() -> str:
    rlcsd_dir = Path(os.environ.get("RLCSD_DIR", "/tmp/RLCSD-runtime")).expanduser().resolve()
    config_dir = rlcsd_dir / "third_party" / "verl" / "verl" / "experimental" / "fully_async_policy" / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Could not find verl fully-async Hydra config dir: {config_dir}")
    return str(config_dir)


def _ensure_legacy_data_config() -> None:
    """RLCSD's vendored verl can reference data/legacy_data without shipping it."""
    rlcsd_dir = Path(os.environ.get("RLCSD_DIR", "/tmp/RLCSD-runtime")).expanduser().resolve()
    data_dir = rlcsd_dir / "third_party" / "verl" / "verl" / "trainer" / "config" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    legacy_data = data_dir / "legacy_data.yaml"
    if legacy_data.exists():
        return
    legacy_data.write_text(
        """tokenizer: null
use_shm: false
train_files: ~/data/rlhf/gsm8k/train.parquet
val_files: ~/data/rlhf/gsm8k/test.parquet
train_max_samples: -1
val_max_samples: -1
prompt_key: prompt
reward_fn_key: data_source
max_prompt_length: 512
max_response_length: 512
train_batch_size: 1024
val_batch_size: null
tool_config_path: ${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path,null}
return_raw_input_ids: false
return_raw_chat: true
return_full_prompt: false
shuffle: true
seed: null
dataloader_num_workers: 8
image_patch_size: 14
validation_shuffle: false
filter_overlong_prompts: false
filter_overlong_prompts_workers: 1
truncation: error
image_key: images
video_key: videos
trust_remote_code: false
custom_cls:
  path: null
  name: null
return_multi_modal_inputs: true
sampler:
  class_path: null
  class_name: null
datagen:
  path: null
  name: null
apply_chat_template_kwargs: {}
""",
        encoding="utf-8",
    )


def _run_colocated(argv: list[str]) -> None:
    overrides = list(argv)
    _ensure_legacy_data_config()
    with initialize_config_dir(config_dir=_config_dir(), job_name="verl_rlcsd", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    from src.self_distill_main import _ensure_legacy_worker_impl

    config = _ensure_legacy_worker_impl(config)
    verl_rlcsd_proof_patch.apply()
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ProofSelfDistillTaskRunner))


def _run_fully_async(argv: list[str]) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    rlcsd_dir = Path(os.environ.get("RLCSD_DIR", "/tmp/RLCSD-runtime")).expanduser().resolve()
    os.chdir(rlcsd_dir / "third_party" / "verl")
    _ensure_legacy_data_config()
    with initialize_config_dir(config_dir=_fully_async_config_dir(), job_name="verl_rlcsd_async", version_base=None):
        config = compose(config_name="fully_async_ppo_trainer", overrides=overrides)
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner

    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)


def main(argv: list[str] | None = None) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("VERL_RLCSD_FULLY_ASYNC", "").strip().lower() in {"1", "true", "yes", "on"}:
        _run_fully_async(overrides)
    else:
        _run_colocated(overrides)


if __name__ == "__main__":
    main()
