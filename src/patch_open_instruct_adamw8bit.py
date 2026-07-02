#!/usr/bin/env python3
"""Patch Open-Instruct's OLMo-core SFT script for torchao AdamW8bit.

The submission image installs Open-Instruct from a pinned fork commit. Keep this
small build-time patch until the fork contains the optimizer branch directly.
"""

from __future__ import annotations

import os
from pathlib import Path


OPEN_INSTRUCT_DIR = Path(os.environ.get("OPEN_INSTRUCT_DIR", "/opt/open-instruct"))
TARGET = OPEN_INSTRUCT_DIR / "open_instruct" / "olmo_core_finetune.py"


def main() -> None:
    text = TARGET.read_text()
    if "TorchAOAdamW8bitConfig" in text:
        return

    text = text.replace(
        '    optimizer_name = os.environ.get("OLMO_OPTIMIZER", "adamw").strip().lower()\n'
        '    if optimizer_name == "adamw":\n',
        '    optimizer_name = os.environ.get("OLMO_OPTIMIZER", "adamw").strip().lower()\n'
        '    if optimizer_name in {"adamw_8bits", "torchao_adamw_8bit"}:\n'
        '        optimizer_name = "adamw_8bit"\n'
        '    if optimizer_name == "adamw":\n',
    )

    text = text.replace(
        '    elif optimizer_name == "skip_step_adamw":\n'
        '        optim_config = optim.SkipStepAdamWConfig(\n'
        '            lr=args.training.learning_rate,\n'
        '            weight_decay=args.training.weight_decay,\n'
        '            betas=(0.9, 0.95),\n'
        '            dtype=optim_dtype,\n'
        '            compile=False,\n'
        '        )\n'
        '    else:\n',
        '    elif optimizer_name == "skip_step_adamw":\n'
        '        optim_config = optim.SkipStepAdamWConfig(\n'
        '            lr=args.training.learning_rate,\n'
        '            weight_decay=args.training.weight_decay,\n'
        '            betas=(0.9, 0.95),\n'
        '            dtype=optim_dtype,\n'
        '            compile=False,\n'
        '        )\n'
        '    elif optimizer_name == "adamw_8bit":\n'
        '        from olmo_torchao_optim import TorchAOAdamW8bitConfig\n'
        '\n'
        '        if optim_dtype is not None:\n'
        '            logger.warning("torchao AdamW8bit ignores OLMO_OPTIM_DTYPE=%s", optim_dtype_env)\n'
        '        optim_config = TorchAOAdamW8bitConfig(\n'
        '            lr=args.training.learning_rate,\n'
        '            weight_decay=args.training.weight_decay,\n'
        '            betas=(0.9, 0.95),\n'
        '            block_size=int(os.environ.get("OLMO_ADAMW_8BIT_BLOCK_SIZE", "256")),\n'
        '            bf16_stochastic_round=os.environ.get(\n'
        '                "OLMO_ADAMW_8BIT_BF16_STOCHASTIC_ROUND", "0"\n'
        '            ).strip().lower()\n'
        '            in {"1", "true", "yes", "on"},\n'
        '        )\n'
        '        logger.warning(\n'
        '            "Using torchao AdamW8bit optimizer with block_size=%d, bf16_stochastic_round=%s.",\n'
        '            optim_config.block_size,\n'
        '            optim_config.bf16_stochastic_round,\n'
        '        )\n'
        '    else:\n',
    )
    if "TorchAOAdamW8bitConfig" not in text:
        raise RuntimeError(f"Could not patch AdamW8bit support into {TARGET}")
    TARGET.write_text(text)


if __name__ == "__main__":
    main()
