from dataclasses import dataclass
from typing import Tuple, Type

import torch
from olmo_core.optim import OptimConfig


def _optimizer_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = value.lower()
    if normalized in {"fp8", "float8", "uint8"}:
        return torch.uint8
    return getattr(torch, normalized)


class TransformerEngineFusedAdamW(torch.optim.Optimizer):
    def __new__(cls, *args, **kwargs):
        from transformer_engine.pytorch.optimizers import FusedAdam

        kwargs["exp_avg_dtype"] = _optimizer_dtype(kwargs.pop("exp_avg_dtype", "bfloat16"))
        kwargs["exp_avg_sq_dtype"] = _optimizer_dtype(
            kwargs.pop("exp_avg_sq_dtype", "bfloat16")
        )
        kwargs["master_weight_dtype"] = _optimizer_dtype(
            kwargs.pop("master_weight_dtype", "float32")
        )
        kwargs.setdefault("adam_w_mode", True)
        return FusedAdam(*args, **kwargs)


@OptimConfig.register("adamw_8bit")
@dataclass
class TorchAOAdamW8bitConfig(OptimConfig[torch.optim.Optimizer]):
    """OLMo-core optimizer config for torchao.optim.AdamW8bit."""

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    block_size: int = 256
    bf16_stochastic_round: bool = False

    @classmethod
    def optimizer(cls) -> Type[torch.optim.Optimizer]:
        from torchao.optim import AdamW8bit

        return AdamW8bit


@OptimConfig.register("te_fused_adamw")
@dataclass
class TransformerEngineFusedAdamWConfig(OptimConfig[torch.optim.Optimizer]):
    """OLMo-core optimizer config for transformer_engine.pytorch.optimizers.FusedAdam."""

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    bias_correction: bool = True
    capturable: bool = False
    master_weights: bool = False
    master_weight_dtype: str = "float32"
    exp_avg_dtype: str = "bfloat16"
    exp_avg_sq_dtype: str = "bfloat16"
    store_param_remainders: bool = False

    @classmethod
    def optimizer(cls) -> Type[torch.optim.Optimizer]:
        return TransformerEngineFusedAdamW
