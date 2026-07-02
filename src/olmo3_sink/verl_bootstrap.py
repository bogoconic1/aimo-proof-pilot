"""Runtime registration for OLMo3 sink models in verl/vLLM workers."""

from __future__ import annotations

import logging


LOGGER = logging.getLogger(__name__)
_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from .register import register_olmo3_sink

    register_olmo3_sink()

    try:
        from vllm import ModelRegistry
    except Exception as exc:
        LOGGER.warning("vLLM is not importable; skipped Olmo3SinkForCausalLM registration: %s", exc)
    else:
        ModelRegistry.register_model(
            "Olmo3SinkForCausalLM",
            "olmo3_sink.vllm_adapter:Olmo3SinkForCausalLM",
        )
        LOGGER.info("Registered Olmo3SinkForCausalLM for vLLM rollouts.")

    _APPLIED = True


apply()
