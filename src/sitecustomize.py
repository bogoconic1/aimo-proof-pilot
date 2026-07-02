"""Small runtime hooks for submission-managed Python processes."""

from __future__ import annotations

import os
import sys


_TRUTHY = {"1", "true", "yes", "on"}

if (
    os.environ.get("VERL_RLCSD_OLMO3_SINK", "").strip().lower() in _TRUTHY
    or os.environ.get("PRIME_RL_OLMO3_SINK", "").strip().lower() in _TRUTHY
):
    try:
        from olmo3_sink.verl_bootstrap import apply

        apply()
    except Exception as exc:
        print(f"[sitecustomize] olmo3_sink bootstrap failed: {exc}", file=sys.stderr)
