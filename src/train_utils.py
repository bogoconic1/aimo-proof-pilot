from __future__ import annotations

import re
from pathlib import Path


def sanitize_slug_part(value: str) -> str:
    """Return a filesystem and HF-path friendly token."""
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-")
    return cleaned or "value"


def truncate_slug(value: str, max_length: int = 180) -> str:
    """Keep generated run names below path limits while preserving readability."""
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip("-_.+")


def path_name_token(value: str | None, fallback: str) -> str:
    """Use the final path component as a stable run-name token."""
    if not value:
        return fallback
    path = Path(value).expanduser()
    name = path.name or fallback
    for suffix in path.suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return sanitize_slug_part(name or fallback)
