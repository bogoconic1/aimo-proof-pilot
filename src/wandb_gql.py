from __future__ import annotations

try:
    from graphql import parse
except Exception:  # pragma: no cover - fallback for disabled W&B paths.
    parse = None


def gql(query: str):
    """Compatibility shim for Prime-RL's legacy wandb_gql import."""
    if parse is None:
        return query
    return parse(query)
