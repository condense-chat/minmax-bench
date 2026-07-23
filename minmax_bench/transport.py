"""Upstream transports — where the bench's own direct model calls land.

A transport only applies where the BENCH holds the request: the ``upstream``
strategy, and the rewrite-invoke (proxy emulation) executor. Strategy proxies
make their own upstream calls and are unaffected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .config import get_settings


@dataclass(frozen=True)
class Upstream:
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    # Bedrock requires inference-profile model ids; pricing keeps the session model.
    model_id_map: Callable[[str], str] | None = None


def resolve_upstream(transport: str) -> Upstream:
    s = get_settings()
    if transport == "bedrock":
        from . import bedrock

        return Upstream(
            base_url=bedrock.anthropic_base(s.bedrock_region),
            headers={
                "authorization": f"Bearer {bedrock.bearer_token(s.bedrock_region)}",
                "anthropic-version": "2023-06-01",
            },
            model_id_map=bedrock.bedrock_model_id,
        )
    if transport != "anthropic":
        raise ValueError(f"unknown transport {transport!r} (anthropic | bedrock)")
    from .auth import anthropic_auth_headers

    return Upstream(
        base_url=s.anthropic_base_url,
        headers={"anthropic-version": "2023-06-01",
                 **anthropic_auth_headers(s.anthropic_beta)},
    )
