"""Exact, cache-aware costing of a *full* reconstructed chain.

SWE-chat's recorded usage reflects the source agent's own compaction (chains
plateau ~165k), so it is not the right "no-compression" baseline. Here we cost the
full reconstructed chain (and, for local rewrite strategies, the rewritten chain)
with a simple, provider-accurate cache model:

    turn t sends prompt of P_t tokens; the prior prompt P_{t-1} is a cache hit,
    the delta (P_t - P_{t-1}) is written to cache.

So per turn: cache_read = P_{t-1}, cache_write = P_t - P_{t-1}, input ≈ 0 — which
matches how Anthropic bills an incrementally-cached, growing conversation (newly
sent tokens are cache_creation, not full-rate input).

P_t is measured either with Anthropic's free ``count_tokens`` endpoint (exact,
same tokenizer the proxies' real usage uses) or locally with tiktoken (offline).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

from .models import Message, Session, Usage
from .providers import render_anthropic
from .tokens import TokenCounter


class LocalCounter:
    """Offline prompt token counter (tiktoken); an approximation for Claude."""

    def __init__(self, encoding: str = "o200k_base"):
        self.tc = TokenCounter(encoding)

    def count(self, session: Session, messages: list[Message]) -> int:
        total = 0
        if session.system:
            total += self.tc.count_text(session.system)
        if session.tools:
            total += self.tc.count_json([t.model_dump() for t in session.tools])
        for m in messages:
            total += self.tc.count_message(m)
        return total


class AnthropicCounter:
    """Exact prompt token counts via Anthropic's free count_tokens endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        anthropic_beta: str | None = "context-1m-2025-08-07",
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        encoding: str = "o200k_base",
    ):
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/v1/messages/count_tokens"
        self.beta = anthropic_beta
        self._client = client or httpx.Client(timeout=timeout)
        self._fallback = LocalCounter(encoding)
        self.fallbacks = 0  # count of points that fell back to local counting

    def count(self, session: Session, messages: list[Message]) -> int:
        body = render_anthropic(session, messages, max_tokens=1, cache=False)
        body.pop("max_tokens", None)
        body.pop("thinking", None)
        if not body.get("messages"):
            return 0
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.beta:
            headers["anthropic-beta"] = self.beta
        try:
            resp = self._client.post(self.url, json=body, headers=headers)
            resp.raise_for_status()
            return int(resp.json().get("input_tokens", 0))
        except Exception:
            # Rare rejects (a content shape count_tokens dislikes) must not kill
            # the run — fall back to the offline count for this one point.
            self.fallbacks += 1
            return self._fallback.count(session, messages)

    def close(self) -> None:
        self._client.close()


def chain_usages(
    session: Session,
    points: list,
    counter,
    output_fn: Callable[[object], int],
    rewrite: Callable[[Session, list[Message]], list[Message]] | None = None,
) -> list[Usage]:
    """Cache-aware baseline/rewrite usages by differencing successive prompts."""
    usages: list[Usage] = []
    prev = 0
    for p in points:
        msgs = rewrite(session, p.prefix) if rewrite else p.prefix
        total = counter.count(session, msgs)
        new = max(0, total - prev)
        usages.append(
            Usage(input_tokens=0, output_tokens=output_fn(p), cache_read=prev, cache_write=new)
        )
        prev = total
    return usages


def make_counter(mode: str, encoding: str, anthropic_base_url: str, anthropic_beta: str | None):
    """``mode`` = 'api' (exact count_tokens) or 'local' (tiktoken)."""
    if mode == "api":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            return AnthropicCounter(key, anthropic_base_url, anthropic_beta)
    return LocalCounter(encoding)
