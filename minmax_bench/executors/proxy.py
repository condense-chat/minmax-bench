"""Proxy executor.

Sends the real request through a strategy's Anthropic/OpenAI-compatible proxy,
capped to a tiny output, and reads the upstream's real, cache-aware usage — which
reflects the proxy's server-side compression. Output tokens are substituted from
the recorded/counted turn for costing (we only pay for 1 generated token).

Points are sent in transcript order so the proxy's cache and any session state
warm up realistically. This path costs real input-token money on the upstream.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable

import httpx

from ..auth import anthropic_auth_headers
from ..gate import EndpointGate
from ..models import Provider, RequestPoint, Session, Usage
from ..pricing import cost_usd
from ..providers import render_request, usage_from_response
from ..strategies.base import ResolvedStrategy as Strategy
from ..tokens import TokenCounter
from .base import Measurement, output_tokens_for


def _endpoint(base_url: str, provider: Provider, path: str | None = None) -> str:
    base = base_url.rstrip("/")
    if path:
        return base + "/" + path.lstrip("/")
    if provider == Provider.openai:
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"
    return base + "/v1/messages"


def _headers(strategy: Strategy) -> dict[str, str]:
    cfg = strategy.proxy
    assert cfg is not None
    headers = {"content-type": "application/json"}
    if cfg.provider == Provider.openai:
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["anthropic-version"] = cfg.anthropic_version
        if cfg.api_key_env:
            # API key, or the user's Claude Code subscription OAuth token — the
            # proxies forward whichever auth header we send, like a real dense run.
            headers.update(anthropic_auth_headers(cfg.anthropic_beta, key_env=cfg.api_key_env))
        elif cfg.anthropic_beta:
            headers["anthropic-beta"] = cfg.anthropic_beta
    headers.update(cfg.extra_headers)
    return headers


class ProxyExecutor:
    def __init__(
        self,
        counter: TokenCounter | None = None,
        max_tokens: int = 1,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        retry_after_cap: float = 65.0,
        gate: EndpointGate | None = None,
    ):
        self.counter = counter or TokenCounter()
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_after_cap = retry_after_cap
        self.gate = gate
        self._client = client or httpx.Client(timeout=timeout)

    def run_session(
        self,
        session: Session,
        points: list[RequestPoint],
        strategy: Strategy,
        on_point: Callable[[Measurement], None] | None = None,
    ) -> list[Measurement]:
        assert strategy.proxy is not None, "proxy executor needs strategy.proxy"
        url = _endpoint(strategy.proxy.base_url, strategy.proxy.provider, strategy.proxy.path)
        headers = _headers(strategy)
        # Per-run session id to bust the proxy's session-keyed chain cache, only
        # for proxies that actually have one (e.g. condense) — never sent to a
        # proxy that doesn't own that header.
        if session.test_uuid and strategy.proxy.session_id_header:
            headers[strategy.proxy.session_id_header] = session.test_uuid
        out: list[Measurement] = []

        def emit(m: Measurement) -> None:
            out.append(m)
            if on_point is not None:
                on_point(m)

        flatten = bool(strategy.proxy and strategy.proxy.flatten_tools)
        for p in points:
            body = render_request(
                session, p.prefix, max_tokens=self.max_tokens, cache=True, flatten_tools=flatten
            )
            if strategy.proxy.model_id_map and body.get("model"):
                body["model"] = strategy.proxy.model_id_map(body["model"])
            if not body.get("messages"):
                # Degenerate prefix (e.g. mid-session capture that is all orphaned
                # tool_results). Skip but keep alignment with the baseline list.
                emit(Measurement(p.index, session.id, Usage(), 0.0, ok=False, error="empty prefix"))
                continue
            usage, detail = self._post_with_retry(url, body, headers, strategy)
            if usage is None:
                emit(Measurement(p.index, session.id, Usage(), 0.0, ok=False, error=detail))
                continue
            # Substitute realistic output tokens for costing (we capped gen).
            out_tokens = output_tokens_for(p, self.counter)
            usage = usage.model_copy(update={"output_tokens": out_tokens})
            emit(Measurement(p.index, session.id, usage, cost_usd(session.model, usage)))
            self._think(strategy, out_tokens, is_last=p is points[-1])
        return out

    def _think(self, strategy: Strategy, out_tokens: int, *, is_last: bool) -> None:
        """Sleep proportional to the just-produced output (async 'user work' time),
        letting background compaction land before the next turn. Not counted against
        the endpoint gate — no request is in flight while the simulated user works."""
        cfg = strategy.proxy
        rate = cfg.think_secs_per_output_token if cfg else 0.0
        if is_last or rate <= 0:
            return
        wait = out_tokens * rate
        cap = cfg.think_cap_seconds if cfg else 0.0
        if cap > 0:
            wait = min(wait, cap)
        if wait > 0:
            time.sleep(wait)

    def _post_with_retry(self, url, body, headers, strategy, parse=None):
        """POST with bounded retry on 429/529, honoring Retry-After. Each attempt
        passes through the shared endpoint gate (concurrency + spacing).
        ``parse`` overrides response parsing for callers that make two differently
        shaped calls per point (e.g. rewrite fetch + upstream invoke)."""
        parse = parse or (lambda resp: self._parse_response(strategy, resp))
        gate_slot = self.gate.slot if self.gate else contextlib.nullcontext
        for attempt in range(self.max_retries + 1):
            try:
                with gate_slot():
                    resp = self._client.post(url, json=body, headers=headers)
                if resp.status_code in (429, 529):
                    ra = resp.headers.get("retry-after")
                    ra_secs = float(ra) if ra and ra.isdigit() else None
                    if self.gate:  # slow every worker on this host, not just us
                        self.gate.penalize(ra_secs)
                    if attempt < self.max_retries:
                        delay = min(ra_secs, self.retry_after_cap) if ra_secs else min(
                            2.0 * (attempt + 1), self.retry_after_cap
                        )
                        time.sleep(delay)
                        continue
                resp.raise_for_status()
                if self.gate:  # endpoint is healthy — relax spacing back down
                    self.gate.reward()
                return parse(resp), None
            except httpx.HTTPStatusError as e:
                return None, f"{e.response.status_code}: {e.response.text[:200]}"
            except Exception as e:  # transport error — retry a couple times
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), self.retry_after_cap))
                    continue
                return None, str(e)
        return None, "retries exhausted"

    def _parse_response(self, strategy: Strategy, resp: httpx.Response):
        return usage_from_response(strategy.proxy.provider, resp.json())

    def close(self) -> None:
        self._client.close()
