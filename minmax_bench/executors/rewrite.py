"""Rewrite executor.

Sends each turn to a strategy's *rewrite* function (e.g. condense's
``x-condense-function: rewrite``): the proxy runs its full compression engine and
returns the **rewritten provider request body** without ever calling the upstream
model. We then cost that rewritten prompt offline with the same cache-aware model
as the baseline (:mod:`minmax_bench.chain`), so token efficiency can be measured
without paying for real model traffic — while pricing stays that of the session's
real model.

Cache model (mirrors the baseline's successive-prefix differencing, but on the
*rewritten* chain): the longest message-prefix shared with the previous turn's
rewritten prompt is a cache read; everything after it is a cache write. When a
compaction lands and rewrites history, the shared prefix shrinks and the diverged
tail is re-written to cache — the same penalty a real prefix cache would charge.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ..models import Provider, RequestPoint, Session, Usage
from ..pricing import cost_usd
from ..providers import render_request, usage_from_response
from ..strategies.base import ResolvedStrategy as Strategy
from ..tokens import TokenCounter
from ..transport import Upstream
from .base import Measurement, output_tokens_for
from .proxy import ProxyExecutor, _endpoint, _headers


def _strip_cache(obj):
    """Deep-copy dropping ``cache_control`` markers so prompts compare/count stably."""
    if isinstance(obj, dict):
        return {k: _strip_cache(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache(v) for v in obj]
    return obj


class RewriteExecutor(ProxyExecutor):
    """Proxy-shaped executor whose responses are rewritten request bodies, not
    completions. Reuses :class:`ProxyExecutor`'s retry/gate/think plumbing."""

    def __init__(
        self,
        chain_counter,
        counter: TokenCounter | None = None,
        **kwargs,
    ):
        super().__init__(counter=counter, **kwargs)
        self.chain_counter = chain_counter  # LocalCounter or AnthropicCounter

    def _parse_response(self, strategy: Strategy, resp: httpx.Response):
        return resp.json()  # the rewritten provider request body

    def _fetch_rewritten(self, url, body, headers, strategy):
        """One rewritten request body for one point (or (None, error))."""
        return self._post_with_retry(url, body, headers, strategy)

    def run_session(
        self,
        session: Session,
        points: list[RequestPoint],
        strategy: Strategy,
        on_point: Callable[[Measurement], None] | None = None,
    ) -> list[Measurement]:
        assert strategy.proxy is not None, "rewrite executor needs strategy.proxy"
        url = _endpoint(strategy.proxy.base_url, strategy.proxy.provider, strategy.proxy.path)
        headers = _headers(strategy)
        if session.test_uuid and strategy.proxy.session_id_header:
            headers[strategy.proxy.session_id_header] = session.test_uuid
        out: list[Measurement] = []

        def emit(m: Measurement) -> None:
            out.append(m)
            if on_point is not None:
                on_point(m)

        prev_body: dict | None = None
        prev_total = 0
        for p in points:
            body = render_request(session, p.prefix, max_tokens=self.max_tokens, cache=True)
            if not body.get("messages"):
                emit(Measurement(p.index, session.id, Usage(), 0.0, ok=False, error="empty prefix"))
                continue
            rewritten, detail = self._fetch_rewritten(url, body, headers, strategy)
            if rewritten is None or not isinstance(rewritten, dict):
                emit(Measurement(
                    p.index, session.id, Usage(), 0.0, ok=False,
                    error=detail or "rewrite returned a non-object body",
                ))
                continue
            stripped = _strip_cache(rewritten)
            total = self.chain_counter.count_body(stripped)
            cached = self._cached_prefix_tokens(prev_body, prev_total, stripped)
            out_tokens = output_tokens_for(p, self.counter)
            usage = Usage(
                input_tokens=0,
                output_tokens=out_tokens,
                cache_read=cached,
                cache_write=max(0, total - cached),
            )
            emit(Measurement(p.index, session.id, usage, cost_usd(session.model, usage)))
            prev_body, prev_total = stripped, total
            self._think(strategy, out_tokens, is_last=p is points[-1])
        return out

    def _cached_prefix_tokens(self, prev: dict | None, prev_total: int, cur: dict) -> int:
        """Tokens of ``cur`` covered by the previous turn's cached prompt: the
        longest shared message-prefix (given identical system + tools)."""
        if prev is None:
            return 0
        if prev.get("system") != cur.get("system") or prev.get("tools") != cur.get("tools"):
            return 0
        pm, cm = prev.get("messages", []), cur.get("messages", [])
        n = 0
        for a, b in zip(pm, cm, strict=False):
            if a != b:
                break
            n += 1
        if n == len(pm):
            return prev_total  # prompt only grew — the whole previous prompt is cached
        if n == 0:
            return 0
        common = {k: v for k, v in cur.items() if k in ("model", "system", "tools")}
        common["messages"] = cm[:n]
        return self.chain_counter.count_body(common)


class RewriteInvokeExecutor(ProxyExecutor):
    """Proxy emulation over an alternate upstream (mode=proxy, transport=bedrock).

    The compression proxy cannot reach the alternate upstream itself, so per point
    we do what it would have done: fetch the strategy's rewritten request body,
    send it to the upstream ourselves (output still capped), and report the
    upstream's actually-reported cache-aware usage. Points stay sequential so the
    upstream's prefix cache warms exactly as it would behind a real proxy.
    """

    def __init__(self, upstream: Upstream, counter: TokenCounter | None = None, **kwargs):
        super().__init__(counter=counter, **kwargs)
        self.upstream = upstream

    def run_session(
        self,
        session: Session,
        points: list[RequestPoint],
        strategy: Strategy,
        on_point: Callable[[Measurement], None] | None = None,
    ) -> list[Measurement]:
        assert strategy.proxy is not None, "rewrite-invoke executor needs strategy.proxy"
        rw_url = _endpoint(strategy.proxy.base_url, strategy.proxy.provider, strategy.proxy.path)
        rw_headers = _headers(strategy)
        if session.test_uuid and strategy.proxy.session_id_header:
            rw_headers[strategy.proxy.session_id_header] = session.test_uuid
        up_url = self.upstream.base_url.rstrip("/") + "/v1/messages"
        up_headers = {"content-type": "application/json", **self.upstream.headers}
        out: list[Measurement] = []

        def emit(m: Measurement) -> None:
            out.append(m)
            if on_point is not None:
                on_point(m)

        for p in points:
            body = render_request(session, p.prefix, max_tokens=self.max_tokens, cache=True)
            if not body.get("messages"):
                emit(Measurement(p.index, session.id, Usage(), 0.0, ok=False, error="empty prefix"))
                continue
            rewritten, detail = self._post_with_retry(
                rw_url, body, rw_headers, strategy, parse=lambda r: r.json()
            )
            if rewritten is None or not isinstance(rewritten, dict):
                emit(Measurement(
                    p.index, session.id, Usage(), 0.0, ok=False,
                    error=detail or "rewrite returned a non-object body",
                ))
                continue
            invoke = dict(rewritten)
            invoke["max_tokens"] = self.max_tokens
            if self.upstream.model_id_map and invoke.get("model"):
                invoke["model"] = self.upstream.model_id_map(invoke["model"])
            usage, detail = self._post_with_retry(
                up_url, invoke, up_headers, strategy,
                parse=lambda r: usage_from_response(Provider.anthropic, r.json()),
            )
            if usage is None:
                emit(Measurement(p.index, session.id, Usage(), 0.0, ok=False, error=detail))
                continue
            out_tokens = output_tokens_for(p, self.counter)
            usage = usage.model_copy(update={"output_tokens": out_tokens})
            emit(Measurement(p.index, session.id, usage, cost_usd(session.model, usage)))
            self._think(strategy, out_tokens, is_last=p is points[-1])
        return out

class _CaptureSink:
    """Loopback HTTP sink posing as the upstream. Whatever body the proxy
    forwards here IS the rewritten request; a minimal Anthropic-shaped stub is
    returned so the proxy's response path stays happy. One sink (and port) per
    executor instance — points in a session are sequential, so the last
    captured body is always the current point's."""

    def __init__(self):
        import http.server
        import json as _json
        import threading

        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                try:
                    sink.body = _json.loads(self.rfile.read(n) or b"{}")
                except _json.JSONDecodeError:
                    sink.body = None
                stub = _json.dumps({
                    "id": "msg_capture", "type": "message", "role": "assistant",
                    "model": (sink.body or {}).get("model", "unknown"),
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(stub)))
                self.end_headers()
                self.wfile.write(stub)

            def log_message(self, *args):
                pass

        self.body: dict | None = None
        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._srv.server_port}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


class CaptureRewriteExecutor(RewriteExecutor):
    """Rewrite mode for proxies with no rewrite function but a per-request
    upstream override header (e.g. headroom's ``x-headroom-base-url``): the
    proxy is pointed at a local capture sink, so the body it would have sent
    upstream is captured and costed offline — the real proxy pipeline, zero
    upstream traffic."""

    def __init__(self, chain_counter, **kwargs):
        super().__init__(chain_counter=chain_counter, **kwargs)
        self._sink = _CaptureSink()

    def _fetch_rewritten(self, url, body, headers, strategy):
        override = strategy.proxy.upstream_override_header
        assert override, "capture-rewrite needs ProxyConfig.upstream_override_header"
        self._sink.body = None
        _, detail = self._post_with_retry(
            url, body, {**headers, override: self._sink.url}, strategy,
            parse=lambda r: r.json(),
        )
        if detail is not None:
            return None, detail
        if self._sink.body is None:
            return None, "proxy never forwarded to the capture sink"
        return self._sink.body, None

    def close(self) -> None:
        self._sink.close()
        super().close()
