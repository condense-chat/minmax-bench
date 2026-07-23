"""Concrete strategy runners — one reusable implementation per technique.

A runner turns a :class:`StrategyConfig` plus the run-wide mode/transport into a
:class:`ResolvedStrategy`. Runners are cheap singletons built at import; anything
that can fail (loading dense creds, minting a Bedrock token) happens lazily in
:meth:`build`, i.e. at run/preflight time, never at import.
"""

from __future__ import annotations

from ..config import get_settings
from ..dense import load_profile
from ..models import Provider
from ..transport import resolve_upstream
from .base import (
    Mode,
    ProxyConfig,
    ResolvedStrategy,
    StrategyConfig,
    StrategyRunner,
    Transport,
    UnsupportedCombo,
)


class NoopRunner(StrategyRunner):
    key = "noop"
    kind = "noop"

    def build(
        self,
        name: str,
        config: StrategyConfig,
        mode: Mode = "proxy",
        transport: Transport = "anthropic",
    ) -> ResolvedStrategy:
        return ResolvedStrategy(
            name=name,
            kind="noop",
            description=self.describe(config),
            cache_model="none",
            mode=mode,
            transport=transport,
        )

    def describe(self, config: StrategyConfig) -> str:
        return "Uncompressed baseline (recorded usage or local token count)."


class UpstreamRunner(StrategyRunner):
    key = "upstream"
    kind = "proxy"

    def build(
        self,
        name: str,
        config: StrategyConfig,
        mode: Mode = "proxy",
        transport: Transport = "anthropic",
    ) -> ResolvedStrategy:
        if mode == "rewrite":
            raise UnsupportedCombo("upstream is a direct call — there is nothing to rewrite")
        up = resolve_upstream(transport)
        suffix = " (via AWS Bedrock)" if transport == "bedrock" else ""
        return ResolvedStrategy(
            name=name,
            kind="proxy",
            description=self.describe(config) + suffix,
            proxy=ProxyConfig(
                base_url=up.base_url,
                provider=Provider.anthropic,
                # Auth (key or subscription OAuth or Bedrock bearer) is fully
                # resolved into the transport's headers.
                api_key_env=None,
                extra_headers=up.headers,
                model_id_map=up.model_id_map,
            ),
            cache_model="none",
            mode=mode,
            transport=transport,
        )

    def describe(self, config: StrategyConfig) -> str:
        return "Direct to the real provider; live cache-aware baseline."


class HeadroomRunner(StrategyRunner):
    """headroom compression proxy. Config var ``mode``: ``cache`` freezes prior
    turns to maximise provider prefix-cache hits (lighter compression); ``token``
    lets Kompress rewrite prior turns for maximum savings. Sent as a per-request
    ``x-headroom-mode`` header (recent headroom versions read the mode from the
    proxy's own ``--mode`` startup flag instead; the header is then inert).

    headroom has no rewrite function, but honors ``x-headroom-base-url`` as a
    per-request upstream override — which yields both extra paths: ``rewrite``
    points it at a local capture sink (the forwarded body IS the rewrite);
    ``transport=bedrock`` points it straight at Bedrock's Anthropic-compatible
    endpoint with bearer auth (true proxy mode, unlike condense's emulation).
    """

    key = "headroom"
    kind = "proxy"
    tool = "headroom"
    supports_rewrite = True

    def build(
        self,
        name: str,
        config: StrategyConfig,
        mode: Mode = "proxy",
        transport: Transport = "anthropic",
    ) -> ResolvedStrategy:
        s = get_settings()
        hmode = config.get("mode", "cache")
        headers = {"x-headroom-mode": hmode}
        ratio = config.get("target_ratio")
        if ratio is not None:
            headers["x-headroom-target-ratio"] = str(ratio)
        kind = "proxy"
        api_key_env: str | None = "ANTHROPIC_API_KEY"
        model_id_map = None
        suffix = ""
        if mode == "rewrite":
            kind = "rewrite_capture"  # sink URL is injected per-run by the executor
        elif transport == "bedrock":
            from .. import bedrock

            headers["x-headroom-base-url"] = bedrock.anthropic_base(s.bedrock_region)
            headers["authorization"] = f"Bearer {bedrock.bearer_token(s.bedrock_region)}"
            api_key_env = None
            model_id_map = bedrock.bedrock_model_id
            suffix = " (via AWS Bedrock)"
        return ResolvedStrategy(
            name=name,
            kind=kind,
            description=self.describe(config) + suffix,
            proxy=ProxyConfig(
                base_url=s.headroom_base_url,
                provider=Provider.anthropic,
                api_key_env=api_key_env,
                anthropic_beta=s.anthropic_beta,
                extra_headers=headers,
                model_id_map=model_id_map,
                upstream_override_header="x-headroom-base-url",
            ),
            cache_model="retroactive",
            mode=mode,
            transport=transport,
        )

    def describe(self, config: StrategyConfig) -> str:
        hmode = config.get("mode", "cache")
        blurb = (
            "max compression (Kompress rewrites prior turns)"
            if hmode == "token"
            else "cache-optimized (freezes prior turns for prefix-cache hits)"
        )
        return f"headroom compression proxy, {blurb}."


class CondenseRunner(StrategyRunner):
    """condense proxy. Config vars: ``profile`` (dense profile name; None = the
    active ``dense`` target) and ``mode`` (``sync`` blocks the request until
    compaction lands; ``async`` compacts in the background and is paced with
    realistic think time so background compaction has wall-clock time to catch up).

    The run-wide mode/transport picks how it is measured: ``proxy`` forwards to
    the real upstream through condense; ``rewrite`` asks condense's rewrite
    function for the rewritten body (costed offline, zero spend); ``proxy`` +
    ``bedrock`` fetches the rewritten body and invokes Bedrock with it (proxy
    emulation, real Bedrock usage).
    """

    key = "condense"
    kind = "proxy"
    tool = "dense"
    supports_rewrite = True

    def build(
        self,
        name: str,
        config: StrategyConfig,
        mode: Mode = "proxy",
        transport: Transport = "anthropic",
    ) -> ResolvedStrategy:
        s = get_settings()
        profile = load_profile(config.get("profile") or s.condense_profile)
        cmode = config.get("mode", "sync")
        is_async = cmode == "async"
        if mode == "rewrite":
            kind = "rewrite"
        elif transport == "bedrock":
            kind = "rewrite_invoke"  # proxy emulation: rewrite here, invoke Bedrock
        else:
            kind = "proxy"
        headers = {
            "x-condense-user-id": profile.user_id or "",
            "x-condense-auto-condense-mode": cmode,
        }
        if kind in ("rewrite", "rewrite_invoke"):
            headers["x-condense-function"] = "rewrite"
        if profile.auth_token:
            headers["x-condense-auth-token"] = profile.auth_token
        think = config.get("think_secs_per_output_token", s.condense_async_secs_per_output_token)
        cap = config.get("think_cap_seconds", s.condense_async_max_think_seconds)
        return ResolvedStrategy(
            name=name,
            kind=kind,
            description=self.describe(config),
            proxy=ProxyConfig(
                base_url=profile.api_url,
                provider=Provider.anthropic,
                api_key_env="ANTHROPIC_API_KEY",
                anthropic_beta=s.anthropic_beta,
                path="/anthropic/v1/messages",
                extra_headers=headers,
                # No fixed request interval: pacing is purely token-driven (the async
                # think time). sync mode has no client-side pacing at all — it relies
                # only on condense's own sync hold.
                min_request_interval=0.0,
                think_secs_per_output_token=think if is_async else 0.0,
                think_cap_seconds=cap if is_async else 0.0,
                session_id_header="x-condense-session-id",
            ),
            cache_model="retroactive",
            mode=mode,
            transport=transport,
        )

    def describe(self, config: StrategyConfig) -> str:
        cmode = config.get("mode", "sync")
        prof = config.get("profile") or "target"
        blurb = (
            "async compaction (background; paced by realistic think time)"
            if cmode == "async"
            else "sync compaction (blocks the request until compacted)"
        )
        return f"condense, {blurb}; dense profile {prof!r}."


NOOP = NoopRunner()
UPSTREAM = UpstreamRunner()
HEADROOM = HeadroomRunner()
CONDENSE = CondenseRunner()

RUNNERS_BY_KEY: dict[str, StrategyRunner] = {
    r.key: r for r in (NOOP, UPSTREAM, HEADROOM, CONDENSE)
}
