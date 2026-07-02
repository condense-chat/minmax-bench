"""Concrete strategy runners — one reusable implementation per technique.

A runner turns a :class:`StrategyConfig` into a :class:`ResolvedStrategy`. Runners
are cheap singletons built at import; anything that can fail (loading dense creds,
loading local resources) happens lazily in :meth:`build`, i.e. at run/preflight
time, never at import.
"""

from __future__ import annotations

from ..config import get_settings
from ..dense import load_profile
from ..models import Provider
from .base import ProxyConfig, ResolvedStrategy, StrategyConfig, StrategyRunner


class NoopRunner(StrategyRunner):
    key = "noop"
    kind = "noop"

    def build(self, name: str, config: StrategyConfig) -> ResolvedStrategy:
        return ResolvedStrategy(
            name=name,
            kind="noop",
            description=self.describe(config),
            cache_model="none",
        )

    def describe(self, config: StrategyConfig) -> str:
        return "Uncompressed baseline (recorded usage or local token count)."


class UpstreamRunner(StrategyRunner):
    key = "upstream"
    kind = "proxy"

    def build(self, name: str, config: StrategyConfig) -> ResolvedStrategy:
        s = get_settings()
        return ResolvedStrategy(
            name=name,
            kind="proxy",
            description=self.describe(config),
            proxy=ProxyConfig(
                base_url=s.anthropic_base_url,
                provider=Provider.anthropic,
                api_key_env="ANTHROPIC_API_KEY",
                anthropic_beta=s.anthropic_beta,
            ),
            cache_model="none",
        )

    def describe(self, config: StrategyConfig) -> str:
        return "Direct to the real provider; live cache-aware baseline."



class HeadroomRunner(StrategyRunner):
    """headroom compression proxy. Config var ``mode``: ``cache`` freezes prior
    turns to maximise provider prefix-cache hits (lighter compression); ``token``
    lets Kompress rewrite prior turns for maximum savings. The mode is sent as a
    per-request ``x-headroom-mode`` header so both variants can share one proxy.
    """

    key = "headroom"
    kind = "proxy"
    tool = "headroom"

    def build(self, name: str, config: StrategyConfig) -> ResolvedStrategy:
        s = get_settings()
        mode = config.get("mode", "cache")
        headers = {"x-headroom-mode": mode}
        ratio = config.get("target_ratio")
        if ratio is not None:
            headers["x-headroom-target-ratio"] = str(ratio)
        return ResolvedStrategy(
            name=name,
            kind="proxy",
            description=self.describe(config),
            proxy=ProxyConfig(
                base_url=s.headroom_base_url,
                provider=Provider.anthropic,
                api_key_env="ANTHROPIC_API_KEY",
                anthropic_beta=s.anthropic_beta,
                extra_headers=headers,
            ),
            cache_model="retroactive",
        )

    def describe(self, config: StrategyConfig) -> str:
        mode = config.get("mode", "cache")
        blurb = (
            "max compression (Kompress rewrites prior turns)"
            if mode == "token"
            else "cache-optimized (freezes prior turns for prefix-cache hits)"
        )
        return f"headroom compression proxy, {blurb}."


class CondenseRunner(StrategyRunner):
    """condense proxy. Config vars: ``profile`` (dense profile name; None = the
    active ``dense`` target) and ``mode`` (``sync`` blocks the request until
    compaction lands; ``async`` compacts in the background and is paced with
    realistic think time so background compaction has wall-clock time to catch up).
    """

    key = "condense"
    kind = "proxy"
    tool = "dense"

    def build(self, name: str, config: StrategyConfig) -> ResolvedStrategy:
        s = get_settings()
        profile = load_profile(config.get("profile") or s.condense_profile)
        mode = config.get("mode", "sync")
        is_async = mode == "async"
        headers = {
            "x-condense-user-id": profile.user_id or "",
            "x-condense-auto-condense-mode": mode,
        }
        if profile.auth_token:
            headers["x-condense-auth-token"] = profile.auth_token
        think = config.get("think_secs_per_output_token", s.condense_async_secs_per_output_token)
        cap = config.get("think_cap_seconds", s.condense_async_max_think_seconds)
        return ResolvedStrategy(
            name=name,
            kind="proxy",
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
        )

    def describe(self, config: StrategyConfig) -> str:
        mode = config.get("mode", "sync")
        prof = config.get("profile") or "target"
        blurb = (
            "async compaction (background; paced by realistic think time)"
            if mode == "async"
            else "sync compaction (blocks the request until compacted)"
        )
        return f"condense proxy, {blurb}; dense profile {prof!r}."


class GeminiRunner(StrategyRunner):
    """Direct to Google Gemini via its OpenAI-compatible chat/completions endpoint.

    A real (uncompressed) executor — like ``upstream`` but pointed at Gemini and
    speaking the OpenAI dialect — so the pipeline can be evaluated on a cheap model
    without Anthropic spend. Applies only to Gemini/OpenAI-dialect models.
    """

    key = "gemini"
    kind = "proxy"

    def build(self, name: str, config: StrategyConfig) -> ResolvedStrategy:
        s = get_settings()
        return ResolvedStrategy(
            name=name,
            kind="proxy",
            description=self.describe(config),
            proxy=ProxyConfig(
                base_url=s.gemini_base_url,
                provider=Provider.openai,
                api_key_env="GEMINI_API_KEY",
                path="/chat/completions",
                flatten_tools=True,
            ),
            cache_model="none",
        )

    def describe(self, config: StrategyConfig) -> str:
        return "Direct to Google Gemini (OpenAI-compatible chat/completions)."


NOOP = NoopRunner()
UPSTREAM = UpstreamRunner()
HEADROOM = HeadroomRunner()
CONDENSE = CondenseRunner()
GEMINI = GeminiRunner()

RUNNERS_BY_KEY: dict[str, StrategyRunner] = {
    r.key: r for r in (NOOP, UPSTREAM, HEADROOM, CONDENSE, GEMINI)
}
