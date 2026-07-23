"""Strategy model.

The benchmark's strategy set is a **matrix** of :class:`Strategy` entries. Each
entry pairs a generic :class:`StrategyRunner` (the reusable *implementation* of a
technique — condense, headroom, …) with a :class:`StrategyConfig` (its vars,
e.g. condense's dense profile + sync/async mode). Resolving an entry yields a
:class:`ResolvedStrategy` — the concrete, executor-facing object (endpoint +
headers) that :mod:`minmax_bench.executors` consumes.

A strategy is only the compression technique. *How* it is measured is a separate,
run-wide choice resolved at :meth:`Strategy.resolve` time:

* ``mode`` — ``proxy`` sends the real request through the strategy's proxy, which
  forwards to the upstream (real usage, real money); ``rewrite`` asks the proxy's
  rewrite function for the transformed request body and costs it offline (zero
  model spend).
* ``transport`` — where direct model traffic lands: ``anthropic`` or ``bedrock``.
  ``mode=proxy`` + ``transport=bedrock`` on a compression strategy *emulates* the
  proxy: fetch the rewritten body, send it to Bedrock ourselves, report Bedrock's
  real usage (the proxy itself cannot reach Bedrock).

The resolved ``kind`` is the executor selector derived from that combination:
``proxy`` | ``rewrite`` | ``rewrite_invoke`` | ``noop``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import Provider

ExecutorKind = Literal["proxy", "rewrite", "rewrite_capture", "rewrite_invoke", "noop"]
CacheModel = Literal["retroactive", "none"]
Mode = Literal["proxy", "rewrite"]
Transport = Literal["anthropic", "bedrock"]

MODES = ("proxy", "rewrite")
TRANSPORTS = ("anthropic", "bedrock")


@dataclass
class ProxyConfig:
    """How to reach a proxy strategy and forward to the real upstream."""

    base_url: str
    provider: Provider = Provider.anthropic
    # Env var holding the UPSTREAM provider key the proxy forwards with.
    # None = no key header; auth must come via extra_headers (e.g. Bedrock bearer).
    api_key_env: str | None = "ANTHROPIC_API_KEY"
    extra_headers: dict[str, str] = field(default_factory=dict)
    anthropic_version: str = "2023-06-01"
    # Optional anthropic-beta header value (e.g. 1M-context) for Anthropic proxies.
    anthropic_beta: str | None = None
    # Full endpoint path override (e.g. condense mounts "/anthropic/v1/messages").
    # When None, the default provider path is used.
    path: str | None = None
    # Minimum seconds between requests to this endpoint (the shared EndpointGate
    # paces global request starts to stay under the proxy's rate limit).
    min_request_interval: float = 0.0
    # "Think time" inserted *after* each turn, proportional to that turn's output
    # tokens (out_tokens * this), to simulate a real user reading/working before the
    # next request. Gives background (async) compaction wall-clock time to land, so
    # async savings are measured realistically. 0 = no wait (e.g. sync mode).
    think_secs_per_output_token: float = 0.0
    # Upper bound on a single turn's think time (0 = uncapped), so one huge output
    # turn doesn't stall the run for minutes.
    think_cap_seconds: float = 0.0
    # Header name this proxy uses to key its own session cache by our per-run test
    # uuid (e.g. condense's "x-condense-session-id"). None = the proxy has no such
    # cache, so no session header is sent (avoids leaking one strategy's header
    # onto another strategy's requests).
    session_id_header: str | None = None
    # Flatten history tool_use/tool_result into text and send no structured tool
    # calls. Needed for OpenAI-dialect endpoints that reject functionCall parts
    # lacking a thought_signature.
    flatten_tools: bool = False
    # Rewrite the request's model id for this upstream (e.g. Bedrock requires
    # inference-profile ids). Pricing keeps using the session's model.
    model_id_map: Callable[[str], str] | None = None
    # Header this proxy honors as a per-request upstream base-url override (e.g.
    # headroom's "x-headroom-base-url"). Enables rewrite-by-capture and direct
    # alternate-transport forwarding for proxies without a rewrite function.
    upstream_override_header: str | None = None


@dataclass
class ResolvedStrategy:
    """Concrete, executor-facing strategy (a matrix entry resolved with its config)."""

    name: str
    kind: ExecutorKind
    description: str = ""
    proxy: ProxyConfig | None = None
    # Informational: how the strategy interacts with prompt caching.
    #   retroactive -> rewrites historical turns; may invalidate cache (condense)
    #   none        -> baseline
    cache_model: CacheModel = "retroactive"
    # The run-wide measurement choice this strategy was resolved under.
    mode: Mode = "proxy"
    transport: Transport = "anthropic"

    def __post_init__(self):
        if self.kind != "noop" and self.proxy is None:
            raise ValueError(f"{self.kind} strategy {self.name!r} needs a ProxyConfig")


@dataclass
class StrategyConfig:
    """The vars for one matrix variant (e.g. condense's ``profile`` + ``mode``)."""

    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        v = self.params.get(key, default)
        return default if v is None else v


class UnsupportedCombo(RuntimeError):
    """The strategy cannot run under the requested mode/transport."""


class StrategyRunner:
    """Generic, reusable implementation of a technique. Subclasses build a
    :class:`ResolvedStrategy` from a :class:`StrategyConfig` plus the run-wide
    mode/transport.

    ``tool`` names the local tool the runner needs (for auto-setup / preflight),
    or ``None`` for pure-local runners (noop, upstream).
    ``supports_rewrite`` gates ``mode=rewrite`` and the bedrock proxy emulation —
    both need the proxy to expose a rewrite function.
    """

    key: str = ""
    kind: ExecutorKind = "noop"
    tool: str | None = None
    supports_rewrite: bool = False

    def build(
        self,
        name: str,
        config: StrategyConfig,
        mode: Mode = "proxy",
        transport: Transport = "anthropic",
    ) -> ResolvedStrategy:
        raise NotImplementedError

    def describe(self, config: StrategyConfig) -> str:  # noqa: ARG002
        return ""


@dataclass
class Strategy:
    """One matrix entry: a named ``runner + config`` variant to benchmark."""

    name: str
    runner: StrategyRunner
    config: StrategyConfig = field(default_factory=StrategyConfig)
    enabled: bool = True     # pre-selected in the wizard
    mandatory: bool = False  # always runs (the vanilla baseline)

    @property
    def kind(self) -> ExecutorKind:
        return self.runner.kind

    @property
    def tool(self) -> str | None:
        return self.runner.tool

    @property
    def description(self) -> str:
        return self.runner.describe(self.config)

    def resolve(
        self, mode: Mode = "proxy", transport: Transport = "anthropic"
    ) -> ResolvedStrategy:
        return self.runner.build(self.name, self.config, mode=mode, transport=transport)
