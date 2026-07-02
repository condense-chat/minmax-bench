"""Orchestration: load -> simulate -> measure (model × strategy × conversation).

Two visible phases: the mandatory **vanilla** baseline ("original cost"), then the
selected **strategies**. Within each phase all (model, strategy, conversation) units
run in parallel on a worker pool; a shared :class:`EndpointGate` per destination
caps concurrency and paces request starts so we never exceed a proxy's rate limit.
Within one unit the turns stay sequential so a proxy's prefix cache warms in order.

Compression proxies (condense/headroom) speak Anthropic, so for OpenAI models only
the provider-neutral strategies (baseline) run. Everything is keyed
by ``(model, conversation)`` with a per-run cache-bust uuid, and only raw usage is
stored — cost is recomputed per model from it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .catalog import is_openai
from .chain import LocalCounter, make_counter
from .config import get_settings
from .dashboard import Dashboard
from .data import load_dataset
from .executors import NoopExecutor, ProxyExecutor
from .executors.base import Measurement
from .gate import EndpointGate
from .harness import simulate, with_test_run
from .models import Provider
from .report import BucketStats
from .runstore import BASELINE, RunStore
from .strategies import get_strategy
from .strategies.base import ResolvedStrategy as Strategy
from .tokens import TokenCounter


@dataclass
class RunResult:
    run_uuid: str = ""
    # {model: {strategy: [BucketStats]}}
    per_model_buckets: dict[str, dict[str, list[BucketStats]]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _run_and_close(ex, session, points, strategy, on_point) -> list[Measurement]:
    try:
        return ex.run_session(session, points, strategy, on_point=on_point)
    finally:
        close = getattr(ex, "close", None)
        if close:
            close()


def key(model: str, kind: str) -> str:
    return f"{model}::{kind}"


def _applicable(strategies: list[Strategy], model: str) -> list[Strategy]:
    """Keep local strategies always; keep a proxy strategy only if it speaks the
    model's dialect (Anthropic proxies for Claude models, the OpenAI/Gemini executor
    for OpenAI-dialect models)."""
    mprov = Provider.openai if is_openai(model) else Provider.anthropic
    out: list[Strategy] = []
    for s in strategies:
        if s.kind != "proxy":
            out.append(s)
        elif s.proxy and s.proxy.provider == mprov:
            out.append(s)
    return out


def run(
    store: RunStore,
    *,
    refresh: set[str] | None = None,
    dashboard: Dashboard | None = None,
) -> RunResult:
    """Measure every (model, strategy) in ``store.manifest`` into the run store."""
    refresh = refresh or set()
    m = store.manifest
    result = RunResult(run_uuid=m.uuid)

    def _refresh(name: str) -> bool:
        return "all" in refresh or name in refresh

    def _err(msg: str) -> None:
        result.errors.append(msg)
        store.log_error(msg)  # instant, crash-safe persistence to errors.log

    base_sessions = load_dataset(m.dataset)
    if m.session_ids is not None:
        keep = set(m.session_ids)
        base_sessions = [s for s in base_sessions if s.id in keep]
    if m.session_limit is not None:
        base_sessions = base_sessions[: m.session_limit]
    if m.longest:
        base_sessions = sorted(
            base_sessions, key=lambda s: len(s.messages), reverse=True
        )[: m.longest]

    settings = get_settings()
    out_counter = TokenCounter(encoding=m.encoding)
    chain_counter = make_counter(
        m.count_mode, m.encoding, settings.anthropic_base_url, settings.anthropic_beta
    )
    noop = get_strategy("noop")
    strategies = []
    for n in m.strategies:
        try:
            strategies.append(get_strategy(n))
        except Exception as e:  # a variant whose creds/tools are missing: skip, don't abort
            _err(f"{n}: unavailable ({e})")

    # One shared gate per destination endpoint, built up front (single-threaded) so
    # workers only read it. condense sync + async share a gate on api.condense.chat.
    gates: dict[str, EndpointGate] = {}
    for strat in strategies:
        if strat.kind == "proxy" and strat.proxy:
            g = gates.get(strat.proxy.base_url)
            if g is None:
                gates[strat.proxy.base_url] = EndpointGate(
                    settings.endpoint_max_concurrency, strat.proxy.min_request_interval
                )
            else:
                g.raise_interval(strat.proxy.min_request_interval)

    def _executor_for(strategy: Strategy):
        if strategy.kind == "proxy":
            gate = gates.get(strategy.proxy.base_url) if strategy.proxy else None
            return ProxyExecutor(counter=out_counter, max_tokens=m.max_tokens, gate=gate)
        return NoopExecutor(chain_counter, out_counter)

    # Cheap offline counter used only to truncate each conversation at the token
    # budget (independent of count_mode so we never pay the API just to size a chain).
    budget = m.token_limit
    trunc = LocalCounter(m.encoding) if budget else None

    def _truncate(session, points):
        """Keep turns up to the first whose chain reaches >= token_limit tokens."""
        if trunc is None or not budget:
            return points
        kept = []
        for p in points:
            kept.append(p)
            if trunc.count(session, p.prefix) >= budget:
                break
        return kept

    # Per model: route sessions to that model/provider, stamp a per-(model,session)
    # cache-bust id, and simulate up front so the dashboard knows each turn count.
    prepared: dict[str, list] = {}
    for model in m.models:
        prov = Provider.openai if is_openai(model) else Provider.anthropic
        rows = []
        for s in base_sessions:
            se = s.model_copy(update={"model": model, "provider": prov})
            se = with_test_run(se, test_uuid=store.test_uuid(model, se.id))
            points = simulate(se)
            if m.point_limit is not None:
                points = points[: m.point_limit]
            points = _truncate(se, points)
            if points:
                rows.append((se, points))
        prepared[model] = rows

    if dashboard is not None:
        for model in m.models:
            total = sum(len(p) for _, p in prepared[model])
            convos = len(prepared[model])
            dashboard.ensure(key(model, BASELINE), f"{model} · baseline", total, convos)
            for strat in _applicable(strategies, model):
                dashboard.ensure(
                    key(model, strat.name), f"{model} · {strat.name}", total, convos
                )

    def measure(model, kind, session, points, run_fresh) -> list[Measurement] | None:
        n = len(points)

        def on_point(mm, model=model, kind=kind):
            if dashboard is not None:
                dashboard.on_point(key(model, kind), mm, model)

        if not _refresh(kind):
            hit = store.get(model, kind, session.id, n)
            if hit is not None:
                for mm in hit:
                    on_point(mm)
                return hit
        measured = run_fresh(on_point)
        store.put(model, kind, session.id, measured)
        return measured

    def unit(model: str, kind: str, strat: Strategy, session, points) -> list[str]:
        """One (model, strategy, conversation) work item; returns its error strings."""
        if dashboard is not None:
            dashboard.begin(key(model, kind))
        try:
            measured = measure(
                model, kind, session, points,
                lambda op: _run_and_close(_executor_for(strat), session, points, strat, op),
            )
        except Exception as e:  # whole-session failure (e.g. proxy unreachable)
            return [f"{kind} / {model} / {session.id}: {e}"]
        finally:
            if dashboard is not None:
                dashboard.finish(key(model, kind))
        return [
            f"{kind}/{model}/{session.id}#{mm.index}: {mm.error}"
            for mm in (measured or [])
            if not mm.ok and mm.error
        ]

    workers = max(1, settings.run_max_workers)

    def run_phase(units: list[tuple]) -> None:
        if not units:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(unit, *u) for u in units]
            for f in futures:
                for e in f.result():
                    _err(e)

    # -- Phase 1: vanilla / baseline ------------------------------------------
    if dashboard is not None:
        dashboard.set_phase("① vanilla cost estimation — uncompressed baseline")
    run_phase(
        [(model, BASELINE, noop, se, pts) for model in m.models for se, pts in prepared[model]]
    )
    store.save()

    # Surface every strategy dropped for a model (dialect mismatch) so a run can
    # never silently measure nothing — e.g. Anthropic proxies vs a Gemini model.
    for model in m.models:
        kept = {s.name for s in _applicable(strategies, model)}
        model_dialect = "OpenAI/Gemini" if is_openai(model) else "Anthropic"
        for strat in strategies:
            if strat.name not in kept:
                strat_dialect = strat.proxy.provider.value if strat.proxy else strat.kind
                _err(
                    f"{strat.name} / {model}: skipped — {strat_dialect} strategy "
                    f"can't run against a {model_dialect}-dialect model"
                )

    # -- Phase 2: strategies ---------------------------------------------------
    if dashboard is not None:
        dashboard.set_phase("② strategies")
    run_phase(
        [
            (model, strat.name, strat, se, pts)
            for model in m.models
            for strat in _applicable(strategies, model)
            for se, pts in prepared[model]
        ]
    )
    store.save()

    del result.errors[30:]  # cap the in-memory list for console; errors.log keeps all

    fallbacks = getattr(chain_counter, "fallbacks", 0)
    if fallbacks:
        _err(f"count_tokens: {fallbacks} point(s) fell back to local tiktoken")
    close = getattr(chain_counter, "close", None)
    if close:
        close()
    store.save()

    result.per_model_buckets = store.buckets()
    return result
