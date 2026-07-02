"""The strategy matrix — the root config of what we benchmark.

A flat, ordered list of :class:`Strategy` entries, each a named ``runner + config``
variant. Add a row to test a new condense config (a different dense profile, mode,
or think time). The first entry is the mandatory *vanilla* baseline that always
runs. This matrix drives the run itself, the report rows, and the wizard (which
rows the user can filter out, and what gets set up).
"""

from __future__ import annotations

from .base import ResolvedStrategy, Strategy, StrategyConfig
from .runners import CONDENSE, GEMINI, HEADROOM, NOOP, RUNNERS_BY_KEY, UPSTREAM

# The mandatory baseline's name == runstore.BASELINE, so it stores as the
# "original cost" cache and every strategy is scored against it.
BASELINE = "baseline"


def _cfg(**params) -> StrategyConfig:
    return StrategyConfig(params=params)


# The competitive set: headroom (cache-optimized) vs headroom-kompress (max
# compression) vs condense sync/async.
STRATEGY_MATRIX: list[Strategy] = [
    Strategy(BASELINE, NOOP, mandatory=True, enabled=True),
    Strategy("upstream", UPSTREAM, enabled=False),
    Strategy("gemini", GEMINI, enabled=False),
    Strategy("headroom", HEADROOM, _cfg(mode="cache"), enabled=True),
    Strategy("headroom-kompress", HEADROOM, _cfg(mode="token"), enabled=True),
    Strategy("condense-sync", CONDENSE, _cfg(mode="sync"), enabled=True),
    Strategy("condense-async", CONDENSE, _cfg(mode="async"), enabled=True),
]


def matrix_names() -> list[str]:
    return [s.name for s in STRATEGY_MATRIX]


def get_entry(name: str) -> Strategy:
    for s in STRATEGY_MATRIX:
        if s.name == name:
            return s
    raise KeyError(name)


def has_entry(name: str) -> bool:
    return any(s.name == name for s in STRATEGY_MATRIX)


def selectable() -> list[Strategy]:
    """Matrix entries the wizard offers (everything except the mandatory baseline)."""
    return [s for s in STRATEGY_MATRIX if not s.mandatory]


def default_selected() -> list[str]:
    return [s.name for s in STRATEGY_MATRIX if s.enabled and not s.mandatory]


def tool_for(name: str) -> str | None:
    try:
        return get_entry(name).tool
    except KeyError:
        return None


def resolve_strategy(name: str) -> ResolvedStrategy:
    """Resolve a matrix entry name; fall back to a runner key with default config
    (so ``resolve_strategy('noop')`` works without a matrix row)."""
    if has_entry(name):
        return get_entry(name).resolve()
    runner = RUNNERS_BY_KEY.get(name)
    if runner is not None:
        return runner.build(name, StrategyConfig())
    raise KeyError(f"unknown strategy {name!r}; available: {', '.join(matrix_names())}")
