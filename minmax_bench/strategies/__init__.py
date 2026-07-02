from .base import (
    ProxyConfig,
    ResolvedStrategy,
    Strategy,
    StrategyConfig,
    StrategyRunner,
)
from .matrix import (
    BASELINE,
    STRATEGY_MATRIX,
    default_selected,
    get_entry,
    has_entry,
    matrix_names,
    resolve_strategy,
    selectable,
    tool_for,
)

# Back-compat: ``get_strategy`` / ``list_strategies`` used across the codebase.
get_strategy = resolve_strategy
list_strategies = matrix_names

__all__ = [
    "ProxyConfig",
    "ResolvedStrategy",
    "Strategy",
    "StrategyConfig",
    "StrategyRunner",
    "BASELINE",
    "STRATEGY_MATRIX",
    "matrix_names",
    "get_entry",
    "has_entry",
    "selectable",
    "default_selected",
    "tool_for",
    "resolve_strategy",
    "get_strategy",
    "list_strategies",
]
