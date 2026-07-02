"""Executor protocol and shared helpers.

An executor measures how a strategy performs on one session. It receives the
session and its ordered :class:`RequestPoint` list (so proxy caches warm and
rewrite caching can be tracked across turns) and returns one
:class:`Measurement` per point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..models import RequestPoint, Session, Usage
from ..strategies.base import ResolvedStrategy as Strategy
from ..tokens import TokenCounter


@dataclass
class Measurement:
    index: int
    session_id: str
    usage: Usage
    cost_usd: float
    ok: bool = True
    error: str | None = None


class Executor(Protocol):
    def run_session(
        self,
        session: Session,
        points: list[RequestPoint],
        strategy: Strategy,
        on_point: Callable[[Measurement], None] | None = None,
    ) -> list[Measurement]: ...


def output_tokens_for(point: RequestPoint, counter: TokenCounter) -> int:
    """Output tokens for costing: the recorded value, else a count of the turn.

    Output is held constant across strategies (deterministic replay), so proxy
    runs cap live generation at 1 token and we substitute this value for cost.
    """
    if point.recorded_usage and point.recorded_usage.output_tokens > 0:
        return point.recorded_usage.output_tokens
    return counter.count_message(point.expected_output)
