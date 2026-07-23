"""Baseline (uncompressed) executor.

Costs the *full* reconstructed chain with an exact, cache-aware model (see
:mod:`minmax_bench.chain`). This is the "no compression" reference every strategy is
scored against; it reaches the true chain length (512k+) rather than the source
agent's already-compacted recorded usage.
"""

from __future__ import annotations

from collections.abc import Callable

from ..chain import iter_chain_usages
from ..models import RequestPoint, Session
from ..pricing import cost_usd
from ..strategies.base import ResolvedStrategy as Strategy
from ..tokens import TokenCounter
from .base import Measurement, output_tokens_for


class NoopExecutor:
    def __init__(self, counter, out_counter: TokenCounter | None = None):
        self.counter = counter  # a chain counter (Local/Anthropic)
        self.out_counter = out_counter or TokenCounter()

    def run_session(
        self,
        session: Session,
        points: list[RequestPoint],
        strategy: Strategy,
        on_point: Callable[[Measurement], None] | None = None,
    ) -> list[Measurement]:
        usages = iter_chain_usages(
            session, points, self.counter,
            lambda p: output_tokens_for(p, self.out_counter), rewrite=None,
        )
        out: list[Measurement] = []
        # zip over the generator: each point emits as soon as it is counted, so
        # the live dashboard advances during (slow) API counting instead of
        # bursting the whole session at the end.
        for p, u in zip(points, usages, strict=False):
            m = Measurement(p.index, session.id, u, cost_usd(session.model, u))
            out.append(m)
            if on_point is not None:
                on_point(m)
        return out
