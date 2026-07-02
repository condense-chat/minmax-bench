"""Shared, self-adjusting per-endpoint rate gate.

When strategies run in parallel, several worker threads hit the same destination
(e.g. condense sync and condense async both go to ``api.condense.chat``). One gate
is shared per destination and enforces:

* **max concurrency** — at most N requests in flight to that endpoint at once.
* **adaptive spacing** — a minimum interval between request *starts*, shared by
  every worker on the host. It follows AIMD: a rate-limit response (429/529)
  **multiplicatively widens** the spacing for everyone (honoring ``Retry-After``),
  and each success **additively relaxes** it back toward the configured floor. So
  the whole run automatically slows down when the endpoint pushes back and speeds
  back up once it recovers — without a hand-tuned interval.

Gates are keyed by base URL, so strategies that share a host share one limit.
Local, un-throttled work (the baseline, local rewrites) uses no gate.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class EndpointGate:
    def __init__(
        self,
        max_concurrency: int,
        min_interval: float,
        *,
        backoff_cap: float = 30.0,
        penalty_initial: float = 1.0,
        decay: float = 0.05,
    ) -> None:
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._base = max(0.0, min_interval)   # configured floor
        self._interval = self._base           # current adaptive spacing
        self._next_start = 0.0                # monotonic clock; next reservable start
        self._cap = backoff_cap
        self._penalty_initial = penalty_initial
        self._decay = decay
        self.penalties = 0                    # observability: rate-limit hits seen

    def raise_interval(self, interval: float) -> None:
        """Widen the floor to the most conservative value seen for this host."""
        with self._lock:
            self._base = max(self._base, max(0.0, interval))
            self._interval = max(self._interval, self._base)

    def penalize(self, retry_after: float | None = None) -> float:
        """A rate-limit was hit — widen spacing for everyone on this host (AIMD)."""
        with self._lock:
            cur = self._interval
            bumped = cur * 2 if cur > 0 else self._penalty_initial
            bumped = max(bumped, self._penalty_initial, retry_after or 0.0)
            self._interval = min(self._cap, bumped)
            self.penalties += 1
            return self._interval

    def reward(self) -> None:
        """A request succeeded — relax spacing back toward the configured floor."""
        with self._lock:
            if self._interval > self._base:
                self._interval = max(self._base, self._interval - self._decay)

    def _reserve(self) -> None:
        with self._lock:
            interval = self._interval
            if interval <= 0:
                return
            # Atomically claim the next start slot; sleep to it outside the lock so
            # concurrent waiters get sequential, evenly-spaced start times.
            start_at = max(time.monotonic(), self._next_start)
            self._next_start = start_at + interval
        wait = start_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    @contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            self._reserve()
            yield
        finally:
            self._sem.release()
