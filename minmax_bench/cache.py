"""Persistent measurement cache.

Proxy runs cost real money and baseline counting costs time, so we persist each
strategy's per-point :class:`Usage` keyed by (cache_key, session_id). A rerun with
the same config replays from disk instead of re-calling the upstream; pass a
refresh set to force specific strategies to recompute.

The cache stores raw usage (not cost) so pricing changes are picked up on reload.
Results are only cached for a session when the *whole* session ran (all points
present) — proxy usage depends on in-order cache warming, so partial reuse would
be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

from .executors.base import Measurement
from .models import Usage

# Bump when measurement semantics change so stale entries are ignored.
CACHE_VERSION = "v1"


class MeasurementCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._data: dict[str, dict[str, list[dict]]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self._dirty = False

    def get(self, key: str, session_id: str, n_points: int) -> list[Measurement] | None:
        rows = self._data.get(key, {}).get(session_id)
        if rows is None or len(rows) != n_points:
            return None
        return [
            Measurement(
                index=r["i"],
                session_id=session_id,
                usage=Usage(
                    input_tokens=r["it"], output_tokens=r["ot"],
                    cache_read=r["cr"], cache_write=r["cw"],
                ),
                cost_usd=0.0,  # recomputed by the caller from usage
                ok=r["ok"],
                error=r.get("err"),
            )
            for r in rows
        ]

    def put(self, key: str, session_id: str, measurements: list[Measurement]) -> None:
        self._data.setdefault(key, {})[session_id] = [
            {
                "i": m.index, "it": m.usage.input_tokens, "ot": m.usage.output_tokens,
                "cr": m.usage.cache_read, "cw": m.usage.cache_write,
                "ok": m.ok, "err": m.error,
            }
            for m in measurements
        ]
        self._dirty = True

    def save(self) -> None:
        if self.path and self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data))
            self._dirty = False


def cache_key(prefix: str, model: str, max_tokens: int) -> str:
    return f"{CACHE_VERSION}|{prefix}|{model}|mt{max_tokens}"
