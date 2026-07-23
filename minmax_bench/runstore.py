"""Per-run, per-model, per-conversation measurement store.

Three tiers of cache:

* **data cache** — the source conversations (``data/cache/``). Shared across every
  run and *not* scoped here: it is input data, not measurement.
* **original-cost cache** — the uncompressed ``baseline`` measurement for each
  conversation, kept **per model** (``run-<uuid>/models/<model>/baseline.json``).
* **per-strategy cache** — each strategy's measurement per conversation, per model
  (``run-<uuid>/models/<model>/strategies/<name>.json``).

The latter two live under a per-run ``run-<uuid>/`` directory. A fresh run mints a
new uuid (clean measurement); passing an existing uuid resumes it. Each
``(model, conversation)`` gets a per-run cache-bust uuid (stored in the manifest,
stable across resume, unique per run) stamped into the request, so one run/model's
proxy chain state can never contaminate another's.

Only raw token :class:`Usage` is stored — never dollars — so cost is always
*recomputed* from the stored usage against current per-model pricing.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .executors.base import Measurement
from .models import Usage
from .report import BucketStats, PairedRow, summarize

BASELINE = "baseline"


@dataclass
class RunManifest:
    uuid: str
    created_utc: str
    dataset: str
    strategies: list[str]
    models: list[str]
    encoding: str
    count_mode: str
    max_tokens: int
    edges: list[int]
    session_limit: int | None = None
    point_limit: int | None = None
    longest: int | None = None
    # Truncate each conversation at the first turn whose chain reaches >= this many
    # tokens (None = run every turn). Shorter conversations are kept whole.
    token_limit: int | None = None
    # Explicit conversation ids to keep (None = all). Set by the wizard's range filter.
    session_ids: list[str] | None = None
    # Run-wide measurement method: proxy (real upstream call) | rewrite (offline cost).
    # Recorded here because cached measurements are keyed by strategy NAME — resuming
    # under a different mode/transport would silently mix incomparable numbers.
    mode: str = "proxy"
    # Where direct model traffic lands: anthropic | bedrock.
    transport: str = "anthropic"
    # "<model>::<session_id>" -> per-run cache-bust uuid (stamped into the request).
    session_test_uuids: dict[str, str] = field(default_factory=dict)


def _safe(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def _dump(ms: list[Measurement]) -> list[list]:
    return [
        [m.index, m.usage.input_tokens, m.usage.output_tokens,
         m.usage.cache_read, m.usage.cache_write, m.ok, m.error]
        for m in ms
    ]


def _load(session_id: str, rows: list[list]) -> list[Measurement]:
    out: list[Measurement] = []
    for r in rows:
        out.append(
            Measurement(
                index=r[0],
                session_id=session_id,
                usage=Usage(
                    input_tokens=r[1], output_tokens=r[2], cache_read=r[3], cache_write=r[4]
                ),
                cost_usd=0.0,  # recomputed from usage by the caller
                ok=r[5],
                error=r[6],
            )
        )
    return out


class _KindCache:
    """One JSON file: ``{session_id: {"model": str, "points": [[...], ...]}}``."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self._dirty = False

    def get(self, session_id: str, n_points: int) -> list[Measurement] | None:
        entry = self._data.get(session_id)
        if entry is None or len(entry["points"]) != n_points:
            return None
        return _load(session_id, entry["points"])

    def put(self, session_id: str, model: str, measurements: list[Measurement]) -> None:
        self._data[session_id] = {"model": model, "points": _dump(measurements)}
        self._dirty = True

    def session_ids(self) -> list[str]:
        return list(self._data)

    def save(self) -> None:
        if self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data))
            self._dirty = False


class RunStore:
    def __init__(self, root: Path, manifest: RunManifest):
        self.root = root
        self.manifest = manifest
        self._caches: dict[tuple[str, str], _KindCache] = {}
        self._lock = threading.RLock()  # get/put/save fire from parallel workers

    # -- lifecycle ------------------------------------------------------------
    @classmethod
    def create(cls, runs_dir: str | Path, manifest_fields: dict) -> RunStore:
        run_uuid = str(uuid.uuid4())
        root = Path(runs_dir) / f"run-{run_uuid}"
        root.mkdir(parents=True, exist_ok=True)
        manifest = RunManifest(uuid=run_uuid, **manifest_fields)
        store = cls(root, manifest)
        store._write_manifest()
        return store

    @classmethod
    def open(cls, runs_dir: str | Path, run_uuid: str) -> RunStore:
        run_uuid = run_uuid.removeprefix("run-")
        root = Path(runs_dir) / f"run-{run_uuid}"
        mpath = root / "run.json"
        if not mpath.exists():
            raise FileNotFoundError(f"no such run: {root}")
        manifest = RunManifest(**json.loads(mpath.read_text()))
        return cls(root, manifest)

    @classmethod
    def latest(cls, runs_dir: str | Path) -> RunStore | None:
        base = Path(runs_dir)
        runs = sorted(base.glob("run-*/run.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            return None
        return cls.open(runs_dir, runs[-1].parent.name)

    def _write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run.json").write_text(json.dumps(asdict(self.manifest), indent=2))

    # -- cache-bust ids -------------------------------------------------------
    def test_uuid(self, model: str, session_id: str) -> str:
        """Per-run, per-(model, conversation) cache-bust id (resume-stable)."""
        key = f"{model}::{session_id}"
        with self._lock:
            tid = self.manifest.session_test_uuids.get(key)
            if tid is None:
                tid = str(uuid.uuid5(uuid.UUID(self.manifest.uuid), key))
                self.manifest.session_test_uuids[key] = tid
                self._write_manifest()
            return tid

    # -- per-(model, kind) caches ---------------------------------------------
    def _cache(self, model: str, kind: str) -> _KindCache:
        ck = (model, kind)
        with self._lock:
            if ck not in self._caches:
                mdir = self.root / "models" / _safe(model)
                if kind == BASELINE:
                    path = mdir / "baseline.json"
                else:
                    path = mdir / "strategies" / f"{kind}.json"
                self._caches[ck] = _KindCache(path)
            return self._caches[ck]

    def get(
        self, model: str, kind: str, session_id: str, n_points: int
    ) -> list[Measurement] | None:
        with self._lock:
            return self._cache(model, kind).get(session_id, n_points)

    def put(self, model: str, kind: str, session_id: str, measurements: list[Measurement]) -> None:
        with self._lock:
            self._cache(model, kind).put(session_id, model, measurements)

    def save(self) -> None:
        with self._lock:
            for c in self._caches.values():
                c.save()

    # -- recompute (cost is always derived from stored usage) -----------------
    def paired_rows(self, model: str, strategy: str) -> list[PairedRow]:
        base = self._cache(model, BASELINE)
        strat = self._cache(model, strategy)
        rows: list[PairedRow] = []
        for sid in strat.session_ids():
            b_entry = base._data.get(sid)
            s_entry = strat._data.get(sid)
            if not b_entry or not s_entry:
                continue
            b_ms = _load(sid, b_entry["points"])
            s_ms = _load(sid, s_entry["points"])
            for b, s in zip(b_ms, s_ms, strict=False):
                rows.append(
                    PairedRow(
                        session_id=sid, index=b.index, chain_tokens=b.usage.total_input,
                        model=model, base=b.usage, strat=s.usage, ok=b.ok and s.ok,
                    )
                )
        return rows

    def flat(self, model: str, kind: str) -> list[tuple[Measurement, str]]:
        """All stored (measurement, model) for a (model, kind), in conversation order."""
        c = self._cache(model, kind)
        out: list[tuple[Measurement, str]] = []
        for sid in c.session_ids():
            for mm in _load(sid, c._data[sid]["points"]):
                out.append((mm, model))
        return out

    def buckets_for(self, model: str, strategies: list[str]) -> dict[str, list[BucketStats]]:
        edges = self.manifest.edges or None
        return {name: summarize(self.paired_rows(model, name), edges) for name in strategies}

    def buckets(self) -> dict[str, dict[str, list[BucketStats]]]:
        """{model: {strategy: [BucketStats]}} for every model in the manifest."""
        return {
            mdl: self.buckets_for(mdl, self.manifest.strategies)
            for mdl in self.manifest.models
        }

    def write_report(self, report: dict) -> None:
        (self.root / "report.json").write_text(json.dumps(report, indent=2))

    def log_error(self, message: str) -> None:
        """Append one error to ``run-<uuid>/errors.log`` immediately.

        Written and flushed the instant an error happens (not at run end), so a
        run that crashes or is interrupted mid-flight still leaves its errors.
        """
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')}  {message}\n"
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "errors.log").open("a", encoding="utf-8") as f:
                f.write(line)
