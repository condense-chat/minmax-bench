"""Where quality-bench runs are saved.

The cost bench auto-mints a unique ``run-<uuid>/`` under ``settings.runs_dir`` so a
re-run never overwrites the last one. The quality bench used to default to a STATIC
``results/jobs/run`` that clobbered on every re-run — this centralizes the same
auto-mint behaviour under ``settings.quality_runs_dir`` (override via
``QUALITY_RUNS_DIR`` in .env or the setup wizard's advanced step).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings


def quality_runs_root() -> str:
    """The configured root for quality-bench run directories (default ``runs/quality``)."""
    return get_settings().quality_runs_dir


def new_run_dir(kind: str, label: str = "run") -> str:
    """A fresh, unique run directory under the quality runs root — auto-minted like the
    cost bench so runs never clobber. ``kind`` is ``full`` | ``incremental``; ``label`` a
    short slug (a session stem or dataset name). Callers may still pass an explicit --out."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in (label or "run"))[:24]
    uid = uuid.uuid4().hex[:6]  # collision guard for two runs within the same second
    return str(Path(quality_runs_root()) / kind / f"{safe}-{stamp}-{uid}")


def default_run_roots() -> tuple[str, ...]:
    """Roots that discovery / the report / the view picker scan by default — the configured
    quality runs root first, then the legacy ``results`` tree and the bundled sample."""
    root = quality_runs_root()
    roots = [root, "results", "runs/quality-sample"]
    # de-dup while preserving order (quality_runs_dir could equal a legacy entry)
    return tuple(dict.fromkeys(roots))
