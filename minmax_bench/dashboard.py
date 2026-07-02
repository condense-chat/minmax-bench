"""Live, animated terminal dashboard for a benchmark run.

Each track (one model · strategy) is a stacked block, full terminal width:

* a bold **header** — ``model · strategy`` plus live ``done/total convos · N running``
  so the parallel batch is visible;
* a **context bar** — the conversation growing turn by turn, one hue per turn so the
  turn count reads at a glance, with the running ``$N / Nt`` at the right;
* a **cost-breakdown bar** — the spend split by tier
  (cache-read / input / cache-write / output).

Many conversations feed one track in parallel; each turn measured anywhere on the
track advances its bars. The same renderer powers a post-hoc ``replay`` of a stored
run, stepping through the recorded per-turn usage with a small delay.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from .executors.base import Measurement
from .pricing import cost_breakdown

# Per-turn context-bar colors (cycled): a cool blue/cyan family so the context bar
# never shares a hue with the warm cost tiers below — each turn gets its own hue so
# the number of turns is legible at a glance.
_TURN_COLORS = [
    "cyan", "blue", "bright_cyan", "bright_blue",
    "deep_sky_blue1", "turquoise2", "dodger_blue1", "steel_blue1",
]
_CTX_EMPTY = "grey19"
# Cost-tier colors for the breakdown bar + legend (warm: green/yellow/orange/magenta).
_TIER_COLORS = {
    "cache_read": "green",
    "input": "yellow",
    "cache_write": "dark_orange3",
    "output": "magenta",
}
_TIERS = ["cache_read", "input", "cache_write", "output"]
_PREFIX_W = 12  # left gutter that holds "context" / "cost split"
_MIN_BAR = 20
_BLOCK = "█"


def _bar_width(console_width: int) -> int:
    """Bar cells after the left gutter — fills the rest of the screen width."""
    return max(_MIN_BAR, console_width - _PREFIX_W - 2)


def _htok(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


@dataclass
class Track:
    label: str
    total_points: int = 0
    done: int = 0
    tokens: int = 0                       # cumulative prompt+output tokens
    cost: float = 0.0
    tier_cost: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in _TIERS})
    seg: list[int] = field(default_factory=list)  # per-turn prompt tokens (context bar)
    errors: int = 0
    convos_total: int = 0                 # conversations feeding this track
    convos_done: int = 0
    active: int = 0                       # conversations in flight right now

    def add(self, m: Measurement, model: str) -> None:
        self.done += 1
        if not m.ok:
            self.errors += 1
            self.seg.append(0)
            return
        self.tokens += m.usage.total_input + m.usage.output_tokens
        self.seg.append(m.usage.total_input)
        b = cost_breakdown(model, m.usage)
        for t in _TIERS:
            self.tier_cost[t] += b[t]
        self.cost += b["total"]


def _header(tr: Track) -> Text:
    """Bold ``model · strategy`` line + live convo progress / in-flight count."""
    t = Text(tr.label, style="bold")
    if tr.convos_total:
        t.append(f"   {tr.convos_done}/{tr.convos_total} convos", style="dim")
        if tr.active:
            t.append(f" · {tr.active} running", style="green")
    return t


def _context_bar(tr: Track, width: int) -> Text:
    """``context`` gutter + turn-segmented bar (one hue per turn) with the running
    ``$cost / Ntok`` at the right. Total line is ``_PREFIX_W + width`` columns."""
    t = Text()
    t.append(f"{'  context':<{_PREFIX_W}}", style="dim")
    right = f" ${tr.cost:,.4f} / {_htok(tr.tokens)}t"
    err = f"  [{tr.errors} err]" if tr.errors else ""
    barw = max(1, width - len(right) - len(err))
    total = max(tr.total_points, 1)
    filled = round(barw * tr.done / total)
    for cell in range(barw):
        if cell < filled:
            turn = int(cell / barw * total)  # which turn this cell belongs to
            t.append(_BLOCK, style=_TURN_COLORS[turn % len(_TURN_COLORS)])
        else:
            t.append(_BLOCK, style=_CTX_EMPTY)
    t.append(right, style="bold white")
    if err:
        t.append(err, style="red")
    return t


def _cost_bar(tr: Track, width: int) -> Text:
    """``cost split`` gutter + bar split by tier (cache_read/input/cache_write/output)."""
    t = Text()
    t.append(f"{'  cost split':<{_PREFIX_W}}", style="dim")
    total = tr.cost
    if total <= 0:
        t.append(_BLOCK * width, style=_CTX_EMPTY)
        return t
    used = 0
    for i, tier in enumerate(_TIERS):
        cells = round(width * tr.tier_cost[tier] / total)
        if i == len(_TIERS) - 1:
            cells = width - used  # last tier absorbs rounding
        cells = max(0, cells)
        t.append(_BLOCK * cells, style=_TIER_COLORS[tier])
        used += cells
    return t


def _legend() -> Text:
    cost = Text("  cost split  ", style="dim")
    for tier in _TIERS:
        cost.append(f"{_BLOCK} {tier}  ", style=_TIER_COLORS[tier])
    cost.append("   context: one hue per turn", style="dim")
    return cost


class Dashboard:
    """Owns a :class:`rich.live.Live`; call :meth:`on_point` as turns are measured."""

    def __init__(
        self,
        order: list[str],
        *,
        title: str = "",
        pace: float = 0.0,
        console: Console | None = None,
    ):
        self.console = console or Console()
        self.title = title
        self.pace = pace  # per-point sleep so a local (instant) phase still animates
        self.tracks: dict[str, Track] = {}
        self.order = order
        self._live: Live | None = None
        self.enabled = self.console.is_terminal
        self._lock = threading.RLock()  # on_point fires from parallel worker threads

    def set_phase(self, title: str) -> None:
        with self._lock:
            self.title = title
            if self._live is not None:
                self._live.update(self.render())

    def ensure(self, kind: str, label: str, total_points: int, convos: int = 0) -> None:
        with self._lock:
            tr = self.tracks.get(kind)
            if tr is None:
                tr = Track(label=label, total_points=total_points)
                self.tracks[kind] = tr
                if kind not in self.order:
                    self.order.append(kind)
            else:
                tr.total_points += total_points
            tr.convos_total += convos

    def _touch(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def begin(self, kind: str) -> None:
        """A conversation started running on this track (for the in-flight count)."""
        with self._lock:
            tr = self.tracks.get(kind)
            if tr is not None:
                tr.active += 1
                self._touch()

    def finish(self, kind: str) -> None:
        """A conversation finished on this track."""
        with self._lock:
            tr = self.tracks.get(kind)
            if tr is not None:
                tr.active = max(0, tr.active - 1)
                tr.convos_done += 1
                self._touch()

    def render(self) -> Group:
        parts: list = []
        if self.title:
            parts.append(Text(self.title, style="bold"))
        width = _bar_width(self.console.width)
        for kind in self.order:
            tr = self.tracks.get(kind)
            if tr is None:
                continue
            parts.append(_header(tr))
            parts.append(_context_bar(tr, width))
            parts.append(_cost_bar(tr, width))
            parts.append(Text(""))
        parts.append(_legend())
        return Group(*parts)

    def on_point(self, kind: str, m: Measurement, model: str) -> None:
        with self._lock:
            tr = self.tracks.get(kind)
            if tr is None:
                return
            tr.add(m, model)
            if self._live is not None:
                self._live.update(self.render())
                pace = self.pace
        if self._live is not None and self.pace:
            time.sleep(pace)

    def __enter__(self) -> Dashboard:
        if self.enabled:
            self._live = Live(self.render(), console=self.console, refresh_per_second=12)
            self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.update(self.render())
            self._live.__exit__(*exc)
            self._live = None


def replay(
    per_kind_points: dict[str, list[tuple[Measurement, str]]],
    order: list[str],
    labels: dict[str, str],
    *,
    title: str = "",
    fps: float = 30.0,
    console: Console | None = None,
) -> None:
    """Animate a finished run from its stored per-turn measurements."""
    dash = Dashboard(order, title=title, console=console)
    for kind in order:
        dash.ensure(kind, labels.get(kind, kind), len(per_kind_points.get(kind, [])))
    # interleave one turn per kind per frame so all bars grow together
    max_len = max((len(v) for v in per_kind_points.values()), default=0)
    delay = 1.0 / fps if fps > 0 else 0.0
    with dash:
        for i in range(max_len):
            for kind in order:
                pts = per_kind_points.get(kind, [])
                if i < len(pts):
                    m, model = pts[i]
                    dash.on_point(kind, m, model)
            if delay:
                time.sleep(delay)
