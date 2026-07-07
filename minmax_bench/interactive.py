"""Guided, animated setup wizard for ``minmax-bench run``.

Walks the user through: a dependency preflight, dataset selection (fetching with a
live progress spinner if needed), a look at the selected conversations and their
lengths, model and strategy multi-selects, and bucket edges — then hands a ready
manifest to the runner. Falls back cleanly to flags in non-interactive contexts.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .catalog import CATALOG, DEFAULT_MODELS
from .chain import LocalCounter
from .config import get_settings
from .data import load_dataset
from .preflight import disabled_strategies, preflight
from .provision import _reachable
from .report import DEFAULT_EDGES
from .strategies import default_selected, matrix_names, selectable, tool_for


@dataclass
class WizardResult:
    dataset: str
    models: list[str]
    strategies: list[str]
    edges: list[int]
    session_ids: list[str] | None
    token_limit: int | None
    setup: list[str] = field(default_factory=list)  # tools to install/start
    count_mode: str = "local"


def _banner(console: Console) -> None:
    console.print(Panel.fit(
        Text.assemble(
            ("minmax-bench\n", "bold cyan"),
            ("estimated token & cost savings for context-reduction proxies", "dim"),
        ),
        border_style="cyan",
    ))


def _multiselect(
    console: Console, title: str, options: list[tuple[str, str, bool, bool]]
) -> list[str]:
    """options: (key, label, enabled, default). Returns chosen keys.

    Enter comma-separated numbers, ``a`` for all enabled, or blank for defaults.
    """
    t = Table(title=f"[bold]{title}", show_header=False, box=None)
    for i, (_key, label, enabled, default) in enumerate(options, 1):
        mark = "[green]•[/]" if default else " "
        row = f"[dim]{i:>2}[/] {mark} {label}"
        if not enabled:
            row = f"[dim]{i:>2}    {label}  (unavailable)[/]"
        t.add_row(row)
    console.print(t)
    defaults = [k for k, _l, en, d in options if d and en]
    raw = Prompt.ask(
        "[cyan]select[/] (numbers, 'a'=all, blank=default)",
        default=",".join(str(i) for i, o in enumerate(options, 1) if o[3] and o[2]),
        console=console,
    ).strip().lower()
    if not raw:
        return defaults
    if raw == "a":
        return [k for k, _l, en, _d in options if en]
    chosen: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok.isdigit():
            continue
        idx = int(tok) - 1
        if 0 <= idx < len(options) and options[idx][2]:
            chosen.append(options[idx][0])
    return chosen or defaults


def _load_with_progress(console: Console, spec: str):
    """Load a dataset, showing a live spinner while a fetch/parse runs."""
    box: dict = {}

    def worker():
        try:
            box["sessions"] = load_dataset(spec)
        except Exception as e:  # surfaced to the caller below
            box["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), TimeElapsedColumn(),
        console=console, transient=True,
    ) as prog:
        prog.add_task(f"loading {spec} (fetches from HuggingFace on first use)…", total=None)
        th.start()
        while th.is_alive():
            prog.refresh()
            th.join(timeout=0.1)
    if "error" in box:
        raise box["error"]
    return box["sessions"]


def _htok(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:.0f}"


def _sized(console: Console, sessions, encoding: str) -> list[tuple]:
    """(session, turns, tokens) for each conversation, largest first (spinner)."""
    counter = LocalCounter(encoding)
    box: dict = {}

    def worker():
        rows = [(s, len(s.messages), counter.count(s, s.messages)) for s in sessions]
        rows.sort(key=lambda r: r[2], reverse=True)
        box["rows"] = rows

    th = threading.Thread(target=worker, daemon=True)
    with Progress(SpinnerColumn(), TextColumn("[cyan]sizing conversations…"),
                  console=console, transient=True) as prog:
        prog.add_task("", total=None)
        th.start()
        while th.is_alive():
            prog.refresh()
            th.join(timeout=0.1)
    return box["rows"]


def _show_conversations(console: Console, rows: list[tuple], kept: set[str] | None) -> None:
    t = Table(title=f"[bold]conversations ({len(rows)}) — largest first")
    t.add_column("#", justify="right", style="dim")
    t.add_column("id")
    t.add_column("turns", justify="right")
    t.add_column("~tokens", justify="right")
    t.add_column("", style="dim")
    for i, (s, turns, tokens) in enumerate(rows[:40], 1):
        mark = "" if kept is None or s.id in kept else "[dim]filtered[/]"
        style = None if kept is None or s.id in kept else "dim"
        t.add_row(str(i), s.id, str(turns), _htok(tokens), mark, style=style)
    if len(rows) > 40:
        t.add_row("…", f"(+{len(rows) - 40} more)", "", "", "")
    console.print(t)


def _parse_range(raw: str) -> tuple[int | None, int | None] | None:
    """'50-300' -> (50,300); '100-' -> (100,None); '-80' -> (None,80); '' -> None."""
    raw = raw.strip()
    if not raw:
        return None
    a, _, b = raw.partition("-")
    lo = int(a) if a.strip().isdigit() else None
    hi = int(b) if b.strip().isdigit() else (lo if "-" not in raw else None)
    return (lo, hi)


def _parse_tokens(raw: str) -> int | None:
    raw = raw.strip().lower().replace(",", "")
    if not raw:
        return None
    mult = 1
    if raw.endswith("k"):
        mult, raw = 1_000, raw[:-1]
    elif raw.endswith("m"):
        mult, raw = 1_000_000, raw[:-1]
    try:
        return int(float(raw) * mult)
    except ValueError:
        return None


def _dense_ready() -> bool:
    s = get_settings()
    if not shutil.which("dense"):
        return False
    try:
        from .dense import load_profile
        p = load_profile(s.condense_profile or None)
        return bool(getattr(p, "auth_token", None) and getattr(p, "user_id", None))
    except Exception:
        return False


def _preflight_step(console: Console, dataset: str) -> dict[str, str]:
    checks = preflight(matrix_names(), DEFAULT_MODELS, dataset)
    t = Table(title="[bold]dependency preflight")
    t.add_column("dependency")
    t.add_column("status")
    t.add_column("detail", style="dim")
    for c in checks:
        badge = "[green]ok[/]" if c.ok else "[red]missing[/]"
        t.add_row(c.name, badge, c.detail)
    console.print(t)
    if not _dense_ready():
        console.print(
            "[yellow]•[/] condense needs the [bold]dense[/] CLI logged in "
            "(run can install it + `dense login`)"
        )
    disabled = disabled_strategies(checks)
    # Only truly-unfixable blockers (missing provider key) gate selection; a missing
    # local tool (headroom/dense) is handled by the setup step below.
    hard = {st: why for st, why in disabled.items() if "API_KEY" in why}
    for st, why in hard.items():
        console.print(f"[yellow]will skip[/] [bold]{st}[/] — {why}")
    if not Confirm.ask("[cyan]continue?[/]", default=True, console=console):
        raise KeyboardInterrupt
    return hard


def _tool_present(tool: str) -> bool:
    s = get_settings()
    if tool == "dense":
        return _dense_ready()
    if tool == "headroom":
        return _reachable(s.headroom_base_url)
    return True


def _needs_setup(strategies: list[str]) -> list[str]:
    """Local tools that selected strategies need but that aren't ready yet."""
    seen: list[str] = []
    for st in strategies:
        tool = tool_for(st)
        if tool and tool not in seen and not _tool_present(tool):
            seen.append(tool)
    return seen


def _setup_step(console: Console, strategies: list[str]) -> list[str]:
    needs = _needs_setup(strategies)
    if not needs:
        return []
    labels = {
        "dense": "dense — install the CLI + `dense login` (condense creds)",
        "headroom": "headroom — install if needed, then start the proxy",
    }
    console.print("[bold]these selected strategies need local tools set up:[/]")
    opts = [(t, labels[t], True, True) for t in needs]
    return _multiselect(console, "set up locally", opts)


def _pick_local_sessions(console: Console) -> str:
    """Interactive multi-pick over ~/.claude/projects -> a claude-code:<paths> spec.

    This is the COST BACKTEST entry point: replay your own sessions turn-by-turn
    through the selected strategies and see what they would have cost. For the
    quality counterfactual (would it have made the same decisions), see
    `minmax-bench counterfactual`.
    """
    from datetime import datetime

    from .counterfactual import scan_sessions

    found = scan_sessions(limit=20)
    if not found:
        console.print("[yellow]no local Claude Code sessions found — enter a path instead[/]")
        return Prompt.ask("[cyan]dataset[/]", default="sample", console=console).strip()
    opts = []
    for s in found:
        when = datetime.fromtimestamp(s.mtime).strftime("%m-%d %H:%M")
        proj = ("…" + s.project[-30:]) if len(s.project) > 31 else s.project
        label = f"{when}  {proj}  [dim]{s.prompt[:40] or '(no text prompt)'}[/]"
        opts.append((str(s.path), label, True, False))
    chosen = _multiselect(console, "your Claude Code sessions (backtest what they'd cost)", opts)
    if not chosen:
        chosen = [opts[0][0]]
    return "claude-code:" + ",".join(chosen)


def run_wizard(console: Console) -> WizardResult:
    _banner(console)

    # 1. dataset -> size every conversation, show them, filter by a turn range.
    dataset = Prompt.ask(
        "[cyan]dataset[/] (sample | swe-chat:N | claude-code = backtest YOUR sessions "
        "| claude-code:/path/*.jsonl)",
        default="sample", console=console,
    ).strip()
    if dataset == "claude-code":  # bare: pick from ~/.claude/projects interactively
        dataset = _pick_local_sessions(console)
    sessions = _load_with_progress(console, dataset)
    rows = _sized(console, sessions, "o200k_base")
    _show_conversations(console, rows, kept=None)

    rng = _parse_range(Prompt.ask(
        "[cyan]keep conversations by turns[/] (e.g. 50-300, 100- for ≥100, blank=all)",
        default="", console=console,
    ))
    if rng is None:
        kept_rows = rows
    else:
        lo, hi = rng
        kept_rows = [
            r for r in rows
            if (lo is None or r[1] >= lo) and (hi is None or r[1] <= hi)
        ]
    if not kept_rows:
        console.print("[yellow]range kept nothing — using all conversations[/]")
        kept_rows = rows
    session_ids = [s.id for s, _t, _tok in kept_rows]
    _show_conversations(console, rows, kept=set(session_ids))
    console.print(f"[green]{len(session_ids)}[/] conversations selected")

    token_limit = _parse_tokens(Prompt.ask(
        "[cyan]token budget per conversation?[/] (truncate each chain at e.g. 200k; blank=full)",
        default="", console=console,
    ))

    # 2. dependency preflight (+ dense hint); returns only unfixable blockers.
    hard = _preflight_step(console, dataset)

    # 3. strategies phase — walk the matrix; check what's present, prompt to select.
    default_names = set(default_selected())
    strat_opts = []
    for entry in selectable():
        name = entry.name
        label = f"{name} [dim]— {entry.description}[/]"
        tool = entry.tool
        if tool and not _tool_present(tool):
            label = f"{name} [dim](needs local setup) — {entry.description}[/]"
        strat_opts.append(
            (name, label, name not in hard, name in default_names and name not in hard)
        )
    strategies = _multiselect(console, "strategies to benchmark", strat_opts)

    # 4. ask whether to set up the local tools those strategies need.
    setup = _setup_step(console, strategies)

    # 5. models (haiku is the cheap default verifier).
    model_opts = [(c.model, f"{c.label} [dim]({c.model})[/]", True, c.default) for c in CATALOG]
    models = _multiselect(console, "models", model_opts)

    # 6. buckets.
    edge_raw = Prompt.ask(
        "[cyan]bucket edges[/] (comma tokens)",
        default=",".join(str(e) for e in DEFAULT_EDGES), console=console,
    ).strip()
    edges = [int(x) for x in edge_raw.split(",") if x.strip().isdigit()] or list(DEFAULT_EDGES)

    budget = f"{_htok(token_limit)} tokens" if token_limit else "full"
    console.print(Panel.fit(
        f"[bold]dataset[/] {dataset}   [bold]conversations[/] {len(session_ids)}   "
        f"[bold]budget[/] {budget}\n"
        f"[bold]models[/] {', '.join(models)}\n"
        f"[bold]strategies[/] {', '.join(strategies)}\n"
        f"[bold]set up[/] {', '.join(setup) or '(none)'}\n"
        f"[bold]buckets[/] {edges}",
        title="ready", border_style="green",
    ))
    if not Confirm.ask("[cyan]run it?[/]", default=True, console=console):
        raise KeyboardInterrupt
    return WizardResult(
        dataset=dataset, models=models, strategies=strategies, edges=edges,
        session_ids=session_ids, token_limit=token_limit, setup=setup,
    )
