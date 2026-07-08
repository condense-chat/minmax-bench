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


# ============================================================================
# Quality / trajectory-preservation bench wizard (minmax-bench quality run)
# ============================================================================


# ============================================================================
# Quality / trajectory-preservation bench wizard (minmax-bench quality run)
# ============================================================================
QUALITY_MODELS = [
    ("claude-sonnet-4-6", "Sonnet 4.6 — the validated default", True),
    ("claude-haiku-4-5", "Haiku 4.5 — cheapest", False),
    ("claude-sonnet-5", "Sonnet 5", False),
    ("claude-opus-4-8", "Opus 4.8 — most capable", False),
]


@dataclass
class QualityWizardResult:
    mode: str                       # "full" | "incremental"
    arms: str
    model: str | None
    out: str
    # full
    tasks: str = "5"
    k: int = 4
    budget_usd: float = 5.0
    milestones: bool = True
    # incremental
    source: str = ""                # "own" | "file" | "swechat"
    session: str | None = None
    swechat: str | None = None
    conv: int = 0
    task: str = "session"
    every: int = 1
    limit: int = 0


def _select_one(console: Console, title: str, options: list[tuple[str, str, bool]],
                default_idx: int = 1) -> str:
    """Single-select: options are (key, label, enabled). Returns the chosen key."""
    t = Table(title=f"[bold]{title}", show_header=False, box=None)
    for i, (_k, label, enabled) in enumerate(options, 1):
        t.add_row(f"[dim]{i:>2}[/] {label}" if enabled
                  else f"[dim]{i:>2}    {label}  (coming soon)[/]")
    console.print(t)
    while True:
        raw = Prompt.ask("[cyan]select[/] (number)", default=str(default_idx),
                         console=console).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options) and options[int(raw) - 1][2]:
            return options[int(raw) - 1][0]
        console.print("[yellow]pick an available option[/]")


def _pick_model(console: Console) -> str:
    mt = Table(title="[bold]model", show_header=False, box=None)
    for i, (mid, label, _d) in enumerate(QUALITY_MODELS, 1):
        mt.add_row(f"[dim]{i:>2}[/] {label} [dim]({mid})[/]")
    console.print(mt)
    raw = Prompt.ask("[cyan]model[/] (number, or an id)", default="1", console=console).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(QUALITY_MODELS):
        return QUALITY_MODELS[int(raw) - 1][0]
    return raw or QUALITY_MODELS[0][0]


def _quality_preflight(console: Console, arms: list[str], *, need_docker: bool) -> None:
    """Auth + arm-specific keys (+ Docker/harbor for full runs). Blocks only on
    the truly-fatal; warns otherwise."""
    import os

    from .quality.engine import auth_mode, load_env
    env = {**load_env(), **os.environ}
    rows = []
    if need_docker:
        rows += [("Docker", bool(shutil.which("docker")), "runs the agent containers"),
                 ("harbor CLI", bool(shutil.which("harbor")), "uv tool install harbor")]
    rows.append(("Anthropic auth", bool(auth_mode(env)),
                 auth_mode(env) or "ANTHROPIC_API_KEY or Claude Code login"))
    if "condense" in arms:
        rows.append(("CONDENSE_API_KEY", bool(env.get("CONDENSE_API_KEY")), "the condense arm"))
    if any(a.startswith("headroom") for a in arms):
        rows.append(("headroom / uvx", bool(shutil.which("headroom") or shutil.which("uvx")),
                     "the headroom proxy"))
    t = Table(title="[bold]dependency preflight")
    t.add_column("dependency")
    t.add_column("status")
    t.add_column("detail", style="dim")
    for name, ok, detail in rows:
        t.add_row(name, "[green]ok[/]" if ok else "[red]missing[/]", str(detail))
    console.print(t)
    fatal = [n for n, ok, _ in rows if not ok and n in ("Docker", "harbor CLI", "Anthropic auth")]
    if fatal:
        console.print(f"[red]can't run without: {', '.join(fatal)}[/]")
        if not Confirm.ask("[cyan]continue anyway?[/]", default=False, console=console):
            raise KeyboardInterrupt


def run_quality_wizard(console: Console) -> QualityWizardResult:
    """Guided setup for a quality run — pick full vs incremental trajectories and a
    source, then walk the knobs. The quality analog of the cost bench's run_wizard."""
    console.print(Panel.fit(
        Text.assemble(("minmax-bench · quality\n", "bold magenta"),
                      ("does compaction preserve the agent's trajectory?", "dim")),
        border_style="magenta"))
    mode = _select_one(console, "trajectory type", [
        ("full", "full trajectories — run the agent end-to-end on tasks "
                 "(behavior, rework, solve, milestones)", True),
        ("incremental", "incremental — teacher-forced per-step replay of a recorded "
                        "session (paired A/B, cheap, no turn noise)", True),
    ])
    return (_full_wizard if mode == "full" else _incremental_wizard)(console)


def _full_wizard(console: Console) -> QualityWizardResult:
    from .quality.engine import DEFAULT_TASKS, resolve_tasks
    _select_one(console, "dataset", [
        ("terminal-bench", "terminal-bench-2-1 — curated coding tasks WITH verifiers", True),
        ("swe-chat", "SWE-chat — real recorded sessions (no verifiers; incremental only)", False),
        ("custom", "another Harbor dataset", False),
    ])
    arms = _multiselect(console, "arms to compare (vanilla baseline always included)", [
        ("condense", "condense — compaction proxy", True, True),
        ("headroom-ccr", "headroom-ccr — token mode + retrieve (full product)", True, True),
        ("headroom", "headroom — cache-mode proxy", True, False),
        ("headroom-kompress", "headroom-kompress — token mode, no retrieval", True, False),
    ])
    t = Table(title="[bold]recommended tasks (short → long)", show_header=False, box=None)
    for i, name in enumerate(DEFAULT_TASKS, 1):
        t.add_row(f"[dim]{i:>2}[/] {name}{'   [green]← default 5[/]' if i == 5 else ''}")
    console.print(t)
    tasks = Prompt.ask("[cyan]tasks[/] (a number N = first N, random:N, comma names, blank = 5)",
                       default="5", console=console).strip() or "5"
    model = _pick_model(console)
    k = int(Prompt.ask("[cyan]trials per arm[/] (k — ≥2 for a verdict; 4 recommended)",
                       default="4", console=console) or 4)
    budget = float(Prompt.ask("[cyan]per-trial $ cap[/]", default="5", console=console) or 5)
    milestones = Confirm.ask("[cyan]also run the LLM milestone judge?[/]", default=True,
                             console=console)
    out = Prompt.ask("[cyan]output dir[/]", default="results/jobs/run", console=console).strip()
    _quality_preflight(console, arms, need_docker=True)
    ntasks = len(resolve_tasks(tasks, "terminal-bench"))
    kv, trials = k + 1, len(resolve_tasks(tasks, "terminal-bench")) * ((k + 1) + k * len(arms))
    console.print(Panel.fit(
        f"[bold]full trajectories[/]   [bold]model[/] {model}   [bold]tasks[/] {ntasks}   "
        f"[bold]k[/] {k} (vanilla {kv})\n[bold]arms[/] vanilla + {', '.join(arms)}\n"
        f"[bold]milestones[/] {'yes' if milestones else 'no'}   [bold]out[/] {out}\n"
        f"[bold]{trials} trials[/], cost ceiling ~[bold]${trials * budget:.0f}[/] "
        f"(${budget:g}/trial cap)", title="ready", border_style="green"))
    if not Confirm.ask("[cyan]run it?[/]", default=False, console=console):
        raise KeyboardInterrupt
    return QualityWizardResult(mode="full", arms=",".join(arms), tasks=tasks, model=model,
                               k=k, budget_usd=budget, milestones=milestones, out=out)


def _incremental_wizard(console: Console) -> QualityWizardResult:
    from pathlib import Path

    src = _select_one(console, "source session", [
        ("own", "your own Claude Code sessions (~/.claude/projects) — pick from a list", True),
        ("file", "a session .jsonl path", True),
        ("swechat", "SWE-chat cached conversations (coming soon here)", False),
    ])
    session = swechat = None
    conv = 0
    task = "session"
    if src == "own":
        from .counterfactual import pick_session
        p = pick_session(console)  # the peak-ctx picker
        session = str(p)
        task = p.stem[:12]
    else:  # file
        raw = Prompt.ask("[cyan]session .jsonl path[/]", console=console).strip()
        p = Path(raw).expanduser()
        if not p.is_file():
            console.print(f"[red]not a file:[/] {p}")
            raise KeyboardInterrupt
        session = str(p)
        task = p.stem[:12]
    # CCR can't engage without tool execution — incremental arms are proxy-only
    arms = _multiselect(console, "arms (vanilla control always included; CCR needs full runs)", [
        ("condense", "condense — compaction proxy", True, True),
        ("headroom", "headroom — proxy only (cache/token via --headroom-mode)", True, False),
    ])
    model = _pick_model(console)
    every = int(Prompt.ask("[cyan]sample every Nth decision point[/]", default="1",
                           console=console) or 1)
    limit = int(Prompt.ask("[cyan]max decision points[/] (0 = all)", default="0",
                           console=console) or 0)
    budget = float(Prompt.ask("[cyan]per-arm $ cap[/]", default="2", console=console) or 2)
    out = Prompt.ask("[cyan]output dir[/]", default="results/jobs/run", console=console).strip()
    _quality_preflight(console, arms, need_docker=False)
    console.print(Panel.fit(
        f"[bold]incremental trajectory[/]   [bold]model[/] {model}\n"
        f"[bold]session[/] {Path(session).name}\n"
        f"[bold]arms[/] control + {', '.join(arms)}   [bold]every[/] {every}   "
        f"[bold]limit[/] {limit or 'all'}\n"
        f"[bold]per-arm cap[/] ${budget:g}   [bold]out[/] {out}",
        title="ready", border_style="green"))
    if not Confirm.ask("[cyan]replay it?[/]", default=False, console=console):
        raise KeyboardInterrupt
    return QualityWizardResult(mode="incremental", source=src, session=session, swechat=swechat,
                               conv=conv, task=task, arms=",".join(arms), model=model,
                               every=every, limit=limit, budget_usd=budget, out=out)
