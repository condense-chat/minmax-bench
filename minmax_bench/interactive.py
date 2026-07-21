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
from pathlib import Path

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
from .tokens import parse_token_count


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


_parse_tokens = parse_token_count  # one parser for every human-entered token amount


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

    This is the COST BACKTEST entry point: run your own sessions turn-by-turn
    through the selected strategies and see what they would have cost. For quality /
    trajectory preservation (would it have made the same decisions), see the
    incremental mode: `minmax-bench quality incremental`.
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
    auth: str = "auto"              # auto | api-key | subscription (choice when both present)
    # full
    tasks: str = "5"
    k: int = 4
    budget_usd: float = 5.0
    milestones: bool = True
    force: bool = False              # True = full re-run (redo completed cells); False = resume
    retries: int = 0                 # extra re-attempts for a crashed/timed-out cell
    # incremental
    source: str = ""                # "own" | "file" | "swechat"
    session: str | None = None
    swechat: str | None = None
    conv: int = 0
    task: str = "session"
    limit: int = 0
    judge: str = "off"
    capture: bool = False
    ctx_gate: int = 50_000     # 0 = deliberately replay a below-gate session
    independent_budgets: bool = False  # True = each arm to own budget; False = cap to control
    resume: bool = True        # skip arms already finished (.done sentinel) on re-run to same out


def _ask_int(console: Console, prompt: str, default: int, lo: int = 0) -> int:
    """Integer prompt that re-asks on junk instead of crashing the wizard."""
    while True:
        raw = Prompt.ask(prompt, default=str(default), console=console).strip()
        try:
            v = int(raw or default)
        except ValueError:
            console.print(f"[yellow]enter a whole number (≥{lo})[/]")
            continue
        if v < lo:
            console.print(f"[yellow]enter a number ≥{lo}[/]")
            continue
        return v


def _ask_float(console: Console, prompt: str, default: float, lo: float = 0.0) -> float:
    """Float prompt that re-asks on junk instead of crashing the wizard."""
    while True:
        raw = Prompt.ask(prompt, default=f"{default:g}", console=console).strip()
        try:
            v = float(raw or default)
        except ValueError:
            console.print(f"[yellow]enter a number (≥{lo:g})[/]")
            continue
        if v < lo:
            console.print(f"[yellow]enter a number ≥{lo:g}[/]")
            continue
        return v


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
    """Dependency preflight, the rich preview of the same checks the full driver
    gates on. Full runs reuse generate._preflight_full (Docker/harbor/auth/keys/port
    — single source of truth); incremental runs need only auth + arm keys."""
    import os

    from .quality.engine import auth_mode, load_env
    env = {**load_env(), **os.environ}
    if need_docker:
        from .quality.generate import _preflight_full
        rows = [(n, ok, d) for n, ok, d, _fatal in _preflight_full(arms, env)]
        fatal_names = {n for n, ok, _d, fatal in _preflight_full(arms, env) if fatal}
    else:
        rows = [("Anthropic auth", bool(auth_mode(env)),
                 auth_mode(env) or "ANTHROPIC_API_KEY or Claude Code login")]
        if "condense" in arms:
            rows.append(("CONDENSE_API_KEY", bool(env.get("CONDENSE_API_KEY")), "the condense arm"))
        if any(a.startswith("headroom") for a in arms):
            rows.append(("headroom / uvx", bool(shutil.which("headroom") or shutil.which("uvx")),
                         "the headroom proxy"))
        fatal_names = {"Anthropic auth"}
    t = Table(title="[bold]dependency preflight")
    t.add_column("dependency")
    t.add_column("status")
    t.add_column("detail", style="dim")
    for name, ok, detail in rows:
        t.add_row(name, "[green]ok[/]" if ok else "[red]missing[/]", str(detail))
    console.print(t)
    fatal = [n for n, ok, _ in rows if not ok and n in fatal_names]
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
        ("incremental", "incremental — teacher-forced per-step run of a recorded "
                        "session (paired A/B, cheap, no turn noise)", True),
        ("view", "view an existing run — pick from stored results and re-render "
                 "(free, never spends)", True),
    ])
    return {"full": _full_wizard, "incremental": _incremental_wizard,
            "view": _view_wizard}[mode](console)


def _view_wizard(console: Console) -> QualityWizardResult:
    """Pick a stored quality run and hand it back for display — the wizard face of
    `quality report` / `quality runs`. Never spends."""
    from datetime import datetime

    from .quality.report import discover_runs
    infos = discover_runs()
    if not infos:
        console.print("[yellow]no stored quality runs found under results/ — "
                      "generate one first[/]")
        raise KeyboardInterrupt
    opts = []
    for i in infos[:20]:
        when = datetime.fromtimestamp(Path(i["dir"]).stat().st_mtime).strftime("%m-%d %H:%M")
        label = (f"{when}  {i['dir']}  [dim]{'+'.join(i['modes'])}"
                 f" · {i['model'] or '?'} · {len(i['tasks'])} task(s)[/]")
        opts.append((i["dir"], label, True))
    chosen = _select_one(console, "stored quality runs — newest first", opts)
    info = next(i for i in infos if i["dir"] == chosen)
    arms = [a for a in info["arms"] if a not in ("vanilla", "control")]
    return QualityWizardResult(
        mode="view", model=None, out=chosen, arms=",".join(arms) or "condense,headroom",
        tasks=",".join(info["tasks"]) or "5")


def _resolve_tasks_safe(spec: str, org: str = "terminal-bench") -> tuple[list[str] | None, str]:
    """resolve_tasks without the sys.exit: (tasks, "") or (None, why) — a bad spec or a
    missing local dataset must re-prompt inside the wizard, not kill it."""
    from .quality.engine import resolve_tasks
    try:
        return resolve_tasks(spec, org), ""
    except SystemExit as e:
        return None, str(e.code or "no tasks match")


def _full_wizard(console: Console) -> QualityWizardResult:
    from .quality.engine import task_meta
    _select_one(console, "dataset", [
        ("terminal-bench", "terminal-bench-2-1 — curated coding tasks WITH verifiers", True),
        ("swe-chat", "SWE-chat — recorded sessions (no verifiers; incremental only)", False),
        ("custom", "another Harbor dataset", False),
    ])
    arms = _multiselect(console, "arms to compare (vanilla baseline always included)", [
        ("condense", "condense — compaction proxy", True, True),
        ("headroom", "headroom — token proxy + retrieve loop (CCR)", True, True),
        ("headroom-kompress", "headroom-kompress — token compression, no retrieval (ablation)",
         True, False),
        ("vanilla-proxy", "vanilla-proxy — passthrough control (isolates the proxy-wiring "
                          "confound every proxy arm shares)", True, False),
    ])
    # Group shortcuts, biased toward sessions long enough that condense/headroom actually
    # compact. The default 5 are SHORT tasks — an agent solves them without ever crossing
    # the compaction threshold, so a compressing arm just passes through (nothing to measure).
    compacts = any(a.startswith(("condense", "headroom")) for a in arms)
    long_group = _resolve_tasks_safe("long")[0] or []
    n_long, n_hard = len(long_group), len(_resolve_tasks_safe("hard")[0] or [])
    n_all = len(_resolve_tasks_safe("all")[0] or [])
    g = Table(title="[bold]pick a group[/] — or a number N, random:N, or comma-separated names",
              show_header=False, box=None, padding=(0, 2, 0, 0))
    g.add_row("[cyan]long[/]", f"{n_long} tasks · author budget ≥30m",
              "[green]most likely to trigger compaction[/]")
    g.add_row("[cyan]hard[/]", f"{n_hard} tasks · difficulty=hard", "")
    g.add_row("[cyan]all[/]", f"{n_all} tasks", "the whole local set")
    g.add_row("[cyan]5[/]", "first 5 recommended",
              "[yellow]short — a compressing arm may just pass through[/]" if compacts
              else "quick smoke set")
    console.print(g)
    if long_group:
        # show the long group so the user sees what they'd get (length from the task timeout)
        lg = Table(title="[dim]long group:[/]", show_header=False, box=None)
        for name in long_group:
            mins = int((task_meta(name)[0] or 0) // 60)
            lg.add_row(f"[dim]{mins:>3}m[/]  {name}")
        console.print(lg)
    default_tasks = "long" if compacts and long_group else "5"
    while True:  # validate the spec now — a typo must re-prompt, not die after preflight
        tasks = Prompt.ask("[cyan]tasks[/] (group long|hard|all, a number N, random:N, "
                           "comma names)", default=default_tasks, console=console
                           ).strip() or default_tasks
        task_list, why = _resolve_tasks_safe(tasks)
        if task_list:
            break
        console.print(f"[yellow]{why}[/]")
    model = _pick_model(console)
    k = _ask_int(console, "[cyan]trials per arm[/] (k — ≥2 for a verdict; 4 recommended)",
                 4, lo=1)
    budget = _ask_float(console, "[cyan]per-trial $ cap[/]", 5.0)
    milestones = Confirm.ask("[cyan]also run the LLM milestone judge?[/]", default=True,
                             console=console)
    out = Prompt.ask("[cyan]output dir[/]", default="results/jobs/run", console=console).strip()
    auth = _auth_choice(console)  # only prompts when both an API key AND a subscription exist
    # resume vs full retry: by default an existing out dir is RESUMED — only cells missing
    # trials (crashes, timeouts, interrupts) re-run; completed cells (incl. reward-0) are kept
    # and cost nothing. "Full retry" forces every cell to re-run from scratch (--force), which
    # re-spends the whole run — only pick it to redo genuinely-completed trials.
    force = Confirm.ask(
        "[cyan]full retry?[/] (re-run ALL cells incl. completed — [red]re-spends everything[/]; "
        "default No = resume, only fill missing)", default=False, console=console)
    # auto-retry transient failures: a cell that crashes or times out (no reward.txt) is
    # re-attempted until it resolves to a verdict (reward 0 or 1) or the attempts run out.
    # A trial that ran and scored — even 0 — is a real result and is NOT retried.
    retries = 0
    if Confirm.ask("[cyan]auto-retry failed cells?[/] (re-attempt crashes/timeouts until they "
                   "resolve — genuine reward-0 fails are kept, not retried)",
                   default=False, console=console):
        retries = _ask_int(console, "  [cyan]max extra attempts per cell[/]", 2, lo=1)
    _quality_preflight(console, arms, need_docker=True)
    ntasks = len(task_list)
    kv = k + 1
    trials = ntasks * (kv + k * len(arms))
    shown = ", ".join(task_list[:6]) + (f", … +{ntasks - 6}" if ntasks > 6 else "")
    console.print(Panel.fit(
        f"[bold]full trajectories[/]   [bold]model[/] {model}   "
        f"[bold]k[/] {k} (vanilla {kv})\n[bold]tasks[/] ({ntasks}) {shown}\n"
        f"[bold]arms[/] vanilla + {', '.join(arms)}\n"
        f"[bold]milestones[/] {'yes' if milestones else 'no'}   [bold]out[/] {out}\n"
        f"[bold]mode[/] {'[red]full retry (re-run all)[/]' if force else 'resume (fill missing)'}"
        f"{f'  ·  auto-retry ×{retries}' if retries else ''}   [bold]auth[/] {auth}\n"
        f"[bold]{trials} trials[/], cost ceiling ~[bold]${trials * budget:.0f}[/] "
        f"(${budget:g}/trial cap)", title="ready", border_style="green"))
    if not Confirm.ask("[cyan]run it?[/]", default=False, console=console):
        raise KeyboardInterrupt
    return QualityWizardResult(mode="full", arms=",".join(arms), tasks=tasks, model=model,
                               k=k, budget_usd=budget, milestones=milestones, out=out,
                               force=force, retries=retries, auth=auth)


def _incremental_wizard(console: Console) -> QualityWizardResult:
    from .quality import engine as eng
    src = _select_one(console, "source session", [
        ("own", "your own Claude Code sessions (~/.claude/projects) — pick from a list", True),
        ("file", "a session .jsonl path", True),
        ("swechat", "SWE-chat cached conversations (coming soon here)", False),
    ])
    swechat = None
    conv = 0
    ctx_gate = 50_000
    # gate the pick NOW: a session whose peak context never crossed the compaction gate
    # is a guaranteed passthrough — surfacing that after every knob is answered (as the
    # runner's late SystemExit did) wastes the whole wizard walk.
    while True:
        if src == "own":
            from .counterfactual import pick_session
            p = pick_session(console)  # the peak-ctx picker
        else:  # file
            raw = Prompt.ask("[cyan]session .jsonl path[/]", console=console).strip()
            p = Path(raw).expanduser()
            if not p.is_file():
                console.print(f"[red]not a file:[/] {p} — try again")
                continue
        peak = eng.peak_ctx(str(p))
        if peak >= ctx_gate:
            break
        console.print(Panel.fit(
            f"[yellow]{p.name}[/] peaks at [bold]{peak / 1000:.0f}k[/] context — below the "
            f"[bold]{ctx_gate / 1000:.0f}k[/] compaction gate.\nNothing would be compacted, "
            f"so there is nothing to compare.",
            title="too short — not comparable", border_style="yellow"))
        if Confirm.ask("[cyan]pick a different session?[/]", default=True, console=console):
            continue
        ctx_gate = 0  # deliberate override: replay it anyway
        break
    session, task = str(p), p.stem[:12]
    # show the session's own model so the inherit default is meaningful
    from .counterfactual import session_meta
    own = session_meta(Path(session)).get("model")
    # the headroom arm runs the CCR retrieve loop here too: the token proxy compresses and
    # retrieve calls are executed via `headroom mcp serve` (injected). Retrieval only fires
    # on sessions with large compressible tool outputs; short ones fall back to kompress —
    # the summary says which.
    arms = _multiselect(console, "arms (vanilla control always included)", [
        ("condense", "condense — compaction proxy", True, True),
        ("headroom", "headroom — token proxy + injected retrieve loop (CCR)", True, False),
    ])
    # inherit the session's OWN model by default — running it faithfully is the point;
    # an arm that can't serve it auto-falls-back at run time (only override deliberately)
    mt = Table(title="[bold]incremental model", show_header=False, box=None)
    mt.add_row(f"[dim] 0[/] [green]inherit from session[/] "
               f"[dim]({own or 'unknown'})[/]   [green]← default[/]")
    for i, (mid, label, _d) in enumerate(QUALITY_MODELS, 1):
        mt.add_row(f"[dim]{i:>2}[/] override → {label} [dim]({mid})[/]")
    console.print(mt)
    raw = Prompt.ask("[cyan]model[/] (0 = inherit, or a number/id to override)",
                     default="0", console=console).strip()
    if raw in ("", "0"):
        model = None  # inherit → replay() auto-detects + falls back
    elif raw.isdigit() and 1 <= int(raw) <= len(QUALITY_MODELS):
        model = QUALITY_MODELS[int(raw) - 1][0]
    else:
        model = raw
    limit = _ask_int(console, "[cyan]max decision points[/] (0 = all; contiguous from the start)", 0)
    budget = _ask_float(console, "[cyan]per-arm $ cap[/]", 2.0)
    # control runs first; by default the later arms are capped at the steps control reached
    # within budget — the paired comparison window, so no arm over-replays steps that have no
    # control counterpart to score against. Opting into independent budgets lets each arm run
    # to its own cap (a "how far does each arm get" reach test) but yields ragged step counts.
    independent_budgets = not Confirm.ask(
        "[cyan]cap arms to control's step window?[/] (recommended — control runs first, later "
        "arms stop at min(control's steps, own budget); No = each arm burns its own budget)",
        default=True, console=console)
    # faithful capture: run the version-matched Claude Code binary once, LOCALLY, to
    # capture the EXACT system prompt + tools + CLAUDE.md your run used — instead of an
    # approximate frozen template. Consent + full transparency, per the user's ask.
    ver = session_meta(Path(session)).get("version") or "newest on disk"
    console.print(Panel.fit(
        f"[bold]capture from your Claude Code binary[/] (recommended)\n"
        f"Runs [bold]Claude Code {ver}[/] once in [bold]{Path(session).parent.name}[/]'s "
        f"project dir to capture the system prompt, tool catalog, and CLAUDE.md your\n"
        f"session used, instead of a stored template.\n"
        f"[dim]• reads & runs your local `claude` binary   • one request to a LOCAL proxy — "
        f"nothing is sent externally, $0\n• falls back to the template if you decline[/]",
        border_style="cyan"))
    capture = Confirm.ask("[cyan]capture from your Claude Code binary?[/]",
                          default=True, console=console)
    # per-step scoring mode. structural (free) matches the exact action; but free-running
    # agents wander valid routes (control agrees only ~26%), so 'goal' is more robust — it
    # rates each action good/degraded/bad toward the TASK on its own merits.
    judge = _select_one(console, "per-step scoring", [
        ("goal", "goal-based — rate each action good/degraded/bad toward the task "
                 "(robust to valid divergence; one LLM call/step) [recommended]", True),
        ("off", "structural only — exact same-action match, free (read vs control floor)", True),
        ("equivalence", "structural + LLM upgrade of near-misses (grep vs rg) to 'agrees'", True),
    ])
    out = Prompt.ask("[cyan]output dir[/]", default="results/jobs/run", console=console).strip()
    # resume: re-running to the same out skips arms that already finished (a .done sentinel),
    # so a cancel mid-run resumes at the next arm instead of re-running control. An interrupted
    # arm has no sentinel and re-runs. Only meaningful when reusing an out; harmless otherwise.
    resume = Confirm.ask("[cyan]resume?[/] (skip arms already complete in this out dir; a "
                         "cancelled arm re-runs)", default=True, console=console)
    auth = _auth_choice(console)  # only prompts when both an API key AND a subscription exist
    _quality_preflight(console, arms, need_docker=False)
    model_lbl = f"inherit ({own})" if model is None else model
    console.print(Panel.fit(
        f"[bold]incremental trajectory[/]   [bold]model[/] {model_lbl}\n"
        f"[bold]session[/] {Path(session).name}\n"
        f"[bold]arms[/] control + {', '.join(arms)}   [bold]every[/] {every}   "
        f"[bold]limit[/] {limit or 'all'}\n"
        f"[bold]source[/] {'exact capture' if capture else 'template'}   "
        f"[bold]scoring[/] {judge}\n"
        f"[bold]per-arm cap[/] ${budget:g}   "
        f"[bold]window[/] {'independent budgets' if independent_budgets else 'cap to control'}   "
        f"[bold]resume[/] {'yes' if resume else 'no'}   [bold]auth[/] {auth}   [bold]out[/] {out}",
        title="ready", border_style="green"))
    if not Confirm.ask("[cyan]run the incremental?[/]", default=False, console=console):
        raise KeyboardInterrupt
    return QualityWizardResult(mode="incremental", source=src, session=session, swechat=swechat,
                               conv=conv, task=task, arms=",".join(arms), model=model,
                               limit=limit, budget_usd=budget, out=out,
                               judge=judge, capture=capture, ctx_gate=ctx_gate,
                               independent_budgets=independent_budgets, resume=resume, auth=auth)


# --------------------------------------------------------------------- auth + setup
def _auth_choice(console: Console) -> str:
    """Which Anthropic credential to spend. 'auto' unless BOTH an API key AND a Claude Code
    subscription are live — then let the user pick, since 'auto' silently prefers the key and a
    subscription user would want to say so. When only one is live, say which (and how to enable
    the other) rather than showing nothing — the missing toggle should never be a mystery."""
    import os

    from .quality.engine import cc_oauth_token, load_env
    env = {**load_env(), **os.environ}
    has_key, has_sub = bool(env.get("ANTHROPIC_API_KEY")), bool(cc_oauth_token())
    if has_key and has_sub:
        return _select_one(console, "both an API key and a Claude Code login are live — "
                           "which should this run spend?", [
            ("api-key", "API key — API billing", True),
            ("subscription", "Claude Code subscription — draws on your plan, no API-key spend", True),
        ])
    if has_key:
        console.print("[dim]auth: API key (only live credential — no toggle). To spend your "
                      "Claude Code subscription instead, run [/][cyan]claude setup-token[/][dim] "
                      "(or open `claude` to refresh an expired login), then this choice appears.[/]")
    elif has_sub:
        console.print("[dim]auth: Claude Code subscription (no API key set).[/]")
    return "auto"


def _upsert_env(path: Path, updates: dict) -> None:
    """Write KEY=value pairs into a .env, replacing existing keys IN PLACE and preserving every
    other line and comment. Seeds from .env.dist when the file doesn't exist yet."""
    if path.exists():
        lines = path.read_text().splitlines()
    elif (path.parent / ".env.dist").exists():
        lines = (path.parent / ".env.dist").read_text().splitlines()
    else:
        lines = []
    seen, out = set(), []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s and s.split("=", 1)[0].strip() in updates:
            k = s.split("=", 1)[0].strip()
            out.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    path.write_text("\n".join(out) + "\n")


def run_setup_wizard(console: Console) -> None:
    """First-run setup: detect what's configured, fill in credentials, write .env — so an
    open-source user goes from a fresh clone to a runnable bench without hand-editing .env.
    Idempotent: existing secrets are shown masked and kept unless explicitly replaced."""
    import os

    from .quality.engine import auth_mode, cc_oauth_token, load_env
    env_path = Path(".env")
    console.print(Panel.fit(
        Text.assemble(("minmax-bench · setup\n", "bold magenta"),
                      ("configure credentials and write .env", "dim")),
        border_style="magenta"))

    env = {**load_env(), **os.environ}
    mask = lambda v: ("…" + v[-4:]) if v and len(v) > 4 else ("set" if v else "[dim]—[/]")
    am, sub = auth_mode(env), bool(cc_oauth_token())
    t = Table(title="[bold]detected", show_header=False, box=None)
    t.add_row("Anthropic auth", f"[green]{am}[/]" if am else "[red]none[/]")
    t.add_row("  ANTHROPIC_API_KEY", mask(env.get("ANTHROPIC_API_KEY")))
    t.add_row("  Claude Code login", "[green]available[/]" if sub else "[dim]not found[/]")
    t.add_row("CONDENSE_API_KEY", mask(env.get("CONDENSE_API_KEY")))
    t.add_row("HF_TOKEN", mask(env.get("HF_TOKEN")))
    console.print(t)

    updates: dict = {}

    console.print("\n[bold]1) Anthropic access[/] — the bench needs EITHER an API key OR your "
                  "Claude Code subscription login.")
    pick = _select_one(console, "Anthropic access", [
        ("keep", f"keep current ({am or 'none'})", True),
        ("api-key", "set / replace an ANTHROPIC_API_KEY (API billing)", True),
        ("subscription", "use my Claude Code subscription "
                         + ("[green](detected ✓)[/]" if sub else "(needs `claude setup-token`)"),
         True),
    ])
    if pick == "api-key":
        key = Prompt.ask("  [cyan]ANTHROPIC_API_KEY[/] (sk-ant-…)", password=True,
                         default="", console=console).strip()
        if key:
            updates["ANTHROPIC_API_KEY"] = key
    elif pick == "subscription":
        if sub:
            console.print("  [green]Claude Code login detected[/] — no key needed; runs use it "
                          "automatically when no API key is set.")
        else:
            console.print("  Run [bold]claude setup-token[/] in another shell, then paste the "
                          "token (or leave blank if you're logged into Claude Code — it's read "
                          "from ~/.claude/.credentials.json / the keychain automatically).")
            tok = Prompt.ask("  [cyan]CLAUDE_CODE_OAUTH_TOKEN[/]", password=True, default="",
                             console=console).strip()
            if tok:
                updates["CLAUDE_CODE_OAUTH_TOKEN"] = tok
        if env.get("ANTHROPIC_API_KEY") or "ANTHROPIC_API_KEY" in updates:
            console.print("  [dim]note: with an API key also present, runs default to it — pick "
                          "subscription per-run in the wizard's auth toggle or with "
                          "--auth subscription.[/]")

    console.print("\n[bold]2) condense arm[/] — the quality-bench condense arm sends this to "
                  "api.condense.chat (skip if you won't run condense).")
    if Confirm.ask("  set CONDENSE_API_KEY?", default=not bool(env.get("CONDENSE_API_KEY")),
                   console=console):
        k = Prompt.ask("  [cyan]CONDENSE_API_KEY[/] (ak_…)", password=True, default="",
                       console=console).strip()
        if k:
            updates["CONDENSE_API_KEY"] = k

    console.print("\n[bold]3) SWE-chat dataset[/] [dim](optional)[/] — HuggingFace token for the "
                  "gated SALT-NLP/SWE-chat dataset.")
    if Confirm.ask("  set HF_TOKEN?", default=False, console=console):
        k = Prompt.ask("  [cyan]HF_TOKEN[/] (hf_…)", password=True, default="",
                       console=console).strip()
        if k:
            updates["HF_TOKEN"] = k

    if not updates:
        console.print("\n[dim]no changes to write.[/]")
        return
    console.print(f"\n[bold]will write to[/] {env_path}: [cyan]{', '.join(updates)}[/]")
    if not Confirm.ask("write .env?", default=True, console=console):
        console.print("[yellow]aborted — nothing written.[/]")
        return
    _upsert_env(env_path, updates)
    env2 = {**load_env(), **os.environ, **updates}
    console.print(Panel.fit(
        f"[green]✓ wrote {env_path}[/]  ([dim]keys: {', '.join(updates)}[/])\n"
        f"auth now resolves to: [bold]{auth_mode(env2) or 'none'}[/]\n"
        f"[dim]verify:[/] minmax-bench info    [dim]·  run:[/] minmax-bench quality run",
        border_style="green", title="setup complete"))
