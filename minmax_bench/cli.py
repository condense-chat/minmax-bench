"""minmax-bench command-line interface."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from .catalog import DEFAULT_MODELS
from .config import get_settings
from .dashboard import Dashboard, replay
from .provision import provision, tools_for
from .report import (
    DEFAULT_EDGES,
    measurements_from_json,
    recompute_buckets,
    render_run,
    render_tables,
    run_report_json,
)
from .runner import key as track_key
from .runner import run as run_bench
from .runstore import BASELINE, RunStore
from .strategies import STRATEGY_MATRIX, default_selected, has_entry, matrix_names

app = typer.Typer(add_completion=False, help="Estimated token/cost savings benchmark for agent-session proxies.")
console = Console()

# ---- quality / trajectory-preservation bench ---------------------------------
# The cost bench (`run`, `report`, `replay`, …) measures how much a proxy SAVES.
# The quality bench measures whether the compressed session still does the same
# work. It lives in minmax_bench.quality; these commands surface it under one
# entrypoint. `run`/`report` are first-class (typed options, like the cost bench);
# they translate to the driver's argv so defaults/validation stay in one place.
quality_app = typer.Typer(add_completion=False, help="Quality / trajectory-preservation bench.")
app.add_typer(quality_app, name="quality")

# defaults mirror minmax_bench.quality.generate's argparse — kept in sync there
_Q_DATASET = "terminal-bench/terminal-bench-2-1"


def _flag(argv: list[str], name: str, value, default=None) -> None:
    """Append `--name value` to argv unless value is the driver's default/None."""
    if value is not None and value != default:
        argv += [name, str(value)]


@quality_app.command("run")
def quality_run(
    tasks: str | None = typer.Option(None, "--tasks", help="N = first N recommended | random:N (with --seed) | a,b,c | omitted = 5. See --list-tasks."),
    arms: str = typer.Option("condense,headroom", "--arms", help="Methods to run; vanilla baseline always included."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model id (default claude-sonnet-4-6)."),
    dataset: str = typer.Option(_Q_DATASET, "--dataset", "-d", help="Harbor dataset (only the default is validated)."),
    k: int = typer.Option(4, "--k", help="Trials per arm/task."),
    k_vanilla: int | None = typer.Option(None, "--k-vanilla", help="Trials for the vanilla baseline (default k+1)."),
    budget_usd: float = typer.Option(5.0, "--budget-usd", help="Per-trial spend cap (Harbor max_budget_usd)."),
    concurrency: int = typer.Option(1, "--concurrency", help="Parallel trials per cell (harbor -n)."),
    milestones: bool = typer.Option(False, "--milestones", help="Also run the LLM milestone judge."),
    out: str = typer.Option("results/jobs/run", "--out", help="Results root."),
    seed: int | None = typer.Option(None, "--seed", help="Seed for --tasks random:N."),
    agent_timeout_mult: int | None = typer.Option(None, "--agent-timeout-mult", help="Harbor agent EXECUTION timeout multiplier (headroom auto-3)."),
    setup_timeout_mult: float = typer.Option(3.0, "--setup-timeout-mult", help="Harbor agent SETUP timeout multiplier, all arms (slow container installs; 3 = ~18min)."),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="Print the known tasks and exit."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the Harbor commands without running."),
    auth: str = typer.Option("auto", "--auth", help="auto | api-key | subscription (force Claude Code login; no API key needed)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the guided wizard; use flags/defaults."),
):
    """Run the agents end-to-end via Harbor and compare trajectories (SPENDS).

    The quality analog of `minmax-bench run`: bare `quality run` launches a guided
    wizard; or drive it with flags, e.g. `-m claude-haiku-4-5 --tasks 5 --milestones`.
    """
    from minmax_bench.quality.generate import main
    if list_tasks:
        main(["--list-tasks", "--dataset", dataset])
        return
    # bare + interactive → guided wizard (like the cost bench's `run`). The wizard
    # can pick EITHER full or incremental trajectories and its own source.
    if (sys.stdin.isatty() and not yes and not dry_run and tasks is None and model is None
            and arms == "condense,headroom" and dataset == _Q_DATASET):
        from .interactive import run_quality_wizard
        try:
            w = run_quality_wizard(console)
        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]aborted.[/]")
            raise typer.Exit(1) from None
        if w.mode == "incremental":
            _run_incremental(session=w.session, arms=w.arms, model=w.model, every=w.every,
                             limit=w.limit, budget_usd=w.budget_usd, max_tokens=6000,
                             out=w.out, task=w.task, auth="auto", assume_yes=True, judge=w.judge,
                             capture=w.capture)
            return
        arms, tasks, model, k, budget_usd, milestones, out = (
            w.arms, w.tasks, w.model, w.k, w.budget_usd, w.milestones, w.out)
    argv = ["--mode", "full", "--arms", arms, "--dataset", dataset, "--out", out,
            "--k", str(k), "--budget-usd", str(budget_usd), "--concurrency", str(concurrency)]
    _flag(argv, "--tasks", tasks)
    _flag(argv, "--model", model)
    _flag(argv, "--k-vanilla", k_vanilla)
    _flag(argv, "--seed", seed)
    _flag(argv, "--agent-timeout-mult", agent_timeout_mult)
    _flag(argv, "--setup-timeout-mult", setup_timeout_mult, default=3.0)
    if auth != "auto":
        argv += ["--auth", auth]
    if milestones:
        argv.append("--milestones")
    if dry_run:
        argv.append("--dry-run")
    main(argv)


@quality_app.command("report")
def quality_report(
    from_: str = typer.Option("results/jobs", "--from", help="Results root produced by `quality run`."),
    tasks: str | None = typer.Option(None, "--tasks", help="N | a,b,c | omitted = 5 (must cover what was run)."),
    arms: str = typer.Option("condense,headroom", "--arms", help="Arms to display."),
    fmt: str = typer.Option("html", "--format", help="html | md."),
    out: str | None = typer.Option(None, "--out", help="Output path (default report.<format>)."),
    ctx_gate: int = typer.Option(50_000, "--ctx-gate", help="Peak-ctx threshold below which compaction can't fire (⊘)."),
):
    """Render the quality bench from stored artifacts — never spends."""
    from minmax_bench.quality.report import main
    argv = ["--from", from_, "--arms", arms, "--format", fmt, "--ctx-gate", str(ctx_gate)]
    _flag(argv, "--tasks", tasks)
    _flag(argv, "--out", out)
    main(argv)


def _run_incremental(*, session: str | None, arms: str, model: str | None, every: int,
                     limit: int, budget_usd: float, max_tokens: int, out: str, task: str,
                     auth: str, assume_yes: bool, judge: bool = False, steps: bool = True,
                     capture: bool = False) -> None:
    """Rich teacher-forced replay of one session — picker when no --session, model
    auto-fallback, cost preview, per-arm progress, a summary table with the recorded
    backtest anchor, and a per-step good/semi/bad/redundant readout. Writes
    <out>/incremental/<task>-<arm>.jsonl for report."""
    from .counterfactual import pick_session, render_steps, render_summary, replay
    sp = Path(session).expanduser() if session else pick_session(console)
    if not sp.is_file():
        raise typer.BadParameter(f"not a session file: {sp}")
    arm_list = [a.strip() for a in arms.split(",") if a.strip() and a.strip() != "control"]
    try:
        summary = replay(sp, arm_list, budget_usd=budget_usd, limit=limit, every=every,
                         max_tokens=max_tokens, out_dir=Path(out), console=console,
                         assume_yes=assume_yes, model=model, auth=auth, task=task, judge=judge,
                         capture=capture)
    except SystemExit as e:
        raise typer.Exit(e.code if isinstance(e.code, int) else 1) from None
    render_summary(summary, console)
    if steps:
        render_steps(summary, console)
    console.print(f"[green]artifacts[/] {out}/incremental/  ([dim]report:[/] "
                  f"minmax-bench quality report --from {out} --tasks {task} --arms {arms})")


@quality_app.command("incremental")
def quality_incremental(
    session: str | None = typer.Argument(None, help="A session .jsonl (default: pick from ~/.claude/projects)."),
    arms: str = typer.Option("condense", "--arms", help="Arms to replay besides control (condense, headroom)."),
    model: str | None = typer.Option(None, "--model", "-m", help="Replay model (default: the session's own, with auto-fallback if an arm can't serve it)."),
    every: int = typer.Option(1, "--every", help="Sample every Nth decision point."),
    limit: int = typer.Option(0, "--limit", "-n", help="Max decision points (0 = all)."),
    budget_usd: float = typer.Option(2.0, "--budget-usd", help="Per-arm spend cap (control included)."),
    max_tokens: int = typer.Option(6000, "--max-tokens", help="Replay output cap per step."),
    out: str | None = typer.Option(None, "--out", help="Output dir (default results/incremental/<session>-<ts>)."),
    task: str = typer.Option("session", "--task", help="Task label for the report join."),
    auth: str = typer.Option("auto", "--auth", help="auto | api-key | subscription (force the Claude Code login)."),
    judge: bool = typer.Option(False, "--judge", help="LLM-adjudicate structural near-misses (grep vs rg etc.) as equivalent."),
    steps: bool = typer.Option(True, "--steps/--no-steps", help="Show the per-step good/semi/bad/redundant readout."),
    capture: bool = typer.Option(False, "--capture", help="Run your version-matched Claude Code binary once (locally) to capture its EXACT system prompt + tools, instead of an approximate template."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Incremental (teacher-forced per-step) trajectories — the paired counterpart
    to `run`'s full trajectories. Replays one session through control + each arm,
    with the session's own model (auto-falling back if an arm can't serve it), a
    cost preview, and a summary table incl. the recorded backtest anchor (SPENDS).

    This is what used to be the `counterfactual` command — replay YOUR own sessions.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    stem = Path(session).stem[:8] if session else "picked"
    out_dir = out or str(Path("results/incremental") / f"{stem}-{stamp}")
    _run_incremental(session=session, arms=arms, model=model, every=every, limit=limit,
                     budget_usd=budget_usd, max_tokens=max_tokens, out=out_dir, task=task,
                     auth=auth, assume_yes=yes, judge=judge, steps=steps, capture=capture)


# judge takes niche flags; pass through to the driver (which owns its --help)
@quality_app.command("judge", context_settings={"allow_extra_args": True,
                                                 "ignore_unknown_options": True,
                                                 "help_option_names": []})
def quality_judge(ctx: typer.Context):
    """Run the LLM milestone judge over existing full-mode runs (SPENDS)."""
    from minmax_bench.quality.generate import main
    main(["--mode", "judge", *ctx.args])


def _meta(m) -> dict:
    return {
        "run_uuid": m.uuid, "created_utc": m.created_utc, "dataset": m.dataset,
        "strategies": m.strategies, "models": m.models, "edges": m.edges,
        "encoding": m.encoding, "count_mode": m.count_mode, "max_tokens": m.max_tokens,
        "session_limit": m.session_limit, "point_limit": m.point_limit, "longest": m.longest,
    }


def _finish(store, result) -> None:
    render_run(result.per_model_buckets, console)
    for err in result.errors:
        console.print(f"[red]error[/] {err}")
    store.write_report({
        "meta": _meta(store.manifest),
        "models": run_report_json(result.per_model_buckets),
        "errors": result.errors,
    })
    (store.root / "README.md").write_text(_bundle_readme(store.manifest))
    console.print(f"[green]run stored[/] {store.root}/  ([dim]replay:[/] minmax-bench replay {store.manifest.uuid})")


def _resolve_setup(opt: str, strategies: list[str]) -> list[str]:
    """Map the --setup flag to a concrete tool list. 'auto' = tools the strategies imply."""
    o = (opt or "").strip().lower()
    if o in ("", "none", "off", "no"):
        return []
    if o == "auto":
        return tools_for(strategies)
    return [t.strip() for t in o.split(",") if t.strip()]


def _parse_token_budget(raw: str | None) -> int | None:
    """'200k'/'1.5m'/'50000' -> int tokens; None/'' -> None."""
    if not raw:
        return None
    r = raw.strip().lower().replace(",", "")
    mult = 1
    if r.endswith("k"):
        mult, r = 1_000, r[:-1]
    elif r.endswith("m"):
        mult, r = 1_000_000, r[:-1]
    try:
        return int(float(r) * mult)
    except ValueError:
        return None


@app.command("run")
def run_cmd(
    dataset: str = typer.Option("sample", "--dataset", "-d", help="Dataset spec, e.g. sample | swe-chat:50 | claude-code:/path/*.jsonl"),
    strategy: list[str] = typer.Option(None, "--strategy", "-s", help="Strategy name(s); repeatable. Default: headroom condense."),
    model: list[str] = typer.Option(None, "--model", "-m", help="Model id(s); repeatable. Default: haiku."),
    edges: str | None = typer.Option(None, "--edges", help="Comma-separated bucket upper edges in tokens."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max conversations to load."),
    longest: int | None = typer.Option(None, "--longest", help="Keep only the N longest conversations."),
    max_points: int | None = typer.Option(None, "--max-points", help="Cap turns per conversation (bounds cost)."),
    token_budget: str | None = typer.Option(None, "--token-budget", help="Truncate each conversation at the first turn whose chain reaches N tokens (e.g. 200k)."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Proxy output cap (default from .env)."),
    encoding: str = typer.Option("o200k_base", "--encoding", help="tiktoken encoding for local counts."),
    count: str = typer.Option("local", "--count", help="Baseline counting: 'api' (exact Anthropic count_tokens, free) or 'local' (tiktoken)."),
    refresh: str | None = typer.Option(None, "--refresh", help="Comma list of strategies (or 'all'/'baseline') to recompute, ignoring cache."),
    run: str | None = typer.Option(None, "--run", help="Resume an existing run-<uuid> (reuse its caches) instead of minting a new one."),
    live: bool = typer.Option(True, "--live/--no-live", help="Animate the live dashboard while measuring."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the guided wizard; use flags/defaults."),
    setup: str = typer.Option("auto", "--setup", help="Local tools to install/start: 'auto' (implied by strategies), 'none', or a comma list of headroom,dense."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Root for run-<uuid> dirs (default from settings)."),
):
    """Guided (or flag-driven) benchmark: measure baseline + strategies per model."""
    settings = get_settings()
    root = runs_dir or settings.runs_dir
    mt = max_tokens if max_tokens is not None else settings.proxy_max_tokens
    refresh_set = {x.strip() for x in refresh.split(",")} if refresh else set()

    chosen_s = list(strategy) if strategy else None
    chosen_m = list(model) if model else None
    for name in chosen_s or []:
        if not has_entry(name):
            raise typer.BadParameter(f"unknown strategy: {name}. Try: {', '.join(matrix_names())}")

    interactive = sys.stdin.isatty() and not yes and not run and not chosen_s and not chosen_m

    wizard_setup: list[str] | None = None
    if run:
        store = RunStore.open(root, run)
        if chosen_s:
            store.manifest.strategies = list(dict.fromkeys(store.manifest.strategies + chosen_s))
        if chosen_m:
            store.manifest.models = list(dict.fromkeys(store.manifest.models + chosen_m))
        store._write_manifest()
    else:
        if interactive:
            from .interactive import run_wizard

            try:
                w = run_wizard(console)
            except (KeyboardInterrupt, EOFError):
                console.print("[yellow]aborted.[/]")
                raise typer.Exit(1) from None
            fields = {
                "dataset": w.dataset, "strategies": w.strategies, "models": w.models,
                "edges": w.edges, "session_ids": w.session_ids, "token_limit": w.token_limit,
                "count_mode": w.count_mode, "encoding": encoding, "max_tokens": mt,
            }
            wizard_setup = w.setup
        else:
            fields = {
                "dataset": dataset, "strategies": chosen_s or default_selected(),
                "models": chosen_m or DEFAULT_MODELS,
                "edges": [int(x) for x in edges.split(",")] if edges else list(DEFAULT_EDGES),
                "session_limit": limit, "point_limit": max_points, "longest": longest,
                "token_limit": _parse_token_budget(token_budget),
                "count_mode": count, "encoding": encoding, "max_tokens": mt,
            }
        store = RunStore.create(root, {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), **fields})

    m = store.manifest
    console.print(f"[dim]run=[/]{m.uuid}  [dim]models=[/]{','.join(m.models)}  [dim]strategies=[/]{','.join(m.strategies)}")

    setup_tools = wizard_setup if wizard_setup is not None else _resolve_setup(setup, m.strategies)

    with provision(setup_tools, console):
        if live and console.is_terminal:
            dash = Dashboard([], title=f"minmax-bench run {m.uuid}", pace=0.02, console=console)
            with dash:
                result = run_bench(store, refresh=refresh_set, dashboard=dash)
        else:
            result = run_bench(store, refresh=refresh_set, dashboard=None)
    _finish(store, result)


@app.command("replay")
def replay_cmd(
    run: str = typer.Argument(..., help="run-<uuid> to replay (animates the recorded evolution, no spend)."),
    fps: float = typer.Option(30.0, "--fps", help="Animation frames per second."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Root for run-<uuid> dirs."),
):
    """Replay a finished run's animated context + cost evolution, then print tables."""
    root = runs_dir or get_settings().runs_dir
    store = RunStore.open(root, run)
    m = store.manifest
    order: list[str] = []
    labels: dict[str, str] = {}
    per_kind: dict[str, list] = {}
    for model in m.models:
        for kind in [BASELINE, *m.strategies]:
            pts = store.flat(model, kind)
            if not pts:
                continue
            k = track_key(model, kind)
            order.append(k)
            labels[k] = f"{model} · {kind}"
            per_kind[k] = pts
    replay(per_kind, order, labels, title=f"replay {m.uuid}", fps=fps, console=console)
    render_run(store.buckets(), console)


@app.command("report")
def report_cmd(
    path: str = typer.Argument(..., help="A run-<uuid> (recompute from its stored usage) or a legacy measurements.json."),
    edges: str | None = typer.Option(None, "--edges", help="Comma-separated bucket upper edges."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Root for run-<uuid> dirs."),
):
    """Recompute and print buckets from stored raw usage — verify results without re-spending."""
    edge_list = [int(x) for x in edges.split(",")] if edges else None
    root = runs_dir or get_settings().runs_dir
    p = Path(path)
    if p.is_file():  # legacy measurements.json bundle
        data = json.loads(p.read_text())
        rows = measurements_from_json(data["rows"])
        el = edge_list or data.get("meta", {}).get("edges")
        render_tables(recompute_buckets(rows, el), console)
        return
    store = RunStore.open(root, path)
    if edge_list is not None:
        store.manifest.edges = edge_list
    render_run(store.buckets(), console)


@app.command("runs")
def runs_cmd(runs_dir: str | None = typer.Option(None, "--runs-dir")):
    """List stored runs (newest last)."""
    root = Path(runs_dir or get_settings().runs_dir)
    for mp in sorted(root.glob("run-*/run.json"), key=lambda p: p.stat().st_mtime):
        try:
            m = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        console.print(f"[bold]{m['uuid']}[/] [dim]{m.get('created_utc','')}[/]  {m.get('dataset')}  "
                      f"models={','.join(m.get('models', []))}  -> {','.join(m.get('strategies', []))}")


@app.command("fetch")
def fetch_cmd(
    dataset: str = typer.Argument("swe-chat", help="Dataset spec to materialize locally, e.g. swe-chat:60"),
    force: bool = typer.Option(False, "--force", help="Re-stream from HF even if a local cache exists."),
):
    """Download/prepare a dataset into the local (shared) data cache."""
    import time

    from .data.loaders import _load_swe_chat_cached, swe_chat_cache_path

    source, _, arg = dataset.partition(":")
    if source != "swe-chat":
        raise typer.BadParameter("fetch currently supports 'swe-chat[:N]'.")
    limit = int(arg) if arg.strip() else None
    path = swe_chat_cache_path(limit)
    if path.exists() and not force:
        console.print(f"[green]already cached[/] {path} (use --force to refresh)")
        return
    console.print(f"[dim]streaming SWE-chat (limit={limit}) from HuggingFace…[/]")
    t = time.time()
    sessions = _load_swe_chat_cached(limit, force=True)
    console.print(f"[green]cached[/] {len(sessions)} sessions -> {path} ({time.time() - t:.0f}s)")


@app.command("strategies")
def strategies_cmd():
    """List the strategy matrix (name, kind, config, and how it resolves)."""
    for entry in STRATEGY_MATRIX:
        tags = []
        if entry.mandatory:
            tags.append("mandatory")
        if entry.enabled and not entry.mandatory:
            tags.append("default")
        cfg = f" {entry.config.params}" if entry.config.params else ""
        tag = f" [dim]({', '.join(tags)})[/]" if tags else ""
        console.print(f"[bold]{entry.name}[/] [dim]{entry.kind}[/]{tag}{cfg}\n    {entry.description}")
        try:
            r = entry.resolve()
            if r.proxy:
                console.print(f"    [dim]-> {r.proxy.base_url}[/]")
        except Exception as e:  # resolution needs creds/tools it may not have yet
            console.print(f"    [red]unavailable:[/] {e}")


@app.command("info")
def info_cmd():
    """Show resolved settings (keys are masked)."""
    from .dense import load_profile

    s = get_settings()

    def mask(v: str | None) -> str:
        return "set" if v else "[red]missing[/]"

    prof = load_profile(s.condense_profile)
    console.print("[bold]minmax-bench settings[/]")
    console.print(f"  ANTHROPIC_API_KEY : {mask(s.anthropic_api_key)}")
    console.print(f"  OPENAI_API_KEY    : {mask(s.openai_api_key)}")
    console.print(f"  HF_TOKEN          : {mask(s.hf_token)}")
    console.print(f"  headroom_base_url : {s.headroom_base_url}")
    console.print(f"  runs_dir          : {s.runs_dir}")
    console.print(f"  condense (dense)  : profile={prof.name} url={prof.api_url} "
                  f"token={mask(prof.auth_token)} user={mask(prof.user_id)}")


def _bundle_readme(m) -> str:
    return f"""# minmax-bench run {m.uuid}

Generated: {m.created_utc}

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report {m.uuid}

## Replay the animated evolution

    minmax-bench replay {m.uuid}

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench run --run {m.uuid} -s condense
"""


if __name__ == "__main__":
    app()
