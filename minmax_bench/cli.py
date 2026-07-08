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


@app.command("counterfactual")
def counterfactual_cmd(
    session: str | None = typer.Argument(None, help="A Claude Code session .jsonl (default: pick interactively from ~/.claude/projects)."),
    arms: str = typer.Option("condense", "--arms", help="Comma list of arms to replay besides control (condense, headroom)."),
    budget_usd: float = typer.Option(2.0, "--budget-usd", help="Max spend per arm (control included)."),
    limit: int = typer.Option(0, "--limit", "-n", help="Max decision points to replay (0 = all)."),
    every: int = typer.Option(1, "--every", help="Replay every Nth decision point."),
    max_tokens: int = typer.Option(6000, "--max-tokens", help="Replay output cap per step."),
    model: str | None = typer.Option(None, "--model", help="Override the replay model (default: the session's own model)."),
    out: str | None = typer.Option(None, "--out", help="Output dir (default results/counterfactual/<session>-<ts>)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    auth: str = typer.Option("auto", "--auth",
                             help="auto (api key if configured, else Claude Code login) | "
                                  "api-key | subscription (force the Claude Code login path)"),
):
    """Replay one of YOUR local Claude Code sessions through condense/headroom, step by step.

    Teacher-forced counterfactual: at every decision point the recorded prefix is sent
    through each arm and the next action is compared to what actually happened, next to
    a control replay (the noise floor). Costs real money; shows an estimate first.
    """
    from .counterfactual import pick_session, render_summary, replay

    sp = Path(session).expanduser() if session else pick_session(console)
    if not sp.is_file():
        raise typer.BadParameter(f"not a session file: {sp}")
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out) if out else Path("results/counterfactual") / f"{sp.stem[:8]}-{stamp}"
    try:
        summary = replay(sp, arm_list, budget_usd=budget_usd, limit=limit, every=every,
                         max_tokens=max_tokens, out_dir=out_dir, console=console,
                         assume_yes=yes, model=model, auth=auth)
    except SystemExit as e:
        raise typer.Exit(e.code if isinstance(e.code, int) else 1) from None
    render_summary(summary, console)
    console.print(f"[green]artifacts[/] {out_dir}/  (per-step jsonl + summary.json)")


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
