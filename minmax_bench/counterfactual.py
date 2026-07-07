"""Counterfactual replay of a *local* Claude Code session through compaction arms.

Answers "how would my session have played out under condense/headroom?" the only way
that is honest without re-executing tools on the user's machine: teacher-forced
per-step replay (the same engine as ``scripts/generate.py --mode incremental``).
At every assistant decision point the recorded prefix is sent through each arm's
endpoint and the replayed next action is scored against what actually happened.
Control (api.anthropic.com, no compaction) is always replayed too — it is the
noise floor; an arm only "diverges" to the extent it falls below that floor.

This module is glue + display: session discovery under ``~/.claude/projects``,
cost preview, privacy notice, and a rich summary. All replay mechanics live in
``minmax_bench/quality/engine.py`` (single source of truth).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from minmax_bench.quality import engine as eng

REPO_ROOT = Path(__file__).resolve().parent.parent
CC_PROJECTS = Path.home() / ".claude" / "projects"
# rough sizing for the preview only; real usage comes back from the API
CHARS_PER_TOKEN = 4.0
PREVIEW_OUTPUT_TOKENS = 300


# --------------------------------------------------------------------------- discovery
@dataclass
class LocalSession:
    path: Path
    project: str          # decoded project dir (best effort)
    mtime: float
    size: int
    prompt: str           # first user text, for the picker
    cwd: str | None


def _peek(path: Path, max_lines: int = 300) -> tuple[str, str | None, bool]:
    """(first user prompt snippet, cwd, has_assistant) from the head of a session file."""
    prompt, cwd, has_assistant = "", None, False
    try:
        with path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n >= max_lines and (prompt and cwd and has_assistant):
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("isSidechain"):
                    return "", None, False  # sub-agent transcript, not a user session
                cwd = cwd or rec.get("cwd")
                t = rec.get("type")
                if t == "assistant":
                    has_assistant = True
                if t == "user" and not prompt:
                    c = rec.get("message", {}).get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = next((b.get("text", "") for b in c
                                     if isinstance(b, dict) and b.get("type") == "text"), "")
                    else:
                        text = ""
                    if text and not text.lstrip().startswith("<"):  # skip harness-injected tags
                        prompt = text
                if prompt and cwd and has_assistant and n >= max_lines:
                    break
    except OSError:
        return "", None, False
    return " ".join(prompt.split())[:90], cwd, has_assistant


def scan_sessions(root: Path = CC_PROJECTS, limit: int = 25) -> list[LocalSession]:
    """Most-recent real sessions (skips sub-agent sidechains and empty files)."""
    found: list[LocalSession] = []
    for f in sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(found) >= limit:
            break
        prompt, cwd, has_assistant = _peek(f)
        if not has_assistant:
            continue
        found.append(LocalSession(
            path=f, project=cwd or f.parent.name,  # cwd is exact; the dir name is a lossy slug
            mtime=f.stat().st_mtime, size=f.stat().st_size, prompt=prompt, cwd=cwd,
        ))
    return found


def pick_session(console: Console) -> Path:
    sessions = scan_sessions()
    if not sessions:
        console.print(f"[red]no Claude Code sessions found under {CC_PROJECTS}[/]")
        raise SystemExit(1)
    t = Table(title="[bold]local Claude Code sessions — newest first")
    t.add_column("#", justify="right", style="dim")
    t.add_column("when", style="dim")
    t.add_column("project")
    t.add_column("size", justify="right")
    t.add_column("first prompt", max_width=52)
    for i, s in enumerate(sessions, 1):
        t.add_row(str(i), datetime.fromtimestamp(s.mtime).strftime("%m-%d %H:%M"),
                  ("…" + s.project[-38:]) if len(s.project) > 39 else s.project,
                  f"{s.size / 1024:.0f}k", s.prompt or "[dim](no text prompt)[/]")
    console.print(t)
    raw = Prompt.ask("[cyan]session[/] (number or a /path/to/session.jsonl)",
                     console=console).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(sessions):
        return sessions[int(raw) - 1].path
    p = Path(raw).expanduser()
    if not p.is_file():
        console.print(f"[red]not a file:[/] {p}")
        raise SystemExit(1)
    return p


# --------------------------------------------------------------------------- session meta
def session_meta(path: Path) -> dict:
    """cwd + recorded model from a session's records (first occurrence wins)."""
    cwd, model = None, None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cwd = cwd or rec.get("cwd")
            if rec.get("type") == "assistant":
                model = model or rec.get("message", {}).get("model")
            if cwd and model:
                break
    return {"cwd": cwd, "model": model}


def _prefix_chars(msgs: list[dict], points: list[int]) -> list[int]:
    """Cumulative JSON size (chars) of each decision point's prefix."""
    sizes, total, j = [], 0, 0
    for i in points:
        while j < i:
            total += len(json.dumps(msgs[j]))
            j += 1
        sizes.append(total)
    return sizes


def estimate_usd(msgs, points, price) -> tuple[float, float]:
    """(control estimate, cache-busted upper bound) per arm, in USD."""
    toks = [c / CHARS_PER_TOKEN for c in _prefix_chars(msgs, points)]
    if not toks:
        return 0.0, 0.0
    out = len(toks) * PREVIEW_OUTPUT_TOKENS * price["output"]
    # sequential replays with a cache breakpoint: each step writes its delta, reads the rest
    control = toks[-1] * price["cache_write"] + sum(toks) * price["cache_read"] + out
    busted = sum(toks) * price["cache_write"] + out  # compaction rewrote history every step
    return control / 1e6, busted / 1e6


# --------------------------------------------------------------------------- replay
def replay(session: Path, arms: list[str], *, budget_usd: float, limit: int, every: int,
           max_tokens: int, out_dir: Path, console: Console, assume_yes: bool = False,
           model: str | None = None) -> dict:
    env = {**eng.load_env(str(REPO_ROOT / ".env")), **dict(os.environ)}

    if "control" in arms:
        console.print("[red]control is always replayed — pass only the arms to compare[/]")
        raise SystemExit(1)
    problems = eng.check_arms(["control"] + arms, env)
    if problems:
        for p in problems:
            console.print(f"[red]{p}[/]")
        raise SystemExit(1)

    template_path = REPO_ROOT / "data" / "cc_request_template.json"
    cap = json.load(open(template_path))
    tmpl_body = cap["body"]
    tmpl_headers = {k.lower(): v for k, v in cap["headers"].items()}

    msgs, points = eng.parse_session(str(session))
    if not points:
        console.print("[red]no assistant decision points in this session[/]")
        raise SystemExit(1)
    meta = session_meta(session)

    # patch the template's capture cwd to this session's real cwd
    if meta["cwd"]:
        eng.patch_cwd(tmpl_body, str(template_path), meta["cwd"])

    # replay on the session's own model so recorded thinking signatures stay valid;
    # on any other model, thinking blocks must be stripped and beta config dropped.
    replay_model = model or meta["model"] or tmpl_body["model"]
    cross_model = replay_model != tmpl_body["model"]
    tmpl_body["model"] = replay_model

    # real template defs for CC tools; permissive stubs for mcp__/Skill/etc.
    used = {b["name"] for m in msgs for b in m["content"] if b.get("type") == "tool_use"}
    tmpl_body["tools"] = eng.build_tools(used, tmpl_body["tools"])

    args = SimpleNamespace(max_tokens=max_tokens, strip_thinking=cross_model,
                           swechat=None, keep_all_tools=True, drop_beta_config=cross_model)

    sel = points[:: max(1, every)]
    if limit:
        sel = sel[:limit]
    lo, hi = estimate_usd(msgs, [i for i in sel], eng.rates_for(replay_model))
    all_arms = ["control"] + arms

    console.print(Panel.fit(
        f"[bold]session[/] {session.name}   [bold]decision points[/] {len(sel)}/{len(points)}"
        f"   [bold]model[/] {replay_model}"
        f"{' [yellow](thinking stripped: cross-model)[/]' if cross_model else ''}\n"
        f"[bold]arms[/] control + {', '.join(arms)}   "
        f"[bold]rough cost/arm[/] ${lo:.2f}–${hi:.2f} (capped at ${budget_usd:.2f} each)\n"
        f"[yellow]this sends your session content to api.anthropic.com"
        f"{' and api.condense.chat' if 'condense' in arms else ''}[/]",
        title="counterfactual replay", border_style="cyan"))
    if not assume_yes and not Confirm.ask("[cyan]replay it?[/]", default=False, console=console):
        raise SystemExit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    summary: dict = {"session": str(session), "model": replay_model, "steps": len(sel), "arms": {}}
    for arm in all_arms:
        path = out_dir / f"{arm}.jsonl"
        spent, n_ok, n_action, n_exact, ctx_total = 0.0, 0, 0, 0, 0
        with path.open("w") as f, Progress(
            TextColumn(f"[cyan]{arm:9}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total} [dim]{task.fields[stat]}[/]"),
            TimeElapsedColumn(), console=console,
        ) as prog:
            tid = prog.add_task("", total=len(sel), stat="")
            for step, i in enumerate(sel):
                if spent >= budget_usd:
                    prog.update(tid, stat=f"budget ${budget_usd} reached")
                    break
                orig = eng.extract_action(msgs[i]["content"])
                req = eng.build_request(tmpl_body, msgs[:i], args, sid)
                resp, err = eng.call_api(arm, req, tmpl_headers, env)
                rec = {"arm": arm, "step": step, "msg_index": i, "orig": orig}
                if err:
                    rec["error"] = err
                else:
                    rep = eng.extract_action(resp.get("content", []))
                    exact, action, sim = eng.score(orig, rep)
                    usage = resp.get("usage", {})
                    c = eng.cost_usd(usage, replay_model)
                    spent += c
                    n_ok += 1
                    n_action += action
                    n_exact += exact
                    ctx_total += (usage.get("input_tokens", 0)
                                  + usage.get("cache_read_input_tokens", 0)
                                  + usage.get("cache_creation_input_tokens", 0))
                    rec.update(replay=rep, agree_exact=exact, agree_action=action,
                               sim=round(sim, 3), usage=usage, cost_usd=round(c, 4))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                prog.update(tid, advance=1,
                            stat=f"agree {n_action}/{n_ok or 1} ${spent:.2f}")
        summary["arms"][arm] = {
            "steps_ok": n_ok, "agree_action": n_action, "agree_exact": n_exact,
            "avg_ctx_tokens": (ctx_total // n_ok) if n_ok else 0,
            "cost_usd": round(spent, 4), "file": str(path),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- verdict
def render_summary(summary: dict, console: Console) -> None:
    ctrl = summary["arms"].get("control", {})
    cn, ca = ctrl.get("steps_ok", 0), ctrl.get("agree_action", 0)
    floor = ca / cn if cn else None
    t = Table(title=f"[bold]counterfactual — {Path(summary['session']).name} "
                    f"({summary['model']}, {summary['steps']} decision points)")
    for col in ("arm", "steps", "same action", "exact", "avg ctx tokens",
                "ctx vs control", "cost", "$ vs control"):
        t.add_column(col, justify="right" if col != "arm" else "left")
    for arm, a in summary["arms"].items():
        n = a["steps_ok"]
        agree = a["agree_action"] / n if n else None
        comp = (1 - a["avg_ctx_tokens"] / ctrl["avg_ctx_tokens"]
                if arm != "control" and ctrl.get("avg_ctx_tokens") else None)
        costd = (1 - a["cost_usd"] / ctrl["cost_usd"]
                 if arm != "control" and ctrl.get("cost_usd") else None)
        t.add_row(
            arm, str(n),
            f"{agree:.0%}" if agree is not None else "—",
            f"{a['agree_exact'] / n:.0%}" if n else "—",
            f"{a['avg_ctx_tokens']:,}",
            f"{comp:+.0%}" if comp is not None else "—",
            f"${a['cost_usd']:.2f}",
            f"{costd:+.0%}" if costd is not None else "—",
        )
    console.print(t)
    if floor is not None:
        console.print(
            f"[dim]control replay agrees with the original {floor:.0%} of the time — that is the "
            f"noise floor (sampling + replay error), not 100%. An arm is only drifting to the "
            f"extent it falls meaningfully below it; a few points on <30 steps is noise.[/]")
