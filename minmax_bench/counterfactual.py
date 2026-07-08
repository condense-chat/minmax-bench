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
from minmax_bench.quality.engine import recorded_usage  # noqa: F401  (re-export)

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
    peak_ctx: int         # peak per-request context tokens the session actually used


# cap the per-file read that computes peak context, so a multi-GB transcript
# doesn't stall the picker; capped reads mark the estimate with a '~'
_PEEK_BYTE_CAP = 12_000_000


def _peek(path: Path, max_lines: int = 300) -> tuple[str, str | None, bool, int, bool]:
    """(prompt, cwd, has_assistant, peak_ctx, capped) from a session file.

    peak_ctx is the largest per-request context (input + cache read + cache write)
    across the session's assistant turns — the number that decides whether a replay
    would ever cross a compaction threshold. Only lines containing a usage block are
    parsed, and the read stops at _PEEK_BYTE_CAP so giant transcripts stay cheap.
    """
    prompt, cwd, has_assistant, peak, capped, read = "", None, False, 0, False, 0
    try:
        with path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                read += len(line)
                if read > _PEEK_BYTE_CAP:
                    capped = True
                    break
                head = n < max_lines
                if '"usage"' not in line and not head:
                    continue  # cheap skip: only usage lines (and the head) need parsing
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("isSidechain"):
                    return "", None, False, 0, False  # sub-agent transcript
                if head:
                    cwd = cwd or rec.get("cwd")
                    if rec.get("type") == "assistant":
                        has_assistant = True
                    if rec.get("type") == "user" and not prompt:
                        c = rec.get("message", {}).get("content")
                        if isinstance(c, str):
                            text = c
                        elif isinstance(c, list):
                            text = next((b.get("text", "") for b in c
                                         if isinstance(b, dict) and b.get("type") == "text"), "")
                        else:
                            text = ""
                        if text and not text.lstrip().startswith("<"):
                            prompt = text
                u = rec.get("message", {}).get("usage") if rec.get("type") == "assistant" else None
                if isinstance(u, dict):
                    has_assistant = True
                    peak = max(peak, u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                               + u.get("cache_creation_input_tokens", 0))
    except OSError:
        return "", None, False, 0, False
    return " ".join(prompt.split())[:90], cwd, has_assistant, peak, capped


def scan_sessions(root: Path = CC_PROJECTS, limit: int = 25) -> list[LocalSession]:
    """Most-recent real sessions (skips sub-agent sidechains and empty files)."""
    found: list[LocalSession] = []
    for f in sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(found) >= limit:
            break
        prompt, cwd, has_assistant, peak, capped = _peek(f)
        if not has_assistant:
            continue
        found.append(LocalSession(
            path=f, project=cwd or f.parent.name,  # cwd is exact; the dir name is a lossy slug
            mtime=f.stat().st_mtime, size=f.stat().st_size, prompt=prompt, cwd=cwd,
            peak_ctx=peak if not capped else -peak,  # negative marks a capped (lower-bound) read
        ))
    return found


def _fmt_ctx(peak: int) -> str:
    """Peak context tokens for the picker; a capped read (peak < 0) shows '≥'."""
    v = abs(peak)
    body = f"{v / 1000:.0f}k" if v >= 1000 else str(v)
    return (">" + body) if peak < 0 else body


def pick_session(console: Console) -> Path:
    sessions = scan_sessions()
    if not sessions:
        console.print(f"[red]no Claude Code sessions found under {CC_PROJECTS}[/]")
        raise SystemExit(1)
    t = Table(title="[bold]local Claude Code sessions — newest first")
    t.add_column("#", justify="right", style="dim")
    t.add_column("when", style="dim")
    t.add_column("project")
    t.add_column("peak ctx", justify="right")  # peak per-request context tokens actually used
    t.add_column("first prompt", max_width=52)
    for i, s in enumerate(sessions, 1):
        t.add_row(str(i), datetime.fromtimestamp(s.mtime).strftime("%m-%d %H:%M"),
                  ("…" + s.project[-38:]) if len(s.project) > 39 else s.project,
                  _fmt_ctx(s.peak_ctx), s.prompt or "[dim](no text prompt)[/]")
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
           model: str | None = None, auth: str = "auto") -> dict:
    env = {**eng.load_env(str(REPO_ROOT / ".env")), **dict(os.environ)}
    if auth == "subscription":
        # force the Claude Code login path even when an API key is configured
        # (.env or environment) — this is how you TEST the no-API-key experience
        env.pop("ANTHROPIC_API_KEY", None)
    elif auth == "api-key" and not env.get("ANTHROPIC_API_KEY"):
        console.print("[red]--auth api-key but no ANTHROPIC_API_KEY configured[/]")
        raise SystemExit(1)

    if "control" in arms:
        console.print("[red]control is always replayed — pass only the arms to compare[/]")
        raise SystemExit(1)
    problems = eng.check_arms(["control"] + arms, env)
    if problems:
        for p in problems:
            console.print(f"[red]{p}[/]")
        raise SystemExit(1)
    mode = eng.auth_mode(env)
    if mode == "subscription":
        console.print("[dim]auth: Claude Code subscription (no API key configured — "
                      "replay usage draws on your plan)[/]")

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

    # replay on the session's own model (auto-detected) so recorded thinking
    # signatures stay valid; on any other model, thinking is stripped + beta config dropped.
    replay_model = model or meta["model"] or tmpl_body["model"]
    template_model = tmpl_body["model"]
    all_arms = ["control"] + arms

    # preflight every arm with a ~30-token probe BEFORE any real spend: a broken arm
    # (bad key, gateway incompatibility with the replay model, dead proxy) must abort
    # here, not after the control arm has already burned the budget. If an arm rejects
    # the session's model, auto-detect a fallback that every arm can serve.
    def _preflight(m):
        probe = {"model": m, "max_tokens": 16, "stream": True,
                 "messages": [{"role": "user", "content": "Say ok."}]}
        for arm in all_arms:
            _, err = eng.call_api(arm, probe, tmpl_headers, env)
            if err:
                return arm, err
        return None, None

    bad_arm, err = _preflight(replay_model)
    if bad_arm:
        console.print(f"[yellow]{bad_arm} can't serve {replay_model}[/] (the session's own "
                      f"model): [dim]{err[:200]}[/]")
        fallbacks = [m for m in (template_model, "claude-sonnet-4-6") if m != replay_model]
        fb = next((m for m in dict.fromkeys(fallbacks) if _preflight(m) == (None, None)), None)
        if not fb:
            console.print("[red]no fallback model passed preflight for every arm — aborting "
                          "before any replay spend (try --model explicitly).[/]")
            raise SystemExit(1)
        console.print(f"[yellow]falling back to {fb} for ALL arms[/] — the comparison stays "
                      f"paired; recorded thinking blocks will be stripped (cross-model).")
        if not assume_yes and not Confirm.ask(f"[cyan]replay on {fb}?[/]", default=True,
                                              console=console):
            raise SystemExit(0)
        replay_model = fb
    else:
        console.print(f"[dim]preflight ok: {', '.join(all_arms)} respond on {replay_model}[/]")

    cross_model = replay_model != template_model
    tmpl_body["model"] = replay_model

    # real template defs for CC tools; permissive stubs for mcp__/Skill/etc. Stub
    # every tool NAME the session references (incl. tool-search-discovered MCP tools
    # never directly called), not just the ones actually used — Anthropic 400s on a
    # reference to a tool absent from the array.
    tmpl_body["tools"] = eng.build_tools(eng.referenced_tool_names(msgs), tmpl_body["tools"])

    args = SimpleNamespace(max_tokens=max_tokens, strip_thinking=cross_model,
                           swechat=None, keep_all_tools=True, drop_beta_config=cross_model)

    sel = points[:: max(1, every)]
    if limit:
        sel = sel[:limit]
    lo, hi = estimate_usd(msgs, [i for i in sel], eng.rates_for(replay_model))

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
        n_err, first_err, ctx_series = 0, None, []
        with path.open("w") as f, Progress(
            TextColumn(f"[cyan]{arm:9}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total} [dim]{task.fields[stat]}[/]"),
            TimeElapsedColumn(), console=console,
        ) as prog:
            tid = prog.add_task("", total=len(sel), stat="")
            consec_err = 0
            for step, i in enumerate(sel):
                if spent >= budget_usd:
                    prog.update(tid, stat=f"budget ${budget_usd} reached")
                    break
                if consec_err >= 5:
                    prog.update(tid, stat=f"aborted: {consec_err} consecutive errors")
                    break
                orig = eng.extract_action(msgs[i]["content"])
                req = eng.build_request(tmpl_body, msgs[:i], args, sid)
                resp, err = eng.call_api(arm, req, tmpl_headers, env)
                rec = {"arm": arm, "step": step, "msg_index": i, "orig": orig}
                if err:
                    rec["error"] = err
                    n_err += 1
                    consec_err += 1
                    first_err = first_err or err
                else:
                    consec_err = 0
                    rep = eng.extract_action(resp.get("content", []))
                    exact, action, sim = eng.score(orig, rep)
                    usage = resp.get("usage", {})
                    c = eng.cost_usd(usage, replay_model)
                    spent += c
                    n_ok += 1
                    n_action += action
                    n_exact += exact
                    step_ctx = (usage.get("input_tokens", 0)
                                + usage.get("cache_read_input_tokens", 0)
                                + usage.get("cache_creation_input_tokens", 0))
                    ctx_total += step_ctx
                    ctx_series.append(step_ctx)
                    rec.update(replay=rep, agree_exact=exact, agree_action=action,
                               sim=round(sim, 3), usage=usage, cost_usd=round(c, 4))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                prog.update(tid, advance=1,
                            stat=f"agree {n_action}/{n_ok or 1} ${spent:.2f}")
        summary["arms"][arm] = {
            "steps_ok": n_ok, "agree_action": n_action, "agree_exact": n_exact,
            "errors": n_err, "first_error": (first_err or "")[:300],
            "avg_ctx_tokens": (ctx_total // n_ok) if n_ok else 0,
            "ctx_series": ctx_series,
            "cost_usd": round(spent, 4), "file": str(path),
        }
    # backtest anchor: what these same turns ACTUALLY consumed when the session ran
    # (recorded usage, the session's own model + live caching) — replays above re-ran
    # them fresh, so this row is the "you actually paid" reference point
    rec = recorded_usage(session)
    pos = {i: k for k, i in enumerate(points)}
    ks = [pos[i] for i in sel if pos[i] < len(rec)]
    if ks:
        ctx = sum(rec[k].get("input_tokens", 0) + rec[k].get("cache_read_input_tokens", 0)
                  + rec[k].get("cache_creation_input_tokens", 0) for k in ks)
        summary["recorded"] = {
            "steps": len(ks), "avg_ctx_tokens": ctx // len(ks),
            "cost_usd": round(sum(eng.cost_usd(rec[k], meta["model"]) for k in ks), 4),
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
    for col in ("arm", "steps", "errors", "same action", "exact", "avg ctx tokens",
                "ctx vs control", "cost", "$ vs control"):
        t.add_column(col, justify="right" if col != "arm" else "left")
    rec = summary.get("recorded")
    if rec:
        t.add_row("[dim]recorded*[/]", f"[dim]{rec['steps']}[/]", "[dim]—[/]", "[dim]—[/]",
                  "[dim]—[/]", f"[dim]{rec['avg_ctx_tokens']:,}[/]", "[dim]—[/]",
                  f"[dim]${rec['cost_usd']:.2f}[/]", "[dim]—[/]")
    for arm, a in summary["arms"].items():
        n = a["steps_ok"]
        agree = a["agree_action"] / n if n else None
        comp = (1 - a["avg_ctx_tokens"] / ctrl["avg_ctx_tokens"]
                if arm != "control" and ctrl.get("avg_ctx_tokens") else None)
        costd = (1 - a["cost_usd"] / ctrl["cost_usd"]
                 if arm != "control" and ctrl.get("cost_usd") else None)
        errs = a.get("errors", 0)
        t.add_row(
            arm, str(n), f"[red]{errs}[/]" if errs else "0",
            f"{agree:.0%}" if agree is not None else "—",
            f"{a['agree_exact'] / n:.0%}" if n else "—",
            f"{a['avg_ctx_tokens']:,}",
            f"{comp:+.0%}" if comp is not None else "—",
            f"${a['cost_usd']:.2f}",
            f"{costd:+.0%}" if costd is not None else "—",
        )
    console.print(t)
    ctrl_series = ctrl.get("ctx_series") or []
    for arm, a in summary["arms"].items():
        if arm == "control":
            continue
        if not a.get("steps_ok"):
            console.print(f"[red]{arm} FAILED every step ({a.get('errors', 0)} errors)[/] — "
                          f"no verdict is possible. First error: {a.get('first_error', '?')}")
            continue
        if not ctrl_series:
            continue
        series = a.get("ctx_series") or []
        pairs = list(zip(series, ctrl_series, strict=False))
        onset = next((i for i, (x, c) in enumerate(pairs) if c and x < c * 0.9), None)
        if onset is None:
            console.print(
                f"[yellow]{arm} never compacted this session[/] — its context matched "
                f"control at every replayed step, so it acted as a passthrough (session "
                f"below its trigger). Agreement/cost deltas above are sampling noise, "
                f"not a compaction effect.")
        else:
            deepest = max((1 - x / c) for x, c in pairs[onset:] if c)
            console.print(
                f"[green]{arm} compacted from step {onset}[/] "
                f"(context −{deepest:.0%} vs control at the deepest point) — fidelity and "
                f"$ deltas from step {onset} on are measuring real compaction.")
    if rec:
        console.print(
            "[dim]* recorded = what these turns actually consumed when the session ran (its own "
            "model + live caching); the replay rows re-ran the same turns fresh, so compare arms "
            "to control for the counterfactual and to recorded for the backtest anchor.[/]")
        # replays reconstruct only the tools the session references; the original CC
        # requests also carried the full system prompt + available tool/MCP catalog
        # (mostly unused). When that overhead is large the recorded row dwarfs the
        # replay rows in absolute terms — flag it so the anchor isn't misread.
        rc = rec.get("avg_ctx_tokens", 0)
        cc = ctrl.get("avg_ctx_tokens", 0)
        if rc and cc and cc < 0.75 * rc:
            console.print(
                f"[yellow]note:[/] replay context ({cc:,} avg) is well below recorded "
                f"({rc:,}) — this session's original requests carried a large fixed "
                f"overhead (system prompt + full tool/MCP catalog) the replay can't "
                f"reconstruct. The arm-vs-control comparison is unaffected, but don't "
                f"read absolute cost against the recorded anchor here.")
    if floor is not None:
        console.print(
            f"[dim]control replay agrees with the original {floor:.0%} of the time — that is the "
            f"noise floor (sampling + replay error), not 100%. An arm is only drifting to the "
            f"extent it falls meaningfully below it; a few points on <30 steps is noise.[/]")
