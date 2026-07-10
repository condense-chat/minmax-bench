"""Rich incremental replay of a *local* session — the engine behind
``minmax-bench quality incremental`` (formerly the ``counterfactual`` command).

Answers "how would my session have played out under condense/headroom?" the only way
that is honest without re-executing tools on the user's machine: teacher-forced
per-step replay. At every assistant decision point the recorded prefix is sent
through each arm's endpoint and the replayed next action is scored against what
actually happened. Control (api.anthropic.com, no compaction) is always replayed
too — it is the noise floor; an arm only "diverges" to the extent it falls below it.

This module is glue + display: session discovery under ``~/.claude/projects``,
model auto-fallback, cost preview, privacy notice, and a rich summary with the
recorded backtest anchor. It writes ``<out>/incremental/<task>-<arm>.jsonl`` so
``quality report`` can join it with full-mode runs. All replay MECHANICS live in
``minmax_bench/quality/engine.py`` (the single stdlib source of truth).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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
    """cwd + recorded model + Claude Code version from a session (first occurrence wins);
    version picks the exact binary to capture the request from."""
    cwd, model, version = None, None, None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cwd = cwd or rec.get("cwd")
            version = version or rec.get("version")
            if rec.get("type") == "assistant":
                model = model or rec.get("message", {}).get("model")
            if cwd and model and version:
                break
    return {"cwd": cwd, "model": model, "version": version}


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
           model: str | None = None, auth: str = "auto", task: str = "session",
           judge: str = "off", capture: bool = False, headroom_mode: str = "token",
           ccr: bool = True) -> dict:
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

    # a headroom arm needs a local proxy on :8787 — this replay path doesn't go through
    # generate.full's proxy management, so start (or reuse a same-mode) proxy here, else
    # every headroom probe is Connection-refused. token mode = compression that can
    # actually change the next action (the meaningful quality test; cache is ~passthrough).
    if any(a.startswith("headroom") for a in arms):
        import atexit

        from minmax_bench.quality import generate as gen
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]starting headroom proxy on :{gen.HRPORT} (mode={headroom_mode})…[/]")
        proxy = gen._start_proxy(str(out_dir), headroom_mode)  # sys.exits with a clear msg on fail
        if proxy:
            atexit.register(gen._stop_proxy, proxy)

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

    def _is_transport(e):
        return e and any(s in e for s in ("Connection refused", "URLError", "Errno 61",
                                          "Max retries", "Failed to establish"))

    bad_arm, err = _preflight(replay_model)
    if bad_arm and _is_transport(err):
        # not a model problem — a model fallback can't fix an unreachable endpoint
        console.print(f"[red]{bad_arm} is unreachable[/]: [dim]{err[:160]}[/]\n"
                      f"[yellow]connectivity issue, not a model one — check the {bad_arm} "
                      f"endpoint/proxy is up (not the model). Aborting before any spend.[/]")
        raise SystemExit(1)
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
    reminders, captured = [], None
    # FAITHFUL path: let the real version-matched CC binary build the request (exact
    # system prompt, full tool catalog, CLAUDE.md/env re-read from disk, exact config)
    # instead of approximating with a frozen template. The caller gates this on consent.
    if capture:
        console.print(f"[dim]capturing the exact request from Claude Code "
                      f"{meta.get('version') or '(newest on disk)'} in {meta['cwd']}… "
                      f"(local only — nothing is sent externally)[/]")
        captured = eng.capture_cc_request(meta["cwd"], model=replay_model,
                                          version=meta.get("version"))
        if captured:
            for k in ("system", "tools", "thinking", "context_management", "output_config"):
                if k in captured:
                    tmpl_body[k] = captured[k]
            # add stubs for any tool the session referenced that the capture didn't list
            # (e.g. an MCP server no longer configured) so the request never 400s
            extra = eng.referenced_tool_names(msgs) - {t["name"] for t in captured["tools"]}
            tmpl_body["tools"] = captured["tools"] + eng.build_tools(extra, [])
            reminders = eng.captured_reminders(captured)
            console.print(f"[green]captured[/] system prompt + "
                          f"{len(captured['tools'])} tools from the binary")
        else:
            why = getattr(eng.capture_cc_request, "last_error", "")
            console.print(f"[yellow]capture failed — falling back to the stored "
                          f"template[/]{(' [dim](' + why + ')[/]') if why else ''}")
    if not captured:
        # real template defs for CC tools; permissive stubs for mcp__/Skill/etc. Stub
        # every tool NAME the session references (incl. tool-search-discovered MCP tools
        # never directly called) — Anthropic 400s on a reference to a tool not in the array.
        tmpl_body["tools"] = eng.build_tools(eng.referenced_tool_names(msgs), tmpl_body["tools"])
    tmpl_body["model"] = replay_model
    msgs = eng.ensure_reminders(msgs, reminders)  # carry CC's injected CLAUDE.md/env context

    # recorded thinking blocks are only valid on the session's own model; the captured
    # config is already built FOR replay_model, so it needn't be dropped as cross-model.
    strip_thinking = replay_model != (meta["model"] or replay_model)
    drop_beta = False if captured else cross_model
    args = SimpleNamespace(max_tokens=max_tokens, strip_thinking=strip_thinking,
                           swechat=None, keep_all_tools=True, drop_beta_config=drop_beta)

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
        title="incremental replay", border_style="cyan"))
    if not assume_yes and not Confirm.ask("[cyan]replay it?[/]", default=False, console=console):
        raise SystemExit(0)

    # write per-arm jsonl under <out>/incremental/<task>-<arm>.jsonl (control included) so
    # `minmax-bench quality report --from <out>` can join these with full-mode runs
    inc_dir = out_dir / "incremental"
    inc_dir.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    # the session's first user text — task context for the equivalence judge
    # the first real user instruction — skip injected <system-reminder>/CLAUDE.md blocks
    # (ensure_reminders may have prepended them) so the judges get the task, not the reminders
    task_hint = next((b.get("text", "") for b in (msgs[0]["content"] if msgs else [])
                      if isinstance(b, dict) and b.get("type") == "text"
                      and not b.get("text", "").lstrip().startswith("<")), "")
    summary: dict = {"session": str(session), "model": replay_model, "steps": len(sel),
                     "judged": judge, "arms": {}}
    from minmax_bench.quality import generate as gen
    for arm in all_arms:
        path = inc_dir / f"{task}-{arm}.jsonl"
        spent, n_ok, n_action, n_exact, ctx_total = 0.0, 0, 0, 0, 0
        n_err, first_err, ctx_series = 0, None, []
        qual = {"good": 0, "degraded": 0, "bad": 0}  # goal-judge tally
        judge_spent = 0.0  # LLM-judge cost, billed into the budget and reported separately
        by_step = {}  # step-index -> {ctx, cost, agree} for common-step fair aggregation
        # CCR injection: for the headroom arm, execute headroom_retrieve calls via
        # `headroom mcp serve` so the real Compress-Cache-Retrieve loop engages (else the
        # arm is only kompress). Retrieves are counted as CCR's step overhead.
        ccr_active = ccr and arm == "headroom"
        n_retrieves = 0
        mcp = None
        with path.open("w") as f, Progress(
            TextColumn(f"[cyan]{arm:9}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total} [dim]{task.fields[stat]}[/]"),
            TimeElapsedColumn(), console=console,
        ) as prog, contextlib.ExitStack() as stack:
            if ccr_active:
                # ExitStack closes the mcp serve subprocess on ANY exit (incl. exception)
                mcp = stack.enter_context(eng.HeadroomMCP(
                    f"http://127.0.0.1:{gen.HRPORT}", str(out_dir / "headroom-mcp.log")))
                console.print("[dim]headroom CCR: retrieve loop active via mcp serve[/]" if mcp.ok
                              else "[yellow]headroom CCR: mcp serve unavailable — arm runs as "
                                   "kompress (no retrieve)[/]")
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
                ccr_overhead = 0.0
                if ccr_active and mcp and mcp.ok:
                    resp, err, nr, ccr_overhead = eng.ccr_step(arm, tmpl_body, msgs[:i], args,
                                                               sid, tmpl_headers, env, mcp)
                    n_retrieves += nr
                else:
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
                    # score against the session's REAL cwd so `cd <proj> && …` artifacts
                    # normalize (local sessions run in their project dir, not /app)
                    exact, action, sim = eng.score(orig, rep, cwd=meta["cwd"] or "/app")
                    # semantic agreement (judge=equivalence): an LLM upgrades a structural
                    # near-miss that is a functionally-equivalent decision (grep vs rg). Only
                    # judge disagreements — matches already agree.
                    agree_sem = action
                    if judge == "equivalence" and not action:
                        eq, jc = eng.judge_equivalent(orig, rep, env, task_hint)
                        agree_sem = bool(eq)
                        judge_spent += jc
                        spent += jc
                    usage = resp.get("usage", {})
                    # bill the CCR retrieve round-trips as part of this decision's cost —
                    # else headroom-CCR looks artificially cheap (it pays for extra calls)
                    c = eng.cost_usd(usage, replay_model) + ccr_overhead
                    spent += c
                    n_ok += 1
                    n_action += action
                    n_exact += exact
                    step_ctx = (usage.get("input_tokens", 0)
                                + usage.get("cache_read_input_tokens", 0)
                                + usage.get("cache_creation_input_tokens", 0))
                    ctx_total += step_ctx
                    ctx_series.append(step_ctx)
                    # per-step, keyed by the shared decision-point index so arm-vs-control
                    # deltas can be computed over the COMMON steps all arms reached (an arm
                    # that hits its budget early must not be compared over a longer step set)
                    by_step[step] = {"ctx": step_ctx, "cost": round(c, 4), "agree": bool(action)}
                    rec.update(replay=rep, agree_exact=exact, agree_action=action,
                               agree_semantic=agree_sem, sim=round(sim, 3),
                               usage=usage, cost_usd=round(c, 4))
                    # goal-based per-step quality: is the arm's action a good/degraded/bad
                    # step toward the TASK, on its own merits (robust to valid divergence)
                    if judge == "goal":
                        # hybrid: a TEXT reply is judged against the ORIGINAL's text (the
                        # reference answer — same conclusion = good); a TOOL action is judged
                        # on its own merits toward the task (robust to valid path divergence)
                        if rep.get("type") == "text" and orig.get("type") == "text":
                            q, jc = eng.judge_text_match(orig, rep, task_hint, env)
                        else:
                            q, jc = eng.judge_action_quality(
                                task_hint, eng.recent_context(msgs, i), rep, env)
                        judge_spent += jc
                        spent += jc
                        rec["quality"] = q
                        by_step[step]["quality"] = q
                        if q in qual:
                            qual[q] += 1
                if ccr_active and mcp and mcp.ok:
                    rec["ccr_retrieves"] = nr
                f.write(json.dumps(rec) + "\n")
                f.flush()
                prog.update(tid, advance=1,
                            stat=f"agree {n_action}/{n_ok or 1} ${spent:.2f}")
        summary["arms"][arm] = {
            "steps_ok": n_ok, "agree_action": n_action, "agree_exact": n_exact,
            "errors": n_err, "first_error": (first_err or "")[:300],
            "avg_ctx_tokens": (ctx_total // n_ok) if n_ok else 0,
            "ctx_series": ctx_series, "quality": qual, "by_step": by_step,
            "ccr_retrieves": n_retrieves, "ccr": bool(ccr_active and mcp and mcp.ok),
            "judge_usd": round(judge_spent, 4),
            # cost_usd is the arm's REPLAY spend only (judge cost is a measurement overhead,
            # reported separately) so the arm-vs-control $ comparison stays clean; the budget
            # cap during the run used spent = replay + judge.
            "cost_usd": round(spent - judge_spent, 4), "file": str(path),
        }
    summary["judge"] = judge
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
def _goal_counts(a: dict, common: set):
    """good/degraded/bad tally over the common steps (fair across arms that stopped at
    different depths); falls back to the arm's whole tally if per-step data is absent."""
    bs = a.get("by_step", {})
    per_step = [bs[s].get("quality") for s in common if s in bs and "quality" in bs[s]]
    if per_step:
        return {k: per_step.count(k) for k in ("good", "degraded", "bad")}
    return a.get("quality", {})


def _render_goal_quality(summary: dict, console: Console, common: set) -> None:
    """Goal-based per-step quality — the HEADLINE when judge=goal. Each action is rated
    good/degraded/bad as a step toward the task (on its own merits), so control ≈100% and
    an arm 'loses' only by taking worse steps. Robust to the valid-divergence noise that
    dominates same-action agreement. Tallies use the common steps for a fair comparison."""
    cq = _goal_counts(summary["arms"].get("control", {}), common)
    ctot = sum(cq.values())
    cgood = cq.get("good", 0) / ctot if ctot else None
    gt = Table(title="[bold]goal-based per-step quality — is each action a good step "
                     "toward the task?")
    for c in ("arm", "good", "degraded", "bad", "good rate"):
        gt.add_column(c, justify="right" if c != "arm" else "left")
    for arm, a in summary["arms"].items():
        q = _goal_counts(a, common)
        tot = sum(q.values())
        if not tot:
            continue
        gp = q.get("good", 0) / tot
        cell = (f"[dim]{gp:.0%} (floor)[/]" if arm == "control" else
                f"[green]{gp:.0%}[/]" if cgood is None or gp >= cgood - 0.05
                else f"[red]{gp:.0%}[/]")
        gt.add_row(arm, str(q.get("good", 0)), str(q.get("degraded", 0)),
                   str(q.get("bad", 0)), cell)
    console.print(gt)
    if cgood is not None:
        for arm, a in summary["arms"].items():
            if arm == "control":
                continue
            q = _goal_counts(a, common)
            tot = sum(q.values())
            if not tot:
                continue
            gp, d = q.get("good", 0) / tot, q.get("good", 0) / tot - cgood
            verdict = ("[green]≈ control → compaction does not degrade decisions[/]"
                       if d >= -0.05 else
                       f"[red]{d:+.0%} vs control → compaction is degrading decisions[/]")
            console.print(f"[bold]{arm}[/]  good-step rate [bold]{gp:.0%}[/]  "
                          f"vs control {cgood:.0%}   →   {verdict}")
    console.print("[dim]goal-based: each action judged a valid step toward the task on its own "
                  "merits (not vs the original) — control scores ~100%, so an arm only 'loses' by "
                  "taking WORSE steps. More robust than same-action agreement.[/]")


def _common_steps(arms: dict) -> set:
    """The decision-point indices EVERY arm reached. An arm that hit its budget (or
    error-bailed) early stopped short; comparing cost/ctx/agreement over a longer step
    set than the shortest arm is unfair, so all vs-control deltas use this intersection."""
    sets = [set(a.get("by_step", {})) for a in arms.values() if a.get("by_step")]
    return set.intersection(*sets) if sets else set()


def _over(a: dict, common: set):
    """An arm's agreement / avg-ctx / total-cost restricted to the common steps."""
    steps = [a["by_step"][s] for s in common if s in a.get("by_step", {})]
    n = len(steps)
    if not n:
        return None
    return {"n": n, "agree": sum(x["agree"] for x in steps) / n,
            "ctx": sum(x["ctx"] for x in steps) / n, "cost": sum(x["cost"] for x in steps)}


def render_summary(summary: dict, console: Console) -> None:
    ctrl = summary["arms"].get("control", {})
    common = _common_steps(summary["arms"])
    ctrl_c = _over(ctrl, common)
    floor = (ctrl_c["agree"] if ctrl_c else
             ctrl.get("agree_action", 0) / ctrl["steps_ok"] if ctrl.get("steps_ok") else None)
    # the comparison window is capped at the fewest steps any arm reached — an arm that
    # hits its budget stops there, and comparing past that point is apples-to-oranges
    reaches = {arm: a.get("steps_ok", 0) for arm, a in summary["arms"].items() if a.get("by_step")}
    cap_n = len(common)
    capper = min(reaches, key=reaches.get) if reaches else None
    capped = capper and len(set(reaches.values())) > 1
    cap_txt = f", where {capper} hit its budget" if capped else ""
    t = Table(title=f"[bold]counterfactual — {Path(summary['session']).name} "
                    f"({summary['model']}, {summary['steps']} decision points; "
                    f"all deltas over the first {cap_n} steps{cap_txt})")
    for col in ("arm", "reached", "errors", "vs original", "exact", "avg ctx tokens",
                "ctx vs control", "cost", "$ vs control"):
        t.add_column(col, justify="right" if col != "arm" else "left")
    rec = summary.get("recorded")
    if rec:
        t.add_row("[dim]recorded*[/]", f"[dim]{rec['steps']}[/]", "[dim]—[/]", "[dim]—[/]",
                  "[dim]—[/]", f"[dim]{rec['avg_ctx_tokens']:,}[/]", "[dim]—[/]",
                  f"[dim]${rec['cost_usd']:.2f}[/]", "[dim]—[/]")
    for arm, a in summary["arms"].items():
        n = a["steps_ok"]
        # vs-original agreement and the vs-control deltas are all computed over the COMMON
        # steps (fair); the 'steps' column still shows each arm's OWN reach so you can see
        # who stopped early. avg ctx / cost columns are each arm's own totals (context).
        ac = _over(a, common)
        if ac and ctrl_c:  # fair: common steps
            agree = ac["agree"]
            comp = 1 - ac["ctx"] / ctrl_c["ctx"] if arm != "control" and ctrl_c["ctx"] else None
            costd = 1 - ac["cost"] / ctrl_c["cost"] if arm != "control" and ctrl_c["cost"] else None
        else:  # fallback: no by_step (old artifact) -> each arm's own aggregates
            agree = a["agree_action"] / n if n else None
            comp = (1 - a["avg_ctx_tokens"] / ctrl["avg_ctx_tokens"]
                    if arm != "control" and ctrl.get("avg_ctx_tokens") else None)
            costd = (1 - a["cost_usd"] / ctrl["cost_usd"]
                     if arm != "control" and ctrl.get("cost_usd") else None)
        errs = a.get("errors", 0)
        # colour each arm's vs-original agreement AGAINST the control floor: at/above the
        # floor is green (no measurable loss), below is red. control itself is the floor.
        if agree is None:
            agree_cell = "—"
        elif arm == "control":
            agree_cell = f"[dim]{agree:.0%} (floor)[/]"
        elif floor is not None:
            agree_cell = (f"[green]{agree:.0%}[/]" if agree >= floor - 0.02
                          else f"[red]{agree:.0%}[/]")
        else:
            agree_cell = f"{agree:.0%}"
        # ctx/cost cells use the common-step aggregates when available, so they reconcile
        # with the deltas beside them (else an arm that ran longer shows a bigger total that
        # contradicts a per-step saving). The 'steps' column shows the arm's own reach.
        ctx_cell = f"{round(ac['ctx']):,}" if ac else f"{a['avg_ctx_tokens']:,}"
        cost_cell = f"${ac['cost']:.2f}" if ac else f"${a['cost_usd']:.2f}"
        # 'reached' = how far this arm got; mark the arm that capped the window (hit budget)
        reached_cell = (f"[yellow]{n} ◀ budget cap[/]"
                        if capper == arm and len(set(reaches.values())) > 1 else str(n))
        t.add_row(
            arm, reached_cell, f"[red]{errs}[/]" if errs else "0",
            agree_cell,
            f"{a['agree_exact'] / n:.0%}" if n else "—",
            ctx_cell,
            f"{comp:+.0%}" if comp is not None else "—",
            cost_cell,
            f"{costd:+.0%}" if costd is not None else "—",
        )
    console.print(t)
    # if arms stopped at different depths (an arm hit its budget / error-bailed), say so —
    # the deltas above use the common steps, but the reader should see who stopped short
    reach = {arm: a.get("steps_ok", 0) for arm, a in summary["arms"].items() if a.get("by_step")}
    if reach and len(set(reach.values())) > 1:
        short = min(reach, key=reach.get)
        console.print(f"[yellow]note:[/] arms stopped at different depths "
                      f"({', '.join(f'{k} {v}' for k, v in reach.items())}) — likely a budget/"
                      f"error cap on [bold]{short}[/]. All vs-control deltas use the "
                      f"{len(common)} steps every arm reached, so they stay apples-to-apples.")
    for arm, a in summary["arms"].items():
        if arm != "headroom" or not a.get("by_step"):
            continue
        if a.get("ccr"):
            nr = a.get("ccr_retrieves", 0)
            console.print(f"[cyan]headroom: CCR retrieve loop engaged[/] — "
                          f"{nr} headroom_retrieve call(s) executed via mcp serve, "
                          f"counted as steps and billed into cost (CCR's overhead).")
        else:
            console.print("[yellow]headroom ran as kompress[/] (retrieve loop unavailable) — "
                          "compression without retrieval. For headroom with CCR, use full mode "
                          "(`quality run`), or check the mcp serve log.")
    judge_total = sum(a.get("judge_usd", 0.0) for a in summary["arms"].values())
    if judge_total:
        console.print(f"[dim]LLM judge ({summary.get('judge')}): [bold]${judge_total:.2f}[/] "
                      f"total across arms (haiku, per step) — billed against the budget cap, "
                      f"excluded from the arm cost columns above.[/]")
    if summary.get("judge") == "goal":
        _render_goal_quality(summary, console, common)
    if floor is not None:
        label = "(secondary — structural) " if summary.get("judge") == "goal" else ""
        console.print(f"[dim]{label}control replays the SAME session with NO compaction; its "
                      f"{floor:.0%} agreement with the original is the NOISE FLOOR (sampling + "
                      f"free-running divergence), not 100%. An arm only loses the trajectory to "
                      f"the extent it falls below it; a few points on <30 steps is noise.[/]")
    ctrl_series = ctrl.get("ctx_series") or []
    for arm, a in summary["arms"].items():
        if arm == "control":
            continue
        if not a.get("steps_ok"):
            console.print(f"[red]{arm} FAILED every step ({a.get('errors', 0)} errors)[/] — "
                          f"no verdict is possible. First error: {a.get('first_error', '?')}")
            continue
        # headline verdict: the arm's vs-original agreement against the control floor
        _ac = _over(a, common)
        agree = _ac["agree"] if _ac else a["agree_action"] / a["steps_ok"]
        if floor is not None:
            delta = agree - floor
            if delta >= -0.02:
                verdict = ("[green]at/above the noise floor → no measurable trajectory loss[/]")
            else:
                verdict = (f"[red]{delta:+.0%} below the floor → possible trajectory loss[/]")
            console.print(f"[bold]{arm}[/]  same action vs original [bold]{agree:.0%}[/]  "
                          f"vs control floor {floor:.0%}   →   {verdict}")
        if not ctrl_series:
            continue
        series = a.get("ctx_series") or []
        pairs = list(zip(series, ctrl_series, strict=False))
        onset = next((i for i, (x, c) in enumerate(pairs) if c and x < c * 0.9), None)
        if onset is None:
            console.print(
                f"[dim]  └ {arm} never compacted this session (context matched control at "
                f"every step) — a passthrough below its trigger, so the delta is sampling "
                f"noise, not a compaction effect.[/]")
        else:
            deepest = max((1 - x / c) for x, c in pairs[onset:] if c)
            console.print(
                f"[dim]  └ {arm} compacted from step {onset} (context −{deepest:.0%} vs "
                f"control at the deepest point) — the delta from step {onset} on is real "
                f"compaction.[/]")
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


# --------------------------------------------------------------------------- per-step view
_CAT_STYLE = {"good": ("green", "✓ good"), "semi": ("yellow", "◐ semi"),
              "bad": ("red", "✗ bad"), "err": ("dim", "· err")}
_REDUND_RE = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl|grep|rg)\b")


def _digest(a):
    """Compact one-line label for an action."""
    if not a or a.get("type") == "text":
        return "answer: " + " ".join((a or {}).get("text", "").split())[:46]
    inp = a.get("input", {})
    tgt = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
    tgt = " ".join(str(tgt).split())
    return f"{a.get('name', '?')}({tgt})"[:52]


def _mark_seen(a, files, cmds):
    if not a or a.get("type") != "tool_use":
        return
    inp = a.get("input", {})
    if inp.get("file_path"):
        files.add(inp["file_path"])
    if a.get("name") == "Bash" and inp.get("command"):
        cmds.add(" ".join(inp["command"].split()))


def _is_refetch(a, files, cmds):
    """Does this replayed action re-fetch info an earlier ORIGINAL step already had?
    (compaction amnesia — the ↻ marker from the old trajectory viz)."""
    if not a or a.get("type") != "tool_use":
        return False
    inp = a.get("input", {})
    if a.get("name") == "Read" and inp.get("file_path") in files:
        return True
    if a.get("name") == "Bash":
        cmd = " ".join(inp.get("command", "").split())
        if cmd in cmds:
            return True
        if _REDUND_RE.search(cmd) and any(f and f in cmd for f in files):
            return True
    return False


def _step_verdict(rec):
    if "error" in rec or not rec.get("replay"):
        return "err"
    # goal-based verdict (judge=goal) takes precedence — it rates the action's quality
    # toward the task, not agreement with the original
    q = rec.get("quality")
    if q:
        return {"good": "good", "degraded": "semi", "bad": "bad"}.get(q, "bad")
    if rec.get("agree_semantic") or rec.get("agree_exact") or rec.get("agree_action"):
        return "good"
    o, p = rec.get("orig", {}), rec["replay"]
    same_tool = o.get("type") == p.get("type") and o.get("name") == p.get("name")
    return "semi" if (same_tool and rec.get("sim", 0) >= 0.5) else "bad"


def render_steps(summary: dict, console: Console, max_rows: int = 60) -> None:
    """Per-step readout for each non-control arm: good / semi / bad / ↻ redundant,
    the terminal version of the trajectory viz. Reads the arm's stored jsonl."""
    for arm, meta in summary["arms"].items():
        if arm == "control":
            continue
        try:
            recs = [json.loads(line) for line in open(meta["file"])]
        except OSError:
            continue
        files, cmds, cats = set(), set(), {"good": 0, "semi": 0, "bad": 0, "err": 0}
        rows, redund = [], 0
        for r in recs:
            cat = _step_verdict(r)
            cats[cat] = cats.get(cat, 0) + 1
            rd = _is_refetch(r.get("replay"), files, cmds)
            redund += rd
            rows.append((r["step"], r.get("orig", {}), r.get("replay"), cat, rd))
            _mark_seen(r.get("orig"), files, cmds)  # the real history the arm should recall

        mode = {"goal": "goal-based quality", "equivalence": "structural + LLM equivalence",
                "off": "structural match"}.get(summary.get("judge"), "structural match")
        anchor = "toward the task" if summary.get("judge") == "goal" else "vs original"
        title = f"[bold]{arm} — per step[/] ({anchor}; {mode})"
        if len(rows) <= max_rows:
            t = Table(title=title)
            t.add_column("step", justify="right", style="dim")
            t.add_column("original", max_width=52)
            t.add_column("replayed", max_width=52)
            t.add_column("verdict")
            for step, o, p, cat, rd in rows:
                style, label = _CAT_STYLE[cat]
                mark = "  [magenta]↻ redundant[/]" if rd else ""
                pdig = "[dim]—[/]" if cat == "err" else _digest(p)
                t.add_row(str(step), _digest(o), pdig,
                          f"[{style}]{label}[/]{mark}")
            console.print(t)
        else:  # compact glyph strip for long sessions
            strip = "".join(f"[{_CAT_STYLE[c][0]}]" + ("↻" if rd else _CAT_STYLE[c][1][0])
                            + "[/]" for _s, _o, _p, c, rd in rows)
            console.print(f"{title}\n{strip}")
        ok = cats["good"] + cats["semi"] + cats["bad"]
        pct = f" ({cats['good'] / ok:.0%} good)" if ok else ""
        console.print(f"[dim]{arm}:[/] [green]{cats['good']} good[/] · "
                      f"[yellow]{cats['semi']} semi[/] · [red]{cats['bad']} bad[/] · "
                      f"[magenta]{redund} ↻ redundant[/]{pct}")
