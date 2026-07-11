#!/usr/bin/env python3
"""report.py — DISPLAY only. Reads what `generate.py` produced and renders it. Never spends.

Mirrors the sibling cost bench's `report` ("recompute from stored data, verify without
re-spending"):
generation writes artifacts; this reads them and computes the offline, deterministic views.

Reads from a results root (`--from`):
  - full run dirs      <root>/**/<arm>-<task>/*/*/{verifier/reward.txt, agent/sessions/.../*.jsonl}
                       plus <arm>-<task>/attempted.json (trials REQUESTED — killed/crashed trials
                       write no reward.txt; counting them as failures avoids survivorship bias)
  - milestones.json    anywhere under <root> (merged; produced by `generate --milestones`)
  - incremental jsonl  <root>/**/incremental/<task>-<arm>.jsonl  (`generate --mode incremental`)

Offline axes (always, no model calls): length / rework / solve, each vs the vanilla noise floor
(OK = bands overlap, DIVERGES = disjoint; needs ≥2 runs/arm). Milestone + incremental axes appear
only if their artifacts exist. Verdicts are deliberately coarse: with k≈3 runs, band overlap can
only catch GROSS divergence — "OK" means "no detectable divergence at this k", not equivalence.

  python3 scripts/report.py --from runs/quality-sample --tasks kv-store-grpc
  python3 scripts/report.py --from results/jobs --tasks a,b --arms condense,headroom --format md
"""
import argparse
import glob
import json
import os
import re

from rich.console import Console
from rich.table import Table

# one session parser for the spend side (generate/engine) and the display side.
from minmax_bench.quality.engine import (
    SESSION_GLOB,
    extract_action,
    parse_session,
    recorded_usage,
    resolve_tasks,
)

AGENT_SESSION_GLOB = {  # only claude-code is wired; others are TODO
    "claude-code": SESSION_GLOB,
}


def actions(path):
    msgs, points = parse_session(path)
    return [extract_action(msgs[i]["content"]) for i in points]


# ---------------------------------------------------------------- offline metrics
def _read_span(inp):
    off = inp.get("offset")
    start = off if isinstance(off, int) and off > 0 else 1
    lim = inp.get("limit")
    return start, (start + lim if isinstance(lim, int) and lim > 0 else float("inf"))


def _covered(spans, s, e):
    cur = s
    for a, b in sorted(spans):
        if a > cur:
            break
        cur = max(cur, b)
        if cur >= e:
            return True
    return cur >= e


def rework_count(acts):
    """Redundant re-fetches: re-read of an already-seen file span (range-aware), re-cat, re-run.

    A Write/Edit invalidates everything known about that file — re-reading, re-catting, or
    re-running a read-only command that touches it afterwards is VERIFICATION, not rework.
    Counting it would penalize verify-heavy behavior (which some compaction methods induce).
    """
    CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
    RO = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc)\b")
    read_spans, last_read, seen, hits = {}, {}, {}, 0
    for a in acts:
        if a.get("type") != "tool_use":
            continue
        name, inp = a.get("name"), a.get("input", {})
        if name in ("Write", "Edit"):
            fp = inp.get("file_path")
            if fp:
                read_spans[fp] = []
                last_read.pop(fp, None)
                seen = {c: 1 for c in seen if fp not in c}
        elif name == "Read":
            fp = inp.get("file_path")
            s, e = _read_span(inp)
            if fp and read_spans.get(fp) and _covered(read_spans[fp], s, e):
                hits += 1
            if fp:
                read_spans.setdefault(fp, []).append((s, e))
                last_read[fp] = 1
        elif name == "Bash":
            cmd = inp.get("command", "")
            if CAT.search(cmd) and any(f and f in cmd for f in last_read):
                hits += 1
            c = " ".join(cmd.split())
            if RO.match(cmd) and c in seen:
                hits += 1
            seen[c] = 1
    return hits


def band(xs):
    return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None


def overlaps(a, b):
    if not a or not b:
        return None
    return a[0] <= b[2] and b[0] <= a[2]


def _is_sidechain(path):
    """A sub-agent transcript: its records carry isSidechain=true from the first lines."""
    try:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n >= 20:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("isSidechain"):
                    return True
                if rec.get("type") in ("user", "assistant"):
                    return False
    except OSError:
        return True
    return False


def index_runs(root, agent):
    """One walk of the results tree -> {'<arm>-<task>': {'runs': [...], 'attempted': n}}.

    Keyed by the literal cell dir name (arm names can contain hyphens — headroom-kompress —
    so the name is not splittable; build() looks up f"{arm}-{task}" directly).
    Discovery anchors on verifier/reward.txt (a finished trial). attempted.json records
    how many trials generate.py REQUESTED, so trials that crashed or were killed by the
    wall timeout surface as missing instead of silently shrinking n.
    """
    idx = {}
    for rt in sorted(glob.glob(f"{root}/**/verifier/reward.txt", recursive=True)):
        inst = os.path.dirname(os.path.dirname(rt))                        # .../<trial>/<inst>
        key = os.path.basename(os.path.dirname(os.path.dirname(inst)))    # <arm>-<task>
        cands = [s for s in sorted(glob.glob(os.path.join(inst, AGENT_SESSION_GLOB[agent])))
                 if not _is_sidechain(s)]
        if not key or not cands:
            continue
        # >1 session file per trial (Task-tool sub-agents, resumes): the main trajectory
        # is the largest remaining transcript, not glob order
        s = max(cands, key=os.path.getsize)
        cell = idx.setdefault(key, {"runs": [], "attempted": None, "seen": set()})
        if s not in cell["seen"]:
            cell["seen"].add(s)
            cell["runs"].append((s, open(rt).read().strip()))
    for ap in glob.glob(f"{root}/**/attempted.json", recursive=True):
        key = os.path.basename(os.path.dirname(ap))
        try:
            k = int(json.load(open(ap)).get("k", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        cell = idx.setdefault(key, {"runs": [], "attempted": None, "seen": set()})
        cell["attempted"] = (cell["attempted"] or 0) + k
        cell["dir"] = os.path.dirname(ap)  # to tell "never ran" from "ran and crashed"
    return idx


# ---------------------------------------------------------------- assemble (from artifacts only)
def _peak_ctx(path):
    """Largest recorded per-request context (input + cache read/write) in a session."""
    return max((u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0) for u in recorded_usage(path)),
               default=0)


def _cell_stats(cell):
    runs = cell["runs"] if cell else []
    lens, rws, peaks = [], [], []
    for p, _ in runs:
        acts = actions(p)
        lens.append(len(acts))
        rws.append(rework_count(acts))
        peaks.append(_peak_ctx(p))
    n = len(runs)
    attempted = cell["attempted"] if cell and cell["attempted"] else n
    started = n  # trial dirs that actually opened (reward or not); >n means some crashed
    cdir = cell.get("dir") if cell else None
    if cdir:
        try:
            started = max(started, sum(1 for e in os.scandir(cdir) if e.is_dir()))
        except OSError:
            pass
    return {
        "n": n, "attempted": max(attempted, n), "lost": max(attempted, n) - n,
        "started": started,
        # a trial that never finished is a failure on the solve axis, not missing data
        "solve": sum(r == "1" for _, r in runs),
        "length": band(lens), "rework": band(rws), "_lens": lens,
        "peak_ctx": max(peaks, default=0),
    }


def build(args):
    root = args.__dict__["from"]
    arms = [a for a in args.arms.split(",") if a]
    idx = index_runs(root, args.agent)
    milestones = {}
    for mp in sorted(glob.glob(f"{root}/**/milestones.json", recursive=True)):
        try:
            milestones.update(json.load(open(mp)))  # {task: {arm: [min,mean,max]}}
        except (OSError, json.JSONDecodeError):
            continue
    incr = _load_incremental(root, arms)
    rows = []
    gate = getattr(args, "ctx_gate", 50_000)
    for task in resolve_tasks(args.tasks):
        v = _cell_stats(idx.get(f"vanilla-{task}"))
        row = {"task": task, "vanilla": v, "arms": {},
               "sub_gate": bool(v["peak_ctx"]) and v["peak_ctx"] < gate}
        for arm in arms:
            a = _cell_stats(idx.get(f"{arm}-{task}"))
            enough = len(v["_lens"]) >= 2 and len(a["_lens"]) >= 2
            mv = (milestones.get(task, {}) or {}).get("vanilla")
            mc = (milestones.get(task, {}) or {}).get(arm)
            a.update(
                length_ok=overlaps(v["length"], a["length"]) if enough else None,
                rework_ok=overlaps(v["rework"], a["rework"]) if enough else None,
                milestone=mc, milestone_ok=(overlaps(mv, mc) if (mv and mc) else None),
                incr=incr.get((task, arm)),
            )
            row["arms"][arm] = a
        rows.append(row)
    model = None
    try:  # the run's model, for the report title (written by generate.full)
        model = json.load(open(f"{root}/run-manifest.json")).get("model")
    except (OSError, json.JSONDecodeError):
        pass
    return {"arms": arms, "rows": rows, "model": model,
            "has_milestone": bool(milestones), "has_incr": bool(incr)}


def _load_incremental(root, arms):
    """generate --mode incremental jsonl -> per (task,arm) comp% / costΔ / fidelity VS CONTROL.

    Everything is computed on the common step set (both arms answered, step > 0 — the
    cold-cache first step is excluded consistently for tokens, cost AND fidelity), and
    the arm's action-fidelity is reported next to control's: control replay is the noise
    floor (sampling + reconstruction error); only the gap below it is signal.
    """
    def rows(p):
        d = {}
        if not os.path.exists(p):
            return d
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in r and "usage" in r:
                d[r["step"]] = r
        return d

    def ctx(u):
        return (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0))

    out = {}
    for cf in sorted(glob.glob(f"{root}/**/incremental/*-control.jsonl", recursive=True)):
        task = os.path.basename(cf)[: -len("-control.jsonl")]
        C = rows(cf)
        for arm in arms:
            A = rows(os.path.join(os.path.dirname(cf), f"{task}-{arm}.jsonl"))
            if not A or not C:
                continue
            common = sorted(s for s in (set(A) & set(C)) if s > 0)
            if not common:
                continue
            cc = sum(ctx(C[s]["usage"]) for s in common) or 1
            ac = sum(ctx(A[s]["usage"]) for s in common)
            oc = sum(C[s].get("cost_usd", 0) for s in common) or 1e-9
            zc = sum(A[s].get("cost_usd", 0) for s in common)
            out[(task, arm)] = {
                "comp": round(1 - ac / cc, 4), "costd": round(1 - zc / oc, 4),
                "fid": sum(bool(A[s].get("agree_action")) for s in common) / len(common),
                "fid_ctrl": sum(bool(C[s].get("agree_action")) for s in common) / len(common),
                "steps": len(common),
            }
    return out


# ---------------------------------------------------------------- rendering
def _b(x):
    return "—" if not x else (f"{x[1]:.0f}[{x[0]}-{x[2]}]" if x[0] != x[2] else f"{x[1]:.0f}")


def _v(ok):
    return "✓ OK" if ok else ("✗ DIVERGES" if ok is False else "—")


def _pct(x):
    return "—" if x is None else f"{x * 100:+.0f}%"


def _solve(c):
    # requested but no trial dir ever opened -> aborted / never ran, not a real 0/k
    if c["n"] == 0 and c.get("started", 0) == 0:
        return "—"
    s = f"{c['solve']}/{c['attempted']}"
    return s + (f" (⚠{c['lost']} lost)" if c["lost"] else "")


def table(d):
    """One (header, rows) build shared by console, md and html renderers.

    Each row is [(cell_text, ok_flag_or_None), ...]; ok drives ✓/✗ colouring in html.
    """
    head = ["task", "arm", "solve v·arm", "ctx peak (v)", "length (v)", "length (arm)",
            "len", "rework"]
    if d["has_milestone"]:
        head.append("milestone")
    if d["has_incr"]:
        head += ["fid (arm·ctrl)", "comp", "$Δ"]
    rows = []
    for r in d["rows"]:
        for arm in d["arms"]:
            a = r["arms"][arm]
            pk = r["vanilla"]["peak_ctx"]
            pk_txt = "—" if not pk else f"{pk / 1000:.0f}k" + (" ⊘" if r["sub_gate"] else "")
            sv, sa = _solve(r["vanilla"]), _solve(a)
            solve_txt = "—" if sv == "—" and sa == "—" else f"{sv} · {sa}"
            cells = [(r["task"], None), (arm, None),
                     (solve_txt, None),
                     (pk_txt, None),
                     (_b(r["vanilla"]["length"]), None), (_b(a["length"]), None),
                     (_v(a["length_ok"]), a["length_ok"]),
                     (_v(a["rework_ok"]), a["rework_ok"])]
            if d["has_milestone"]:
                cells.append((_v(a["milestone_ok"]), a["milestone_ok"]))
            if d["has_incr"]:
                inc = a.get("incr") or {}
                if inc.get("fid") is None:
                    fid = "—"
                elif abs(inc.get("comp") or 0) < 0.02:
                    fid = "⊘ passthrough"  # no compression -> fid deltas are noise
                else:
                    fid = f"{inc['fid']:.0%} · {inc['fid_ctrl']:.0%}"
                cells += [(fid, None), (_pct(inc.get("comp")), None),
                          (_pct(inc.get("costd")), None)]
            rows.append(cells)
    return head, rows


SUB = ("vanilla = the noise floor; ✓ = the arm's band overlaps vanilla's, ✗ = disjoint "
       "(needs ≥2 finished runs/arm). In solve, ⚠ lost = trials that opened but crashed / "
       "timed out (counted as unsolved), while — = the cell was never run in this pass "
       "(no trial ever opened — aborted or out of scope, not a failure). "
       "⊘ = vanilla's peak context stayed below the compaction gate "
       "(--ctx-gate, default 50k): condense's whole-conversation compaction cannot have "
       "triggered and headroom only compresses individual tool outputs >200 tokens, so a "
       "len ✗ on a ⊘ task is a BEHAVIORAL effect of the arm's wiring, not compaction damage. "
       "fid = per-step action agreement vs the control noise floor; it is only shown when "
       "the arm actually compressed (|comp| ≥ 2%) — ⊘ passthrough means the replay proved "
       "no compaction happened, so agreement deltas would be noise. "
       "No model was called to produce this.")


def render_md(d):
    head, rows = table(d)
    o = ["# Trajectory preservation\n", SUB + "\n",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    o += ["| " + " | ".join(c for c, _ in row) + " |" for row in rows]
    return "\n".join(o) + "\n"


def render_html(d):
    import html as H
    head, rows = table(d)
    cls = {True: "ok", False: "bad", None: ""}
    trs = ["<tr>" + "".join(
        f"<td><span class='{cls[ok]}'>{H.escape(c)}</span></td>" if ok is not None
        else f"<td>{H.escape(c)}</td>" for c, ok in row) + "</tr>" for row in rows]
    style = ("body{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;"
             "background:#0f1117;color:#d8dce6} h1{font-size:19px}"
             ".sub{color:#8b93a3;margin-bottom:14px;max-width:900px}"
             "table{border-collapse:collapse} th,td{border:1px solid #232838;"
             "padding:6px 10px;text-align:center;font-size:12px}"
             "th{background:#161a22;color:#9aa4b5} .ok{color:#3fb950;font-weight:700}"
             ".bad{color:#e5534b;font-weight:700}")
    ths = "".join(f"<th>{H.escape(h)}</th>" for h in head)
    return ("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<title>Trajectory preservation</title>"
            f"<style>{style}</style></head><body><h1>Trajectory preservation</h1>"
            f"<div class=\"sub\">{H.escape(SUB)}</div>"
            f"<table><tr>{ths}</tr>{''.join(trs)}</table></body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="display the quality bench (reads generate.py's artifacts)")
    ap.add_argument("--from", default="results/jobs", help="results root produced by generate.py")
    ap.add_argument("--tasks", default=None,
                    help="comma list, a number N for the first N curated defaults, "
                         "or omitted = 5 (must cover what generate.py ran)")
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--agent", default="claude-code", choices=list(AGENT_SESSION_GLOB))
    ap.add_argument("--ctx-gate", type=int, default=50_000,
                    help="peak-context threshold below which compaction cannot have "
                         "triggered (marks the task ⊘)")
    ap.add_argument("--format", default="html", choices=["html", "md"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    d = build(args)
    render_console(d)
    out = args.out or f"report.{args.format}"
    open(out, "w").write(render_md(d) if args.format == "md" else render_html(d))
    Console().print(f"[dim]wrote {out}[/]")


# The wide table() view (all axes as separate columns) is the HTML/md layout. The terminal
# gets a headline projection with control on its own row: method, solve%, length vs control,
# and a single faithfulness score (per-step agreement docked for redundant re-work). A task
# whose context never crossed the compaction gate is nulled — nothing was compacted, so the
# arms aren't worth comparing on it. rework/milestone/ctx detail stays in report.html.
_LEGEND = (
    "control = the vanilla baseline (its own row); every arm is measured against it. "
    "solve = trials solved / attempted (⚠ = some crashed; — = not run in this pass). "
    "method = how the row was measured — N× full trials, +replay = teacher-forced per-step. "
    "length = arm trajectory length vs control's median (+longer / −shorter; ✓/✗ = within "
    "control's noise band, needs ≥2 runs/arm). faithful = per-step action agreement with the "
    "original trajectory, docked for redundant re-work (replay only). cost = $ vs control "
    "(replay). ⊘ too short = control's context never crossed the compaction gate, so nothing "
    "was compacted and the arm isn't comparable on this task. No model was called.")
_SHORT = "[dim]⊘ short[/]"
_DASH = "[dim]—[/]"


def _icon(ok):
    return "✓" if ok is True else ("✗" if ok is False else "—")


def _colok(text, ok):
    return f"[green]{text}[/]" if ok is True else (f"[red]{text}[/]" if ok is False else text)


def _solve_pct(c):
    """Solve rate as a %, yellow with ⚠lost when some trials crashed; — when never run."""
    if c["n"] == 0 and c.get("started", 0) == 0:
        return _DASH
    pct = f"{c['solve'] / c['attempted'] * 100:.0f}%" if c["attempted"] else "0%"
    return f"[yellow]{pct} ⚠{c['lost']}[/]" if c["lost"] else pct


def _method(c):
    """How the row was measured: N full trials, + replay when teacher-forced data exists."""
    if c["n"] == 0 and c.get("started", 0) == 0:
        return _DASH
    m = f"{c['n']}× full"
    return m + " +replay" if (c.get("incr") or {}).get("fid") is not None else m


def _len_delta(v, a, sub_gate):
    """Arm length vs control's median as a signed %, + ✓/✗ verdict; nulled when too short."""
    vm = v["length"][1] if v["length"] else None
    am = a["length"][1] if a["length"] else None
    if am is None or not vm:
        return _DASH
    if sub_gate:
        return _SHORT
    core = f"{(am / vm - 1) * 100:+.0f}%"
    if a["length_ok"] is not None:
        core += f" {_icon(a['length_ok'])}"
    return _colok(core, a["length_ok"])


def _faithful_cost(v, a, sub_gate):
    """(faithfulness, cost). Faithfulness = per-step agreement (replay) docked for the arm's
    excess redundant re-work over control. Nulled when the task is too short or nothing was
    actually compressed (agreement would be measuring noise)."""
    if a["n"] == 0 and a.get("started", 0) == 0:   # arm never ran -> no data, not "too short"
        return _DASH, _DASH
    if sub_gate:
        return _SHORT, _SHORT
    inc = a.get("incr") or {}
    fid = inc.get("fid")
    if fid is None:
        return _DASH, _DASH
    cost = _pct(inc.get("costd"))
    if abs(inc.get("comp") or 0) < 0.02:  # nothing compressed -> not comparable
        return "[dim]⊘ n/a[/]", cost
    vr = v["rework"][1] if v["rework"] else 0.0
    ar = a["rework"][1] if a["rework"] else 0.0
    al = a["length"][1] if a["length"] else 1.0
    penalty = max(0.0, ar - vr) / max(al, 1.0)   # extra re-fetches per step
    return f"{max(0.0, min(1.0, fid - penalty)):.0%}", cost


def render_console(d):
    """Headline terminal view: control + each arm as rows, scored on solve% · length ·
    faithfulness · cost. The full axis grid (rework, milestone, ctx) lives in report.html."""
    console = Console()
    model = d.get("model")
    title = "[bold]quality — trajectory preservation" + (f" · {model}" if model else "") + "[/]"
    t = Table(title=title, caption=_LEGEND, caption_justify="left", caption_style="dim")
    t.add_column("task", no_wrap=True)
    t.add_column("arm", no_wrap=True)
    t.add_column("method", no_wrap=True)
    t.add_column("solve", justify="right")
    t.add_column("length", justify="right")
    t.add_column("faithful", justify="center")
    t.add_column("cost", justify="right")
    for ri, r in enumerate(d["rows"]):
        if ri:
            t.add_section()
        v, sub = r["vanilla"], r["sub_gate"]
        # control is the baseline: its own row, length shown absolute, nothing to score against
        base = f"{v['length'][1]:.0f}" if v["length"] else _DASH
        t.add_row(r["task"], "control", _method(v), _solve_pct(v), f"[dim]{base}[/]",
                  _DASH, _DASH)
        for arm in d["arms"]:
            a = r["arms"][arm]
            faith, cost = _faithful_cost(v, a, sub)
            t.add_row("", arm, _method(a), _solve_pct(a), _len_delta(v, a, sub), faith, cost)
    console.print(t)
    _takeaway(console, d)


def _takeaway(console, d):
    """One-glance summary: how many tasks are actually comparable (graded AND compaction
    could fire), how many were too short, how many arm-cells never ran, and any divergence."""
    total = len(d["rows"])
    comparable, tooshort = set(), set()
    for r in d["rows"]:
        graded = (len(r["vanilla"]["_lens"]) >= 2
                  and any(len(r["arms"][arm]["_lens"]) >= 2 for arm in d["arms"]))
        if graded:
            (tooshort if r["sub_gate"] else comparable).add(r["task"])
    notrun = sum(1 for r in d["rows"] for arm in d["arms"]
                 if r["arms"][arm]["n"] == 0 and r["arms"][arm].get("started", 0) == 0)
    diverge = [f"{r['task']}/{arm}" for r in d["rows"] if not r["sub_gate"]
               for arm in d["arms"] if r["arms"][arm].get("length_ok") is False]
    parts = [f"{len(comparable)}/{total} tasks comparable"]
    if tooshort:
        parts.append(f"[dim]{len(tooshort)} too short (compaction never fired)[/]")
    if notrun:
        parts.append(f"[dim]{notrun} arm-cells not run[/]")
    console.print("[bold]takeaway[/]  " + " · ".join(parts))
    if diverge:
        console.print("[red]  ✗ length diverges vs control:[/] " + ", ".join(diverge))
    elif comparable:
        console.print("[green]  ✓ no divergence on comparable tasks[/]")


if __name__ == "__main__":
    main()
