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

try:  # rich renders the terminal tables, but analysis must work on a bare python3
    from rich.console import Console
    from rich.table import Table
    HAVE_RICH = True
except ImportError:  # fresh clone, no install — fall back to the plain-text views
    HAVE_RICH = False

# one session parser for the spend side (generate/engine) and the display side.
from minmax_bench.quality.engine import (
    SESSION_GLOB,
    ctx_tokens,
    extract_action,
    parse_session,
    peak_ctx,
    resolve_tasks,
)

AGENT_SESSION_GLOB = {  # only claude-code is wired; others are TODO
    "claude-code": SESSION_GLOB,
}

# longest names first — cell dir names are '<arm>-<task>' and arm names contain hyphens,
# so splitting is a longest-prefix match against the arms the bench knows about
KNOWN_ARMS = ("headroom-kompress", "vanilla-proxy", "headroom", "condense", "vanilla", "control")


def split_cell(name):
    """'<arm>-<task>' -> (arm, task) by longest known-arm prefix; (None, name) if no match."""
    for arm in KNOWN_ARMS:
        if name.startswith(arm + "-"):
            return arm, name[len(arm) + 1:]
    return None, name


# legacy static default; the settings-aware default (incl. quality_runs_dir) is
# paths.default_run_roots(), used when discover_runs is called with roots=None
DEFAULT_RUN_ROOTS = ("results", "runs/quality-sample")


def _run_info(d):
    """What one results dir holds: modes (full/incremental), arms, tasks, model."""
    modes, arms, tasks, model = set(), set(), set(), None
    for mf in ("run-manifest.json", "summary.json"):  # full manifest first, then incremental
        try:
            model = model or json.load(open(os.path.join(d, mf))).get("model")
        except (OSError, json.JSONDecodeError):
            continue
    cells = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(os.path.join(d, "*", "attempted.json"))}
    for p in glob.glob(os.path.join(d, "*", "*", "*", "verifier", "reward.txt")):
        cell = p
        for _ in range(4):
            cell = os.path.dirname(cell)
        cells.add(os.path.basename(cell))
    for c in cells:
        arm, task = split_cell(c)
        if arm:
            modes.add("full")
            arms.add(arm)
            tasks.add(task)
    for p in glob.glob(os.path.join(d, "incremental", "*.jsonl")):
        label, _, arm = os.path.basename(p)[:-len(".jsonl")].rpartition("-")
        modes.add("incremental")
        arms.add(arm or label)
        tasks.add(label or arm)
    return {"dir": d, "modes": sorted(modes), "arms": sorted(arms),
            "tasks": sorted(tasks), "model": model}


def discover_runs(roots=None):
    """Quality result dirs under `roots`, newest first — pure filesystem walk, never spends.

    A run dir is any directory holding full-mode artifacts (run-manifest.json /
    attempted.json cells / verifier rewards) or incremental artifacts
    (incremental/*.jsonl, summary.json). Used by `quality runs` and the wizard's
    view mode so stored results are discoverable without remembering paths. `roots=None`
    uses the settings-aware default (the configured quality_runs_dir first, then the
    legacy results tree) so freshly-saved runs are found without passing --roots.
    """
    if roots is None:
        from .paths import default_run_roots
        roots = default_run_roots()
    dirs = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat, up in (("run-manifest.json", 1), ("summary.json", 1), ("attempted.json", 2),
                        (os.path.join("incremental", "*.jsonl"), 2),
                        (os.path.join("verifier", "reward.txt"), 5)):
            for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
                d = p
                for _ in range(up):
                    d = os.path.dirname(d)
                if d:
                    dirs.add(d)
    infos = [i for i in (_run_info(d) for d in dirs) if i["modes"]]
    infos.sort(key=lambda i: os.path.getmtime(i["dir"]), reverse=True)
    return infos


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


def _trial_metrics(trial_dir):
    """cost_usd + total tokens + wall-clock seconds from a trial's result.json ({} if absent).
    Tokens = input + cache + output (the whole trajectory's spend), matching the cost basis."""
    try:
        r = json.load(open(os.path.join(trial_dir, "result.json")))
    except (OSError, json.JSONDecodeError):
        return {}
    ar = r.get("agent_result") or {}
    m = {}
    if ar.get("cost_usd") is not None:
        m["cost"] = ar["cost_usd"]
        m["tok"] = (ar.get("n_input_tokens", 0) + ar.get("n_cache_tokens", 0)
                    + ar.get("n_output_tokens", 0))
    try:
        from datetime import datetime
        s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
        f = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00"))
        m["lat"] = (f - s).total_seconds()
    except (KeyError, ValueError, TypeError):
        pass
    return m


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
            cell["runs"].append((s, open(rt).read().strip(), _trial_metrics(inst)))
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
def _cell_stats(cell):
    runs = cell["runs"] if cell else []
    lens, rws, peaks = [], [], []
    costs, toks, lats = [], [], []
    for p, _, m in runs:
        acts = actions(p)
        lens.append(len(acts))
        rws.append(rework_count(acts))
        peaks.append(peak_ctx(p))
        if m.get("cost") is not None:
            costs.append(m["cost"])
        if m.get("tok") is not None:
            toks.append(m["tok"])
        if m.get("lat") is not None:
            lats.append(m["lat"])
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
        "solve": sum(r == "1" for _, r, _m in runs),
        "length": band(lens), "rework": band(rws), "_lens": lens,
        "peak_ctx": max(peaks, default=0),
        # cost/token/latency for the $ and tokens columns; lists so a band can be shown
        "_costs": costs, "_toks": toks, "_lats": lats,
    }


def build(args, include_incremental_only=True):
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
    # curated full-run tasks first, then (unless scoped off — e.g. the inline render after a
    # full run) any task that only has incremental-replay data, so session-labelled replays
    # (swe-long, a uuid) still get a row when the report is pointed at an incremental dir
    curated = list(resolve_tasks(args.tasks))
    extra = sorted({t for t, _a in incr} - set(curated)) if include_incremental_only else []
    for task in curated + extra:
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
    the arm's action-fidelity is reported next to control's: the control incremental run is
    the noise floor (sampling + reconstruction error); only the gap below it is signal.
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

    ctx = ctx_tokens  # the shared context-size definition (engine.ctx_tokens)

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
                # per-step wall-clock (only on newer artifacts) — mean over the common steps
                "latency": _mean_latency(A, common), "latency_ctrl": _mean_latency(C, common),
                "fid": sum(_faithful_step(A[s]) for s in common) / len(common),
                "fid_ctrl": sum(_faithful_step(C[s]) for s in common) / len(common),
                "redund": sum(bool(A[s].get("redundant")) for s in common),
                # CCR engagement (headroom): retrieves the agent made, and whether the arm ran
                # with the retrieve loop wired at all (the field is only written when it is)
                "retrieves": sum(r.get("ccr_retrieves", 0) for r in A.values()),
                "ccr": any("ccr_retrieves" in r for r in A.values()),
                "scoring": _scoring(A), "steps": len(common),
            }
    return out


def _mean_latency(rowset, common):
    """Mean per-step wall-clock over the common steps; None when nothing was timed."""
    v = [rowset[s].get("latency_s") for s in common]
    v = [x for x in v if isinstance(x, (int, float))]
    return (sum(v) / len(v)) if v else None


def _scoring(recset):
    """Infer per-step scoring from the stored records: an LLM goal judge (each action rated
    good/degraded/bad toward the task), an LLM equivalence judge (structural near-misses
    upgraded), or plain structural agreement (exact/action match, no LLM)."""
    vals = list(recset.values())
    if any("quality" in r for r in vals):
        return "llm:goal"
    if any(r.get("agree_semantic") not in (None, r.get("agree_action")) for r in vals):
        return "llm:equiv"
    return "struct"


def _faithful_step(r):
    """A faithful step took a valid action and didn't redundantly re-fetch info it already had.

    'Valid' depends on how the run was scored. With a GOAL judge (each action rated on its own
    merit toward the task), a good step is faithful — control replaying itself scores ~100%,
    so an arm only loses by taking WORSE steps: a trustworthy floor. Without a judge, we fall
    back to STRUCTURAL agreement with the original recorded action, whose ceiling is low even
    for control (the model rarely reproduces its own past sampling) — noisy, not a true 100%.
    """
    q = r.get("quality")
    if q is not None:
        good = q == "good"                                   # goal judge
    else:                                                    # structural (equiv upgrades a
        good = bool(r.get("agree_semantic", r.get("agree_action")))  # near-miss if it ran)
    return good and not r.get("redundant")


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
       "the arm actually compressed (|comp| ≥ 2%) — ⊘ passthrough means the incremental run proved "
       "no compaction happened, so agreement deltas would be noise. "
       "No model was called to produce this.")


def render_md(d):
    head, rows = table(d)
    o = ["# Trajectory preservation\n", SUB + "\n",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    o += ["| " + " | ".join(c for c, _ in row) + " |" for row in rows]
    return "\n".join(o) + "\n"


_HTML_STYLE = """
:root{--bg:#f7f9fb;--panel:#fff;--panel2:#f1f4f8;--line:#e2e7ee;--ink:#17202e;--ink2:#586274;
--ink3:#8a94a5;--good:#1f9d5b;--bad:#d1453a;--warn:#b9812a;--mut:#8791a2;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#0d1219;--panel:#151d27;--panel2:#1b2530;
--line:#28323f;--ink:#e2e8f1;--ink2:#9aa6b6;--ink3:#6b7686;--good:#33c37e;--bad:#f26a5c;
--warn:#e0a93f;--mut:#7d8798}}
:root[data-theme=dark]{--bg:#0d1219;--panel:#151d27;--panel2:#1b2530;--line:#28323f;--ink:#e2e8f1;
--ink2:#9aa6b6;--ink3:#6b7686;--good:#33c37e;--bad:#f26a5c;--warn:#e0a93f;--mut:#7d8798}
:root[data-theme=light]{--bg:#f7f9fb;--panel:#fff;--panel2:#f1f4f8;--line:#e2e7ee;--ink:#17202e;
--ink2:#586274;--ink3:#8a94a5;--good:#1f9d5b;--bad:#d1453a;--warn:#b9812a;--mut:#8791a2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
font-size:14px;line-height:1.5}.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-family:var(--mono);font-size:19px;margin:0 0 4px}h2{font-family:var(--mono);font-size:15px;
margin:30px 0 8px}.dim{color:var(--ink3);font-weight:400}.sub{color:var(--ink2);max-width:70ch;font-size:13px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink3);font-weight:600;
text-align:right;padding:9px 11px;background:var(--panel2);border-bottom:1px solid var(--line)}
th.l{text-align:left}td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line);
font-family:var(--mono);font-size:12.5px}td.l{text-align:left;font-family:var(--sans)}
tr.row{cursor:pointer}tr.row:hover td{background:var(--panel2)}tr.row.open td{background:var(--panel2)}
.task b{font-weight:600}.task .s{display:block;font-family:var(--mono);font-size:10.5px;color:var(--ink3)}
.v{color:var(--ink3)}.d{font-size:11px;display:block}.good{color:var(--good)}.bad{color:var(--bad)}
.warn{color:var(--warn)}.mut{color:var(--ink2)}
.pill{font-family:var(--sans);font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:20px;
white-space:nowrap;display:inline-block}.pill.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.pill.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.pill.warn{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.pill.na{background:var(--panel2);color:var(--ink2)}
tr.det td{background:var(--panel2);font-family:var(--sans);text-align:left;font-size:12.5px;color:var(--ink2)}
.det .grid{display:flex;flex-wrap:wrap;gap:8px 22px;padding:4px 2px}.det b{color:var(--ink)}
.det .k{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.caret{color:var(--ink3);display:inline-block;width:12px}tr.row.open .caret{transform:rotate(90deg)}
.foot{margin-top:22px;font-size:12px;color:var(--ink3)}
"""


def _html_report_body(d):
    """Interactive per-arm HTML derived from the SAME _cmp/_verdict the console uses."""
    import html as H

    def fL(x):
        return f"{x:.0f}"

    def fT(x):
        return f"{x / 1e6:.1f}M"

    def fU(x):
        return f"${x:.2f}"

    ST = {"good": "good", "bad": "bad", "within": "mut", "warn": "warn", "na": "mut"}

    def mcell(ctrl, arm, fmt):
        c = _cmp(ctrl, arm)
        if not c:
            return '<td>—</td>'
        st = ST[c["state"]]
        return (f'<td><span class="v">{fmt(c["van"])}→</span><span class="{st}">{fmt(c["arm"])}</span>'
                f'<span class="d {st}">{c["pct"]:+.0f}%</span></td>')

    def detail(v, a):
        bits = [f'<span><span class="k">length</span> vanilla [{", ".join(map(str, v["_lens"]))}] '
                f'· arm [{", ".join(map(str, a["_lens"]))}]</span>',
                f'<span><span class="k">$/trial</span> vanilla [{", ".join(f"{x:.2f}" for x in v["_costs"])}] '
                f'· arm [{", ".join(f"{x:.2f}" for x in a["_costs"])}]</span>']
        if a.get("_lats") and v.get("_lats"):
            bits.append(f'<span><span class="k">latency</span> vanilla '
                        f'{sum(v["_lats"]) / len(v["_lats"]):.0f}s · arm '
                        f'{sum(a["_lats"]) / len(a["_lats"]):.0f}s</span>')
        inc = a.get("incr") or {}
        if inc.get("fid") is not None:
            bits.append(f'<span><span class="k">incremental</span> compaction '
                        f'<b>{_pct(inc.get("comp"))}</b> · faithful <b>{inc["fid"]:.0%}</b> '
                        f'(ctrl {inc.get("fid_ctrl", 0):.0%}) · $ savings <b>{_pct(inc.get("costd"))}</b></span>')
        return '<div class="grid">' + "".join(bits) + '</div>'

    secs = []
    for arm in d["arms"]:
        arm_rows = [r for r in d["rows"] if r["arms"][arm]["_lens"] or r["arms"][arm]["n"]]
        if not arm_rows:
            continue
        msh = '<th>milestone</th>' if d["has_milestone"] else ''
        span = 6 if d["has_milestone"] else 5
        trs = []
        for r in arm_rows:
            v, a, sub = r["vanilla"], r["arms"][arm], r["sub_gate"]
            label, state = _verdict(v, a, sub)
            pk = v["peak_ctx"] // 1000
            ms = ""
            if d["has_milestone"]:
                m = a.get("milestone")
                mst = ("good" if a.get("milestone_ok") else "bad"
                       if a.get("milestone_ok") is False else "mut")
                ms = f'<td><span class="{mst}">{m[1] * 100:.0f}%</span></td>' if m else '<td>—</td>'
            trs.append(
                f'<tr class="row" onclick="tg(this)"><td class="l task"><span class="caret">▸</span> '
                f'<b>{H.escape(r["task"])}</b><span class="s">peak {pk}k{" ⊘" if sub else ""} · '
                f'solve {a["solve"]}/{a["attempted"]}</span></td>'
                + mcell(v["_lens"], a["_lens"], fL) + mcell(v["_toks"], a["_toks"], fT)
                + mcell(v["_costs"], a["_costs"], fU)
                + f'<td class="l"><span class="pill {state}">{H.escape(label)}</span></td>{ms}</tr>'
                + f'<tr class="det" style="display:none"><td class="l" colspan="{span}">{detail(v, a)}</td></tr>')
        secs.append(f'<h2>{H.escape(arm)} <span class="dim">vs vanilla</span></h2>'
                    f'<table><thead><tr><th class="l">task ▸</th><th>length</th><th>tokens</th>'
                    f'<th>$</th><th class="l">verdict</th>{msh}</tr></thead><tbody>'
                    + "".join(trs) + '</tbody></table>')
    return "".join(secs)


def render_html(d):
    import html as H
    model = d.get("model") or ""
    title = "Trajectory preservation" + (f" · {model}" if model else "")
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{H.escape(title)}</title>'
            f'<style>{_HTML_STYLE}</style></head><body><div class="wrap">'
            f'<h1>trajectory preservation <span class="dim">{H.escape("· " + model) if model else ""}</span></h1>'
            f'<div class="sub">{H.escape(SUB)}</div>'
            + _html_report_body(d)
            + '<div class="foot">length/tokens/$ show vanilla→arm with the signed delta, coloured '
            'green (shorter/saved) / red (longer/costlier) / grey (within vanilla\'s band) '
            '— independently, so token-savings that cost more show green next to red. '
            'Click a row for per-trial values and incremental compaction / faithful / $ savings.</div>'
            '</div><script>function tg(r){var d=r.nextElementSibling;'
            'var o=d.style.display!=="none";d.style.display=o?"none":"";'
            'r.classList.toggle("open",!o);}</script></body></html>')


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
    if HAVE_RICH:
        Console().print(f"[dim]wrote {out}[/]")
    else:
        print(f"wrote {out}")


# The wide table() view (all axes as separate columns) is the HTML/md layout. The terminal
# shows two focused tables so the modes aren't confounded — FULL trajectories (solve, length)
# and INCREMENTAL teacher-forced per-step (scoring, compression, faithfulness, cost) — each
# with control on its own row. rework/milestone/ctx detail stays in report.html.
_FULL_LEGEND = (
    "one table per arm; each row a task. length / tokens / $ each show [dim]vanilla[/]→arm (mean "
    "steps, total tokens, USD/trial) with the signed delta below, coloured INDEPENDENTLY: green "
    "below vanilla's spread (shorter / saved), red above (longer / costlier), dim within — so an "
    "arm that cuts tokens yet costs more, the cache-bust tax, shows green next to red. verdict = "
    "length preservation (the load-bearing "
    "axis): same work (within vanilla's band) / drifted ↑ longer / shorter ↓ / ⊘ too short "
    "(vanilla never crossed the compaction gate — nothing compacted, so any change is "
    "behavioural, not compaction). n1 = single run, a trend not a verdict (needs ≥2/arm). "
    "milestone = mean % of the task's subgoals the arm reached (LLM-judged, approach-agnostic), "
    "green when it matches vanilla's band / red when below (only shown with --milestones). "
    "Under each task: peak context + solve rate (⚠lost trials count as fails). No model was called.")
_INCR_LEGEND = (
    "teacher-forced per-step run of a recorded session. control = the baseline (its own row); "
    "its faithful is the noise floor, its s/step the latency baseline. Every delta is vs "
    "control and signed so [bold]+ = BETTER[/], − = worse. scoring = how each step was judged: "
    "llm:goal rates each action on its own merit (control ≈100% — trustworthy floor; an arm "
    "only loses by taking WORSE steps), whereas struct / llm:equiv match the ORIGINAL recorded "
    "action, whose ceiling is sampling-driven and low even for control (a noisy floor — prefer "
    "--judge goal). compaction % / $ savings / speed up = context compressed / cost saved / "
    "per-step wall-clock faster vs control (+ = better = less context / cheaper / faster). ↻N = "
    "CCR retrieve calls a headroom arm made (net compaction can be ~0 yet still engaged). "
    "faithful = the % of control's own faithfulness the arm keeps (control = 100% by "
    "construction; its absolute good-rate is shown in parens) — normalising divides out the "
    "sampling/judge floor, so 100% = no measurable loss. Coloured green ≥95% / yellow 85–95% / "
    "red <85% when the arm engaged (compressed or CCR-retrieved); else dim (≈100% by "
    "construction — no verdict). — = no incremental data. speed up is wall-clock — read it as a "
    "trend, not exact.")
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


def _runs(c):
    """Finished-trial count (0× when trials opened but all crashed; — when never run)."""
    if c["n"] == 0 and c.get("started", 0) == 0:
        return _DASH
    return f"{c['n']}×"


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


def _cmp(ctrl_list, arm_list):
    """The ONE metric comparison both the console and HTML derive from, so they can't drift.
    Returns {van, arm, pct, state} or None. state: good = arm below vanilla's spread
    (shorter/saved), bad = above (longer/costlier), within = inside the band."""
    if not ctrl_list or not arm_list:
        return None
    lo, hi = min(ctrl_list), max(ctrl_list)
    cm, am = sum(ctrl_list) / len(ctrl_list), sum(arm_list) / len(arm_list)
    state = "good" if am < lo else "bad" if am > hi else "within"
    return {"van": cm, "arm": am, "pct": (am / cm - 1) * 100 if cm else 0.0, "state": state}


def _cmp_cell(ctrl_list, arm_list, fmt):
    """Console cell: vanilla→arm on line 1, signed delta below, coloured by _cmp's state."""
    c = _cmp(ctrl_list, arm_list)
    if not c:
        return _DASH
    color = {"good": "green", "bad": "red", "within": "dim"}[c["state"]]
    return f"[dim]{fmt(c['van'])}[/]→[{color}]{fmt(c['arm'])}[/]\n[{color}]{c['pct']:+.0f}%[/]"


def _milestone_cell(a):
    """Arm's milestone coverage (mean % of the task's subgoals reached), green when it overlaps
    vanilla's band (same subgoals) / red when it falls below / plain when ungraded."""
    m = a.get("milestone")
    if not m:
        return _DASH
    return _colok(f"{m[1] * 100:.0f}%", a.get("milestone_ok"))


def _TOKFMT(x):
    return f"{x / 1e6:.1f}M"


def _USDFMT(x):
    return f"${x:.2f}"


def _engaged(inc):
    """Did the arm actually do something to the context? Compression that moved the context
    (condense, headroom-kompress) OR a CCR retrieve (headroom fetched a compressed output back).
    If neither, faithfulness is just measuring reconstruction noise."""
    return abs(inc.get("comp") or 0) >= 0.02 or inc.get("retrieves", 0) > 0


def _comp_cell(inc):
    """Context compression %, with the CCR retrieve count (↻) appended for a headroom arm so
    engagement is visible even when net compression is ~0 (it retrieved content back)."""
    txt = _pct(inc.get("comp"))
    if inc.get("ccr"):
        txt += f" [magenta]↻{inc.get('retrieves', 0)}[/]"
    return txt


def _latency_cell(inc):
    """Per-step wall-clock SAVED vs control, signed % (+ = faster, − = slower); — when not
    timed (old artifact). Same sign convention as ctx/$ saved: positive is better."""
    la, lc = inc.get("latency"), inc.get("latency_ctrl")
    return _DASH if la is None or not lc else _pct(1 - la / lc)


def _incr_field(r, arms, key):
    """The first arm's incremental value for `key` — used for control-side fields
    (fid_ctrl / latency_ctrl / scoring), which are ≈equal across a task's arms."""
    return next((r["arms"][a]["incr"][key] for a in arms
                 if (r["arms"][a].get("incr") or {}).get(key) is not None), None)


def _latency_floor(r, arms):
    """Control's own per-step wall-clock for a task (the latency baseline)."""
    return _incr_field(r, arms, "latency_ctrl")


def _faithful_norm(fid, floor):
    """The arm's faithfulness as a fraction of control's — control replaying itself is 1.0
    (100%) by construction, so this is 'how much of control's faithfulness the arm keeps',
    with the absolute sampling/judge floor divided out. None when there's no floor to divide by.
    Capped at 1.0: an arm can't be MORE faithful than the no-compaction reference (excess is
    noise)."""
    if fid is None or not floor:
        return None
    return min(fid / floor, 1.0)


def _faithful_cost(a, floor):
    """(faithfulness, cost) — incremental metrics. Faithfulness is normalised to control (=100%):
    the % of control's own faithfulness the arm retains, coloured green ≥95% (no measurable loss)
    / yellow 85–95% / red <85% when the arm engaged (compressed or made a CCR retrieve). When it
    barely touched the context (comp <2%, no CCR) the value is shown DIM (≈100% by construction —
    no verdict). — when there's no incremental data or no control floor to normalise against."""
    inc = a.get("incr") or {}
    cost = _pct(inc.get("costd"))
    norm = _faithful_norm(inc.get("fid"), floor)
    if norm is None:
        return _DASH, cost
    txt = f"{norm:.0%}"
    if not _engaged(inc):
        return f"[dim]{txt}[/]", cost
    return (f"[green]{txt}[/]" if norm >= 0.95
            else f"[red]{txt}[/]" if norm < 0.85 else f"[yellow]{txt}[/]"), cost


def _has_full(r, arms):
    """A task has full-run data if control or any arm opened at least one trial."""
    return any(c["n"] or c.get("started", 0)
               for c in [r["vanilla"]] + [r["arms"][a] for a in arms])


def _has_incr(r, arms):
    return any((r["arms"][a].get("incr") or {}).get("fid") is not None for a in arms)


def _scoring_for(r, arms):
    """The scoring method used for a task's incremental run (uniform across its arms)."""
    return _incr_field(r, arms, "scoring")


def _grouped(t, rows, render_row):
    """Add task-grouped rows to a table: a section divider between tasks, the task name on
    the control row, blank on the arm rows (render_row yields cell tuples per arm)."""
    for ri, r in enumerate(rows):
        if ri:
            t.add_section()
        for first, arm, cells in render_row(r):
            t.add_row(r["task"] if first else "", arm, *cells)


def _LENFMT(x):
    return f"{x:.0f}"


def _verdict(v, a, sub_gate):
    """Shared length-preservation verdict → (label, state). state ∈ good|bad|warn|na. Statistical
    (band overlap) when both arms have ≥2 runs; else DIRECTIONAL vs vanilla's spread, tagged n1 (a
    single run reads a trend, not significance). Both console and HTML render from this."""
    if not a["_lens"] or not v["length"]:
        return ("—", "na")
    if sub_gate:
        return ("⊘ too short", "na")
    lo, hi, am = v["length"][0], v["length"][2], a["length"][1]
    if a["length_ok"] is True:
        return ("same work", "good")
    if a["length_ok"] is False:
        return ("drifted ↑ longer", "bad") if am > hi else ("shorter ↓", "warn")
    if am > hi:                                           # k<2 → directional, not a real verdict
        return ("drifted ↑ longer n1", "bad")
    if am < lo:
        return ("shorter ↓ n1", "warn")
    return ("~ same n1", "good")


def _verdict_cell(v, a, sub_gate):
    """Console verdict cell — rich-markup wrapper around the shared _verdict."""
    label, state = _verdict(v, a, sub_gate)
    color = {"good": "green", "bad": "red", "warn": "yellow", "na": "dim"}[state]
    return f"[{color}]{label}[/]"


def _full_table(console, d, model):
    """One table PER ARM: each row a task, vanilla's length/tokens/$ then the arm's (colored
    vs vanilla), verdict last. Vanilla is the shared reference, repeated in each arm's table."""
    rows = [r for r in d["rows"] if _has_full(r, d["arms"])]
    if not rows:
        return
    for ai, arm in enumerate(d["arms"]):
        arm_rows = [r for r in rows if r["arms"][arm]["_lens"] or r["arms"][arm]["n"]]
        if not arm_rows:
            continue
        title = f"[bold]{arm} vs vanilla" + (f" · {model}" if model else "") + "[/]"
        t = Table(title=title, caption=_FULL_LEGEND if ai == len(d["arms"]) - 1 else None,
                  caption_justify="left", caption_style="dim", pad_edge=False)
        has_ms = d.get("has_milestone")
        t.add_column("task", no_wrap=True)
        # one column per metric, each showing vanilla→arm + delta (see _cmp_cell) — far less
        # crammed than separate vanilla/arm columns, and the arm's absolute value is still there
        for c in ("length", "tokens", "$"):
            t.add_column(c, justify="right")
        t.add_column("verdict", justify="left", no_wrap=True)
        if has_ms:
            t.add_column("milestone", justify="right")  # subgoals reached vs vanilla
        for r in arm_rows:
            v, a, sub = r["vanilla"], r["arms"][arm], r["sub_gate"]
            peak = v["peak_ctx"] // 1000
            taskcell = (f"{r['task']}\n[dim]peak {peak}k{' ⊘' if sub else ''} · "
                        f"solve {a['solve']}/{a['attempted']}[/]")
            cells = [
                taskcell,
                _cmp_cell(v["_lens"], a["_lens"], _LENFMT),
                _cmp_cell(v["_toks"], a["_toks"], _TOKFMT),
                _cmp_cell(v["_costs"], a["_costs"], _USDFMT),
                _verdict_cell(v, a, sub),
            ]
            if has_ms:
                cells.append(_milestone_cell(a))
            t.add_row(*cells)
        console.print(t)


def _incremental_table(console, d, model):
    rows = [r for r in d["rows"] if _has_incr(r, d["arms"])]
    if not rows:
        return
    title = ("[bold]quality — incremental (teacher-forced per-step)"
             + (f" · {model}" if model else "") + "[/]")
    t = Table(title=title, caption=_INCR_LEGEND, caption_justify="left", caption_style="dim")
    for col in ("task", "arm", "scoring"):
        t.add_column(col, no_wrap=True)
    for col in ("compaction %", "faithful", "$ savings", "speed up"):
        t.add_column(col, justify="right" if col != "faithful" else "center")

    def render_row(r):
        floor = _floor_for(r, d["arms"])
        scoring = _scoring_for(r, d["arms"]) or _DASH
        # control is the 100% reference (arms are normalised to it); show its own absolute
        # good-rate in parens so a harsh judge / low sampling floor is visible
        ctrl_faith = f"[dim]100% ({floor:.0%})[/]" if floor is not None else _DASH
        lat0 = _latency_floor(r, d["arms"])
        ctrl_lat = f"[dim]{lat0:.1f}s[/]" if lat0 is not None else _DASH
        yield True, "control", (scoring, _DASH, ctrl_faith, _DASH, ctrl_lat)
        for arm in d["arms"]:
            inc = r["arms"][arm].get("incr") or {}
            if inc.get("fid") is None:
                yield False, arm, ("", _DASH, _DASH, _DASH, _DASH)
                continue
            faith, cost = _faithful_cost(r["arms"][arm], floor)
            yield False, arm, ("", _comp_cell(inc), faith, cost, _latency_cell(inc))

    _grouped(t, rows, render_row)
    console.print(t)


def render_console(d):
    """Terminal view: two focused tables (full trajectories, then incremental) so the two
    measurement modes aren't confounded, followed by a one-glance takeaway."""
    if not HAVE_RICH:
        # bare-python3 fallback (fresh clone, nothing installed): the md view carries
        # every axis in one wide table — less pretty, same numbers
        print(render_md(d))
        return
    console = Console()
    model = d.get("model")
    _full_table(console, d, model)
    _incremental_table(console, d, model)
    _takeaway(console, d)


def _floor_for(r, arms):
    """The control incremental fidelity floor for a task (the noise floor)."""
    return _incr_field(r, arms, "fid_ctrl")


def _takeaway(console, d):
    """One-glance summary across both measurement modes: full-run comparability (graded AND
    compaction could fire) and incremental coverage, plus divergences — length (full) or
    fidelity below the control floor (incremental)."""
    comparable, tooshort = set(), set()
    for r in d["rows"]:
        graded = (len(r["vanilla"]["_lens"]) >= 2
                  and any(len(r["arms"][arm]["_lens"]) >= 2 for arm in d["arms"]))
        if graded:
            (tooshort if r["sub_gate"] else comparable).add(r["task"])
    replayed = {r["task"] for r in d["rows"] for arm in d["arms"]
                if (r["arms"][arm].get("incr") or {}).get("fid") is not None}
    diverge = [f"{r['task']}/{arm} length" for r in d["rows"] if not r["sub_gate"]
               for arm in d["arms"] if r["arms"][arm].get("length_ok") is False]
    for r in d["rows"]:                                   # fidelity meaningfully below floor
        floor = _floor_for(r, d["arms"])
        if floor is None or floor < 0.9:  # control should score ≥90% under a calibrated judge;
            continue                      # below that the baseline is noisy (warned below)
        for arm in d["arms"]:
            inc = r["arms"][arm].get("incr") or {}
            norm = _faithful_norm(inc.get("fid"), floor)
            if norm is not None and _engaged(inc) and norm < 0.85:  # kept <85% of control
                diverge.append(f"{r['task']}/{arm} fidelity ({norm:.0%} of control)")
    parts = []
    if comparable or tooshort:
        full = f"full: {len(comparable)} comparable"
        if tooshort:
            full += f", {len(tooshort)} too short"
        parts.append(full)
    if replayed:
        parts.append(f"incremental: {len(replayed)} tasks")
    console.print("[bold]takeaway[/]  " + " · ".join(parts or ["nothing comparable yet"]))
    if diverge:
        console.print("[red]  ✗ diverges vs control:[/] " + ", ".join(diverge))
    elif comparable or replayed:
        console.print("[green]  ✓ no divergence detected[/]")
    # a low control floor means faithfulness was scored by structural match (no goal judge),
    # whose ceiling is sampling-driven — the comparison is noise-dominated, not trustworthy
    lowfloor = sum(1 for r in d["rows"]
                   if (_floor_for(r, d["arms"]) or 1) < 0.9 and _has_incr(r, d["arms"]))
    if lowfloor:
        console.print(f"[yellow]  ⚠ {lowfloor} incremental task(s) have control <90% faithful — "
                      "the judge is miscalibrated (or the run wasn't goal-judged), so the "
                      "baseline is noisy. Re-score with `minmax-bench quality rejudge --from "
                      "<dir>` (cheap) or re-run with --judge goal.[/]")


if __name__ == "__main__":
    main()
