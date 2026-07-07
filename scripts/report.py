#!/usr/bin/env python3
"""report.py — the one command for the quality / trajectory-preservation bench.

Two measurement modes, shared output/filters:

  --mode full    Read recorded FULL agent runs and test whether each method preserves the
                 trajectory vs a vanilla-vs-vanilla noise floor (length / rework / milestone /
                 solve, per-axis ✓ overlap / ✗ disjoint). Offline, no API calls.
  --mode replay  TEACHER-FORCED per-step replay of one trajectory through each arm's endpoint
                 (control / condense / headroom): per-step action agreement + cache-aware
                 compaction, paired (no turn-count noise). Calls the live APIs — costs money.

Shared: --arms (which methods vs the floor), --agent (cc|codex session format), --format (html|md).
Generation is a separate step (scripts/run_cc_matrix.sh drives Harbor); this only reports.

Examples:
  # offline preservation report (the common path)
  python3 scripts/report.py --mode full --tasks kv-store-grpc --root results/sample
  python3 scripts/report.py --mode full --tasks a,b,c --arms condense,headroom --format md --milestones

  # teacher-forced replay of one session (online, needs the request template)
  python3 scripts/report.py --mode replay --session <trial>/.../x.jsonl --arms condense
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# fidelity_replay is the shared session-I/O + replay engine (the one module report.py leans on).
from fidelity_replay import (  # noqa: E402
    build_request, call_api, cost_usd, extract_action, load_env, parse_session, score,
)

MODEL = "claude-sonnet-4-6"
HEADERS = {"anthropic-version": "2023-06-01", "anthropic-beta": "", "user-agent": "report"}
# agent -> glob for a run's session jsonl under a Harbor trial dir (cc is wired; codex is TODO)
SESSION_GLOB = {
    "cc": "agent/sessions/projects/-app/*.jsonl",
    "codex": "agent/sessions/**/*.jsonl",  # best-effort; codex layout not yet confirmed
}


# ------------------------------------------------------------------ trajectory + rework (folded)
def actions(path):
    """List of tool_use/text actions (decision points) for a recorded session."""
    msgs, points = parse_session(path)
    return [extract_action(msgs[i]["content"]) for i in points]


def _read_span(inp):
    """(start, end) line span of a Read; end=inf when no limit. Pagination -> disjoint spans."""
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
    """Redundant re-fetches: re-read of an already-seen file span (range-aware), re-cat, re-run."""
    import re
    CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
    RO = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc)\b")
    read_spans, last_read, last_mod, seen, hits = {}, {}, {}, {}, 0
    for a in acts:
        if a.get("type") != "tool_use":
            continue
        name, inp = a.get("name"), a.get("input", {})
        if name in ("Write", "Edit"):
            fp = inp.get("file_path")
            if fp:
                last_mod[fp] = 1
                read_spans[fp] = []
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
            if CAT.search(cmd) and any(f and f in cmd and f in last_read for f in last_read):
                hits += 1
            c = " ".join(cmd.split())
            if RO.match(cmd) and c in seen:
                hits += 1
            seen[c] = 1
    return hits


def _label(a):
    if a.get("type") == "text":
        return "(answer)"
    inp = a.get("input", {})
    tgt = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
    return f"{a.get('name', '?')} {str(tgt)[:60]}".strip()


# ------------------------------------------------------------------ milestones (folded, LLM)
EXTRACT = """You are analyzing a coding agent's SOLVED trajectory.
TASK:
{task}

TRAJECTORY (step. action => result):
{summary}

Extract the ORDERED KEY MILESTONES — substantive checkpoints of real progress. Make them
APPROACH-AGNOSTIC: checkpoints ANY valid solution hits, NOT tied to this run's tactic. Ignore
tactical noise and tool specifics. 5-8 milestones covering understanding, strategy, implementing,
verifying, meeting the goal.
Return ONLY JSON: {{"milestones":[{{"id":1,"name":"...","evidence":"..."}}]}}"""

COVER = """TASK:
{task}

MILESTONES (from a reference solution):
{milestones}

ANOTHER agent's trajectory (step. action => result):
{summary}

For EACH milestone, did THIS agent achieve it (judge from evidence)?
Return ONLY JSON: {{"coverage":[{{"id":1,"achieved":true}}]}}"""


def summarize(path, max_steps=80):
    msgs, points = parse_session(path)
    task = next((b["text"] for b in msgs[0]["content"] if b.get("type") == "text"), "")[:600]
    lines = []
    for n, i in enumerate(points[:max_steps]):
        act = _label(extract_action(msgs[i]["content"]))
        res = ""
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
            for b in msgs[i + 1]["content"]:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    res = (c if isinstance(c, str)
                           else " ".join(x.get("text", "") for x in c if isinstance(x, dict)))[:180]
                    break
        lines.append(f"{n}. {act} => {res}".replace("\n", " "))
    return task, "\n".join(lines)


def ask(prompt, env):
    req = {"model": MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}
    resp, err = call_api("control", req, HEADERS, env)
    if err:
        return None
    text = "".join(b.get("text", "") for b in resp["content"]).strip()
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------------ distributions / verdict
def band(xs):
    return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None


def overlaps(a, b):
    if not a or not b:
        return None
    return a[0] <= b[2] and b[0] <= a[2]


def sessions(root, arm, task, agent):
    """Pooled (session_path, reward) for (arm, task) across every job dir under root."""
    out, seen = [], set()
    for rt in sorted(glob.glob(f"{root}/*/{arm}-{task}/*/*/verifier/reward.txt")):
        inst = os.path.dirname(os.path.dirname(rt))
        s = glob.glob(os.path.join(inst, SESSION_GLOB[agent]), recursive=True)
        if s and s[0] not in seen:
            seen.add(s[0])
            out.append((s[0], open(rt).read().strip()))
    return out


# ------------------------------------------------------------------ FULL mode
def full_report(args, env):
    arms = [a for a in args.arms.split(",") if a]
    rows = []
    for task in args.tasks.split(","):
        van = sessions(args.root, "vanilla", task, args.agent)
        vlen = [len(actions(p)) for p, _ in van]
        vrw = [rework_count(actions(p)) for p, _ in van]
        vsolve = sum(r == "1" for _, r in van)
        row = {"task": task, "vanilla": {"n": len(van), "solve": vsolve,
                                         "length": band(vlen), "rework": band(vrw)}, "arms": {}}
        ms = None
        if args.milestones and van:
            ref = next((p for p, r in van if r == "1"), van[0][0])
            t, s = summarize(ref)
            m = ask(EXTRACT.format(task=t, summary=s), env)
            ms = (ref, t, m["milestones"]) if m else None
        for arm in arms:
            runs = sessions(args.root, arm, task, args.agent)
            alen = [len(actions(p)) for p, _ in runs]
            arw = [rework_count(actions(p)) for p, _ in runs]
            enough = len(vlen) >= 2 and len(alen) >= 2
            mcov = None
            if ms and runs:
                _, t, mil = ms
                covs = []
                for p, _ in runs[:3]:
                    _, s = summarize(p)
                    c = ask(COVER.format(task=t, milestones=json.dumps(mil), summary=s), env)
                    if c:
                        covs.append(sum(bool(x.get("achieved")) for x in c["coverage"]) / len(mil))
                mcov = band(covs)
            row["arms"][arm] = {
                "n": len(runs), "solve": sum(r == "1" for _, r in runs),
                "length": band(alen), "rework": band(arw), "milestone": mcov,
                "length_ok": (overlaps(band(vlen), band(alen)) if enough else None),
                "rework_ok": (overlaps(band(vrw), band(arw)) if enough else None),
            }
        rows.append(row)
        L = row["arms"].get(arms[0], {}) if arms else {}
        print(f"{task:24} vanilla len={_b(row['vanilla']['length'])} "
              + " ".join(f"{a} len={_b(row['arms'][a]['length'])} "
                         f"{_v(row['arms'][a]['length_ok'])}" for a in arms))
    return {"mode": "full", "arms": arms, "milestones": args.milestones, "rows": rows}


# ------------------------------------------------------------------ REPLAY mode
def replay_report(args, env):
    import copy
    import uuid
    cap = json.load(open(args.template))
    tmpl_body = cap["body"]
    tmpl_headers = {k.lower(): v for k, v in cap["headers"].items()}
    msgs, points = parse_session(args.session)
    arms = ["control"] + [a for a in args.arms.split(",") if a]
    sid = "report-replay-fixed"
    per_arm = {a: [] for a in arms}
    spent = 0.0
    for step, i in enumerate(points[:: max(1, args.every)]):
        if args.limit and step >= args.limit:
            break
        orig = extract_action(msgs[i]["content"])
        for arm in arms:
            if spent >= args.budget_usd:
                break
            req = build_request(tmpl_body, copy.deepcopy(msgs[:i]), args, sid)
            resp, err = call_api(arm, req, tmpl_headers, env)
            if err:
                continue
            rep = extract_action(resp.get("content", []))
            _, agree, _ = score(orig, rep)
            u = resp.get("usage", {})
            ctx = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get(
                "cache_creation_input_tokens", 0)
            c = cost_usd(u)
            spent += c
            per_arm[arm].append({"step": step, "agree": agree, "ctx": ctx, "cost": c})
        print(f"  step {step}: " + " ".join(f"{a}={'✓' if per_arm[a] and per_arm[a][-1]['agree'] else '·'}"
                                             for a in arms))
    ctrl = per_arm["control"]

    def agg(a):
        rows = per_arm[a]
        n = len(rows)
        rfid = sum(r["agree"] for r in rows) / n if n else None
        comp = costd = None
        if a != "control" and ctrl:
            common = min(len(rows), len(ctrl))
            cc = sum(ctrl[k]["ctx"] for k in range(1, common)) or 1  # exclude step 0 (cold cache)
            ac = sum(rows[k]["ctx"] for k in range(1, common))
            oc = sum(ctrl[k]["cost"] for k in range(1, common)) or 1e-9
            zc = sum(rows[k]["cost"] for k in range(1, common))
            comp, costd = round(1 - ac / cc, 4), round(1 - zc / oc, 4)
        return {"n": n, "rfid": rfid, "comp": comp, "costd": costd}
    return {"mode": "replay", "session": os.path.basename(args.session),
            "arms": arms, "results": {a: agg(a) for a in arms}, "spent": round(spent, 2)}


# ------------------------------------------------------------------ rendering
def _b(x):
    return "—" if not x else (f"{x[1]:.0f}[{x[0]}-{x[2]}]" if x[0] != x[2] else f"{x[1]:.0f}")


def _v(ok):
    return "OK" if ok else ("DIVERGES" if ok is False else "—")


def _pct(x):
    return "—" if x is None else f"{x * 100:+.0f}%"


def render_md(data):
    out = []
    if data["mode"] == "full":
        out.append("# Trajectory preservation\n")
        out.append("vanilla `k≈3` = the floor; a method **✓** if its distribution overlaps vanilla's, "
                   "**✗** if disjoint (needs ≥2 runs/arm).\n")
        cols = ["task", "solve (v·arm)", "length vanilla", "length arm", "len ✓?", "rework ✓?"]
        for arm in data["arms"]:
            out.append(f"\n## arm: `{arm}`\n")
            out.append("| " + " | ".join(cols) + " |")
            out.append("|" + "---|" * len(cols))
            for r in data["rows"]:
                a = r["arms"][arm]
                out.append(f"| {r['task']} | {r['vanilla']['solve']}/{r['vanilla']['n']} · "
                           f"{a['solve']}/{a['n']} | {_b(r['vanilla']['length'])} | {_b(a['length'])} "
                           f"| {_v(a['length_ok'])} | {_v(a['rework_ok'])} |")
    else:
        out.append(f"# Teacher-forced replay — `{data['session']}`  (spent ${data['spent']})\n")
        out.append("| arm | steps | rfid | comp | $Δ |")
        out.append("|---|---|---|---|---|")
        for a in data["arms"]:
            r = data["results"][a]
            out.append(f"| {a} | {r['n']} | {_pct(r['rfid'])} | "
                       f"{'ref' if a == 'control' else _pct(r['comp'])} | "
                       f"{'ref' if a == 'control' else _pct(r['costd'])} |")
    return "\n".join(out) + "\n"


def render_html(data):
    body = render_md(data)
    # minimal: wrap the markdown-ish table in <pre> is ugly; instead build a real table.
    import html as _h
    rows_html = []
    if data["mode"] == "full":
        title = "Trajectory preservation"
        sub = "vanilla k≈3 = floor · ✓ overlaps vanilla · ✗ disjoint (≥2 runs/arm)"
        head = "<tr><th>task</th><th>arm</th><th>solve v·arm</th><th>vanilla len</th><th>arm len</th><th>length</th><th>rework</th><th>milestone</th></tr>"
        for r in data["rows"]:
            for arm in data["arms"]:
                a = r["arms"][arm]
                mc = a.get("milestone")
                rows_html.append(
                    f"<tr><td>{_h.escape(r['task'])}</td><td>{arm}</td>"
                    f"<td>{r['vanilla']['solve']}/{r['vanilla']['n']} · {a['solve']}/{a['n']}</td>"
                    f"<td>{_b(r['vanilla']['length'])}</td><td>{_b(a['length'])}</td>"
                    f"<td class='{ 'ok' if a['length_ok'] else 'bad' if a['length_ok'] is False else '' }'>{_v(a['length_ok'])}</td>"
                    f"<td class='{ 'ok' if a['rework_ok'] else 'bad' if a['rework_ok'] is False else '' }'>{_v(a['rework_ok'])}</td>"
                    f"<td>{'' if not mc else str(round(mc[1]*100))+'%'}</td></tr>")
    else:
        title = f"Teacher-forced replay — {_h.escape(data['session'])}"
        sub = f"paired per-step; spent ${data['spent']}"
        head = "<tr><th>arm</th><th>steps</th><th>rfid</th><th>comp</th><th>$Δ</th></tr>"
        for a in data["arms"]:
            r = data["results"][a]
            rows_html.append(
                f"<tr><td>{a}</td><td>{r['n']}</td><td>{_pct(r['rfid'])}</td>"
                f"<td>{'ref' if a=='control' else _pct(r['comp'])}</td>"
                f"<td>{'ref' if a=='control' else _pct(r['costd'])}</td></tr>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title><style>
 body{{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;background:#0f1117;color:#d8dce6}}
 h1{{font-size:19px}}.sub{{color:#8b93a3;margin-bottom:14px}}
 table{{border-collapse:collapse}} th,td{{border:1px solid #232838;padding:6px 10px;text-align:center;font-size:12px}}
 th{{background:#161a22;color:#9aa4b5}} .ok{{color:#3fb950;font-weight:700}}.bad{{color:#e5534b;font-weight:700}}
</style></head><body><h1>{title}</h1><div class="sub">{sub}</div>
<table>{head}{''.join(rows_html)}</table></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="quality / trajectory-preservation bench — one report command")
    ap.add_argument("--mode", choices=["full", "replay"], default="full")
    ap.add_argument("--arms", default="condense", help="comma list: condense[,headroom]")
    ap.add_argument("--agent", choices=["cc", "codex"], default="cc")
    ap.add_argument("--format", choices=["html", "md"], default="html")
    ap.add_argument("--out", default=None)
    # full mode
    ap.add_argument("--tasks", help="comma list (full mode)")
    ap.add_argument("--root", default="results/jobs")
    ap.add_argument("--milestones", action="store_true")
    # replay mode
    ap.add_argument("--session", help="reference trajectory jsonl (replay mode)")
    ap.add_argument("--template", default="data/cc_request_template.json")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--strip-thinking", action="store_true")
    ap.add_argument("--swechat", default=None)
    args = ap.parse_args()
    env = {**load_env(), **os.environ}

    if args.agent == "codex" and args.mode == "full":
        sys.exit("--agent codex is not yet wired for --mode full (codex session layout unconfirmed); "
                 "it works for --mode replay via fidelity_replay's SWE-chat parser.")
    if args.mode == "full":
        if not args.tasks:
            sys.exit("--mode full needs --tasks")
        data = full_report(args, env)
    else:
        if not args.session:
            sys.exit("--mode replay needs --session")
        data = replay_report(args, env)

    out = args.out or f"report.{args.format}"
    txt = render_md(data) if args.format == "md" else render_html(data)
    with open(out, "w") as f:
        f.write(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
