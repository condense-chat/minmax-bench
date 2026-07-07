#!/usr/bin/env python3
"""report.py — DISPLAY only. Reads what `generate.py` produced and renders it. Never spends.

Mirrors the sibling cost bench's `report` ("recompute from stored data, verify without re-spending"):
generation writes artifacts; this reads them and computes the offline, deterministic views.

Reads from a results root (`--from`):
  - full run dirs      <root>/*/<arm>-<task>/*/*/{verifier/reward.txt, agent/sessions/.../*.jsonl}
  - milestones.json    <root>/milestones.json     (optional; produced by `generate --milestones`)
  - incremental jsonl  <root>/incremental/<task>-<arm>.jsonl  (optional; `generate --mode incremental`)

Offline axes (always, no model calls): length / rework / solve, each vs the vanilla noise floor
(✓ overlap = indistinguishable, ✗ disjoint; needs ≥2 runs/arm). Milestone + incremental axes appear
only if their artifacts exist.

  python3 scripts/report.py --from results/sample --tasks kv-store-grpc
  python3 scripts/report.py --from results/jobs --tasks a,b --arms condense,headroom --format md
"""
import argparse
import copy
import glob
import json
import os
import re
import sys

AGENT_SESSION_GLOB = {  # only claude-code is wired; others are TODO
    "claude-code": "agent/sessions/projects/-app/*.jsonl",
}


# ---------------------------------------------------------------- session parsing (folded, offline)
def parse_session(path):
    """CC session JSONL -> API-shaped messages + assistant decision-point indices."""
    groups = []
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("isSidechain") or rec.get("type") not in ("user", "assistant"):
            continue
        m = rec["message"]
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        content = copy.deepcopy(content)
        req = rec.get("requestId") if rec["type"] == "assistant" else None
        if groups and groups[-1]["role"] == m["role"] and (
                rec["type"] == "user" or groups[-1]["req"] == req):
            groups[-1]["content"].extend(content)
        else:
            groups.append({"role": m["role"], "content": content, "req": req,
                           "stop": m.get("stop_reason")})
        if rec["type"] == "assistant":
            groups[-1]["stop"] = m.get("stop_reason") or groups[-1]["stop"]
    msgs = []
    for g in groups:
        thinking_only = all(b.get("type") in ("thinking", "redacted_thinking") for b in g["content"])
        if g["role"] == "assistant" and g["stop"] == "max_tokens" and thinking_only:
            continue
        if msgs and msgs[-1]["role"] == g["role"]:
            msgs[-1]["content"].extend(g["content"])
        else:
            msgs.append({"role": g["role"], "content": g["content"]})
    return msgs, [i for i, m in enumerate(msgs) if m["role"] == "assistant"]


def extract_action(content):
    for b in content:
        if b.get("type") == "tool_use":
            return {"type": "tool_use", "name": b["name"], "input": b.get("input", {})}
    return {"type": "text"}


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
    """Redundant re-fetches: re-read of an already-seen file span (range-aware), re-cat, re-run."""
    CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
    RO = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc)\b")
    read_spans, last_read, seen, hits = {}, {}, {}, 0
    for a in acts:
        if a.get("type") != "tool_use":
            continue
        name, inp = a.get("name"), a.get("input", {})
        if name in ("Write", "Edit"):
            if inp.get("file_path"):
                read_spans[inp["file_path"]] = []
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


def runs_for(root, arm, task, agent):
    # match <arm>-<task> whether directly under root (a single generate --out) or one/more
    # dirs deep (pooling several job dirs). ** matches zero-or-more path segments.
    out, seen = [], set()
    for rt in sorted(glob.glob(f"{root}/**/{arm}-{task}/*/*/verifier/reward.txt", recursive=True)):
        inst = os.path.dirname(os.path.dirname(rt))
        s = glob.glob(os.path.join(inst, AGENT_SESSION_GLOB[agent]), recursive=True)
        if s and s[0] not in seen:
            seen.add(s[0])
            out.append((s[0], open(rt).read().strip()))
    return out


# ---------------------------------------------------------------- assemble (from artifacts only)
def build(args):
    arms = [a for a in args.arms.split(",") if a]
    milestones = {}
    mpath = os.path.join(args.__dict__["from"], "milestones.json")
    if os.path.exists(mpath):
        milestones = json.load(open(mpath))  # {task: {arm: [min,mean,max]}}
    incr = _load_incremental(os.path.join(args.__dict__["from"], "incremental"), arms)
    rows = []
    for task in args.tasks.split(","):
        van = runs_for(args.__dict__["from"], "vanilla", task, args.agent)
        vlen, vrw = [], []
        for p, _ in van:
            acts = actions(p)
            vlen.append(len(acts))
            vrw.append(rework_count(acts))
        row = {"task": task, "vanilla": {"n": len(van), "solve": sum(r == "1" for _, r in van),
                                         "length": band(vlen), "rework": band(vrw)}, "arms": {}}
        for arm in arms:
            runs = runs_for(args.__dict__["from"], arm, task, args.agent)
            alen, arw = [], []
            for p, _ in runs:
                acts = actions(p)
                alen.append(len(acts))
                arw.append(rework_count(acts))
            enough = len(vlen) >= 2 and len(alen) >= 2
            mv = (milestones.get(task, {}) or {}).get("vanilla")
            mc = (milestones.get(task, {}) or {}).get(arm)
            row["arms"][arm] = {
                "n": len(runs), "solve": sum(r == "1" for _, r in runs),
                "length": band(alen), "rework": band(arw),
                "length_ok": overlaps(band(vlen), band(alen)) if enough else None,
                "rework_ok": overlaps(band(vrw), band(arw)) if enough else None,
                "milestone": mc, "milestone_ok": (overlaps(mv, mc) if (mv and mc) else None),
                "incr": incr.get((task, arm)),
            }
        rows.append(row)
    return {"arms": arms, "rows": rows, "has_milestone": bool(milestones), "has_incr": bool(incr)}


def _load_incremental(dirpath, arms):
    """Read generate --mode incremental jsonl -> per (task,arm) comp% / costΔ / fidelity vs control."""
    if not os.path.isdir(dirpath):
        return {}
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
        return u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get(
            "cache_creation_input_tokens", 0)
    out = {}
    for cf in glob.glob(f"{dirpath}/*-control.jsonl"):
        task = os.path.basename(cf)[:-len("-control.jsonl")]
        C = rows(cf)
        for arm in arms:
            A = rows(f"{dirpath}/{task}-{arm}.jsonl")
            if not A or not C:
                continue
            common = sorted(s for s in (set(A) & set(C)) if s > 0)  # exclude step 0 (cold cache)
            cc = sum(ctx(C[s]["usage"]) for s in common) or 1
            ac = sum(ctx(A[s]["usage"]) for s in common)
            oc = sum(C[s].get("cost_usd", 0) for s in common) or 1e-9
            zc = sum(A[s].get("cost_usd", 0) for s in common)
            rfid = sum(bool(A[s].get("agree_action")) for s in A) / len(A) if A else None
            out[(task, arm)] = {"comp": round(1 - ac / cc, 4), "costd": round(1 - zc / oc, 4),
                                "rfid": rfid, "steps": len(A)}
    return out


# ---------------------------------------------------------------- rendering
def _b(x):
    return "—" if not x else (f"{x[1]:.0f}[{x[0]}-{x[2]}]" if x[0] != x[2] else f"{x[1]:.0f}")


def _v(ok):
    return "OK" if ok else ("DIVERGES" if ok is False else "—")


def _pct(x):
    return "—" if x is None else f"{x * 100:+.0f}%"


def render_md(d):
    o = ["# Trajectory preservation\n",
         "vanilla = the noise floor; a method is **✓** if its distribution overlaps vanilla's, "
         "**✗** if disjoint (needs ≥2 runs/arm). No model was called to produce this.\n"]
    for arm in d["arms"]:
        o.append(f"\n## `{arm}` vs vanilla\n")
        cols = ["task", "solve v·arm", "length (v)", "length (arm)", "len", "rework"]
        if d["has_milestone"]:
            cols.append("milestone")
        if d["has_incr"]:
            cols += ["comp", "$Δ"]
        o.append("| " + " | ".join(cols) + " |")
        o.append("|" + "---|" * len(cols))
        for r in d["rows"]:
            a = r["arms"][arm]
            cells = [r["task"], f"{r['vanilla']['solve']}/{r['vanilla']['n']} · {a['solve']}/{a['n']}",
                     _b(r["vanilla"]["length"]), _b(a["length"]), _v(a["length_ok"]), _v(a["rework_ok"])]
            if d["has_milestone"]:
                cells.append(_v(a["milestone_ok"]))
            if d["has_incr"]:
                inc = a.get("incr") or {}
                cells += [_pct(inc.get("comp")), _pct(inc.get("costd"))]
            o.append("| " + " | ".join(cells) + " |")
    return "\n".join(o) + "\n"


def render_html(d):
    import html as H
    head = ["task", "arm", "solve v·arm", "length (v)", "length (arm)", "len", "rework"]
    if d["has_milestone"]:
        head.append("milestone")
    if d["has_incr"]:
        head += ["comp", "$Δ"]
    trs = []
    for r in d["rows"]:
        for arm in d["arms"]:
            a = r["arms"][arm]
            cls = lambda ok: "ok" if ok else ("bad" if ok is False else "")  # noqa: E731
            tds = [H.escape(r["task"]), arm,
                   f"{r['vanilla']['solve']}/{r['vanilla']['n']} · {a['solve']}/{a['n']}",
                   _b(r["vanilla"]["length"]), _b(a["length"]),
                   f"<span class='{cls(a['length_ok'])}'>{_v(a['length_ok'])}</span>",
                   f"<span class='{cls(a['rework_ok'])}'>{_v(a['rework_ok'])}</span>"]
            if d["has_milestone"]:
                tds.append(f"<span class='{cls(a['milestone_ok'])}'>{_v(a['milestone_ok'])}</span>")
            if d["has_incr"]:
                inc = a.get("incr") or {}
                tds += [_pct(inc.get("comp")), _pct(inc.get("costd"))]
            trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in tds) + "</tr>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Trajectory preservation</title>
<style>body{{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;background:#0f1117;color:#d8dce6}}
h1{{font-size:19px}}.sub{{color:#8b93a3;margin-bottom:14px;max-width:900px}}
table{{border-collapse:collapse}} th,td{{border:1px solid #232838;padding:6px 10px;text-align:center;font-size:12px}}
th{{background:#161a22;color:#9aa4b5}} .ok{{color:#3fb950;font-weight:700}}.bad{{color:#e5534b;font-weight:700}}</style>
</head><body><h1>Trajectory preservation</h1>
<div class="sub">vanilla = noise floor · ✓ overlaps vanilla · ✗ disjoint (≥2 runs/arm). Display only — no model called.</div>
<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(trs)}</table></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="display the quality bench (reads generate.py's artifacts)")
    ap.add_argument("--from", default="results/jobs", help="results root produced by generate.py")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--agent", default="claude-code", choices=list(AGENT_SESSION_GLOB))
    ap.add_argument("--format", default="html", choices=["html", "md"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = build(args)
    for r in d["rows"]:
        for arm in d["arms"]:
            a = r["arms"][arm]
            print(f"{r['task']:22} {arm:9} len v={_b(r['vanilla']['length'])} "
                  f"arm={_b(a['length'])} {_v(a['length_ok'])}")
    out = args.out or f"report.{args.format}"
    open(out, "w").write(render_md(d) if args.format == "md" else render_html(d))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
