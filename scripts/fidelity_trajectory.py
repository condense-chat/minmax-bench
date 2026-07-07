#!/usr/bin/env python3
"""Full-trajectory comparison: align vanilla vs each method (condense, headroom) node-by-node,
and score COMPRESSION (total tokens) + FAITHFULNESS (aligned-node agreement, vs a vanilla-vs-
vanilla noise floor). Complements the teacher-forced replay: this captures END-TO-END behavior
(including cross-turn methods like headroom CCR) at the cost of free-running divergence — which
the noise floor accounts for.

Alignment: Needleman-Wunsch (order-preserving) on action sequences; equivalence = same tool +
same target. Method-specific tool calls (retrieve/memory) are tallied as CCR overhead, not
counted against faithfulness.

Usage:
  python3 scripts/fidelity_trajectory.py --out results/fidelity/trajectory.html \
    --vanilla LABEL=<session.jsonl> --vanilla2 <session.jsonl> \
    --method condense=<session.jsonl> --method headroom=<session.jsonl>
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity_replay import extract_action, parse_session  # noqa: E402

_TOK = re.compile(r"[A-Za-z0-9_./-]+")
OVERHEAD = ("retrieve", "memory")  # headroom CCR tool names (substring match)


def is_overhead(a):
    n = (a.get("name") or "").lower()
    return any(w in n for w in OVERHEAD)


_CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
_RO = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc)\b")


def _read_span(inp):
    """(start, end) line span of a Read. end=inf when no limit (reads to EOF). Paginating a file
    (offset 1/500/1000…) yields disjoint spans — NOT redundant; only a re-read of an already-seen
    span is."""
    off = inp.get("offset")
    start = off if isinstance(off, int) and off > 0 else 1
    lim = inp.get("limit")
    return start, (start + lim if isinstance(lim, int) and lim > 0 else float("inf"))


def _covered(spans, s, e):
    """True if [s,e) is fully covered by the union of prior [start,end) spans."""
    cur = s
    for a, b in sorted(spans):
        if a > cur:
            break
        cur = max(cur, b)
        if cur >= e:
            return True
    return cur >= e


def redundant_indices(acts):
    """Action indices that re-fetch info the run already had (re-read of an already-seen file span /
    re-cat / identical read-only re-run) — the compaction-amnesia signal, self-contained per run.
    Range-aware: sequential pagination of a file is not redundant; re-reading a covered span is."""
    read_spans, last_read, last_mod, seen = {}, {}, {}, {}
    out = set()
    for idx, a in enumerate(acts):
        if a.get("type") != "tool_use":
            continue
        name, inp = a.get("name"), a.get("input", {})
        if name in ("Write", "Edit"):
            fp = inp.get("file_path")
            if fp:
                last_mod[fp] = idx
                read_spans[fp] = []          # content changed: prior reads are stale
        elif name == "Read":
            fp = inp.get("file_path")
            s, e = _read_span(inp)
            if fp and read_spans.get(fp) and _covered(read_spans[fp], s, e):
                out.add(idx)
            if fp:
                read_spans.setdefault(fp, []).append((s, e))
                last_read[fp] = idx
        elif name == "Bash":
            cmd = inp.get("command", "")
            if _CAT.search(cmd) and any(f and f in cmd and last_mod.get(f, -1) <= last_read[f]
                                        for f in last_read):
                out.add(idx)
            c = " ".join(cmd.split())
            if _RO.match(cmd) and c in seen:
                out.add(idx)
            seen[c] = idx
    return out


def label(a):
    if a.get("type") == "text":
        return "final"
    n = a.get("name", "?")
    inp = a.get("input", {})
    if n in ("Read", "Write", "Edit"):
        return f"{n}({os.path.basename(inp.get('file_path', '?'))})"
    if n == "Bash":
        return "Bash: " + " ".join(_TOK.findall(inp.get("command", ""))[:5])
    return n


def detail(a):
    if a.get("type") == "text":
        return a.get("text", "")[:900]
    return f"{a.get('name')}: " + json.dumps(a.get("input", {}))[:900]


def equiv(a, b):
    if a.get("type") != b.get("type"):
        return False
    if a.get("type") == "text":
        return True
    if a.get("name") != b.get("name"):
        return False
    ia, ib = a.get("input", {}), b.get("input", {})
    if a["name"] in ("Read", "Write", "Edit"):
        return ia.get("file_path") == ib.get("file_path")
    if a["name"] == "Bash":
        ta, tb = set(_TOK.findall(ia.get("command", ""))), set(_TOK.findall(ib.get("command", "")))
        return bool(ta) and len(ta & tb) / len(ta | tb) >= 0.5
    return json.dumps(ia, sort_keys=True) == json.dumps(ib, sort_keys=True)


def nw_align(A, B):
    """Needleman-Wunsch -> list of (i|None, j|None) aligned columns. sub 0/1, gap 1."""
    n, m = len(A), len(B)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = 0 if equiv(A[i - 1], B[j - 1]) else 1
            dp[i][j] = min(dp[i - 1][j - 1] + s, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if equiv(A[i - 1], B[j - 1]) else 1):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def observation(msgs, i):
    """The tool_result (observation) following decision point i, truncated."""
    if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
        for b in msgs[i + 1]["content"]:
            if b.get("type") == "tool_result":
                c = b.get("content")
                return (c if isinstance(c, str)
                        else " ".join(x.get("text", "") for x in c if isinstance(x, dict)))[:1200]
    return ""


def load(path):
    msgs, points = parse_session(path)
    acts = [extract_action(msgs[i]["content"]) for i in points]
    obs = [observation(msgs, i) for i in points]
    tot = 0
    for line in open(path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            u = e["message"].get("usage", {})
            tot += (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                    + u.get("cache_creation_input_tokens", 0))
    return acts, obs, tot


def score(vacts, macts):
    """faithfulness (aligned agreement over non-overhead pairs), overhead count."""
    al = nw_align(vacts, macts)
    matched = pairs = overhead = 0
    for i, j in al:
        if j is not None and is_overhead(macts[j]):
            overhead += 1
            continue
        if i is not None and j is not None:
            pairs += 1
            matched += equiv(vacts[i], macts[j])
    return (matched / pairs if pairs else 0.0), overhead, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", required=True, help="session.jsonl (the reference)")
    ap.add_argument("--vanilla2", help="a 2nd vanilla run for the noise floor")
    ap.add_argument("--method", action="append", default=[], help="name=session.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vacts, vobs, vtot = load(args.vanilla)
    floor = None
    if args.vanilla2:
        v2, _, _ = load(args.vanilla2)
        floor = score(vacts, v2)[0]

    cols = {"vanilla": vacts}
    obs_by = {"vanilla": vobs}
    metrics = {}
    aligns = {}
    for spec in args.method:
        name, path = spec.split("=", 1)
        macts, mobs, mtot = load(path)
        cols[name] = macts
        obs_by[name] = mobs
        faith, overhead, al = score(vacts, macts)
        aligns[name] = al
        metrics[name] = {
            "compression": round(1 - mtot / vtot, 3) if vtot else 0,
            "faithfulness_raw": round(faith, 3),
            "faithfulness_vs_floor": round(faith / floor, 3) if floor else None,
            "overhead": overhead, "steps": len(macts), "tokens": mtot,
        }

    # vanilla-anchored rows: for each vanilla step, the method steps aligned to it (+ insertions)
    rows = []
    per = {name: {} for name in aligns}  # name -> {vanilla_idx: [method_idx...]}, plus 'ins' before
    ins = {name: {} for name in aligns}
    for name, al in aligns.items():
        last_v = -1
        for i, j in al:
            if i is not None:
                last_v = i
                if j is not None:
                    per[name].setdefault(i, []).append(j)
            elif j is not None:
                ins[name].setdefault(last_v, []).append(j)
    for v in range(len(vacts)):
        # leading insertions for this vanilla anchor (methods did an extra step here)
        maxins = max((len(ins[n].get(v - 1, [])) for n in aligns), default=0)
        for k in range(maxins):
            r = {"v": None}
            for n in aligns:
                lst = ins[n].get(v - 1, [])
                r[n] = lst[k] if k < len(lst) else None
            rows.append(r)
        r = {"v": v}
        for n in aligns:
            lst = per[n].get(v, [])
            r[n] = lst[0] if lst else None
        rows.append(r)

    redset = {n: redundant_indices(cols[n]) for n in cols}
    for n in cols:
        metrics.setdefault(n, {})["redundant"] = len(redset[n])
        metrics[n].setdefault("steps", len(cols[n]))
    metrics["vanilla"]["compression"] = 0

    def cell(name, idx):
        if idx is None:
            return None
        a = cols[name][idx]
        return {"label": label(a), "detail": detail(a), "obs": obs_by[name][idx],
                "eq": None, "overhead": is_overhead(a), "redundant": idx in redset[name]}
    data = {"names": list(cols), "vtot": vtot, "floor": floor, "metrics": metrics,
            "rows": []}
    for r in rows:
        row = {"vanilla": cell("vanilla", r["v"])}
        vidx = r["v"]
        for n in aligns:
            c = cell(n, r[n])
            if c and vidx is not None and r[n] is not None and not c["overhead"]:
                c["eq"] = equiv(vacts[vidx], cols[n][r[n]])
            row[n] = c
        data["rows"].append(row)

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")
    print(f"vanilla: {len(vacts)} steps, {vtot:,} tok; floor faith={floor}")
    for n, mt in metrics.items():
        print(f"  {n}: {mt.get('steps', 0)} steps, compression {mt.get('compression', 0):+.0%}, "
              f"redundant {mt.get('redundant', 0)}"
              + (f", node-faith {mt['faithfulness_raw']:.2f}" if "faithfulness_raw" in mt else ""))


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trajectory explorer — how each solved run reached the goal</title>
<style>
 body{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:20px;background:#0f1117;color:#d8dce6}
 h1{font-size:18px;margin-bottom:4px}.sub{color:#8b93a3;max-width:960px;margin-bottom:14px}
 .legend{display:flex;gap:16px;margin-bottom:14px;font-size:12px;color:#9aa4b5;flex-wrap:wrap;align-items:center}
 .legend b{color:#d8dce6} .chip{display:inline-flex;align-items:center;gap:6px}
 .sw{width:22px;height:14px;border-radius:3px;display:inline-block}
 .sw.red{background:#2d1618;box-shadow:inset 0 0 0 2px #e5534b}.sw.pur{background:#211a2e;box-shadow:inset 0 0 0 2px #a371f7}.sw.norm{background:#1b2029}
 .summary{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
 .card{background:#161a22;border-radius:8px;padding:10px 14px;min-width:150px;border-top:3px solid #333}
 .card h3{margin:0 0 4px;font-size:13px}.card .big{font-size:15px}.card .row{font-size:12px;color:#9aa4b5;margin-top:2px}
 .wrap{overflow-x:auto;border:1px solid #1b2029;border-radius:8px}
 table{border-collapse:collapse;width:100%;table-layout:fixed}
 col.idxcol{width:34px} col.runcol{width:340px}
 th{position:sticky;top:0;background:#161a22;font-size:12px;padding:8px;text-align:left;border-bottom:2px solid #232838;z-index:2}
 td{padding:5px 8px;border-bottom:1px solid #14181f;border-left:1px solid #14181f;font-size:11px;vertical-align:top;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .idx{color:#4a5160;text-align:right;background:#12161e}
 .rw{background:#2d1618;box-shadow:inset 3px 0 0 #e5534b}
 .ovh{background:#211a2e;box-shadow:inset 3px 0 0 #a371f7;color:#c9a3ff}
 .gap{color:#2a3038;background:#0d1016;text-align:center}
 td:not(.gap):hover{background:#232a35}
 .marker{color:#e5534b;font-weight:700;margin-right:3px}.pmark{color:#a371f7}
 #det{position:sticky;bottom:0;background:#12161e;border-top:1px solid #232838;padding:10px 12px;font-size:11px;max-height:230px;overflow:auto}
 #det .h{color:#9aa4b5;margin-top:6px}
 pre{white-space:pre-wrap;word-break:break-word;margin:3px 0;color:#c9d1e0;background:#0d1016;padding:6px;border-radius:5px}
</style></head><body>
<h1>Trajectory explorer — how each <span style="color:#3fb950">solved</span> run reached the goal</h1>
<div class="sub">Every column is a <b>successful</b> run of the same task. Columns are aligned to vanilla only to line up comparable moments — <b>divergence is normal, not error</b> (free-running agents wander different valid routes; two vanilla runs already share only ~{FLOOR}% of steps). The signal that actually reflects compression <i>quality</i> is <b class="marker">↻ redundant re-work</b> — a step that re-fetches info the run already had (compaction amnesia). Click any cell to see the action <b>and its observation</b>.</div>
<div class="legend">
 <span class="chip"><i class="sw norm"></i> action</span>
 <span class="chip"><i class="sw red"></i> <b class="marker">↻</b> redundant re-work (amnesia)</span>
 <span class="chip"><i class="sw pur"></i> retrieve/memory overhead</span>
 <span class="chip"><i class="sw" style="background:#0d1016"></i> · no aligned step</span>
</div>
<div class="summary" id="sum"></div>
<div class="wrap"><table id="tbl"><colgroup id="cg"></colgroup><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
<div id="det">Click a cell to inspect its action + observation.</div>
<script>
const D=__DATA__;
const C={vanilla:'#8b93a3',vanilla2:'#8b93a3',condense:'#f778ba',headroom:'#58a6ff','headroom-ccr':'#a371f7'};
const col=n=>C[n]||'#9aa4b5';
document.querySelector('.sub').innerHTML=document.querySelector('.sub').innerHTML.replace('{FLOOR}',D.floor!=null?Math.round(D.floor*100):'—');
let sh='';
for(const n of D.names){const m=D.metrics[n]||{};const red=m.redundant||0;
 sh+=`<div class="card" style="border-top-color:${col(n)}"><h3 style="color:${col(n)}">${n}</h3>`+
 `<div class="big">${m.steps||0} steps${n!=='vanilla'&&m.compression!=null?` · ${m.compression>=0?'−':'+'}${Math.abs(Math.round(m.compression*100))}% tok`:''}</div>`+
 `<div class="row">${red?`<span class="marker">↻ ${red}</span> redundant re-work`:'✓ no redundant re-work'}</div></div>`;}
document.getElementById('sum').innerHTML=sh;
document.getElementById('cg').innerHTML='<col class="idxcol">'+D.names.map(()=>'<col class="runcol">').join('');
document.getElementById('head').innerHTML='<th class="idx">#</th>'+D.names.map(n=>`<th style="color:${col(n)}">${n}</th>`).join('');
let b='';
D.rows.forEach((r,ri)=>{
 b+=`<tr><td class="idx">${r.vanilla?ri:''}</td>`;
 for(const n of D.names){const c=r[n];
  if(!c){b+=`<td class="gap">·</td>`;continue;}
  const cls=c.redundant?'rw':(c.overhead?'ovh':'');
  const mk=c.redundant?'<span class="marker">↻</span>':(c.overhead?'<span class="pmark">⇲</span> ':'');
  b+=`<td class="${cls}" title="${esc(c.label)}" onclick="show(${ri},'${n}')">${mk}${esc(c.label)}</td>`;}
 b+='</tr>';});
document.getElementById('body').innerHTML=b;
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function show(ri,n){const c=D.rows[ri][n];if(!c)return;
 let tag='';if(c.redundant)tag=' · <b style="color:#e5534b">↻ REDUNDANT (re-fetched info it already had)</b>';else if(c.overhead)tag=' · <b style="color:#a371f7">retrieve/memory overhead</b>';
 document.getElementById('det').innerHTML=`<b style="color:${col(n)}">${n}</b> · aligned row ${ri}${tag}`+
  `<div class="h">action</div><pre>${esc(c.detail)}</pre>`+(c.obs?`<div class="h">observation (tool result)</div><pre>${esc(c.obs)}</pre>`:'<div class="h" style="color:#4a5160">(no observation captured)</div>');}
</script></body></html>
"""

if __name__ == "__main__":
    main()
