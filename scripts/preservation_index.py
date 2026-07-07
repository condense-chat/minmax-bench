#!/usr/bin/env python3
"""Trajectory-PRESERVATION index — the load-bearing test for the cost eval.

The cost eval measures savings ASSUMING compaction preserves the trajectory (same work, same
length). This bench proves (or breaks) that assumption. The bar is NOT "identical to control"
(two vanilla runs already differ) — it's "indistinguishable from the vanilla-vs-vanilla spread."

Per task: vanilla k=3 establishes the noise FLOOR; condense k=3 is tested against it.
  length    : trajectory length (# decision points) — THE axis the cost assumption rests on
  rework    : redundant re-fetch actions (range-aware; compaction amnesia)
  solve     : verifier pass rate
  milestone : approach-agnostic subgoal coverage (LLM) vs a vanilla reference
Verdict/axis: ✓ if the condense range OVERLAPS the vanilla range (statistically indistinguishable),
✗ if disjoint (compaction changed the trajectory). A task is PRESERVED if every axis passes.
No $/savings column — dollars belong to the separate cost eval, which is only valid where these pass.

Usage:
  python3 scripts/preservation_index.py --kdir results/jobs/kfloor \
    --tasks "bn-fit-modify,..." --out results/fidelity/index.html [--milestones]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity_trajectory import load, redundant_indices  # noqa: E402
from fidelity_milestones import summarize, ask, EXTRACT, COVER  # noqa: E402
from fidelity_replay import load_env  # noqa: E402


def sessions(root, arm, task):
    """All trial sessions for (arm, task), POOLED across every job dir under `root`
    (e.g. results/jobs/kfloor + results/jobs/cc-matrix + ...), so long tasks that only
    fit 1 trial in the clean k=3 batch still get a floor from earlier runs. The `arm-task`
    dir-name prefix keeps this to vanilla/condense only (CCR/headroom dirs are excluded)."""
    out, seen = [], set()
    for rt in sorted(glob.glob(f"{root}/*/{arm}-{task}/*/*/verifier/reward.txt")):
        inst = os.path.dirname(os.path.dirname(rt))
        s = glob.glob(inst + "/agent/sessions/projects/-app/*.jsonl")
        if s and s[0] not in seen:
            seen.add(s[0])
            out.append((s[0], open(rt).read().strip()))
    return out


def run_metrics(path):
    """(length, rework) for one session."""
    acts, _, _ = load(path)
    return len(acts), len(redundant_indices(acts))


def band(xs):
    """(min, mean, max) or None if empty."""
    return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None


def overlaps(a, b):
    """Do ranges a=(min,_,max), b=(min,_,max) overlap? (indistinguishable distributions)."""
    if not a or not b:
        return None
    return a[0] <= b[2] and b[0] <= a[2]


def milestone_cov(ref_sess, sess_list, env):
    """Coverage per session vs milestones extracted from ref_sess (approach-agnostic)."""
    task_txt, vsum = summarize(ref_sess)
    ms, err = ask(EXTRACT.format(task=task_txt, summary=vsum), env)
    if err:
        return {}
    milestones = ms["milestones"]
    cov = {}
    for p in sess_list:
        _, s = summarize(p)
        c, err = ask(COVER.format(task=task_txt, milestones=json.dumps(milestones), summary=s), env)
        cov[p] = (sum(bool(x.get("achieved")) for x in c["coverage"]) / len(milestones)
                  if not err else None)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/jobs",
                    help="pool vanilla/condense runs across every job dir under here")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--milestones", action="store_true")
    args = ap.parse_args()
    env = {**load_env(), **os.environ}

    rows = []
    for task in args.tasks.split(","):
        v = sessions(args.root, "vanilla", task)
        c = sessions(args.root, "condense", task)
        vlen = [run_metrics(p)[0] for p, _ in v]
        clen = [run_metrics(p)[0] for p, _ in c]
        vrw = [run_metrics(p)[1] for p, _ in v]
        crw = [run_metrics(p)[1] for p, _ in c]
        vsolve = sum(r == "1" for _, r in v)
        csolve = sum(r == "1" for _, r in c)

        mcov = {"v": None, "c": None}
        if args.milestones and v:
            ref = next((p for p, r in v if r == "1"), v[0][0])
            # cap runs scored per arm (a band needs a few, not all) to bound LLM cost
            others_v = [p for p, _ in v if p != ref][:3]
            cond_sess = [p for p, _ in c][:3]
            cov = milestone_cov(ref, others_v + cond_sess, env)
            vv = [cov[p] for p in others_v if cov.get(p) is not None]
            cc = [cov[p] for p in cond_sess if cov.get(p) is not None]
            mcov = {"v": band(vv), "c": band(cc)}

        # A verdict needs a distribution on BOTH arms (>=2 samples each); else it's inconclusive (—),
        # not a pass/fail — this is what kills the k=1 mirage.
        enough = len(vlen) >= 2 and len(clen) >= 2
        vd = lambda a, b: (overlaps(a, b) if enough else None)
        row = {
            "task": task,
            "n": {"v": len(v), "c": len(c)},
            "solve": {"v": vsolve, "c": csolve},
            "length": {"v": band(vlen), "c": band(clen), "ok": vd(band(vlen), band(clen))},
            "rework": {"v": band(vrw), "c": band(crw), "ok": vd(band(vrw), band(crw))},
            "milestone": {"v": mcov["v"], "c": mcov["c"],
                          "ok": (overlaps(mcov["v"], mcov["c"])
                                 if (mcov["v"] and mcov["c"]) else None)},
        }
        rows.append(row)
        vb, cb = row["length"]["v"], row["length"]["c"]
        print(f"{task:24} len v={_b(vb)} c={_b(cb)} {'OK' if row['length']['ok'] else 'DIVERGES' if row['length']['ok'] is False else '—'}"
              f"  rework v={_b(row['rework']['v'])} c={_b(row['rework']['c'])}"
              f"  solve {vsolve}/{len(v)} vs {csolve}/{len(c)}")

    with open(args.out, "w") as f:
        f.write(TEMPLATE.replace("__DATA__", json.dumps({"rows": rows, "milestones": args.milestones})))
    print(f"wrote {args.out} ({len(rows)} tasks)")


def _b(b):
    return "—" if not b else (f"{b[1]:.0f}[{b[0]}-{b[2]}]" if b[0] != b[2] else f"{b[1]:.0f}")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trajectory preservation — does compaction change the run?</title>
<style>
 body{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;background:#0f1117;color:#d8dce6}
 h1{font-size:20px;margin-bottom:4px}.sub{color:#8b93a3;max-width:1040px;margin-bottom:8px}
 .legend{color:#8b93a3;max-width:1040px;margin-bottom:16px;font-size:12px}.legend b{color:#c9d1e0}
 .wrap{overflow-x:auto} table{border-collapse:collapse;white-space:nowrap}
 th,td{border:1px solid #232838;padding:6px 10px;font-size:12px;text-align:center}
 th{background:#161a22;color:#9aa4b5} td.task{text-align:left;font-weight:600}
 .van{color:#8b93a3}.cnd{color:#f778ba}
 .grp{border-left:2px solid #33394d}
 .pass{color:#3fb950;font-weight:700}.fail{color:#e5534b;font-weight:700}.dim{color:#6b7280}
 tr.tot td{border-top:2px solid #3a4152;background:#12151d;font-weight:700}
 .band{font-size:11px;color:#8b93a3}
</style></head><body>
<h1>Trajectory preservation — is a compacted run indistinguishable from vanilla?</h1>
<div class="sub">The cost eval assumes compaction preserves the trajectory (same work, <b>same length</b>). This bench tests that. Bar = <b>not "identical"</b> but <b>"within the vanilla-vs-vanilla spread."</b> vanilla k=3 = the noise <b>floor</b>; condense k=3 tested against it.</div>
<div class="legend">
Each axis shows <b><span class="van">vanilla</span></b> mean[min–max] vs <b><span class="cnd">condense</span></b> mean[min–max], and a verdict: <b class="pass">✓</b> if the ranges <b>overlap</b> (indistinguishable) · <b class="fail">✗</b> if <b>disjoint</b> (compaction changed the trajectory).
<br><b>length</b>=# decision points — <b>the load-bearing axis</b>: if condense inflates length beyond the vanilla band, the cost eval's "same trajectory" assumption fails there and its savings are partly illusory. <b>rework</b>=redundant re-fetches (amnesia). <b>solve</b>=verifier pass rate. <b>milestone</b>=approach-agnostic subgoal coverage. A task is <b>PRESERVED</b> only if every measured axis overlaps.
</div>
<div class="wrap"><table id="t"></table></div>
<script>
const D=__DATA__;
function bstr(b){if(!b)return '<span class="dim">—</span>';const[mn,mean,mx]=b;
  return mn===mx?`${Math.round(mean)}`:`${Math.round(mean)}<span class="band"> [${mn}–${mx}]</span>`;}
function verdict(ok){return ok===true?'<span class="pass">✓</span>':ok===false?'<span class="fail">✗</span>':'<span class="dim">—</span>';}
const axes=D.milestones?['length','rework','milestone']:['length','rework'];
let h='<tr><th rowspan="2" class="task" style="text-align:left">task</th><th rowspan="2">solve<br>v · c</th>'
  +axes.map(a=>`<th class="grp" colspan="3">${a}</th>`).join('')+'</tr>';
h+='<tr>'+axes.map(()=>'<th class="grp van">vanilla</th><th class="cnd">condense</th><th>✓?</th>').join('')+'</tr>';
for(const r of D.rows){
 h+=`<tr><td class="task">${r.task}</td>`;
 h+=`<td>${r.solve.v}/${r.n.v} · ${r.solve.c}/${r.n.c}</td>`;
 for(const a of axes){const x=r[a];
  const fmt=(a==='milestone')?(b=>b?`${Math.round(b[1]*100)}%`:'<span class="dim">—</span>'):bstr;
  h+=`<td class="grp van">${fmt(x.v)}</td><td class="cnd">${fmt(x.c)}</td><td>${verdict(x.ok)}</td>`;}
 h+='</tr>';}
// totals: how many tasks preserve each axis
h+='<tr class="tot"><td class="task">Σ preserved</td><td>—</td>';
for(const a of axes){const ok=D.rows.filter(r=>r[a].ok===true).length,n=D.rows.filter(r=>r[a].ok!=null).length;
 h+=`<td class="grp" colspan="3">${ok}/${n} tasks within floor</td>`;}
h+='</tr>';
document.getElementById('t').innerHTML=h;
</script></body></html>
"""

if __name__ == "__main__":
    main()
