#!/usr/bin/env python3
"""Milestone/subgoal-based trajectory faithfulness (robust to free-running tactical variance).

Node-by-node matching of free-running agent trajectories is dominated by exploratory-step
divergence (a ~14% vanilla-vs-vanilla floor here) — a documented failure mode of step-wise
action matching. Following milestone/subgoal eval (BEACON-style phases; AdaRubric-style rubric
coverage), we instead:
  1. LLM extracts the ORDERED KEY MILESTONES from the vanilla (reference) solved run.
  2. For each run (condense / headroom / a 2nd vanilla = noise floor), LLM scores which
     milestones it achieved + where it first diverged.
Faithfulness = milestone coverage. This ignores which exact command was run and asks the real
question: did compression change what the agent ACCOMPLISHED? Works identically for proxy-side
(condense) and cross-turn (headroom CCR) methods.

Usage:
  python3 scripts/fidelity_milestones.py --out results/fidelity/milestones.html \
    --vanilla vanilla=<session.jsonl> \
    --run "vanilla2=<session.jsonl>" --run "condense=<session.jsonl>" --run "headroom=<session.jsonl>"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity_replay import call_api, load_env, parse_session, extract_action  # noqa: E402
from fidelity_trajectory import label, load as load_traj  # noqa: E402
from fidelity_redundancy import redundancy  # noqa: E402

MODEL = "claude-sonnet-4-6"
HEADERS = {"anthropic-version": "2023-06-01", "anthropic-beta": "", "user-agent": "tmb-milestones"}


def summarize(path, max_steps=80):
    """Compact 'step. action => result-snippet' summary of a trajectory."""
    msgs, points = parse_session(path)
    task = next((b["text"] for b in msgs[0]["content"] if b.get("type") == "text"), "")[:600]
    lines = []
    for n, i in enumerate(points[:max_steps]):
        act = label(extract_action(msgs[i]["content"]))
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
        return None, err
    text = "".join(b.get("text", "") for b in resp["content"]).strip()
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1]), None
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"parse: {e}: {text[:200]}"


EXTRACT = """You are analyzing a coding agent's SOLVED trajectory.
TASK:
{task}

TRAJECTORY (step. action => result):
{summary}

Extract the ORDERED KEY MILESTONES — substantive checkpoints of real progress toward solving the
task. CRITICAL: make them APPROACH-AGNOSTIC — checkpoints that ANY valid solution would hit,
NOT tied to this run's particular tactic. E.g. write "implemented a faster custom solver" NOT
"compiled a Cython extension"; write "chose an optimization strategy" NOT "chose numba". A
different correct solution using a different library/technique MUST still hit every milestone.
Ignore tactical/exploratory noise (individual greps, failed probes) and tool/library specifics.
Aim for 5-8 milestones covering: understanding the task, choosing a strategy, implementing,
verifying correctness, and meeting the goal/target.
Return ONLY JSON: {{"milestones":[{{"id":1,"name":"...","evidence":"..."}}]}}"""

COVER = """TASK:
{task}

MILESTONES to check (from a reference solution):
{milestones}

ANOTHER agent's trajectory (step. action => result):
{summary}

For EACH milestone, did THIS agent achieve it (judge from the trajectory evidence)? Also give the
id of the FIRST milestone it did NOT achieve (or null if all achieved).
Return ONLY JSON: {{"coverage":[{{"id":1,"achieved":true,"evidence":"..."}}],"first_missed":null}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", required=True, help="name=session.jsonl (reference)")
    ap.add_argument("--run", action="append", default=[], help="name=session.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = {**load_env(), **os.environ}

    vname, vpath = args.vanilla.split("=", 1)
    task, vsum = summarize(vpath)
    ms, err = ask(EXTRACT.format(task=task, summary=vsum), env)
    if err:
        sys.exit("milestone extraction failed: " + err)
    milestones = ms["milestones"]
    print(f"extracted {len(milestones)} milestones from {vname}:")
    for m in milestones:
        print(f"  {m['id']}. {m['name']}")

    vtok = load_traj(vpath)[2]
    runs = {}
    for spec in [args.vanilla] + args.run:
        name, path = spec.split("=", 1)
        _, s = summarize(path)
        cov, err = ask(COVER.format(task=task, milestones=json.dumps(milestones), summary=s), env)
        if err:
            print(f"  {name}: coverage ERROR {err[:120]}")
            continue
        achieved = {c["id"]: bool(c.get("achieved")) for c in cov["coverage"]}
        n_hit = sum(achieved.get(m["id"], False) for m in milestones)
        # axes 2 & 3: compression (token ratio vs vanilla) + redundant re-work (self-contained)
        tok = load_traj(path)[2]
        hits, nact = redundancy(path)
        runs[name] = {"achieved": achieved, "coverage": n_hit / len(milestones),
                      "first_missed": cov.get("first_missed"),
                      "compression": round(1 - tok / vtok, 3) if vtok else 0,
                      "redundant": len(hits), "actions": nact}
        print(f"  {name}: {n_hit}/{len(milestones)} milestones ({n_hit/len(milestones):.0%}), "
              f"compression {runs[name]['compression']:+.0%}, redundant {len(hits)}/{nact}, "
              f"first missed = {cov.get('first_missed')}")

    data = {"task": task[:300], "milestones": milestones, "runs": runs}
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Milestone faithfulness</title>
<style>
 body{font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;background:#0f1117;color:#d8dce6}
 h1{font-size:18px}.sub{color:#8b93a3;max-width:820px;margin-bottom:16px}
 table{border-collapse:collapse;margin-top:8px}
 th,td{border:1px solid #232838;padding:8px 12px;text-align:center;font-size:12px}
 th{background:#161a22;color:#9aa4b5} td.ms{text-align:left;max-width:420px}
 .hit{color:#3fb950;font-weight:700}.miss{color:#e5534b;font-weight:700}
 .score{font-size:15px;font-weight:600}
 .cov{font-size:12px;color:#9aa4b5}
</style></head><body>
<h1>Trajectory quality scorecard — compression methods on full runs</h1>
<div class="sub">Three axes, each a different question. <b>Compression</b> = tokens saved vs vanilla (savings). <b>Milestone coverage</b> = did it accomplish the same subgoals (approach-agnostic milestones extracted by LLM from the vanilla run — robust to tactical variance, but saturates for solved runs). <b>Redundant re-work</b> = info-gathering actions that re-fetch content the run already had (re-Read/re-cat/re-run) — the compaction-amnesia cost that <i>discriminates</i> among solved runs. A 2nd vanilla run is the noise floor (should be 100% / 0 redundant).</div>
<table id="t"></table>
<script>
const D=__DATA__;
const C={vanilla:'#8b93a3',vanilla2:'#8b93a3',condense:'#f778ba',headroom:'#58a6ff','headroom-ccr':'#a371f7'};
const names=Object.keys(D.runs);
let h='<tr><th class="ms">milestone</th>'+names.map(n=>`<th style="color:${C[n]||'#ccc'}">${n}</th>`).join('')+'</tr>';
for(const m of D.milestones){
 h+=`<tr><td class="ms"><b>${m.id}.</b> ${esc(m.name)}</td>`;
 for(const n of names){const a=D.runs[n].achieved[m.id];h+=`<td class="${a?'hit':'miss'}">${a?'✓':'✗'}</td>`;}
 h+='</tr>';}
h+='<tr><td class="ms score">▸ Milestone coverage</td>'+names.map(n=>{const r=D.runs[n];return `<td class="score" style="color:${C[n]||'#ccc'}">${Math.round(r.coverage*100)}%</td>`;}).join('')+'</tr>';
h+='<tr><td class="ms cov">first milestone missed</td>'+names.map(n=>`<td class="cov">${D.runs[n].first_missed??'—'}</td>`).join('')+'</tr>';
h+='<tr><td class="ms score">▸ Compression (tokens saved)</td>'+names.map(n=>{const r=D.runs[n];const c=r.compression||0;return `<td class="score">${n==="vanilla"?'—':(c>=0?'−':'+')+Math.abs(Math.round(c*100))+'%'}</td>`;}).join('')+'</tr>';
h+='<tr><td class="ms score">▸ Redundant re-work</td>'+names.map(n=>{const r=D.runs[n];const base=D.runs.vanilla?D.runs.vanilla.redundant:0;const ex=r.redundant-base;return `<td class="score ${r.redundant>base?'miss':(r.redundant===0?'hit':'')}">${r.redundant}${r.actions?` <span class="cov">(${Math.round(100*r.redundant/r.actions)}%)</span>`:''}</td>`;}).join('')+'</tr>';
document.getElementById('t').innerHTML=h;
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
</script></body></html>
"""

if __name__ == "__main__":
    main()
