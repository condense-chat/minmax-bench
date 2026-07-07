#!/usr/bin/env python3
"""generate.py — produce trajectory data (this is the part that SPENDS). `report.py` then displays.

Two ways to generate:
  --mode full         run the agent end-to-end via Harbor (Docker) → results/<out>/<arm>-<task>/**
  --mode incremental  teacher-force one trajectory step-by-step through each arm's endpoint →
                      results/<out>/incremental/<task>-<arm>.jsonl   (paired, cache-aware, no turn noise)

Shared:
  --agent claude-code   default; codex / opencode = TODO (not implemented yet)
  --arms condense,headroom   the methods to run (vanilla baseline is always included)
Optional (full): --milestones runs an LLM judge over the runs → results/<out>/milestones.json.

Nothing here is read by report.py except the files it writes. Keep generation and display separate.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import incremental_engine as eng  # noqa: E402  (the teacher-forced replay engine; generation side)

DATASET = "terminal-bench/terminal-bench-2-1"
DEFAULT_MODEL = "claude-sonnet-4-6"
# agent -> (harbor agent id). Only claude-code is wired; others are TODO.
AGENTS = {"claude-code": "claude-code", "codex": None, "opencode": None}
HRPORT = int(os.environ.get("HRPORT", "8787"))


# ------------------------------------------------------------------ FULL: drive Harbor per arm/task
def _arm_wiring(arm, env):
    """(base_url, allow_host, harbor_agent, extra_flags) for an arm — how a real user deploys it."""
    if arm == "vanilla":
        return "https://api.anthropic.com", "api.anthropic.com", "claude-code", []
    if arm == "condense":
        tok = env.get("CONDENSE_API_KEY", "")
        return ("https://api.condense.chat/anthropic", "api.condense.chat", "claude-code",
                ["--ae", f"ANTHROPIC_CUSTOM_HEADERS=X-Condense-Auth-Token: {tok}"])
    if arm == "headroom":
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal", "claude-code", [])
    if arm == "headroom-ccr":
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal",
                "harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode",
                ["--ae", f"TMB_HEADROOM_PROXY_URL=http://host.docker.internal:{HRPORT}"])
    sys.exit(f"unknown arm: {arm}")


def _start_proxy(out):
    p = subprocess.Popen(["headroom", "proxy", "--port", str(HRPORT), "--mode", "token"],
                         stdout=open(f"{out}/headroom.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(20):
        try:
            import socket
            socket.create_connection(("127.0.0.1", HRPORT), timeout=1).close()
            print(f"[headroom] proxy up on :{HRPORT}")
            return p
        except OSError:
            time.sleep(1)
    print("[headroom] proxy failed to start", file=sys.stderr)
    return p


def full(args, env):
    arms = ["vanilla"] + [a for a in args.arms.split(",") if a]
    tasks = args.tasks.split(",")
    out = args.out
    os.makedirs(out, exist_ok=True)
    model = args.model or DEFAULT_MODEL
    for arm in arms:
        proxy = None
        if arm in ("headroom", "headroom-ccr") and not args.dry_run:
            proxy = _start_proxy(out)
        base, allow, agent, extra = _arm_wiring(arm, env)
        for task in tasks:
            cmd = ["harbor", "run", "-d", DATASET, "-a", agent, "-m", model,
                   "-i", f"terminal-bench/{task}", "-k", str(args.k), "-n", "1",
                   "-o", f"{out}/{arm}-{task}", "--ak", f"max_budget_usd={args.budget_usd}",
                   "--allow-agent-host", allow, *extra]
            if args.agent_timeout_mult:
                cmd += ["--agent-timeout-multiplier", str(args.agent_timeout_mult)]
            renv = {**os.environ, "ANTHROPIC_BASE_URL": base,
                    "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}
            print(f"### {arm} / {task} (k={args.k}, base={base}) ###")
            if args.dry_run:
                print("   " + " ".join(cmd))
            else:
                subprocess.run(["timeout", str(args.wall_timeout), *cmd], env=renv,
                               stdout=open(f"{out}/_runlog.txt", "a"), stderr=subprocess.STDOUT)
        if proxy:
            proxy.terminate()
    if args.milestones and not args.dry_run:
        judge_milestones(args, env)


# ------------------------------------------------------------------ INCREMENTAL: teacher-forced replay
def incremental(args, env):
    out = os.path.join(args.out, "incremental")
    os.makedirs(out, exist_ok=True)
    cap = json.load(open(args.template))
    tmpl_body, tmpl_headers = cap["body"], {k.lower(): v for k, v in cap["headers"].items()}
    tmpl_body["system"] = json.loads(json.dumps(tmpl_body["system"]))  # (cwd patch omitted for brevity)
    msgs, points = eng.parse_session(args.session)
    arms = ["control"] + [a for a in args.arms.split(",") if a and a != "vanilla"]
    task = args.task or "session"
    sid = str(uuid.uuid4())
    for arm in arms:
        path = os.path.join(out, f"{task}-{arm}.jsonl")
        spent = 0.0
        with open(path, "w") as f:
            for step, i in enumerate(points[:: max(1, args.every)]):
                if args.limit and step >= args.limit or spent >= args.budget_usd:
                    break
                orig = eng.extract_action(msgs[i]["content"])
                req = eng.build_request(tmpl_body, copy.deepcopy(msgs[:i]), args, sid)
                resp, err = eng.call_api(arm, req, tmpl_headers, env)
                rec = {"arm": arm, "step": step, "orig": orig}
                if err:
                    rec["error"] = err
                else:
                    rep = eng.extract_action(resp.get("content", []))
                    _, agree, _ = eng.score(orig, rep)
                    rec.update({"agree_action": agree, "usage": resp.get("usage", {}),
                                "cost_usd": eng.cost_usd(resp.get("usage", {}))})
                    spent += rec["cost_usd"]
                f.write(json.dumps(rec) + "\n")
        print(f"[incremental] {arm}: wrote {path} (${spent:.2f})")


# ------------------------------------------------------------------ milestone judge (LLM, gen-side)
EXTRACT = ('Analyze this SOLVED coding-agent trajectory and extract 5-8 ORDERED, APPROACH-AGNOSTIC '
           'milestones any valid solution would hit.\nTASK:\n{task}\nTRAJECTORY:\n{summary}\n'
           'Return ONLY JSON {{"milestones":[{{"id":1,"name":"..."}}]}}')
COVER = ('TASK:\n{task}\nMILESTONES:\n{ms}\nANOTHER trajectory:\n{summary}\nFor each milestone, did '
         'THIS agent achieve it? Return ONLY JSON {{"coverage":[{{"id":1,"achieved":true}}]}}')


def _summary(path):
    msgs, points = eng.parse_session(path)
    task = next((b["text"] for b in msgs[0]["content"] if b.get("type") == "text"), "")[:600]
    out = []
    for n, i in enumerate(points[:80]):
        a = eng.extract_action(msgs[i]["content"])
        out.append(f"{n}. {a.get('name', 'answer')}")
    return task, "\n".join(out)


def _ask(prompt, env):
    req = {"model": DEFAULT_MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}
    resp, err = eng.call_api("control", req, {"anthropic-version": "2023-06-01"}, env)
    if err:
        return None
    t = "".join(b.get("text", "") for b in resp["content"]).strip()
    try:
        return json.loads(t[t.index("{"): t.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def judge_milestones(args, env):
    import glob
    arms = ["vanilla"] + [a for a in args.arms.split(",") if a]
    result = {}
    for task in args.tasks.split(","):
        def sess(arm):
            ss = []
            for rt in glob.glob(f"{args.out}/{arm}-{task}/*/*/verifier/reward.txt"):
                s = glob.glob(os.path.dirname(os.path.dirname(rt)) + "/agent/sessions/projects/-app/*.jsonl")
                if s:
                    ss.append((s[0], open(rt).read().strip()))
            return ss
        van = sess("vanilla")
        ref = next((p for p, r in van if r == "1"), van[0][0] if van else None)
        if not ref:
            continue
        t, s = _summary(ref)
        m = _ask(EXTRACT.format(task=t, summary=s), env)
        if not m:
            continue
        ms = m["milestones"]
        result[task] = {}
        for arm in arms:
            covs = []
            for p, _ in sess(arm)[:3]:
                _, s2 = _summary(p)
                c = _ask(COVER.format(task=t, ms=json.dumps(ms), summary=s2), env)
                if c:
                    covs.append(sum(bool(x.get("achieved")) for x in c["coverage"]) / len(ms))
            if covs:
                result[task][arm] = [min(covs), sum(covs) / len(covs), max(covs)]
    json.dump(result, open(f"{args.out}/milestones.json", "w"), indent=1)
    print(f"[milestones] wrote {args.out}/milestones.json")


def main():
    ap = argparse.ArgumentParser(description="generate trajectory data (spends); report.py displays it")
    ap.add_argument("--mode", choices=["full", "incremental", "judge"], default="full",
                    help="full/incremental = generate trajectories; judge = LLM milestone scoring of existing runs")
    ap.add_argument("--agent", default="claude-code")
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--tasks", help="comma list (full mode / milestone judge)")
    ap.add_argument("--out", default="results/jobs/run")
    ap.add_argument("--model", default=None)
    # full
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--wall-timeout", type=int, default=2400)
    ap.add_argument("--agent-timeout-mult", type=int, default=None)
    ap.add_argument("--milestones", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the Harbor commands without running")
    # incremental
    ap.add_argument("--session", help="reference trajectory jsonl (incremental mode)")
    ap.add_argument("--task", default=None)
    ap.add_argument("--template", default="data/cc_request_template.json")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--strip-thinking", action="store_true")
    ap.add_argument("--swechat", default=None)
    args = ap.parse_args()

    if AGENTS.get(args.agent) is None:
        sys.exit(f"--agent {args.agent} is not implemented yet (TODO). Only 'claude-code' is wired.")
    # export .env into the process env so Harbor / headroom subprocesses inherit the keys
    for k, v in eng.load_env().items():
        os.environ.setdefault(k, v)
    env = dict(os.environ)
    if args.mode == "full":
        if not args.tasks:
            sys.exit("--mode full needs --tasks")
        full(args, env)
    elif args.mode == "judge":
        if not args.tasks:
            sys.exit("--mode judge needs --tasks")
        judge_milestones(args, env)
    else:
        if not args.session:
            sys.exit("--mode incremental needs --session")
        incremental(args, env)


if __name__ == "__main__":
    main()
