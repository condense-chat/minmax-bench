#!/usr/bin/env python3
"""generate.py — produce trajectory data (this is the part that SPENDS). `report.py` then displays.

Two ways to generate:
  --mode full         run the agent end-to-end via Harbor (Docker) → results/<out>/<arm>-<task>/**
  --mode incremental  teacher-force one trajectory step-by-step through each arm's endpoint →
                      results/<out>/incremental/<task>-<arm>.jsonl
                      (paired, cache-aware, no turn noise)

Shared:
  --agent claude-code   default; codex / opencode = TODO (not implemented yet)
  --arms condense,headroom   the methods to run (vanilla baseline is always included).
                             headroom       = cache-mode proxy (cost bench's 'headroom')
                             headroom-ccr   = token-mode proxy + the mcp retrieve loop — the
                                              full Compress-Cache-Retrieve product, headroom's
                                              intended token-mode deployment
                             headroom-kompress = token mode WITHOUT retrieval (ablation only;
                                              matches the cost-bench strategy of that name)
Optional (full): --milestones runs an LLM judge over the runs → results/<out>/milestones.json.

Nothing here is read by report.py except the files it writes. Keep generation and display separate.
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from minmax_bench.quality import engine as eng  # the teacher-forced replay engine

DATASET = "terminal-bench/terminal-bench-2-1"
DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_AGENTS = {"claude-code"}  # codex / opencode: TODO; harbor agent ids in _arm_wiring
HRPORT = int(os.environ.get("HRPORT", "8787"))
UTC = timezone.utc  # noqa: UP017  (scripts run on system python3, may be 3.10)

# arm -> headroom proxy mode. 'headroom' MUST stay cache mode: it is the cost bench's
# 'headroom' strategy (matrix.py). For token mode, headroom's intended deployment is CCR
# (Compress-Cache-Retrieve): proxy compression PLUS the mcp retrieve loop -> 'headroom-ccr'
# is the canonical token-mode arm. 'headroom-kompress' (compression with NO retrieval,
# the cost bench's strategy of that name) is kept as an ABLATION, not a headline arm.
HEADROOM_MODES = {"headroom": "cache", "headroom-kompress": "token", "headroom-ccr": "token"}


# ------------------------------------------------------------------ FULL: drive Harbor per arm/task
def _arm_wiring(arm, env):
    """(base_url, allow_host, harbor_agent, extra_flags) for an arm — how a real user deploys it."""
    if arm == "vanilla":
        return "https://api.anthropic.com", "api.anthropic.com", "claude-code", []
    if arm == "condense":
        tok = env.get("CONDENSE_API_KEY", "")
        return ("https://api.condense.chat/anthropic", "api.condense.chat", "claude-code",
                ["--ae", f"ANTHROPIC_CUSTOM_HEADERS=X-Condense-Auth-Token: {tok}"])
    if arm in ("headroom", "headroom-kompress"):
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal", "claude-code", [])
    if arm == "headroom-ccr":
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal",
                "harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode",
                ["--ae", f"TMB_HEADROOM_PROXY_URL=http://host.docker.internal:{HRPORT}"])
    sys.exit(f"unknown arm: {arm}")


def _proxy_up():
    try:
        socket.create_connection(("127.0.0.1", HRPORT), timeout=1).close()
        return True
    except OSError:
        return False


def _start_proxy(out, mode):
    log = open(f"{out}/headroom-{mode}.log", "w")
    p = subprocess.Popen(["headroom", "proxy", "--port", str(HRPORT), "--mode", mode],
                         stdout=log, stderr=subprocess.STDOUT)
    p._tmb_log = log  # closed in _stop_proxy
    for _ in range(20):
        if _proxy_up():
            print(f"[headroom] proxy up on :{HRPORT} (mode={mode})")
            return p
        time.sleep(1)
    print("[headroom] proxy failed to start", file=sys.stderr)
    return p


def _stop_proxy(p):
    if p:
        p.terminate()
        log = getattr(p, "_tmb_log", None)
        if log:
            log.close()


def _validate_full(args, env):
    arms = ["vanilla"] + [a for a in args.arms.split(",") if a]
    if "condense" in arms and not env.get("CONDENSE_API_KEY") and not args.dry_run:
        sys.exit("CONDENSE_API_KEY missing — the condense arm would run with an EMPTY auth "
                 "token and silently produce garbage. Set it in .env or drop the arm.")
    return arms


def full(args, env):
    arms = _validate_full(args, env)
    tasks = args.tasks.split(",")
    out = args.out
    os.makedirs(out, exist_ok=True)
    model = args.model or DEFAULT_MODEL
    for arm in arms:
        proxy = None
        try:
            if arm in HEADROOM_MODES and not args.dry_run:
                proxy = _start_proxy(out, HEADROOM_MODES[arm])
            base, allow, agent, extra = _arm_wiring(arm, env)
            for task in tasks:
                cell = f"{out}/{arm}-{task}"
                cmd = ["harbor", "run", "-d", DATASET, "-a", agent, "-m", model,
                       "-i", f"terminal-bench/{task}", "-k", str(args.k),
                       "-n", str(args.concurrency),
                       "-o", cell, "--ak", f"max_budget_usd={args.budget_usd}",
                       "--allow-agent-host", allow, *extra]
                if args.agent_timeout_mult:
                    cmd += ["--agent-timeout-multiplier", str(args.agent_timeout_mult)]
                renv = {**os.environ, "ANTHROPIC_BASE_URL": base,
                        "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}
                if arm == "headroom-kompress":
                    print("[note] headroom-kompress = token-mode compression WITHOUT the CCR "
                          "retrieve loop (ablation); headroom's intended token-mode config "
                          "is the headroom-ccr arm")
                print(f"### {arm} / {task} (k={args.k}, base={base}) ###")
                if args.dry_run:
                    print("   " + " ".join(cmd))
                    continue
                # record intent BEFORE running so killed/crashed trials still count as
                # attempted — report.py uses this to expose survivorship instead of
                # silently shrinking n (a reward.txt-less trial is a failure, not noise)
                os.makedirs(cell, exist_ok=True)
                json.dump({"k": args.k, "arm": arm, "task": task,
                           "started_utc": datetime.now(UTC).isoformat(timespec="seconds")},
                          open(f"{cell}/attempted.json", "w"))
                # per-trial wall budget: the timeout must cover all k trials of the cell,
                # otherwise slow arms systematically lose their later trials.
                # subprocess timeout (not the GNU `timeout` binary) — stock macOS has none.
                with open(f"{out}/_runlog.txt", "a") as runlog:
                    try:
                        subprocess.run(cmd, env=renv, stdout=runlog, stderr=subprocess.STDOUT,
                                       timeout=args.wall_timeout * args.k)
                    except subprocess.TimeoutExpired:
                        print(f"[timeout] {arm}/{task} killed after "
                              f"{args.wall_timeout * args.k}s — partial trials recorded",
                              file=sys.stderr)
        finally:
            _stop_proxy(proxy)
    if args.milestones and not args.dry_run:
        judge_milestones(args, env)


# ------------------------------------------------------------------ INCREMENTAL: teacher-forced
def _replay_arm(arm, msgs, sel, tmpl_body, tmpl_headers, args, env, sid, path):
    """Teacher-force one arm over the selected decision points; returns a summary line."""
    spent, n_ok, n_action, n_err = 0.0, 0, 0, 0
    model = tmpl_body["model"]
    with open(path, "w") as f:
        for step, i in enumerate(sel):
            if spent >= args.budget_usd:
                break
            orig = eng.extract_action(msgs[i]["content"])
            req = eng.build_request(tmpl_body, msgs[:i], args, sid)
            t0 = time.time()
            resp, err = eng.call_api(arm, req, tmpl_headers, env)
            rec = {"arm": arm, "step": step, "msg_index": i, "orig": orig,
                   "latency_s": round(time.time() - t0, 1)}
            if err:
                n_err += 1
                rec["error"] = err
            else:
                rep = eng.extract_action(resp.get("content", []))
                exact, agree, sim = eng.score(orig, rep)
                usage = resp.get("usage", {})
                c = eng.cost_usd(usage, model)
                spent += c
                n_ok += 1
                n_action += agree
                rec.update(replay=rep, agree_exact=exact, agree_action=agree,
                           sim=round(sim, 3), usage=usage, cost_usd=round(c, 4))
            f.write(json.dumps(rec) + "\n")
            f.flush()
    fid = f"{n_action}/{n_ok} ({n_action / n_ok:.0%})" if n_ok else "n/a"
    return f"[incremental] {arm}: action-agree {fid}, errors {n_err}, ${spent:.2f} -> {path}"


def incremental(args, env):
    arms = ["control"] + [a for a in args.arms.split(",") if a and a != "vanilla"]
    problems = eng.check_arms(arms, env)
    if problems:
        sys.exit("refusing to start (would crash mid-run after spending):\n  - "
                 + "\n  - ".join(problems))
    out = os.path.join(args.out, "incremental")
    os.makedirs(out, exist_ok=True)
    cap = json.load(open(args.template))
    tmpl_body = cap["body"]
    tmpl_headers = {k.lower(): v for k, v in cap["headers"].items()}

    if args.swechat:
        msgs, points, model, used = eng.load_swechat(args.swechat, args.conv)
        tmpl_body["model"] = model
        tmpl_body["tools"] = eng.build_tools(used, tmpl_body["tools"])
        print(f"[swechat #{args.conv}] model={model}, {len(used)} tools", file=sys.stderr)
    else:
        eng.patch_cwd(tmpl_body, args.template, args.cwd_patch)
        msgs, points = eng.parse_session(args.session)

    if "headroom" in arms and not _proxy_up():
        _start_proxy(args.out, args.headroom_mode)

    sel = points[:: max(1, args.every)]
    if args.limit:
        sel = sel[: args.limit]
    sid = str(uuid.uuid4())
    print(f"[incremental] {len(msgs)} msgs, replaying {len(sel)}/{len(points)} decision points "
          f"per arm ({', '.join(arms)}), session {sid}")
    # arms hit independent endpoints with independent budgets — replay them concurrently;
    # steps WITHIN an arm stay sequential (incremental prompt caching needs nested prefixes)
    with ThreadPoolExecutor(max_workers=len(arms)) as ex:
        futures = [ex.submit(_replay_arm, arm, msgs, sel, tmpl_body, tmpl_headers,
                             args, env, sid, os.path.join(out, f"{args.task}-{arm}.jsonl"))
                   for arm in arms]
        for fut in futures:
            print(fut.result())


# ------------------------------------------------------------------ milestone judge (LLM, gen-side)
EXTRACT = ('Analyze this coding task and ONE solved reference trajectory, and extract 5-8 '
           'ORDERED, APPROACH-AGNOSTIC milestones ANY valid solution would hit (do not encode '
           'this particular approach).\nTASK:\n{task}\nREFERENCE TRAJECTORY:\n{summary}\n'
           'Return ONLY JSON {{"milestones":[{{"id":1,"name":"..."}}]}}')
COVER = ('TASK:\n{task}\nMILESTONES:\n{ms}\nANOTHER trajectory:\n{summary}\nFor each milestone, '
         'did THIS agent achieve it? Return ONLY JSON {{"coverage":[{{"id":1,"achieved":true}}]}}')


def _digest(a):
    """Short, judge-readable label for one action: tool + its target."""
    if a.get("type") != "tool_use":
        return "answer"
    inp = a.get("input", {})
    tgt = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
    tgt = " ".join(str(tgt).split())[:70]
    return f"{a.get('name')}({tgt})" if tgt else a.get("name", "?")


def _summary(path):
    msgs, points = eng.parse_session(path)
    task = next((b["text"] for b in msgs[0]["content"] if b.get("type") == "text"), "")[:600]
    out = [f"{n}. {_digest(eng.extract_action(msgs[i]['content']))}"
           for n, i in enumerate(points[:80])]
    return task, "\n".join(out)


def _ask(prompt, env):
    req = {"model": DEFAULT_MODEL, "max_tokens": 1500, "temperature": 0,
           "messages": [{"role": "user", "content": prompt}]}
    resp, err = eng.call_api("control", req, {"anthropic-version": "2023-06-01"}, env)
    if err:
        print(f"[milestones] judge call failed: {err[:160]}", file=sys.stderr)
        return None
    t = "".join(b.get("text", "") for b in resp["content"]).strip()
    try:
        return json.loads(t[t.index("{"): t.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _runs(root, arm, task):
    """(session_path, reward) per trial — same discovery convention as report.py."""
    out = []
    for rt in sorted(glob.glob(f"{root}/{arm}-{task}/*/*/verifier/reward.txt")):
        s = glob.glob(os.path.join(os.path.dirname(os.path.dirname(rt)), eng.SESSION_GLOB))
        if s:
            out.append((sorted(s)[0], open(rt).read().strip()))
    return out


def judge_milestones(args, env):
    problems = eng.check_arms(["control"], env)
    if problems:
        sys.exit("milestone judge needs an API key:\n  - " + "\n  - ".join(problems))
    arms = ["vanilla"] + [a for a in args.arms.split(",") if a]
    result, mpath = {}, f"{args.out}/milestones.json"
    for task in args.tasks.split(","):
        sess = {arm: _runs(args.out, arm, task) for arm in arms}  # glob once per (task, arm)
        ref = next((p for p, r in sess["vanilla"] if r == "1"), None)
        if not ref:
            print(f"[milestones] {task}: no SOLVED vanilla run to ground milestones — skipped")
            continue
        t, s = _summary(ref)
        m = _ask(EXTRACT.format(task=t, summary=s), env)
        ms = (m or {}).get("milestones") or []
        if not ms:
            print(f"[milestones] {task}: judge returned no usable milestones — skipped")
            continue
        result[task] = {}
        for arm in arms:
            covs = []
            # the reference run trivially covers its own milestones — exclude it from
            # vanilla's coverage so the vanilla band isn't inflated vs the arms
            paths = [p for p, _ in sess[arm] if p != ref][:3]
            for p in paths:
                _, s2 = _summary(p)
                c = _ask(COVER.format(task=t, ms=json.dumps(ms), summary=s2), env)
                cov = (c or {}).get("coverage") or []
                if cov:
                    covs.append(sum(bool(x.get("achieved")) for x in cov) / len(ms))
            if covs:
                result[task][arm] = [min(covs), sum(covs) / len(covs), max(covs)]
        json.dump(result, open(mpath, "w"), indent=1)  # per-task: a late crash keeps earlier work
    json.dump(result, open(mpath, "w"), indent=1)
    print(f"[milestones] wrote {mpath}")


def main():
    ap = argparse.ArgumentParser(
        description="generate trajectory data (spends); report.py displays it")
    ap.add_argument("--mode", choices=["full", "incremental", "judge"], default="full",
                    help="full/incremental = generate trajectories; "
                         "judge = LLM milestone scoring of existing runs")
    ap.add_argument("--agent", default="claude-code")
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--tasks", help="comma list (full mode / milestone judge)")
    ap.add_argument("--out", default="results/jobs/run")
    ap.add_argument("--model", default=None)
    # full
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--wall-timeout", type=int, default=2400,
                    help="PER-TRIAL wall budget in seconds (the cell gets wall_timeout * k)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="harbor -n: parallel trials per cell (parallel containers can add "
                         "resource-contention noise to trajectories; 1 = cleanest)")
    ap.add_argument("--agent-timeout-mult", type=int, default=None)
    ap.add_argument("--milestones", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the Harbor commands without running")
    # incremental
    ap.add_argument("--session", help="reference trajectory jsonl (incremental mode)")
    ap.add_argument("--task", default=None,
                    help="task name for incremental artifacts; MUST match the --tasks name "
                         "you will pass to report.py or the comp/$Δ/fid columns can't join")
    ap.add_argument("--template", default="data/cc_request_template.json")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--strip-thinking", action="store_true")
    ap.add_argument("--cwd-patch", default="/app",
                    help="rewrite the capture cwd in the template system prompt to this")
    ap.add_argument("--headroom-mode", default="cache", choices=["cache", "token"],
                    help="proxy mode for the incremental 'headroom' arm")
    ap.add_argument("--swechat", default=None, help="SWE-chat jsonl (alternative to --session)")
    ap.add_argument("--conv", type=int, default=0, help="conversation index within --swechat")
    args = ap.parse_args()

    if args.agent not in SUPPORTED_AGENTS:
        sys.exit(f"--agent {args.agent} is not implemented yet (TODO). "
                 "Only 'claude-code' is wired.")
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
        if not (args.session or args.swechat):
            sys.exit("--mode incremental needs --session (or --swechat)")
        if not args.task:
            sys.exit("--mode incremental needs --task <name> (report.py joins incremental "
                     "artifacts to its --tasks list by this name)")
        incremental(args, env)


if __name__ == "__main__":
    main()
