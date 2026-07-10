#!/usr/bin/env python3
"""generate.py — produce trajectory data (this is the part that SPENDS). `report.py` then displays.

Two ways to generate:
  --mode full         run the agent end-to-end via Harbor (Docker) → results/<out>/<arm>-<task>/**
  --mode incremental  teacher-force one trajectory step-by-step through each arm's endpoint →
                      results/<out>/incremental/<task>-<arm>.jsonl
                      (paired, cache-aware, no turn noise)

Shared:
  --agent claude-code   default; codex / opencode = TODO (not implemented yet)
  --arms condense,headroom   the methods to run (vanilla baseline is always included;
                             this default pits condense against headroom's full product).
                             Two headroom arms, mirroring the cost bench's names:
                             headroom          = the REGULAR/full product — token-mode proxy
                                              + the mcp retrieve loop (Compress-Cache-Retrieve)
                             headroom-kompress = token-mode compression WITHOUT retrieval
                                              (ablation only)
Optional (full): --milestones runs an LLM judge over the runs → results/<out>/milestones.json.

Nothing here is read by report.py except the files it writes. Keep generation and display separate.
"""
import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from minmax_bench.quality import engine as eng  # the teacher-forced replay engine
from minmax_bench.quality.engine import DEFAULT_TASKS, dataset_tasks, resolve_tasks

DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"  # the only validated dataset so far
DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_AGENTS = {"claude-code"}  # codex / opencode: TODO; harbor agent ids in _arm_wiring
HRPORT = int(os.environ.get("HRPORT", "8787"))
UTC = timezone.utc  # noqa: UP017  (scripts run on system python3, may be 3.10)

# arm -> headroom proxy mode. Two arms, mirroring the cost bench's headroom names:
# 'headroom' is the REGULAR/full product (token-mode proxy + the mcp retrieve loop,
# i.e. Compress-Cache-Retrieve — the CCR agent); 'headroom-kompress' is the ABLATION
# (token-mode compression with NO retrieval). Both need a token-mode proxy.
HEADROOM_MODES = {"headroom": "token", "headroom-kompress": "token"}

# the harbor child currently running (for cleanup on interrupt) — a 1-slot box so
# the signal handler / atexit reaper can terminate it before removing its network
_HARBOR = [None]

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
        # regular headroom = the full product: token-mode proxy + the MCP retrieve
        # loop (Compress-Cache-Retrieve), via the CCR agent
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal",
                "harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode",
                ["--ae", f"TMB_HEADROOM_PROXY_URL=http://host.docker.internal:{HRPORT}"])
    if arm == "headroom-kompress":
        # ablation: token-mode compression WITHOUT the retrieve loop (plain agent)
        return (f"http://host.docker.internal:{HRPORT}", "host.docker.internal", "claude-code", [])
    sys.exit(f"unknown arm: {arm}")


def _proxy_up():
    try:
        socket.create_connection(("127.0.0.1", HRPORT), timeout=1).close()
        return True
    except OSError:
        return False


def _proxy_mode():
    """The mode a headroom proxy already on HRPORT reports (via /stats), or None."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{HRPORT}/stats", timeout=3) as r:
            return (json.load(r).get("summary") or {}).get("mode")
    except Exception:  # noqa: BLE001
        return None


def _start_proxy(out, mode):
    if _proxy_up():
        # a proxy is already up — reuse it iff it's the mode this arm needs (a leftover
        # from an earlier run is common); refuse only on a genuine MODE mismatch, which
        # would silently mislabel results. A reused proxy isn't ours, so we don't stop it.
        existing = _proxy_mode()
        if existing == mode:
            print(f"[headroom] reusing existing {mode}-mode proxy on :{HRPORT}")
            return None
        sys.exit(f"port {HRPORT} is serving a {existing or 'unknown'}-mode proxy but this arm "
                 f"needs mode={mode}; stop it (`kill $(lsof -ti :{HRPORT})`) or set HRPORT.")
    log = open(f"{out}/headroom-{mode}.log", "w")
    # nothing is installed into the user's global environment: if the headroom CLI
    # isn't already on PATH (e.g. project venv), run it via uvx in an isolated,
    # ephemeral environment
    base_cmd = (["headroom"] if shutil.which("headroom")
                else ["uvx", "--from", "headroom-ai[proxy]", "headroom"])
    p = subprocess.Popen([*base_cmd, "proxy", "--port", str(HRPORT), "--mode", mode],
                         stdout=log, stderr=subprocess.STDOUT)
    p._tmb_log = log  # closed in _stop_proxy
    for _ in range(90):  # token mode cold-starts slowly (model/tokenizer downloads)
        if _proxy_up():
            print(f"[headroom] proxy up on :{HRPORT} (mode={mode})")
            return p
        if p.poll() is not None:
            break
        time.sleep(1)
    _stop_proxy(p)
    sys.exit(f"[headroom] proxy failed to start (mode={mode}) — see {out}/headroom-{mode}.log")


def _stop_proxy(p):
    if p:
        p.terminate()
        log = getattr(p, "_tmb_log", None)
        if log:
            log.close()


def _reap_docker():
    """Remove leftover harbor trial containers/networks. Harbor cleans up after a
    NORMAL trial, but a killed/aborted/crashed run leaks its `<task>__<id>__env*`
    containers and networks — which exhaust Docker's address pools over time. Harbor
    names EVERY trial resource with the `__env` marker, so that pattern targets only
    harbor's leftovers (k3d clusters, user projects, and defaults don't match).
    Best-effort: silent if Docker is absent."""
    if not shutil.which("docker"):
        return
    # if a harbor child is still tearing down its trial, terminate + wait for it first
    # — otherwise its network is still 'in use', `network rm` skips it, and it dangles
    proc = _HARBOR[0]
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    reaped = 0
    kinds = ((["ps", "-aq"], ["rm", "-f"]), (["network", "ls", "-q"], ["network", "rm"]))
    for lister, remover in kinds:
        ids = subprocess.run(["docker", *lister, "--filter", "name=__env"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", *remover, *ids], capture_output=True)
            reaped += len(ids)
    if reaped:
        print(f"[cleanup] removed {reaped} leftover harbor docker container(s)/network(s)")


def _agent_auth_env(env):
    """Harbor --ae args so the container's Claude Code authenticates. Prefer an API
    key if configured; otherwise forward the user's Claude Code SUBSCRIPTION token
    (so a full run needs no API key — same default as the replay paths). Harbor
    forwards ANTHROPIC_API_KEY from the host env on its own; the OAuth token must be
    passed explicitly."""
    if env.get("ANTHROPIC_API_KEY"):
        return []
    tok = eng.cc_oauth_token()
    return ["--ae", f"CLAUDE_CODE_OAUTH_TOKEN={tok}"] if tok else []


def _validate_full(args, env):
    arms = ["vanilla"] + [a for a in args.arms.split(",") if a]
    if not eng.auth_mode(env) and not args.dry_run:
        sys.exit("no Anthropic auth — set ANTHROPIC_API_KEY, or log in to Claude Code "
                 "(run `claude setup-token` + `export CLAUDE_CODE_OAUTH_TOKEN=…`); the "
                 "container's agent needs it to run.")
    if "condense" in arms and not env.get("CONDENSE_API_KEY") and not args.dry_run:
        sys.exit("CONDENSE_API_KEY missing — the condense arm would run with an EMPTY auth "
                 "token and silently produce garbage. Set it in .env or drop the arm.")
    return arms


def _docker_alive():
    """True iff the docker daemon actually RESPONDS — `which docker` only proves the
    binary exists; a stopped/starting Docker Desktop still fails every trial with
    'Docker daemon is not running'. `docker info` is the cheap liveness probe."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _preflight_full(arms, env):
    """Dependency preflight before spending (like the cost bench's run preflight):
    every dependency each requested arm needs, as (name, ok, detail, fatal). Docker
    and harbor run the containers; auth runs the agent; per-arm: the condense key and
    the headroom proxy/port. Printed as a table; a fatal miss aborts before any spend."""
    rows = [
        ("Docker daemon", _docker_alive(), "must be running (not just installed)", True),
        ("harbor CLI", bool(shutil.which("harbor")), "uv tool install harbor", True),
        ("Anthropic auth", bool(eng.auth_mode(env)),
         eng.auth_mode(env) or "ANTHROPIC_API_KEY or `claude setup-token`", True),
    ]
    if "condense" in arms:
        rows.append(("CONDENSE_API_KEY", bool(env.get("CONDENSE_API_KEY")),
                     "the condense arm", True))
    if any(a in HEADROOM_MODES for a in arms):
        rows.append(("headroom / uvx", bool(shutil.which("headroom") or shutil.which("uvx")),
                     "the headroom proxy", True))
        # the port must be free OR already serving the token mode the headroom arms need
        # (this is exactly the mid-run failure the preflight is here to catch up front)
        pm = _proxy_mode() if _proxy_up() else None
        ok = (not _proxy_up()) or pm == "token"
        detail = ("free" if not _proxy_up() else
                  "reusable token-mode proxy up" if pm == "token" else
                  f"busy with a {pm or 'unknown'}-mode proxy — `kill $(lsof -ti :{HRPORT})`")
        rows.append((f"port {HRPORT}", ok, detail, True))
    return rows


def full(args, env):
    # --auth subscription forces the Claude Code login even if an API key is
    # configured — drop the key so both the auth check and the container agent use
    # the subscription token. 'auto' (default) prefers a key if present.
    if getattr(args, "auth", "auto") == "subscription":
        env = {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"}
        os.environ.pop("ANTHROPIC_API_KEY", None)  # so harbor doesn't forward it either
    arms = _validate_full(args, env)
    if not args.dry_run:
        rows = _preflight_full(arms, env)
        print("[preflight]")
        for name, ok, detail, _fatal in rows:
            print(f"   {'ok  ' if ok else 'MISS'}  {name:16} {detail}")
        missing = [n for n, ok, _d, fatal in rows if not ok and fatal]
        if missing:
            sys.exit(f"[preflight] blocked by: {', '.join(missing)} — fix and re-run "
                     f"(nothing spent).")
        # never leave a dead docker trail: reap harbor's leftover trial containers/
        # networks on exit — normal, exception, or Ctrl-C (atexit) and kill (SIGTERM).
        # (harbor cleans up healthy trials; this catches aborted/crashed ones.)
        import atexit
        import signal
        atexit.register(_reap_docker)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # -> SystemExit -> atexit
    org = args.dataset.split("/", 1)[0]
    tasks = resolve_tasks(args.tasks, org, args.seed)
    out = args.out
    os.makedirs(out, exist_ok=True)
    model = args.model or DEFAULT_MODEL
    kv = args.k_vanilla or args.k + 1
    trials = len(tasks) * (kv + args.k * (len(arms) - 1))
    print(f"[plan] {len(arms)} arms ({', '.join(arms)}) x {len(tasks)} tasks "
          f"({', '.join(tasks)}) x k={args.k} (vanilla {kv}) = {trials} trials, "
          f"cost ceiling ~${trials * args.budget_usd:.0f} "
          f"(--budget-usd {args.budget_usd:g}/trial cap)")
    for arm in arms:
        # the vanilla band is the noise floor every arm is judged against — one extra
        # vanilla run sharpens every verdict, so it defaults to k+1
        k = (args.k_vanilla or args.k + 1) if arm == "vanilla" else args.k
        proxy = None
        try:
            if arm in HEADROOM_MODES and not args.dry_run:
                proxy = _start_proxy(out, HEADROOM_MODES[arm])
            base, allow, agent, extra = _arm_wiring(arm, env)
            for task in tasks:
                cell = f"{out}/{arm}-{task}"
                cmd = ["harbor", "run", "-d", args.dataset, "-a", agent, "-m", model,
                       "-i", f"{org}/{task}", "-k", str(k),
                       "-n", str(args.concurrency),
                       "-o", cell, "--ak", f"max_budget_usd={args.budget_usd}",
                       "--allow-agent-host", allow, *_agent_auth_env(env), *extra]
                # agent SETUP (install Claude Code + node/deps per container) has its OWN
                # timeout (harbor default 360s), separate from EXECUTION. On a cold image
                # or a loaded machine even the plain agent blows past 360s and every trial
                # dies before the agent runs — so bump the setup timeout for EVERY arm.
                cmd += ["--agent-setup-timeout-multiplier", str(args.setup_timeout_mult)]
                # agent EXECUTION timeout: the CCR agent (headroom) also installs headroom-ai
                # + the mcp SDK and runs a heavier loop, so give it more execution time too.
                mult = args.agent_timeout_mult or (3 if arm == "headroom" else None)
                if mult:
                    cmd += ["--agent-timeout-multiplier", str(mult)]
                renv = {**os.environ, "ANTHROPIC_BASE_URL": base,
                        "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}
                if arm == "headroom-kompress":
                    print("[note] headroom-kompress = token-mode compression WITHOUT the retrieve "
                          "loop (ablation); the full product is the 'headroom' arm (with CCR)")
                print(f"### {arm} / {task} (k={k}, base={base}) ###")
                if args.dry_run:
                    print("   " + " ".join(cmd))
                    continue
                # resume: skip a cell that already has its k finished trials (so a
                # re-run after an abort fills in only the missing arms, no re-spend).
                done = len(glob.glob(f"{cell}/*/*/verifier/reward.txt"))
                if done >= k and not args.force:
                    print(f"[skip] {arm}/{task} already has {done}/{k} trials (--force to rerun)")
                    continue
                # record intent BEFORE running so killed/crashed trials still count as
                # attempted — report.py uses this to expose survivorship instead of
                # silently shrinking n (a reward.txt-less trial is a failure, not noise)
                os.makedirs(cell, exist_ok=True)
                json.dump({"k": k, "arm": arm, "task": task,
                           "started_utc": datetime.now(UTC).isoformat(timespec="seconds")},
                          open(f"{cell}/attempted.json", "w"))
                # per-trial wall budget: the timeout must cover all k trials of the cell,
                # otherwise slow arms systematically lose their later trials.
                # subprocess timeout (not the GNU `timeout` binary) — stock macOS has none.
                with open(f"{out}/_runlog.txt", "a") as runlog:
                    # tracked Popen (not subprocess.run) so an interrupt/timeout can
                    # terminate this harbor child and wait for its container/network
                    # teardown BEFORE the reaper runs
                    proc = subprocess.Popen(cmd, env=renv, stdout=runlog,
                                            stderr=subprocess.STDOUT)
                    _HARBOR[0] = proc
                    try:
                        proc.wait(timeout=args.wall_timeout * k)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        print(f"[timeout] {arm}/{task} killed after "
                              f"{args.wall_timeout * k}s — partial trials recorded",
                              file=sys.stderr)
                    finally:
                        _HARBOR[0] = None
        finally:
            _stop_proxy(proxy)
    if args.milestones and not args.dry_run:
        judge_milestones(args, env)


# ------------------------------------------------------------------ INCREMENTAL: teacher-forced
def _replay_arm(arm, msgs, sel, tmpl_body, tmpl_headers, args, env, sid, path):
    """Teacher-force one arm over the selected decision points; returns a summary line."""
    spent, n_ok, n_action, n_err = 0.0, 0, 0, 0
    consec_err = 0
    model = tmpl_body["model"]
    with open(path, "w") as f:
        for step, i in enumerate(sel):
            if spent >= args.budget_usd or consec_err >= 5:
                break
            orig = eng.extract_action(msgs[i]["content"])
            req = eng.build_request(tmpl_body, msgs[:i], args, sid)
            t0 = time.time()
            resp, err = eng.call_api(arm, req, tmpl_headers, env)
            rec = {"arm": arm, "step": step, "msg_index": i, "orig": orig,
                   "latency_s": round(time.time() - t0, 1)}
            if err:
                n_err += 1
                consec_err += 1
                rec["error"] = err
            else:
                consec_err = 0
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
        tmpl_body["model"] = args.model or model  # --model replays on a cheaper model
        tmpl_body["tools"] = eng.build_tools(used, tmpl_body["tools"])
        print(f"[swechat #{args.conv}] model={tmpl_body['model']} "
              f"(recorded {model}), {len(used)} tools", file=sys.stderr)
    else:
        eng.patch_cwd(tmpl_body, args.template, args.cwd_patch)
        msgs, points = eng.parse_session(args.session)
        # stub every tool the session references (incl. tool-search-discovered MCP
        # tools) so a tool-search session doesn't 400 on an unresolved reference
        tmpl_body["tools"] = eng.build_tools(eng.referenced_tool_names(msgs),
                                             tmpl_body["tools"])

    # start a proxy only if we need one and none is up; _start_proxy returns None when
    # it reuses an existing same-mode proxy, so we only ever stop the one WE started
    proxy = None
    if "headroom" in arms:
        proxy = _start_proxy(args.out, args.headroom_mode)

    sel = points[:: max(1, args.every)]
    if args.limit:
        sel = sel[: args.limit]
    sid = str(uuid.uuid4())
    print(f"[incremental] {len(msgs)} msgs, replaying {len(sel)}/{len(points)} decision points "
          f"per arm ({', '.join(arms)}), session {sid}")
    # arms hit independent endpoints with independent budgets — replay them concurrently;
    # steps WITHIN an arm stay sequential (incremental prompt caching needs nested prefixes)
    try:
        with ThreadPoolExecutor(max_workers=len(arms)) as ex:
            futures = [ex.submit(_replay_arm, arm, msgs, sel, tmpl_body, tmpl_headers,
                                 args, env, sid, os.path.join(out, f"{args.task}-{arm}.jsonl"))
                       for arm in arms]
            for fut in futures:
                print(fut.result())
    finally:
        _stop_proxy(proxy)  # always close the proxy we started (no-op if reused/none)


# ------------------------------------------------------------------ milestone judge (LLM, gen-side)
EXTRACT = ('Analyze this coding task and ONE solved reference trajectory, and extract 5-8 '
           'ORDERED, APPROACH-AGNOSTIC milestones ANY valid solution would hit (do not encode '
           'this particular approach). Each milestone must be checkable from a trajectory of '
           'tool calls, their results, and the final message.\nTASK:\n{task}\n'
           'REFERENCE TRAJECTORY:\n{summary}\n'
           'Return ONLY JSON {{"milestones":[{{"id":1,"name":"..."}}]}}')
COVER = ('TASK:\n{task}\nMILESTONES:\n{ms}\nANOTHER trajectory (steps show tool(target) -> '
         'result snippet; "answer:" lines and FINAL MESSAGE are the agent\'s own text):\n'
         '{summary}\nFor each milestone, did THIS agent achieve it? Mark achieved=true when '
         'the actions, their results, or the final message provide evidence it was reached; '
         'mark false when the trajectory shows it was not attempted or clearly failed.\n'
         'Return ONLY JSON {{"coverage":[{{"id":1,"achieved":true}}]}}')


def _digest(a):
    """Short, judge-readable label for one action: tool + its target."""
    if a.get("type") != "tool_use":
        return "answer"
    inp = a.get("input", {})
    tgt = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
    tgt = " ".join(str(tgt).split())[:70]
    return f"{a.get('name')}({tgt})" if tgt else a.get("name", "?")


def _result_snippet(msgs, i, limit=110):
    """Short tool-result snippet for the decision at msgs[i] (the next user turn)."""
    if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "user":
        return ""
    for b in msgs[i + 1]["content"]:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            s = " ".join(str(c or "").split())[:limit]
            if s:
                return (" -> ERROR: " if b.get("is_error") else " -> ") + s
            return ""
    return ""


def _summary(path):
    """Judge-readable trajectory: tool(target) -> result per step, answers, final message.

    Results and the agent's own text are what let the judge VERIFY a milestone rather
    than guess from tool names — without them, solved runs score near-zero coverage.
    """
    msgs, points = eng.parse_session(path)
    task = next((b["text"] for b in msgs[0]["content"] if b.get("type") == "text"), "")[:600]
    out, final = [], ""
    for n, i in enumerate(points[:80]):
        a = eng.extract_action(msgs[i]["content"])
        if a.get("type") == "text":
            text = " ".join(a.get("text", "").split())
            out.append(f"{n}. answer: {text[:200]}")
            final = text  # last text answer wins
        else:
            out.append(f"{n}. {_digest(a)}{_result_snippet(msgs, i)}")
    if final:
        out.append(f"FINAL MESSAGE: {final[:600]}")
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
    for task in resolve_tasks(args.tasks, args.dataset.split('/', 1)[0]):
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="generate trajectory data (spends); report.py displays it")
    ap.add_argument("--mode", choices=["full", "incremental", "judge"], default="full",
                    help="full/incremental = generate trajectories; "
                         "judge = LLM milestone scoring of existing runs")
    ap.add_argument("--agent", default="claude-code")
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--tasks", default=None,
                    help="comma list of terminal-bench task ids, or a number N for the "
                         "first N curated defaults; omitted = 5 (see --list-tasks)")
    ap.add_argument("--list-tasks", action="store_true",
                    help="print the locally known tasks (curated marked) and exit")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="harbor dataset (org/name-version); only the default is "
                         "validated so far")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --tasks random:N (omit = fresh sample; the plan "
                         "line always prints the chosen tasks for reproducibility)")
    ap.add_argument("--out", default="results/jobs/run")
    ap.add_argument("--model", default=None)
    # full
    ap.add_argument("--k", type=int, default=4,
                    help="trials per arm/task; band-overlap verdicts need >=2 and get "
                         "fragile below 4 (a low-k verdict can flip as bands widen)")
    ap.add_argument("--k-vanilla", type=int, default=None,
                    help="trials for the vanilla baseline (default k+1: the noise floor "
                         "is shared by every comparison)")
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--wall-timeout", type=int, default=2400,
                    help="PER-TRIAL wall budget in seconds (the cell gets wall_timeout * k)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="harbor -n: parallel trials per cell (parallel containers can add "
                         "resource-contention noise to trajectories; 1 = cleanest)")
    ap.add_argument("--agent-timeout-mult", type=int, default=None,
                    help="harbor agent EXECUTION timeout multiplier (headroom auto-3)")
    ap.add_argument("--setup-timeout-mult", type=float, default=3.0,
                    help="harbor agent SETUP timeout multiplier for every arm (installing "
                         "Claude Code + deps per container is slow; default 3 = ~18min)")
    ap.add_argument("--auth", default="auto", choices=["auto", "api-key", "subscription"],
                    help="auto = API key if set, else Claude Code login; subscription = force "
                         "the Claude Code login (container agent gets CLAUDE_CODE_OAUTH_TOKEN)")
    ap.add_argument("--milestones", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the Harbor commands without running")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have their k trials (default: skip them)")
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
    ap.add_argument("--headroom-mode", default="token", choices=["cache", "token"],
                    help="proxy mode for the incremental 'headroom' arm (token = compression, "
                         "the meaningful test; cache = ~passthrough)")
    ap.add_argument("--swechat", default=None, help="SWE-chat jsonl (alternative to --session)")
    ap.add_argument("--conv", type=int, default=0, help="conversation index within --swechat")
    args = ap.parse_args(argv)

    if args.list_tasks:
        for i, t in enumerate(dataset_tasks(args.dataset.split("/", 1)[0]), 1):
            mark = " *curated" if t in DEFAULT_TASKS else ""
            print(f"{i}. {t}{mark}{'   <- default 5' if i == 5 else ''}")
        print("(*curated = cost-ordered defaults; the rest is whatever harbor has cached "
              "locally — `harbor datasets download terminal-bench/terminal-bench-2-1` "
              "fetches the full dataset)")
        return
    if args.agent not in SUPPORTED_AGENTS:
        sys.exit(f"--agent {args.agent} is not implemented yet (TODO). "
                 "Only 'claude-code' is wired.")
    # export .env into the process env so Harbor / headroom subprocesses inherit the keys
    for k, v in eng.load_env().items():
        os.environ.setdefault(k, v)
    env = dict(os.environ)
    if args.mode == "full":
        full(args, env)
    elif args.mode == "judge":
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
