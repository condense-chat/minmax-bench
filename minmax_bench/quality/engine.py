#!/usr/bin/env python3
"""Teacher-forced trajectory replay: per-step context-fidelity scoring.

Takes a logged vanilla Claude Code session (Harbor trial session JSONL), and at
each assistant decision point replays the *same message prefix* through an arm's
endpoint (control = api.anthropic.com, condense = api.condense.chat/anthropic).
Scores whether the replayed next action agrees with the original one.

This removes free-running trajectory divergence: every step is an independent,
paired A/B on identical history. Control-replay agreement vs the original is the
noise floor (sampling temperature + reconstruction error); a method's agreement
is only signal to the extent it falls below that floor.

Caveat (report alongside solve-rate, not instead of it): this measures
information preserved in context, not end-to-end outcome — agents recover from
flipped actions, and per-step agreement doesn't capture compounding.

This module is the LIBRARY: session I/O, request building, calling, scoring.
The drivers are `scripts/generate.py --mode incremental` (batch, artifacts for
report.py) and `minmax-bench counterfactual` (interactive, local sessions).

Template = a real CC request captured via a local recording server (has the
version-matched system prompt, tools, beta headers, thinking/context_management
config). mcp__* tools are dropped by default (container runs had plain CC tools).
"""
import copy
import difflib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# where a Harbor claude-code trial stores its session transcript (cwd=/app slug);
# shared by report.py and generate.py so the path convention lives in one place
SESSION_GLOB = "agent/sessions/projects/-app/*.jsonl"

# USD per Mtok: input / output / cache_write (5-min TTL, 1.25x) / cache_read (0.1x).
# Matched by model-id prefix, longest match wins; extend when replaying new models.
PRICES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
    "claude-opus-4": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 1.00},
}
DEFAULT_PRICE_MODEL = "claude-sonnet-4-6"


def rates_for(model):
    """Longest-prefix price lookup; falls back to sonnet rates with a warning."""
    best = ""
    for prefix in PRICES:
        if model and model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    if not best:
        print(f"warn: no price entry for model {model!r}; using {DEFAULT_PRICE_MODEL} rates",
              file=sys.stderr)
        return PRICES[DEFAULT_PRICE_MODEL]
    return PRICES[best]

# Curated terminal-bench-2-1 tasks, validated in earlier bench runs and roughly ordered
# from short/cheap to long/expensive. `--tasks 5` (or omitting --tasks) takes the first N;
# any dataset task id also works by name (browse: https://hub.harborframework.com/datasets).
DEFAULT_TASKS = [
    "kv-store-grpc",                # short; known to expose planning-induction divergence
    "fix-code-vulnerability",       # short, focused edit
    "pypi-server",                  # medium service setup
    "write-compressor",             # medium algorithmic
    "schemelike-metacircular-eval", # medium-long interpreter work
    "dna-assembly",                 # long
    "qemu-alpine-ssh",              # long, infra-heavy
    "torch-pipeline-parallelism",   # long, GPU-flavored
    "torch-tensor-parallelism",     # long, GPU-flavored
]


def dataset_tasks(org="terminal-bench"):
    """Every task id harbor has locally for a dataset org (sorted); curated ones first.

    Harbor materializes task packages under ~/.cache/harbor/tasks/packages/<org>/;
    `harbor datasets download <org>/<dataset>` fetches the full set.
    """
    cache = os.path.expanduser(f"~/.cache/harbor/tasks/packages/{org}")
    local = sorted(d for d in (os.listdir(cache) if os.path.isdir(cache) else [])
                   if os.path.isdir(os.path.join(cache, d)))
    curated = DEFAULT_TASKS if org == "terminal-bench" else []
    return curated + [t for t in local if t not in curated]


def resolve_tasks(raw, org="terminal-bench", seed=None):
    """--tasks forms: omitted -> first 5 recommended; N -> first N known (recommended
    order first, then the rest alphabetically); random:N -> N sampled from everything
    known locally (use --seed to reproduce); else comma-separated names."""
    import random as _random
    raw = (raw or "").strip()
    rand_n = None
    if raw.lower().startswith("random:"):
        rand_n = raw.split(":", 1)[1]
        if not rand_n.isdigit():
            sys.exit(f"--tasks {raw}: expected random:<N>")
    if not raw or raw.isdigit() or rand_n:
        n = int(rand_n or raw or 5)
        pool = dataset_tasks(org)
        if n > len(pool):
            sys.exit(f"--tasks {raw or n}: only {len(pool)} {org} tasks are known locally — "
                     f"run `harbor datasets download <org>/<dataset>` to fetch the full "
                     f"dataset, or name tasks explicitly")
        if rand_n:
            return sorted(_random.Random(seed).sample(pool, n))
        return pool[:n]
    return [t for t in raw.split(",") if t.strip()]


# NOTE: teacher-forced replay executes no tools, so headroom's CCR retrieve loop cannot
# engage here — the 'headroom' arm below measures the proxy only (cache or token mode,
# per the driver's --headroom-mode). CCR is only measurable in generate.py --mode full.
ARMS = {
    "control": {"base": "https://api.anthropic.com"},
    "condense": {"base": "https://api.condense.chat/anthropic", "condense_auth": True},
    "headroom": {"base": os.environ.get("HEADROOM_PROXY", "http://localhost:8787")},
}


def load_env(path=".env"):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    return env


_CC_TOKEN_CACHE = ["unset"]  # one keychain read per process, not per request


def cc_oauth_token():
    """The user's own Claude Code subscription credential, read locally.

    Most Claude Code users have no API key — their auth is the OAuth token the
    Claude Code login stores. The bench replays THEIR sessions in Claude Code's
    own request shape, so when no API key is configured we authenticate the same
    way Claude Code does. The token stays in-process and is only ever sent where
    the arm points (api.anthropic.com or the user's chosen gateway).

    Sources, in order: CLAUDE_CODE_OAUTH_TOKEN env var, ~/.claude/.credentials.json,
    the macOS keychain item Claude Code maintains. Returns None when absent/expired.
    """
    if _CC_TOKEN_CACHE[0] != "unset":
        return _CC_TOKEN_CACHE[0]

    def parse(raw):
        try:
            oauth = json.loads(raw).get("claudeAiOauth") or {}
        except (json.JSONDecodeError, AttributeError):
            return None
        exp = oauth.get("expiresAt") or 0
        if exp and exp / 1000 < time.time():
            print("warn: Claude Code OAuth token is expired — open `claude` once to "
                  "refresh it, or set ANTHROPIC_API_KEY", file=sys.stderr)
            return None
        return oauth.get("accessToken")

    tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or None
    if not tok:
        cred = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(cred):
            tok = parse(open(cred).read())
    if not tok and sys.platform == "darwin":
        try:
            r = subprocess.run(["security", "find-generic-password",
                                "-s", "Claude Code-credentials", "-w"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                tok = parse(r.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            tok = None
    _CC_TOKEN_CACHE[0] = tok
    return tok


def auth_mode(env):
    """'api-key' | 'subscription' | None — how upstream calls will authenticate."""
    if env.get("ANTHROPIC_API_KEY"):
        return "api-key"
    if cc_oauth_token():
        return "subscription"
    return None


def check_arms(arms, env):
    """Fail-fast validation BEFORE any money is spent: known arms + required keys.

    Returns a list of human-readable problems (empty = good to go). call_api
    dereferences ARMS[arm] unconditionally, so skipping this check crashes
    mid-run — typically after the control arm already spent budget.
    """
    problems = []
    for arm in arms:
        if arm not in ARMS:
            problems.append(f"arm {arm!r} has no replay endpoint "
                            f"(known: {', '.join(sorted(ARMS))})")
    if not auth_mode(env):
        problems.append(
            "no auth found. Either:\n"
            "    - set ANTHROPIC_API_KEY (API billing), or\n"
            "    - use your Claude Code subscription: run `claude setup-token` and\n"
            "      `export CLAUDE_CODE_OAUTH_TOKEN=<the printed token>` "
            "(it is not persisted where the bench can read it otherwise)")
    if any(ARMS.get(a, {}).get("condense_auth") for a in arms) and not env.get("CONDENSE_API_KEY"):
        problems.append("CONDENSE_API_KEY missing — needed for the condense arm")
    return problems


def patch_cwd(tmpl_body, template_path, new_cwd):
    """Rewrite the template system prompt's advertised working directory to new_cwd.

    The template is a real CC request captured in SOME project; its ``<env>`` block
    hardcodes that project's cwd (e.g. ``working directory: /Users/x/dev/foo``) and
    Claude Code's project-dir SLUG form (``-Users-x-dev-foo``). Replaying a DIFFERENT
    session against that template tells the model it is in the wrong directory, so it
    cd's there and reads that project's files — the trajectory is lost from step 0.

    We auto-detect the cwd the template actually advertises (the old heuristic guessed
    a ``<repo>/ccwork`` path that no longer exists in the template, so the rewrite was
    a silent no-op) and replace BOTH the path and slug forms with the session's cwd.
    """
    s = json.dumps(tmpl_body["system"])
    m = re.search(r'working directory:\s*(/[^\s\\"]+)', s)
    cap_cwd = m.group(1) if m else None
    if cap_cwd and cap_cwd != new_cwd:
        s = s.replace(cap_cwd, new_cwd)
        s = s.replace(cap_cwd.replace("/", "-"), new_cwd.replace("/", "-"))  # CC dir slug
        tmpl_body["system"] = json.loads(s)
    return tmpl_body


def recorded_usage(path):
    """Per-decision-point usage as RECORDED in the session (one per assistant response).

    Ordered by first occurrence of each requestId — matching parse_session's merged
    assistant messages closely enough to anchor a backtest: "what did these turns
    actually consume when the session ran", next to what the replays consumed.
    """
    seen, out = set(), []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("isSidechain") or rec.get("type") != "assistant":
                continue
            rid = rec.get("requestId")
            u = rec.get("message", {}).get("usage") or {}
            if not u or rid in seen:
                continue  # a response split across records repeats the same usage
            seen.add(rid)
            out.append(u)
    return out


def parse_session(path):
    """Session JSONL -> API-shaped message list + decision point indices.

    Assistant records are grouped by requestId (one API response can span several
    records). A response that hit max_tokens mid-thinking and produced nothing but
    thinking blocks is dropped — CC itself omits it from history, and replaying a
    cross-response merge fails thinking-signature validation.
    """
    groups = []  # [{'role','content','req','stop'}]
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("isSidechain"):
            continue
        t = rec.get("type")
        if t not in ("user", "assistant"):
            continue  # attachments/queue-operations aren't part of the API history
        m = rec["message"]
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        content = copy.deepcopy(content)
        req = rec.get("requestId") if t == "assistant" else None
        if groups and groups[-1]["role"] == m["role"] and (t == "user" or groups[-1]["req"] == req):
            groups[-1]["content"].extend(content)
        else:
            groups.append({"role": m["role"], "content": content, "req": req,
                           "stop": m.get("stop_reason")})
        if t == "assistant":
            groups[-1]["stop"] = m.get("stop_reason") or groups[-1]["stop"]

    msgs = []
    for g in groups:
        thinking_only = all(b.get("type") in ("thinking", "redacted_thinking")
                            for b in g["content"])
        if g["role"] == "assistant" and g["stop"] == "max_tokens" and thinking_only:
            continue  # truncated thinking-only response; CC drops it from history
        if msgs and msgs[-1]["role"] == g["role"]:
            if g["role"] == "assistant":
                print(f"warn: merging adjacent assistant responses "
                      f"({msgs[-1].get('req')} + {g['req']}) — signature check may reject",
                      file=sys.stderr)
            msgs[-1]["content"].extend(g["content"])
        else:
            msgs.append({"role": g["role"], "content": g["content"], "req": g["req"]})
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return msgs, points


def load_swechat(path, idx):
    """Load conversation `idx` from a SWE-chat jsonl -> (msgs, points, model, used_tools).

    SWE-chat blocks use `kind` (text/tool_use/tool_result), role 'tool' for tool results,
    and have thinking signatures stripped (so we drop thinking). Real public Claude Code
    sessions on real repos — long, exploratory, no verifier reward.
    """
    rec = [json.loads(line) for line in open(path)][idx]
    used = set()
    msgs = []
    for m in rec["messages"]:
        role = "user" if m["role"] == "tool" else m["role"]
        content = []
        for b in m.get("blocks", []):
            k = b.get("kind")
            if k == "text" and b.get("text"):
                content.append({"type": "text", "text": b["text"]})
            elif k == "tool_use":
                used.add(b["tool_name"])
                content.append({"type": "tool_use", "id": b["tool_use_id"],
                                "name": b["tool_name"], "input": b.get("tool_input") or {}})
            elif k == "tool_result":
                blk = {"type": "tool_result", "tool_use_id": b["tool_use_id"],
                       "content": b.get("content") or ""}
                if b.get("is_error"):
                    blk["is_error"] = True
                content.append(blk)
            # thinking blocks: signatures stripped in the public dataset -> drop
        if not content:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].extend(content)
        else:
            msgs.append({"role": role, "content": content})

    # repair tool_use<->tool_result pairing (Anthropic: each assistant tool_use needs a
    # matching tool_result FIRST in the next user turn; orphan results / dangling uses 400).
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        use_ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
        if not use_ids:
            continue
        if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "user":
            msgs.insert(i + 1, {"role": "user", "content": []})
        nxt = msgs[i + 1]
        results = {b["tool_use_id"]: b for b in nxt["content"] if b.get("type") == "tool_result"}
        others = [b for b in nxt["content"] if b.get("type") != "tool_result"]
        paired = [results.get(uid, {"type": "tool_result", "tool_use_id": uid,
                                    "content": "(result omitted)"}) for uid in use_ids]
        nxt["content"] = paired + others  # matched results first, drops orphans
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return msgs, points, rec["model"], used


_MCP_TOOL_RE = re.compile(r"mcp__[A-Za-z0-9_]+")


def referenced_tool_names(msgs):
    """Every tool name the messages reference — not just the ones actually CALLED.

    Direct ``tool_use`` names are the obvious set, but sessions that use Claude
    Code's tool-search feature also reference MCP tools BY NAME inside tool-search
    results without ever calling them. Anthropic validates every such reference
    against the request's ``tools`` array (``Tool reference '…' not found in
    available tools`` → 400), so a replay must stub the search-discovered tools
    too. tool_use names are collected structurally; mcp__ tools (the only ones
    surfaced by search in practice) are swept from the serialized content.
    """
    names = {b["name"] for m in msgs for b in m.get("content", [])
             if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")}
    names.update(_MCP_TOOL_RE.findall(json.dumps(msgs)))
    return names


# ------------------------------------------------------------------ faithful capture
# The most faithful reconstruction of a session's request isn't to hand-rebuild the
# system prompt / CLAUDE.md / tools — it is to let the ACTUAL Claude Code binary do it.
# CC keeps every version it has run under ~/.local/share/claude/versions/<version>, and
# each session records the `version` it ran on. We run the version-matched binary once
# in the session's cwd, pointed at a LOCAL capture proxy that returns a canned response
# (no upstream call, no spend), and grab the exact request it emits: base prompt (from
# the binary), CLAUDE.md + env (re-read from disk), and the full tool catalog (MCP
# re-connected). Teacher-forcing the recorded messages through THAT is the real thing.
CC_VERSIONS_DIR = os.path.expanduser("~/.local/share/claude/versions")


def cc_binary_for(version=None):
    """Path to the Claude Code binary matching `version` (exact if kept on disk), else
    the newest kept version, else whatever `claude` is on PATH. None if none found."""
    import shutil
    if version:
        exact = os.path.join(CC_VERSIONS_DIR, version)
        if os.path.isfile(exact):
            return exact
    if os.path.isdir(CC_VERSIONS_DIR):
        kept = sorted(os.listdir(CC_VERSIONS_DIR))  # semver-ish; newest last
        if kept:
            return os.path.join(CC_VERSIONS_DIR, kept[-1])
    return shutil.which("claude")


def _canned_sse(model):
    evs = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_cap", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "OK"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in evs).encode()


def capture_cc_request(cwd, model=None, version=None, port=8791, timeout=90):
    """Run the version-matched CC binary once in `cwd` against a local capture proxy and
    return the EXACT request it builds (system, tools, thinking/context_management/…), or
    None. Nothing leaves the machine: the proxy returns a canned response so `claude -p`
    exits. `model` is passed to `--model` so the config matches the session's model.
    Reads the user's own binary — the CALLER must have obtained consent."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    binary = cc_binary_for(version)
    if not binary:
        return None
    captured = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self._send(200, b'{"ok":true}', "application/json")

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = self.rfile.read(n) if n else b"{}"
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                req = {}
            if "count_tokens" in self.path:
                self._send(200, b'{"input_tokens":3}', "application/json")
                return
            if "/v1/messages" in self.path:
                captured.append(req)  # the real agent turn carries `tools`
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(_canned_sse(req.get("model") or model or "claude-sonnet-4-6"))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send(self, code, b, ct):
            self.send_response(code)
            self.send_header("content-type", ct)
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    try:
        srv = HTTPServer(("127.0.0.1", port), H)
    except OSError:
        return None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        cmd = [binary, "-p", "reply with just: OK"]
        if model:
            cmd += ["--model", model]
        renv = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}
        subprocess.run(cmd, cwd=cwd, env=renv, timeout=timeout, capture_output=True)
    except (subprocess.SubprocessError, OSError):
        pass
    finally:
        srv.shutdown()
        t.join(timeout=3)
    if not captured:
        return None
    return max(captured, key=lambda r: len(r.get("tools", [])))  # the turn with the catalog


def captured_reminders(captured_req):
    """The <system-reminder> context blocks CC injected into its first user message
    (CLAUDE.md, env/git, the agent+tool catalog) — everything except the throwaway
    capture prompt. CC adds these at request-build time and the transcript usually does
    NOT log them, so a faithful replay must carry them into the recorded first user turn."""
    msgs = (captured_req or {}).get("messages", [])
    c = msgs[0].get("content", []) if msgs else []
    if not isinstance(c, list):
        return []
    return [b for b in c if isinstance(b, dict) and b.get("type") == "text"
            and "system-reminder" in b.get("text", "")]


def ensure_reminders(msgs, reminders):
    """Return msgs with the captured injected reminders present in the first user turn
    (prepended) — unless it already carries them (some CC versions log them). Non-
    mutating. Makes a replay see the same CLAUDE.md/env context the original run did."""
    if not reminders or not msgs:
        return msgs
    first = json.dumps(msgs[0].get("content"))
    if "system-reminder" in first and "claudeMd" in first:
        return msgs  # already present in the recording — don't duplicate
    out = [dict(m) for m in msgs]
    c = out[0].get("content")
    body = c if isinstance(c, list) else [{"type": "text", "text": c or ""}]
    out[0]["content"] = [*reminders, *body]
    return out


def build_tools(used, tmpl_tools):
    """Real template defs for known tools + permissive stubs for the rest (for replay)."""
    by_name = {t["name"]: t for t in tmpl_tools}
    out = []
    for name in sorted(used):
        if name in by_name:
            out.append(by_name[name])
        else:
            out.append({"name": name, "description": f"{name} tool",
                        "input_schema": {"type": "object", "additionalProperties": True}})
    return out


def extract_action(content):
    """First tool_use in an assistant content list, else the text."""
    for b in content:
        if b.get("type") == "tool_use":
            return {"type": "tool_use", "name": b["name"], "input": b.get("input", {})}
    text = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
    return {"type": "text", "text": text}


def norm_ws(s):
    return " ".join(str(s).split())


_CD_PREFIX = re.compile(r"(?:^|(?<=&&)|(?<=;))\s*cd\s+\S+\s*&&\s*")
_SEG_SPLIT = re.compile(r"&&|\|\||;|\n|\|")


def _norm_cmd(cmd, cwd="/app"):
    """Strip replay cwd artifacts: `cd <dir> && ` prefixes and absolute-cwd paths.

    A replayed model that is unsure of its cwd defensively writes `cd /app && python x`
    or `-I/app` where the original wrote `python x` / `-I.` — same action, different
    phrasing. Normalizing both sides keeps the comparison about the decision.
    """
    c = _CD_PREFIX.sub("", str(cmd))
    c = c.replace(cwd.rstrip("/") + "/", "").replace(cwd.rstrip("/"), ".")
    return norm_ws(c)


def _programs(cmd):
    """The sequence of programs a shell command invokes (first token per segment)."""
    progs = []
    for seg in _SEG_SPLIT.split(cmd):
        toks = [t for t in seg.split() if "=" not in t]  # skip leading VAR=… assignments
        if toks and not toks[0].startswith("#"):
            progs.append(toks[0])
    return progs


def score(orig, replay, cwd="/app"):
    """-> (agree_exact, agree_action, sim). agree_action = same tool + same target."""
    if orig["type"] != replay["type"]:
        return False, False, 0.0
    if orig["type"] == "text":
        sim = difflib.SequenceMatcher(None, orig["text"], replay["text"]).ratio()
        # both chose to stop and answer, but only count it as the same decision when
        # the answers are in the same ballpark — a bail-out ("can't proceed") must not
        # agree with a substantive final answer
        return sim > 0.9, sim > 0.5, sim
    if orig["name"] != replay["name"]:
        return False, False, 0.0
    a, b = orig["input"], replay["input"]
    na, nb = norm_ws(json.dumps(a, sort_keys=True)), norm_ws(json.dumps(b, sort_keys=True))
    exact = na == nb
    sim = 1.0 if exact else difflib.SequenceMatcher(None, na, nb).ratio()
    if orig["name"] == "Bash":
        ca, cb = _norm_cmd(a.get("command", ""), cwd), _norm_cmd(b.get("command", ""), cwd)
        # same action = same command modulo cwd artifacts, or the same program
        # sequence with near-identical text (path spelling, ls target, flag order)
        action = ca == cb or (_programs(ca) == _programs(cb)
                              and difflib.SequenceMatcher(None, ca, cb).ratio() > 0.7)
    elif "file_path" in a or "file_path" in b:
        action = a.get("file_path") == b.get("file_path")
    else:
        action = sim > 0.9
    return exact, exact or action, sim


def build_request(tmpl_body, prefix, args, session_id):
    # SWE-chat replays pre-build their tool list and target older models, so they keep
    # all tools and drop the 4.6-era config keys; both knobs are also available on their
    # own (counterfactual replay of local sessions keeps mcp__ tool stubs, and drops the
    # beta config only when replaying on a model other than the template's).
    keep_all_tools = getattr(args, "swechat", None) or getattr(args, "keep_all_tools", False)
    drop_beta_config = getattr(args, "swechat", None) or getattr(args, "drop_beta_config", False)
    # never mutate the caller's messages: per-message dict copies, and content lists are
    # rebuilt wherever we change them (strip-thinking, cache breakpoint below) — callers
    # replay the same msgs list once per step, so leaked mutations would compound
    # (e.g. cache_control accumulating on interior blocks until the API rejects >4)
    prefix = [dict(m) for m in prefix]
    req = {
        "model": tmpl_body["model"],
        "max_tokens": args.max_tokens,
        "system": tmpl_body["system"],
        "tools": (tmpl_body["tools"] if keep_all_tools else
                  [t for t in tmpl_body["tools"] if not t.get("name", "").startswith("mcp__")]),
        "messages": prefix,
        "stream": True,  # condense's gateway 504s on long non-streaming requests; CC streams
        "metadata": {"user_id": json.dumps({"device_id": "tmb-fidelity-replay",
                                            "account_uuid": "", "session_id": session_id})},
    }
    if not drop_beta_config:  # these are 4.6-era; older models 400 on them
        for k in ("thinking", "context_management", "output_config"):
            if k in tmpl_body:
                req[k] = tmpl_body[k]
    if args.strip_thinking:
        req.pop("thinking", None)
        for m in req["messages"]:
            m["content"] = [b for b in m["content"]
                            if b.get("type") not in ("thinking", "redacted_thinking")]
    # incremental prompt caching across sequential replays (prefixes are nested)
    last_msg = req["messages"][-1]
    last = last_msg["content"]
    if last and isinstance(last[-1], dict) and last[-1].get("type") in ("text", "tool_result"):
        patched = {**last[-1], "cache_control": {"type": "ephemeral"}}
        req["messages"][-1] = {**last_msg, "content": list(last[:-1]) + [patched]}
    return req


def read_sse(resp):
    """Accumulate an SSE /v1/messages stream into a response dict."""
    content, usage = [], {}
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "message_start":
            usage.update(ev["message"].get("usage", {}))
        elif t == "content_block_start":
            b = dict(ev["content_block"])
            if b.get("type") == "tool_use":
                b["_json"] = ""
            content.append(b)
        elif t == "content_block_delta":
            d, b = ev["delta"], content[ev["index"]]
            if d["type"] == "text_delta":
                b["text"] = b.get("text", "") + d["text"]
            elif d["type"] == "input_json_delta":
                b["_json"] = b.get("_json", "") + d["partial_json"]
            elif d["type"] == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d["thinking"]
        elif t == "message_delta":
            usage.update(ev.get("usage", {}))
        elif t == "error":
            return None, json.dumps(ev)[:2000]
    for b in content:
        if "_json" in b:
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except json.JSONDecodeError as e:
                b["input"] = {}
                b["_input_parse_error"] = str(e)
    return {"content": content, "usage": usage}, None


def call_api(arm, req, tmpl_headers, env):
    url = ARMS[arm]["base"] + "/v1/messages?beta=true"
    headers = {
        "content-type": "application/json",
        "anthropic-version": tmpl_headers.get("anthropic-version", "2023-06-01"),
        "user-agent": tmpl_headers.get("user-agent", "tmb-fidelity-replay"),
    }
    if tmpl_headers.get("anthropic-beta"):
        headers["anthropic-beta"] = tmpl_headers["anthropic-beta"]
    if env.get("ANTHROPIC_API_KEY"):
        headers["x-api-key"] = env["ANTHROPIC_API_KEY"]
    else:  # Claude Code subscription auth (validated up front by check_arms)
        headers["Authorization"] = f"Bearer {cc_oauth_token()}"
        beta = headers.get("anthropic-beta", "")
        if "oauth-2025-04-20" not in beta:
            headers["anthropic-beta"] = (beta + "," if beta else "") + "oauth-2025-04-20"
    if ARMS[arm].get("condense_auth"):
        headers["X-Condense-Auth-Token"] = env["CONDENSE_API_KEY"]
    data = json.dumps(req).encode()
    for attempt in range(3):
        try:
            r = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(r, timeout=600) as resp:
                if req.get("stream"):
                    return read_sse(resp)
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            return None, f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001
            if attempt < 2:
                time.sleep(10)
                continue
            return None, f"{type(e).__name__}: {e}"
    return None, "unreachable"


def cost_usd(usage, model=None):
    price = rates_for(model or DEFAULT_PRICE_MODEL)
    return (usage.get("input_tokens", 0) * price["input"]
            + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]) / 1e6


JUDGE_MODEL = "claude-haiku-4-5"  # cheap; the equivalence call is easy
_JUDGE_PROMPT = (
    "You are judging whether two candidate NEXT ACTIONS of a coding agent are FUNCTIONALLY "
    "EQUIVALENT decisions. The agent is mid-task; both were proposed from the same history.\n"
    "TASK (truncated):\n{task}\n"
    "ACTION A (what the agent originally did):\n{a}\n"
    "ACTION B (replayed under compaction):\n{b}\n"
    "Equivalent = same kind of step toward the task with the same target and intent (e.g. grep "
    "vs rg for the same pattern, the same file read a different way, the same command with "
    "reordered flags). NOT equivalent = different target/file, different approach, running vs "
    "answering, testing vs editing, or a step that sends the trajectory elsewhere.\n"
    'Return ONLY JSON {{"equivalent": true|false}}.')


def _action_text(a):
    if a.get("type") == "text":
        return f"(final answer) {a.get('text', '')[:280]}"
    inp = a.get("input", {})
    tgt = (inp.get("command") or inp.get("file_path") or inp.get("pattern")
           or json.dumps(inp, sort_keys=True))
    return f"{a.get('name', '?')}: {' '.join(str(tgt).split())[:280]}"


def judge_equivalent(orig, replay, env, task_hint=""):
    """LLM adjudication: are these two next-actions functionally equivalent? Returns
    True/False (None on error). Meant for structural NEAR-MISSES — it upgrades a
    replay that chose an equivalent-but-differently-spelled action to 'agrees'."""
    prompt = _JUDGE_PROMPT.format(task=(task_hint or "")[:400],
                                  a=_action_text(orig), b=_action_text(replay))
    req = {"model": JUDGE_MODEL, "max_tokens": 60, "temperature": 0,
           "messages": [{"role": "user", "content": prompt}]}
    resp, err = call_api("control", req, {"anthropic-version": "2023-06-01"}, env)
    if err:
        return None
    t = "".join(b.get("text", "") for b in resp.get("content", [])).strip()
    try:
        return bool(json.loads(t[t.index("{"): t.rindex("}") + 1]).get("equivalent"))
    except (ValueError, json.JSONDecodeError):
        return None
