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


def check_arms(arms, env):
    """Fail-fast validation BEFORE any money is spent: known arms + required keys.

    Returns a list of human-readable problems (empty = good to go). call_api
    dereferences ARMS[arm] and the key env vars unconditionally, so skipping this
    check crashes mid-run — typically after the control arm already spent budget.
    """
    problems = []
    for arm in arms:
        if arm not in ARMS:
            problems.append(f"arm {arm!r} has no replay endpoint "
                            f"(known: {', '.join(sorted(ARMS))})")
    if not env.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY missing (.env or environment)")
    if any(ARMS.get(a, {}).get("condense_auth") for a in arms) and not env.get("CONDENSE_API_KEY"):
        problems.append("CONDENSE_API_KEY missing — needed for the condense arm")
    return problems


def patch_cwd(tmpl_body, template_path, new_cwd):
    """Rewrite the capture machine's cwd in the template system prompt to new_cwd.

    The template was captured from a live CC session whose cwd was <repo>/ccwork
    (sibling of the template's data/ dir); replayed sessions ran elsewhere
    (containers: /app, local sessions: their own project dir), and a mismatched
    advertised cwd depresses action fidelity for every arm.
    """
    cap_dir = os.path.dirname(os.path.abspath(template_path))
    cap_cwd = os.path.join(os.path.dirname(cap_dir), "ccwork")
    tmpl_body["system"] = json.loads(
        json.dumps(tmpl_body["system"]).replace(cap_cwd, new_cwd))
    return tmpl_body


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
        "x-api-key": env["ANTHROPIC_API_KEY"],
    }
    if tmpl_headers.get("anthropic-beta"):
        headers["anthropic-beta"] = tmpl_headers["anthropic-beta"]
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
