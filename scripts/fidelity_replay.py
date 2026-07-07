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

Usage:
  python3 scripts/fidelity_replay.py \
    --session <trial>/agent/sessions/projects/-app/<id>.jsonl \
    --template <captured req_00.json> \
    --arm control --out results/fidelity/control.jsonl \
    [--every 1] [--limit 0] [--max-tokens 6000] [--budget-usd 5] [--strip-thinking]

Template = a real CC request captured via a local recording server (has the
version-matched system prompt, tools, beta headers, thinking/context_management
config). mcp__* tools are dropped (container runs had plain CC tools).
"""
import argparse
import copy
import difflib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC alias is 3.11+; this repo's python3 is 3.10

# claude-sonnet-4-6 pricing, USD per Mtok
PRICE = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}

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
    rec = [json.loads(l) for l in open(path)][idx]
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


def score(orig, replay):
    """-> (agree_exact, agree_action, sim). agree_action = same tool + same target."""
    if orig["type"] != replay["type"]:
        return False, False, 0.0
    if orig["type"] == "text":
        sim = difflib.SequenceMatcher(None, orig["text"], replay["text"]).ratio()
        return sim > 0.9, True, sim  # same decision: stop and answer
    if orig["name"] != replay["name"]:
        return False, False, 0.0
    a, b = orig["input"], replay["input"]
    sim = difflib.SequenceMatcher(
        None, norm_ws(json.dumps(a, sort_keys=True)), norm_ws(json.dumps(b, sort_keys=True))
    ).ratio()
    exact = norm_ws(json.dumps(a, sort_keys=True)) == norm_ws(json.dumps(b, sort_keys=True))
    if orig["name"] == "Bash":
        action = norm_ws(a.get("command", "")) == norm_ws(b.get("command", ""))
    elif "file_path" in a or "file_path" in b:
        action = a.get("file_path") == b.get("file_path")
    else:
        action = sim > 0.9
    return exact, exact or action, sim


def build_request(tmpl_body, prefix, args, session_id):
    req = {
        "model": tmpl_body["model"],
        "max_tokens": args.max_tokens,
        "system": tmpl_body["system"],
        "tools": (tmpl_body["tools"] if getattr(args, "swechat", None)
                  else [t for t in tmpl_body["tools"] if not t.get("name", "").startswith("mcp__")]),
        "messages": prefix,
        "stream": True,  # condense's gateway 504s on long non-streaming requests; CC streams
        "metadata": {"user_id": json.dumps({"device_id": "tmb-fidelity-replay",
                                            "account_uuid": "", "session_id": session_id})},
    }
    if not getattr(args, "swechat", None):  # these are 4.6-era; drop for SWE-chat (older
        for k in ("thinking", "context_management", "output_config"):  # models 400; no thinking blocks anyway)
            if k in tmpl_body:
                req[k] = tmpl_body[k]
    if args.strip_thinking:
        req.pop("thinking", None)
        for m in req["messages"]:
            m["content"] = [b for b in m["content"]
                            if b.get("type") not in ("thinking", "redacted_thinking")]
    # incremental prompt caching across sequential replays (prefixes are nested)
    last = req["messages"][-1]["content"]
    if last and isinstance(last[-1], dict) and last[-1].get("type") in ("text", "tool_result"):
        last[-1] = {**last[-1], "cache_control": {"type": "ephemeral"}}
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


def cost_usd(usage):
    return (usage.get("input_tokens", 0) * PRICE["input"]
            + usage.get("output_tokens", 0) * PRICE["output"]
            + usage.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * PRICE["cache_read"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--swechat", help="SWE-chat jsonl (alternative to --session)")
    ap.add_argument("--conv", type=int, default=0, help="conversation index within --swechat")
    ap.add_argument("--template", default="data/cc_request_template.json",
                    help="captured CC request template (regenerate via scripts/capture_cc_template.py)")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="replay every Nth decision point")
    ap.add_argument("--limit", type=int, default=0, help="max decision points (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--strip-thinking", action="store_true")
    ap.add_argument("--cwd-patch", default="/app",
                    help="rewrite the capture cwd in the system prompt to this")
    args = ap.parse_args()

    env = {**load_env(), **os.environ}
    if not env.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing (.env or env)")
    if ARMS[args.arm].get("condense_auth") and not env.get("CONDENSE_API_KEY"):
        sys.exit("CONDENSE_API_KEY missing (.env or env)")

    cap = json.load(open(args.template))
    tmpl_body, tmpl_headers = cap["body"], {k.lower(): v for k, v in cap["headers"].items()}
    # patch the local capture cwd out of the system prompt so it matches the container run
    cap_cwd = os.path.dirname(os.path.abspath(args.template))
    tmpl_body["system"] = json.loads(
        json.dumps(tmpl_body["system"]).replace(os.path.join(os.path.dirname(cap_cwd), "ccwork"),
                                                args.cwd_patch))

    if args.swechat:
        msgs, points, model, used = load_swechat(args.swechat, args.conv)
        tmpl_body["model"] = model
        tmpl_body["tools"] = build_tools(used, tmpl_body["tools"])
        print(f"[swechat #{args.conv}] model={model}, {len(used)} tools", file=sys.stderr)
    else:
        msgs, points = parse_session(args.session)
    sel = points[:: args.every]
    if args.limit:
        sel = sel[: args.limit]
    session_id = str(uuid.uuid4())
    print(f"[{args.arm}] {len(msgs)} msgs, {len(points)} decision points, "
          f"replaying {len(sel)} (every={args.every}), replay session {session_id}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    spent, n_exact, n_action, n_err = 0.0, 0, 0, 0
    with open(args.out, "w") as out:
        for step, i in enumerate(sel):
            if spent >= args.budget_usd:
                print(f"budget ${args.budget_usd} reached, stopping at step {step}")
                break
            orig = extract_action(msgs[i]["content"])
            req = build_request(tmpl_body, copy.deepcopy(msgs[:i]), args, session_id)
            ts_start = datetime.now(UTC).isoformat()
            t0 = time.time()
            resp, err = call_api(args.arm, req, tmpl_headers, env)
            ts_end = datetime.now(UTC).isoformat()
            rec = {"arm": args.arm, "step": step, "msg_index": i,
                   "n_prefix_msgs": i, "orig": orig, "ts_start": ts_start, "ts_end": ts_end,
                   "latency_s": round(time.time() - t0, 1)}
            if err:
                n_err += 1
                rec["error"] = err
                print(f"  step {step} (msg {i}): ERROR {err[:160]}")
            else:
                replay = extract_action(resp.get("content", []))
                exact, action, sim = score(orig, replay)
                usage = resp.get("usage", {})
                c = cost_usd(usage)
                spent += c
                n_exact += exact
                n_action += action
                rec.update(replay=replay, agree_exact=exact, agree_action=action,
                           sim=round(sim, 3), usage=usage, cost_usd=round(c, 4))
                o = orig.get("name", "text")
                r = replay.get("name", "text")
                print(f"  step {step} (msg {i}): {o} vs {r} "
                      f"exact={exact} action={action} sim={sim:.2f} "
                      f"ctx={usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0) + usage.get('input_tokens', 0)} "
                      f"${spent:.2f}")
            out.write(json.dumps(rec) + "\n")
            out.flush()

    n_ok = len(sel) - n_err
    if n_ok:
        print(f"\n[{args.arm}] SUMMARY: exact {n_exact}/{n_ok} ({n_exact / n_ok:.0%}), "
              f"action {n_action}/{n_ok} ({n_action / n_ok:.0%}), "
              f"errors {n_err}, cost ${spent:.2f}")


if __name__ == "__main__":
    main()
