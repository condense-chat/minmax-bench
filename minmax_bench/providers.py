"""Render normalized messages/tools into real provider request bodies.

We build request payloads a real harness would send, including prompt-cache
breakpoints, so that a proxy under test sees a realistic request and the upstream
returns realistic cache-aware usage. Two provider dialects are supported:
Anthropic Messages API and OpenAI Chat Completions.
"""

from __future__ import annotations

from .models import Block, BlockKind, Message, Provider, Role, Session, Usage

_EPHEMERAL = {"type": "ephemeral"}


# --------------------------------------------------------------------------- #
# Anthropic Messages API
# --------------------------------------------------------------------------- #
def _anthropic_block(b: Block) -> dict | None:
    if b.kind == BlockKind.text:
        if not (b.text and b.text.strip()):
            return None
        return {"type": "text", "text": b.text}
    if b.kind == BlockKind.thinking:
        out = {"type": "thinking", "thinking": b.text or ""}
        if b.signature:
            out["signature"] = b.signature
        return out
    if b.kind == BlockKind.tool_use:
        return {
            "type": "tool_use",
            "id": b.tool_use_id or "",
            "name": b.tool_name or "",
            "input": b.tool_input or {},
        }
    if b.kind == BlockKind.tool_result:
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id or "",
            "content": b.content or "",
            "is_error": b.is_error,
        }
    return None


def _anthropic_messages(prefix: list[Message]) -> list[dict]:
    """Fold normalized messages into Anthropic user/assistant turns.

    Normalized ``tool`` messages and their tool_result blocks are attached to a
    ``user`` turn, matching how Anthropic expects tool results to be returned.
    """
    out: list[dict] = []
    for msg in prefix:
        role = "assistant" if msg.role == Role.assistant else "user"
        content = [c for b in msg.blocks if (c := _anthropic_block(b)) is not None]
        if not content:
            continue
        # Merge consecutive same-role turns (e.g. tool_result user turn following
        # a user turn) so the user/assistant alternation stays valid.
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(content)
        else:
            out.append({"role": role, "content": content})
    return out


def _sanitize_anthropic(messages: list[dict]) -> list[dict]:
    """Make a (possibly mid-conversation) prefix a valid Anthropic request.

    SWE-chat and other truncated captures can begin partway through a session,
    so a request prefix may start with a ``tool_result`` whose ``tool_use`` isn't
    in the request, or start with an assistant turn. Anthropic rejects both. We
    drop tool_result blocks with no preceding tool_use in this request, remove any
    emptied turns, re-merge adjacent same-role turns, and ensure the sequence
    starts with a user turn.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    for m in messages:
        blocks = []
        for b in m["content"]:
            if b.get("type") == "tool_use":
                seen.add(b.get("id"))
                blocks.append(b)
            elif b.get("type") == "tool_result":
                if b.get("tool_use_id") in seen:  # drop orphaned results
                    blocks.append(b)
            else:
                blocks.append(b)
        if blocks:
            kept.append({"role": m["role"], "content": blocks})

    merged: list[dict] = []
    for m in kept:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)

    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continued)"}]})
    return _pair_tool_uses(merged)


def _pair_tool_uses(messages: list[dict]) -> list[dict]:
    """Ensure every assistant ``tool_use`` is immediately followed by a user turn
    containing its ``tool_result`` — Anthropic rejects unpaired tool_uses (which
    occur when a source transcript is missing a tool result). Missing results get
    a small placeholder so the turn structure stays valid.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
            if ids:
                nxt = (
                    messages[i + 1]
                    if i + 1 < len(messages) and messages[i + 1]["role"] == "user"
                    else None
                )
                present = (
                    {b.get("tool_use_id") for b in nxt["content"] if b.get("type") == "tool_result"}
                    if nxt
                    else set()
                )
                stubs = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "[tool result unavailable]",
                    }
                    for tid in ids
                    if tid not in present
                ]
                if stubs and nxt is not None:
                    nxt["content"] = stubs + nxt["content"]  # results must lead the turn
                elif stubs:
                    out.append({"role": "user", "content": stubs})
        i += 1
    return out


def _mark_cache(messages: list[dict], breakpoints: int = 2) -> None:
    """Put ``cache_control`` on the last block of the last ``breakpoints`` turns.

    Mirrors how Claude Code marks a moving cache breakpoint near the tail so each
    subsequent call reads the prior prefix from cache.
    """
    marked = 0
    for msg in reversed(messages):
        if marked >= breakpoints:
            break
        content = msg["content"]
        if content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = _EPHEMERAL
            marked += 1


def render_anthropic(
    session: Session, prefix: list[Message], *, max_tokens: int = 1, cache: bool = True
) -> dict:
    body: dict = {
        "model": session.model,
        "max_tokens": max_tokens,
        # Explicitly opt out of extended thinking: keeps max_tokens=1 valid and
        # stops compression proxies (condense) from injecting a thinking budget
        # that would exceed our tiny output cap. We measure input-side usage, so
        # generated reasoning is irrelevant.
        "thinking": {"type": "disabled"},
        "messages": _sanitize_anthropic(_anthropic_messages(prefix)),
    }
    if session.system:
        sys_block = {"type": "text", "text": session.system}
        if cache:
            sys_block["cache_control"] = _EPHEMERAL
        body["system"] = [sys_block]
    if session.tools:
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema or {"type": "object", "properties": {}},
            }
            for t in session.tools
        ]
        if cache:
            tools[-1]["cache_control"] = _EPHEMERAL
        body["tools"] = tools
    if cache:
        _mark_cache(body["messages"])
    return body


def usage_from_anthropic(resp: dict) -> Usage:
    u = resp.get("usage", {}) or {}
    return Usage(
        input_tokens=u.get("input_tokens", 0) or 0,
        output_tokens=u.get("output_tokens", 0) or 0,
        cache_read=u.get("cache_read_input_tokens", 0) or 0,
        cache_write=u.get("cache_creation_input_tokens", 0) or 0,
    )


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions
# --------------------------------------------------------------------------- #
def _openai_messages(
    session: Session, prefix: list[Message], *, flatten_tools: bool = False
) -> list[dict]:
    out: list[dict] = []
    if session.system:
        out.append({"role": "system", "content": session.system})
    for msg in prefix:
        if msg.role == Role.assistant:
            text_parts = [b.text for b in msg.blocks if b.kind == BlockKind.text and b.text]
            tu_blocks = [b for b in msg.blocks if b.kind == BlockKind.tool_use]
            tool_calls: list[dict] = []
            if flatten_tools:
                # Gemini's OpenAI endpoint rejects functionCall parts without a
                # thought_signature; render each call as text so the token anatomy
                # is preserved while the request stays acceptable.
                for b in tu_blocks:
                    text_parts.append(
                        f"[tool_call {b.tool_name or ''}({_json(b.tool_input or {})})]"
                    )
            else:
                tool_calls = [
                    {
                        "id": b.tool_use_id or "",
                        "type": "function",
                        "function": {
                            "name": b.tool_name or "",
                            "arguments": _json(b.tool_input or {}),
                        },
                    }
                    for b in tu_blocks
                ]
            m: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                m["tool_calls"] = tool_calls
            out.append(m)
        elif msg.role == Role.tool or any(
            b.kind == BlockKind.tool_result for b in msg.blocks
        ):
            for b in msg.blocks:
                if b.kind == BlockKind.tool_result:
                    if flatten_tools:
                        out.append({"role": "user", "content": f"[tool_result] {b.content or ''}"})
                    else:
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.tool_use_id or "",
                                "content": b.content or "",
                            }
                        )
                elif b.kind == BlockKind.text and b.text:
                    out.append({"role": "user", "content": b.text})
        else:  # user
            text = "\n".join(b.text for b in msg.blocks if b.kind == BlockKind.text and b.text)
            out.append({"role": "user", "content": text})
    return out


def _json(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def render_openai(
    session: Session, prefix: list[Message], *, max_tokens: int = 1, cache: bool = True,
    flatten_tools: bool = False,
) -> dict:
    body: dict = {
        "model": session.model,
        "max_tokens": max_tokens,
        "messages": _openai_messages(session, prefix, flatten_tools=flatten_tools),
    }
    if session.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            }
            for t in session.tools
        ]
    return body


def usage_from_openai(resp: dict) -> Usage:
    u = resp.get("usage", {}) or {}
    cached = (u.get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0) or 0
    prompt = u.get("prompt_tokens", 0) or 0
    return Usage(
        input_tokens=max(0, prompt - cached),
        output_tokens=u.get("completion_tokens", 0) or 0,
        cache_read=cached,
        cache_write=0,
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def render_request(
    session: Session, prefix: list[Message], *, flatten_tools: bool = False, **kwargs
) -> dict:
    if session.provider == Provider.openai:
        return render_openai(session, prefix, flatten_tools=flatten_tools, **kwargs)
    return render_anthropic(session, prefix, **kwargs)


def usage_from_response(provider: Provider, resp: dict) -> Usage:
    if provider == Provider.openai:
        return usage_from_openai(resp)
    return usage_from_anthropic(resp)
