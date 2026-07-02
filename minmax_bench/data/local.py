"""Converters for *local* agent session files into normalized :class:`Session`.

Format-only, no network. Three sources, each confirmed by inspecting real files
on disk (paths noted per function):

- **Claude Code** ``~/.claude/projects/**/*.jsonl`` -- one JSON *event* per line.
  User/assistant events carry ``message`` with Anthropic-style content blocks
  (``text`` / ``thinking`` / ``tool_use`` / ``tool_result``); assistant events
  carry ``message.usage``.
- **Codex CLI** ``~/.codex/sessions/**/rollout-*.jsonl`` -- one record per line
  with a ``type`` (``session_meta`` / ``turn_context`` / ``response_item`` /
  ``event_msg``) and a ``payload``. Response items are ``message`` /
  ``function_call`` / ``function_call_output`` / ``reasoning``; token usage
  arrives as an ``event_msg`` of ``type: token_count``.
- **OpenCode** -- file-based storage under
  ``~/.local/share/opencode/storage/{message,part}/...``. Best-effort: the exact
  on-disk JSON schema is not officially documented, so the loader is defensive
  and raises :class:`NotImplementedError` if it cannot locate messages.
"""

from __future__ import annotations

import json
from pathlib import Path

from minmax_bench.models import (
    Block,
    BlockKind,
    Message,
    Provider,
    Role,
    Session,
    Usage,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file, skipping blank/corrupt lines."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _stringify(content: object) -> str:
    """Flatten Anthropic-style content (str | list-of-blocks | dict) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                # tool_result content blocks look like {"type": "text", "text": ...}
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or json.dumps(content)
    return str(content)


# --------------------------------------------------------------------------- #
# Claude Code
# --------------------------------------------------------------------------- #
def _cc_blocks(content: object) -> list[Block]:
    """Convert a Claude Code ``message.content`` into normalized blocks.

    Anthropic block shapes (confirmed on disk):
      text        -> {"type": "text", "text": ...}
      thinking    -> {"type": "thinking", "thinking": ..., "signature": ...}
      tool_use    -> {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
      tool_result -> {"type": "tool_result", "tool_use_id": ..., "content": str|list,
                      "is_error": bool}
    """
    if isinstance(content, str):
        return [Block.text_block(content)] if content else []

    blocks: list[Block] = []
    if not isinstance(content, list):
        return blocks

    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            blocks.append(Block.text_block(b.get("text") or ""))
        elif btype == "thinking":
            blocks.append(
                Block(
                    kind=BlockKind.thinking,
                    text=b.get("thinking") or b.get("text") or "",
                    signature=b.get("signature"),
                )
            )
        elif btype == "tool_use":
            blocks.append(
                Block.tool_use(
                    tool_use_id=b.get("id") or "",
                    name=b.get("name") or "",
                    tool_input=b.get("input") if isinstance(b.get("input"), dict) else {},
                )
            )
        elif btype == "tool_result":
            blocks.append(
                Block.tool_result(
                    tool_use_id=b.get("tool_use_id") or "",
                    content=_stringify(b.get("content")),
                    is_error=bool(b.get("is_error", False)),
                )
            )
    return blocks


def _cc_usage(usage: object) -> Usage | None:
    """Map ``message.usage`` to :class:`Usage` (cache_creation -> cache_write)."""
    if not isinstance(usage, dict):
        return None
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_write=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read=int(usage.get("cache_read_input_tokens") or 0),
    )


def load_claude_code(path: str | Path) -> Session:
    """Load a single Claude Code ``.jsonl`` transcript into a :class:`Session`."""
    path = Path(path)
    records = _read_jsonl(path)

    session_id = ""
    model: str | None = None
    system: str | None = None
    meta: dict = {}
    messages: list[Message] = []

    for rec in records:
        rtype = rec.get("type")
        # Session id / provenance appear on every message event.
        session_id = session_id or rec.get("sessionId") or ""
        if not meta:
            meta = {
                k: rec.get(k)
                for k in ("cwd", "gitBranch", "version")
                if rec.get(k) is not None
            }

        if rtype not in ("user", "assistant", "system"):
            continue  # skip mode/permission/snapshot/last-prompt meta events

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or rtype
        model = model or msg.get("model")

        blocks = _cc_blocks(msg.get("content"))
        if not blocks:
            continue

        if role == "system":
            # First system message doubles as the session system prompt.
            if system is None:
                system = _stringify(msg.get("content"))
            continue

        if role == "assistant":
            new = Message(
                role=Role.assistant,
                blocks=blocks,
                recorded_usage=_cc_usage(msg.get("usage")),
            )
        else:
            # A user turn made entirely of tool_result blocks is really a tool
            # message in our normalized model (Anthropic folds tool -> user).
            is_tool = bool(blocks) and all(b.kind == BlockKind.tool_result for b in blocks)
            new = Message(role=Role.tool if is_tool else Role.user, blocks=blocks)

        # Coalesce consecutive same-role turns. Claude Code streams one logical
        # assistant turn across several jsonl lines (tool_use, then text), so a
        # tool_use and its follow-up land in separate records; without merging,
        # a request prefix (messages[:i]) can end on an orphaned assistant
        # tool_use whose result is in the next same-role record → upstream 400.
        if messages and messages[-1].role == new.role:
            messages[-1].blocks.extend(new.blocks)
            if new.recorded_usage is not None:
                messages[-1].recorded_usage = new.recorded_usage
        else:
            messages.append(new)

    return Session(
        id=session_id or path.stem,
        source="claude-code",
        provider=Provider.anthropic,
        model=model or "claude-sonnet-4-5",
        system=system,
        tools=[],  # Claude Code transcripts do not record tool schemas.
        messages=messages,
        meta={"path": str(path), **meta},
    )


# --------------------------------------------------------------------------- #
# Codex CLI
# --------------------------------------------------------------------------- #
def _codex_message_blocks(content: object) -> list[Block]:
    """Codex message content is a list of {"type": input_text|output_text, "text"}."""
    blocks: list[Block] = []
    if isinstance(content, str):
        return [Block.text_block(content)] if content else []
    if not isinstance(content, list):
        return blocks
    for part in content:
        if isinstance(part, dict) and part.get("text"):
            blocks.append(Block.text_block(part["text"]))
        elif isinstance(part, str) and part:
            blocks.append(Block.text_block(part))
    return blocks


def _codex_usage(info: object) -> Usage | None:
    """Map a Codex ``token_count.info`` to :class:`Usage`.

    ``last_token_usage`` has ``input_tokens`` (OpenAI: *total* prompt incl.
    cached), ``cached_input_tokens``, ``output_tokens`` (incl. reasoning), and
    ``reasoning_output_tokens``. We treat cached as cache_read and subtract it
    from input_tokens so ``input_tokens`` is the non-cached remainder (OpenAI has
    no separate cache-write tier).
    """
    if not isinstance(info, dict):
        return None
    usage = info.get("last_token_usage") or info.get("total_token_usage")
    if not isinstance(usage, dict):
        return None
    cached = int(usage.get("cached_input_tokens") or 0)
    total_in = int(usage.get("input_tokens") or 0)
    return Usage(
        input_tokens=max(0, total_in - cached),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read=cached,
        cache_write=0,
    )


def load_codex(path: str | Path) -> Session:
    """Load a single Codex ``rollout-*.jsonl`` session into a :class:`Session`."""
    path = Path(path)
    records = _read_jsonl(path)

    session_id = ""
    model: str | None = None
    provider_name: str | None = None
    system: str | None = None
    meta: dict = {}
    messages: list[Message] = []

    def _last_assistant() -> Message | None:
        for m in reversed(messages):
            if m.role == Role.assistant:
                return m
        return None

    for rec in records:
        rtype = rec.get("type")
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue

        if rtype == "session_meta":
            session_id = payload.get("id") or session_id
            provider_name = payload.get("model_provider") or provider_name
            base = payload.get("base_instructions")
            if system is None and isinstance(base, dict):
                system = base.get("text")
            meta = {
                k: payload.get(k)
                for k in ("cwd", "cli_version", "originator")
                if payload.get(k) is not None
            }
            continue

        if rtype == "turn_context":
            model = model or payload.get("model")
            continue

        if rtype == "event_msg" and payload.get("type") == "token_count":
            # Attach the per-turn usage to the assistant message that produced it.
            usage = _codex_usage(payload.get("info"))
            last = _last_assistant()
            if last is not None and usage is not None and last.recorded_usage is None:
                last.recorded_usage = usage
            continue

        if rtype != "response_item":
            continue  # ignore other event_msg (user_message echoes, task_*, etc.)

        ptype = payload.get("type")
        if ptype == "message":
            role = payload.get("role")
            blocks = _codex_message_blocks(payload.get("content"))
            if not blocks:
                continue
            if role == "assistant":
                messages.append(Message(role=Role.assistant, blocks=blocks))
            elif role == "developer":
                # Codex 'developer' turns are system-style instructions.
                if system is None:
                    system = _stringify(payload.get("content"))
                else:
                    messages.append(Message(role=Role.system, blocks=blocks))
            else:  # user
                messages.append(Message(role=Role.user, blocks=blocks))

        elif ptype == "function_call":
            args = payload.get("arguments")
            try:
                tool_input = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                tool_input = {"raw": args}
            if not isinstance(tool_input, dict):
                tool_input = {"value": tool_input}
            messages.append(
                Message(
                    role=Role.assistant,
                    blocks=[
                        Block.tool_use(
                            tool_use_id=payload.get("call_id") or "",
                            name=payload.get("name") or "",
                            tool_input=tool_input,
                        )
                    ],
                )
            )

        elif ptype == "function_call_output":
            messages.append(
                Message(
                    role=Role.tool,
                    blocks=[
                        Block.tool_result(
                            tool_use_id=payload.get("call_id") or "",
                            content=_stringify(payload.get("output")),
                        )
                    ],
                )
            )

        elif ptype == "reasoning":
            # Reasoning is usually encrypted; keep any summary text + signature.
            summary = _stringify(payload.get("summary")) or _stringify(payload.get("content"))
            messages.append(
                Message(
                    role=Role.assistant,
                    blocks=[
                        Block(
                            kind=BlockKind.thinking,
                            text=summary,
                            signature=payload.get("encrypted_content"),
                        )
                    ],
                )
            )

    provider = (
        Provider.openai
        if not provider_name or "anthropic" not in provider_name.lower()
        else Provider.anthropic
    )

    return Session(
        id=session_id or path.stem,
        source="codex",
        provider=provider,
        model=model or "gpt-5",
        system=system,
        tools=[],  # Codex tool schemas are not in the rollout file.
        messages=messages,
        meta={"path": str(path), "model_provider": provider_name, **meta},
    )


# --------------------------------------------------------------------------- #
# OpenCode (best-effort)
# --------------------------------------------------------------------------- #
def _opencode_part_blocks(part: dict) -> list[Block]:
    """Best-effort OpenCode part -> blocks.

    Observed/assumed part shapes (file-based storage, unofficial):
      text      -> {"type": "text", "text": ...}
      reasoning -> {"type": "reasoning", "text": ...}
      tool      -> {"type": "tool", "tool": <name>, "callID": ...,
                    "state": {"status", "input": {...}, "output": <str>}}
    """
    ptype = part.get("type")
    if ptype == "text" and part.get("text"):
        return [Block.text_block(part["text"])]
    if ptype == "reasoning" and part.get("text"):
        return [Block(kind=BlockKind.thinking, text=part["text"])]
    if ptype == "tool":
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        call_id = part.get("callID") or part.get("id") or ""
        name = part.get("tool") or ""
        out_blocks: list[Block] = [
            Block.tool_use(
                tool_use_id=call_id,
                name=name,
                tool_input=state.get("input") if isinstance(state.get("input"), dict) else {},
            )
        ]
        if state.get("output") is not None:
            out_blocks.append(
                Block.tool_result(
                    tool_use_id=call_id,
                    content=_stringify(state.get("output")),
                    is_error=state.get("status") == "error",
                )
            )
        return out_blocks
    return []


def _opencode_usage(msg: dict) -> Usage | None:
    """OpenCode assistant message ``tokens`` block, if present."""
    tokens = msg.get("tokens")
    if not isinstance(tokens, dict):
        return None
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    return Usage(
        input_tokens=int(tokens.get("input") or 0),
        output_tokens=int(tokens.get("output") or 0),
        cache_read=int(cache.get("read") or 0),
        cache_write=int(cache.get("write") or 0),
    )


def load_opencode(path: str | Path) -> Session:
    """Best-effort load of an OpenCode session.

    ``path`` should be the session's message directory, i.e.
    ``~/.local/share/opencode/storage/message/{sessionID}`` (containing
    ``msg_*.json`` files). Part files are looked up next door under
    ``storage/part/{sessionID}/{messageID}/prt_*.json``; if parts are instead
    embedded on the message as a ``parts`` list, that is used as a fallback.

    The OpenCode on-disk schema is not officially documented; this reflects the
    community-confirmed layout and degrades gracefully. Raises
    :class:`NotImplementedError` if no messages can be located.
    """
    path = Path(path)
    if not path.exists():
        raise NotImplementedError(
            f"OpenCode session path not found: {path}. Expected a message directory "
            "like ~/.local/share/opencode/storage/message/{sessionID}."
        )

    msg_files = sorted(path.glob("msg_*.json")) if path.is_dir() else [path]
    if not msg_files:
        raise NotImplementedError(
            f"No OpenCode message files (msg_*.json) under {path}. The OpenCode "
            "storage format could not be confirmed for this path."
        )

    session_id = path.name if path.is_dir() else path.stem
    # storage/message/{sid} -> storage/part/{sid}
    part_root = path.parent.parent / "part" / session_id if path.is_dir() else None

    model: str | None = None
    messages: list[Message] = []

    for mf in msg_files:
        try:
            msg = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        model = model or msg.get("modelID")
        message_id = msg.get("id") or mf.stem

        # Gather parts: sibling part dir first, else embedded 'parts'.
        parts: list[dict] = []
        if part_root is not None:
            pdir = part_root / message_id
            if pdir.is_dir():
                for pf in sorted(pdir.glob("prt_*.json")):
                    try:
                        p = json.loads(pf.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if isinstance(p, dict):
                        parts.append(p)
        if not parts and isinstance(msg.get("parts"), list):
            parts = [p for p in msg["parts"] if isinstance(p, dict)]

        blocks: list[Block] = []
        for p in parts:
            blocks.extend(_opencode_part_blocks(p))
        if not blocks:
            continue

        if role == "assistant":
            messages.append(
                Message(role=Role.assistant, blocks=blocks, recorded_usage=_opencode_usage(msg))
            )
        else:
            is_tool = all(b.kind == BlockKind.tool_result for b in blocks)
            messages.append(Message(role=Role.tool if is_tool else Role.user, blocks=blocks))

    if not messages:
        raise NotImplementedError(
            f"Could not extract any messages from OpenCode session at {path}; "
            "the storage format may differ from the confirmed layout."
        )

    provider = Provider.openai if (model and "gpt" in model.lower()) else Provider.anthropic
    return Session(
        id=session_id,
        source="opencode",
        provider=provider,
        model=model or "claude-sonnet-4-5",
        system=None,
        tools=[],  # OpenCode tool schemas are not stored per-session.
        messages=messages,
        meta={"path": str(path)},
    )


__all__ = ["load_claude_code", "load_codex", "load_opencode"]
