"""Loader for the gated HuggingFace dataset ``SALT-NLP/SWE-chat``.

Confirmed against the dataset card / data-files viewer
(https://huggingface.co/datasets/SALT-NLP/SWE-chat). The ``conversations`` config
has one row per *turn* with (relevant) columns:

- ``session_id``          (str)  grouping key
- ``turn_number``         (int)  global ordering across all rows of a session
- ``role``                (str)  ``user`` | ``assistant`` | ``tool_use`` |
                                 ``tool_result`` | ``metadata``
- ``content``             (str)  prompt / response / thinking text (tool results
                                 are capped at ~10KB in the source)
- ``model``               (str)  model name (NULL for user/tool_result/metadata)
- ``agent``               (str)  agent name (e.g. Claude Code / Gemini CLI)
- ``input_tokens`` / ``output_tokens``                       (int)
- ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` (int)
- ``tool_name``           (str)  Read/Write/Edit/Bash/Grep/...
- ``tool_call_id``        (str)  links a ``tool_use`` row to its ``tool_result``
- ``tool_input_json``     (str)  full tool parameters, JSON-encoded

There are also ``sessions`` / ``session_logs`` tables and raw transcripts under
``transcripts/{session_id}``; we only need ``conversations`` here.

Every field is read with ``.get()`` so a schema drift (renamed/absent column)
degrades gracefully rather than raising. Any column whose exact name we are less
sure of is noted inline.
"""

from __future__ import annotations

import json

from minmax_bench.models import (
    Block,
    Message,
    Provider,
    Role,
    Session,
    Usage,
)

# Fallbacks when the row does not record a model/agent.
_DEFAULT_PROVIDER = Provider.anthropic
_DEFAULT_MODEL = "claude-sonnet-4-5"


def _to_int(value: object) -> int:
    """Coerce a possibly-None / possibly-str token count to a non-negative int."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _detect_provider(model: str | None, agent: str | None) -> Provider:
    """Anthropic unless the model/agent clearly names an OpenAI model."""
    hint = f"{model or ''} {agent or ''}".lower()
    if any(k in hint for k in ("gpt", "openai", "o1", "o3", "o4", "codex")):
        return Provider.openai
    return _DEFAULT_PROVIDER


def _parse_tool_input(raw: object) -> dict:
    """``tool_input_json`` is a JSON string; tolerate dicts / bad JSON."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": raw}
    return {}


def _row_to_message(row: dict) -> Message | None:
    """Map one ``conversations`` row to a normalized :class:`Message`.

    Returns ``None`` for rows we intentionally drop (e.g. ``metadata``).
    """
    role = (row.get("role") or "").strip().lower()
    content = row.get("content")
    text = content if isinstance(content, str) else ("" if content is None else str(content))

    if role == "user":
        return Message(role=Role.user, blocks=[Block.text_block(text)])

    if role == "assistant":
        blocks: list[Block] = []
        if text:
            blocks.append(Block.text_block(text))
        usage = Usage(
            input_tokens=_to_int(row.get("input_tokens")),
            output_tokens=_to_int(row.get("output_tokens")),
            # SWE-chat: cache_creation -> our cache_write, cache_read -> cache_read.
            cache_write=_to_int(row.get("cache_creation_input_tokens")),
            cache_read=_to_int(row.get("cache_read_input_tokens")),
        )
        return Message(role=Role.assistant, blocks=blocks, recorded_usage=usage)

    if role == "tool_use":
        # A standalone tool-call turn is emitted by the assistant.
        block = Block.tool_use(
            tool_use_id=row.get("tool_call_id") or "",
            name=row.get("tool_name") or "",
            tool_input=_parse_tool_input(row.get("tool_input_json")),
        )
        return Message(role=Role.assistant, blocks=[block])

    if role == "tool_result":
        block = Block.tool_result(
            tool_use_id=row.get("tool_call_id") or "",
            content=text,
            # No explicit error flag in the schema; assume success.
            is_error=bool(row.get("is_error", False)),
        )
        return Message(role=Role.tool, blocks=[block])

    # metadata / unknown roles carry no conversational payload.
    return None


def _order_key(row: dict) -> tuple[int, int]:
    """Prefer ``turn_number`` for ordering; fall back to conversation index."""
    for key in ("turn_number", "conversation_turn_number"):
        val = row.get(key)
        if val is not None:
            try:
                return (0, int(val))
            except (TypeError, ValueError):
                pass
    return (1, 0)  # unordered rows sort after ordered ones, stable within.


def load_swe_chat(
    limit: int | None = None,
    hf_token: str | None = None,
    config: str = "conversations",
    split: str = "train",
    streaming: bool = True,
) -> list[Session]:
    """Load ``SALT-NLP/SWE-chat`` and normalize it into :class:`Session` objects.

    One :class:`Session` is built per ``session_id``; rows are grouped and ordered
    by turn index. ``limit`` caps the number of *sessions* returned (not rows).
    ``hf_token`` is passed through for the gated repo (falls back to the ambient
    HuggingFace login / ``HF_TOKEN`` env when ``None``).

    ``streaming`` (default True) iterates the parquet shards lazily so a small
    ``limit`` doesn't download the whole multi-GB ``conversations`` config; it
    relies on rows of a session being contiguous in the stream (they are, in
    turn order). Pass ``streaming=False`` for a full local materialization.
    """
    # HF's download progress bars create a multiprocessing.RLock (a POSIX
    # semaphore) that the resource tracker reports as "leaked" at shutdown.
    # Disabling them before importing datasets avoids that benign warning.
    import contextlib
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from datasets import load_dataset  # lazy: optional extra
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "The 'datasets' package is required to load SWE-chat. "
            "Install the optional HF extra with: uv sync --extra hf"
        ) from exc
    with contextlib.suppress(Exception):
        from datasets import disable_progress_bars

        disable_progress_bars()

    dataset = load_dataset(
        "SALT-NLP/SWE-chat",
        config,
        split=split,
        token=hf_token,
        streaming=streaming,
    )

    # Group rows by session, preserving arrival order for a stable tie-break.
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in dataset:
        row = dict(row)
        sid = row.get("session_id") or row.get("id") or "unknown"
        if sid not in grouped:
            grouped[sid] = []
            order.append(sid)
            if limit is not None and len(order) > limit:
                # We've now started a new session beyond the limit; drop it and
                # stop scanning further rows.
                grouped.pop(sid)
                order.pop()
                break
        if sid in grouped:
            grouped[sid].append(row)

    sessions: list[Session] = []
    for sid in order:
        rows = sorted(grouped[sid], key=_order_key)

        # Session-level model/agent: first non-empty across the turns.
        model = next((r.get("model") for r in rows if r.get("model")), None)
        agent = next((r.get("agent") for r in rows if r.get("agent")), None)
        provider = _detect_provider(model, agent)

        messages = [m for m in (_row_to_message(r) for r in rows) if m is not None]

        sessions.append(
            Session(
                id=str(sid),
                source="swe-chat",
                provider=provider,
                model=model or _DEFAULT_MODEL,
                system=None,  # not present in the conversations table
                tools=[],  # tool schemas are not recorded per-conversation
                messages=messages,
                meta={
                    "agent": agent,
                    "repo_id": next((r.get("repo_id") for r in rows if r.get("repo_id")), None),
                    "turn_count": len(rows),
                },
            )
        )

    return sessions
