"""Harness simulator.

A real agent harness calls the model once, gets an assistant turn (possibly with
tool_use blocks), executes the tools, appends the tool_results, and calls again —
growing the message chain each round. This module replays a recorded
:class:`Session` into exactly that sequence of model calls, without a live model:
each *assistant* message in the transcript marks one call, whose request prefix is
every message that preceded it and whose expected output is the assistant turn
itself.

Because we replay deterministically from the recorded transcript, strategies and
executors observe the identical chain a real harness would have sent — the only
thing that varies is how each strategy rewrites/compresses that chain.
"""

from __future__ import annotations

import uuid

from ..models import Message, RequestPoint, Role, Session


def with_test_run(session: Session, test_uuid: str | None = None) -> Session:
    """Return a copy of ``session`` stamped with a fresh test-run id (rotates on
    every call unless one is passed in).

    The id is (a) prepended to the system prompt — so the request content is
    unique per run and the proxy's content-keyed chain matching can't reuse a
    prior run's compacted chain — and (b) exposed as ``test_uuid``, which the
    proxy executor sends as whichever session header a given strategy declares
    (``ProxyConfig.session_id_header``, e.g. condense's ``x-condense-session-id``)
    to bust that proxy's own session-keyed chain state. This guarantees each fresh
    rerun of a session is independent of previous runs. The marker is fixed
    length, so baseline token counts are unaffected in value.
    """
    tid = test_uuid or str(uuid.uuid4())
    marker = f"[minmax-bench test-run: {tid}]"
    system = f"{marker}\n{session.system}" if session.system else marker
    return session.model_copy(update={"system": system, "test_uuid": tid})


class HarnessSimulator:
    """Turns a :class:`Session` into an ordered list of :class:`RequestPoint`."""

    def __init__(
        self,
        *,
        min_prefix_messages: int = 1,
        skip_empty_assistant: bool = True,
        coalesce: bool = True,
    ):
        self.min_prefix_messages = min_prefix_messages
        self.skip_empty_assistant = skip_empty_assistant
        self.coalesce = coalesce

    def points(self, session: Session) -> list[RequestPoint]:
        messages = _coalesce(session.messages) if self.coalesce else session.messages
        points: list[RequestPoint] = []
        for i, msg in enumerate(messages):
            if msg.role != Role.assistant:
                continue
            prefix = messages[:i]
            if len(prefix) < self.min_prefix_messages:
                continue
            if self.skip_empty_assistant and not _has_content(msg):
                continue
            points.append(
                RequestPoint(
                    index=len(points),
                    session_id=session.id,
                    prefix=prefix,
                    expected_output=msg,
                    recorded_usage=msg.recorded_usage,
                )
            )
        return points


def _has_content(msg: Message) -> bool:
    return any(
        (b.text and b.text.strip()) or b.tool_name or (b.content and b.content.strip())
        for b in msg.blocks
    )


def _coalesce(messages: list[Message]) -> list[Message]:
    """Merge consecutive same-role messages into one.

    Some sources (e.g. SWE-chat) split a single assistant turn into separate
    rows per block (text, then tool_use); a real harness sends those as one
    model call. Merging consecutive same-role messages restores one message per
    turn. For a merged assistant run we keep the recorded usage of the row that
    looks like the real call (largest total input), so baseline costing stays
    right; other roles just concatenate blocks.
    """
    out: list[Message] = []
    for msg in messages:
        if out and out[-1].role == msg.role:
            prev = out[-1]
            prev.blocks.extend(msg.blocks)
            prev.recorded_usage = _better_usage(prev.recorded_usage, msg.recorded_usage)
        else:
            out.append(msg.model_copy(deep=True))
    return out


def _better_usage(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a.total_input >= b.total_input else b


def simulate(session: Session, **kwargs) -> list[RequestPoint]:
    """Convenience wrapper: ``simulate(session)`` -> list of request points."""
    return HarnessSimulator(**kwargs).points(session)
