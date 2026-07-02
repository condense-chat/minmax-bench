"""Provider-neutral normalized data model for agent sessions.

A *session* is a full recorded agent run (system prompt + tools + an ordered list
of messages, where messages carry typed content blocks: text, tool_use,
tool_result, thinking). Loaders normalize any source (SWE-chat, Claude Code,
Codex, OpenCode) into this shape.

A *harness simulator* replays a session into a growing sequence of
:class:`RequestPoint` objects — one per model call a real harness would make —
which strategies then compress and executors then price.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Provider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    # Normalized tool-result carrier. The Anthropic renderer folds these into a
    # user message of tool_result blocks; OpenAI renders them as role="tool".
    tool = "tool"


class BlockKind(str, Enum):
    text = "text"
    tool_use = "tool_use"
    tool_result = "tool_result"
    thinking = "thinking"


class Block(BaseModel):
    """A single typed content block within a message."""

    kind: BlockKind
    # text / thinking
    text: str | None = None
    # tool_use
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    # tool_result
    content: str | None = None
    is_error: bool = False
    # thinking signature (opaque, preserved verbatim for Anthropic)
    signature: str | None = None

    @classmethod
    def text_block(cls, text: str) -> Block:
        return cls(kind=BlockKind.text, text=text)

    @classmethod
    def tool_use(cls, tool_use_id: str, name: str, tool_input: dict) -> Block:
        return cls(
            kind=BlockKind.tool_use,
            tool_use_id=tool_use_id,
            tool_name=name,
            tool_input=tool_input,
        )

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error: bool = False) -> Block:
        return cls(
            kind=BlockKind.tool_result,
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )


class Usage(BaseModel):
    """Token usage for a single model call, split for cache-aware costing.

    ``input_tokens`` is the *non-cached* prompt tokens billed at the full input
    rate. ``cache_read`` / ``cache_write`` are the cached-prefix tokens billed at
    the (much cheaper) cache-read and (slightly dearer) cache-write rates.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total_input(self) -> int:
        """All prompt-side tokens regardless of cache tier."""
        return self.input_tokens + self.cache_read + self.cache_write


class ToolDef(BaseModel):
    """A tool the agent had available (its schema counts toward prompt tokens)."""

    name: str
    description: str = ""
    input_schema: dict = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    blocks: list[Block] = Field(default_factory=list)
    # Usage the source recorded for the *assistant* call that produced this
    # message, if any (SWE-chat carries this; most trajectory dumps do not).
    recorded_usage: Usage | None = None


class Session(BaseModel):
    """A normalized, provider-neutral agent session."""

    id: str
    source: str  # swe-chat | claude-code | codex | opencode | sample | ...
    provider: Provider = Provider.anthropic
    model: str = "claude-sonnet-4-5"
    system: str | None = None
    tools: list[ToolDef] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    # Free-form provenance (repo, instance_id, original agent, etc.).
    meta: dict = Field(default_factory=dict)
    # Per-run cache-busting id (see minmax_bench.harness.with_test_run): stamped into
    # the system prompt, and also sent as a proxy-specific session header (e.g.
    # condense's x-condense-session-id, see ProxyConfig.session_id_header) for
    # proxies that key their own chain state by session — so a fresh rerun is not
    # contaminated by a prior run's cached/compacted chain state on that proxy.
    test_uuid: str | None = None


class RequestPoint(BaseModel):
    """One reconstructed model call a real harness would issue.

    ``prefix`` is every message sent to the model at this point (system/tools are
    carried on the owning :class:`Session`). ``expected_output`` is the assistant
    message the source recorded as the response — we replay it deterministically
    to build the next request rather than trusting a live 1-token generation.
    """

    index: int
    session_id: str
    prefix: list[Message]
    expected_output: Message
    # Usage the source recorded for this exact call, if available. Lets us use
    # true baseline cost without re-billing when a session already carries usage.
    recorded_usage: Usage | None = None
