"""A tiny, redacted, fully offline sample session.

Lets ``minmax-bench run --dataset sample`` work with no dataset download and no
network, so the pipeline (simulate -> execute -> report) can be smoke-tested. The
tool outputs are deliberately verbose/repetitive so compression strategies have
something to bite on.
"""

from __future__ import annotations

from ..models import Block, Message, Role, Session, ToolDef, Usage

_TOOLS = [
    ToolDef(
        name="Grep",
        description="Search file contents with a regex.",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    ),
    ToolDef(
        name="Read",
        description="Read a file from disk.",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    ),
    ToolDef(
        name="Edit",
        description="Replace a string in a file.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    ),
]

_GREP_OUT = "\n".join(
    f"src/app/handlers.py:{i}:    logger.debug('processing item %s', item_id)" for i in range(40, 140)
)
_READ_OUT = "\n".join(
    [f"{i}\tdef handle_request(req):" if i == 12 else f"{i}\t    # boilerplate line {i}" for i in range(1, 90)]
)


def sample_sessions() -> list[Session]:
    msgs = [
        Message(role=Role.user, blocks=[Block.text_block("The timeout handler never fires. Find and fix it.")]),
        Message(
            role=Role.assistant,
            blocks=[
                Block.text_block("Let me search for the timeout handling."),
                Block.tool_use("t1", "Grep", {"pattern": "timeout", "path": "src/app"}),
            ],
            recorded_usage=Usage(input_tokens=1800, output_tokens=40, cache_read=0, cache_write=1800),
        ),
        Message(role=Role.tool, blocks=[Block.tool_result("t1", _GREP_OUT)]),
        Message(
            role=Role.assistant,
            blocks=[
                Block.text_block("Let me read the handler."),
                Block.tool_use("t2", "Read", {"file_path": "src/app/handlers.py"}),
            ],
            recorded_usage=Usage(input_tokens=350, output_tokens=30, cache_read=1800, cache_write=1600),
        ),
        Message(role=Role.tool, blocks=[Block.tool_result("t2", _READ_OUT)]),
        Message(
            role=Role.assistant,
            blocks=[
                Block.text_block("The default timeout is 0, disabling it. Fixing."),
                Block.tool_use(
                    "t3",
                    "Edit",
                    {
                        "file_path": "src/app/handlers.py",
                        "old_string": "timeout = 0",
                        "new_string": "timeout = 30",
                    },
                ),
            ],
            recorded_usage=Usage(input_tokens=420, output_tokens=60, cache_read=3400, cache_write=1400),
        ),
        Message(role=Role.tool, blocks=[Block.tool_result("t3", "Edited src/app/handlers.py (1 replacement).")]),
        Message(
            role=Role.assistant,
            blocks=[Block.text_block("Fixed: the default timeout was 0. Set it to 30s so the handler fires.")],
            recorded_usage=Usage(input_tokens=180, output_tokens=45, cache_read=4800, cache_write=200),
        ),
    ]
    session = Session(
        id="sample:timeout-fix:0",
        source="sample",
        model="claude-sonnet-4-5",
        system="You are a coding agent. Use the tools to inspect and edit the repository.",
        tools=_TOOLS,
        messages=msgs,
        meta={"note": "synthetic redacted sample"},
    )
    return [session]
