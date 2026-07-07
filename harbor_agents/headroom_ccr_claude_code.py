"""Custom Harbor agent: Claude Code + Headroom CCR, wired fully self-contained.

Headroom's Compress-Cache-Retrieve (CCR) needs two cooperating pieces: the `headroom proxy`
(compresses stale tool outputs into hash-marker summaries) AND an MCP server exposing
`headroom_retrieve`, which the agent CALLS to fetch the full content back. The proxy-only arm
never engages CCR because plain Claude Code has no executor for that retrieve tool.

This agent supplies the missing executor the reproducible way — no `headroom mcp install`, no
mutation of the user's machine:
  1. install `headroom-ai` INTO the task container (per-container, ephemeral);
  2. register headroom's stdio MCP server in the container's USER-scoped ~/.claude.json
     (loaded without a trust dialog), pointing its retrieve backend at the host proxy.
Combined with ANTHROPIC_BASE_URL routed through `headroom proxy`, Claude Code sees compressed
markers and calls mcp__headroom__headroom_retrieve on demand — the real CCR loop. Permission mode
defaults to bypassPermissions, so the MCP tool is callable without prompts.

Run via:  -a harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode
with:      ANTHROPIC_BASE_URL=http://host.docker.internal:<port>
           --ae TMB_HEADROOM_PROXY_URL=http://host.docker.internal:<port>   (MCP retrieve backend)
"""
from __future__ import annotations

import json
import shlex

from harbor.agents.installed.claude_code import ClaudeCode


class HeadroomCcrClaudeCode(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "headroom-ccr-claude-code"

    async def install(self, environment) -> None:
        # Base installs Claude Code (node/npm). Then we add python3 + headroom-ai.
        # Harbor's agent-setup phase has a hard timeout (360s by default) and a cold
        # apt-get update alone can eat most of it — skip anything already present and
        # pair this agent with --agent-timeout-multiplier (generate.py raises it
        # automatically for the headroom-ccr arm).
        await super().install(environment)
        await self.exec_as_root(
            environment,
            command=(
                "if command -v python3 >/dev/null 2>&1 "
                "&& python3 -m pip --version >/dev/null 2>&1; "
                "then echo 'python3+pip present, skipping'; "
                "elif command -v apk >/dev/null 2>&1; then apk add --no-cache python3 py3-pip; "
                "elif command -v apt-get >/dev/null 2>&1; then apt-get update && "
                "  apt-get install -y --no-install-recommends python3 python3-pip; "
                "elif command -v yum >/dev/null 2>&1; then yum install -y python3 python3-pip; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # headroom-ai does NOT pull the `mcp` SDK, but `headroom mcp serve` needs it — install both.
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                "python3 -m pip install --user --quiet --break-system-packages headroom-ai mcp "
                "|| python3 -m pip install --user --quiet headroom-ai mcp; "
                "headroom --version"
            ),
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Write headroom's stdio MCP server into the container's user-scoped config.

        Spawned through a shell that puts ~/.local/bin (pip --user) on PATH so `headroom`
        resolves. Its --proxy-url points back at the same host proxy that ANTHROPIC_BASE_URL uses.

        Servers configured on the base agent (self.mcp_servers) are preserved — the base
        method serializes them into the same file, so clobbering it would silently strip
        tools the vanilla claude-code arm has and make the arms incomparable.
        """
        proxy = self._get_env("TMB_HEADROOM_PROXY_URL") or "http://host.docker.internal:8787"
        servers: dict = {}
        for server in self.mcp_servers or []:  # mirror the base method's serialization
            if server.transport == "stdio":
                servers[server.name] = {"type": "stdio", "command": server.command,
                                        "args": server.args}
            else:
                transport = "http" if server.transport == "streamable-http" else server.transport
                servers[server.name] = {"type": transport, "url": server.url}
        servers["headroom"] = {
            "type": "stdio",
            "command": "sh",
            "args": [
                "-lc",
                f'export PATH="$HOME/.local/bin:$PATH"; '
                f"exec headroom mcp serve --proxy-url {proxy}",
            ],
        }
        cfg = {"mcpServers": servers}
        return f"echo {shlex.quote(json.dumps(cfg))} > $CLAUDE_CONFIG_DIR/.claude.json"
