"""Custom Harbor agent: Claude Code + RTK, wired fully self-contained.

RTK (https://github.com/rtk-ai/rtk) is an *observation-transform* method — a third class,
distinct from both the history transforms (condense, headroom) and the generation policy
(caveman). It never touches the conversation and never changes how the agent writes. A
PreToolUse hook rewrites Bash commands to their rtk equivalents (`git status` ->
`rtk git status`), and rtk runs the command and filters its output before the result reaches
the model. So it shrinks what the agent READS BACK — the one part of the prefix caveman
explicitly cannot touch, and the part that dominates a coding session.

Two consequences the rest of the bench has to know about:

  - Like caveman it is NOT a proxy: ANTHROPIC_BASE_URL stays default, so it reads against
    plain `vanilla` rather than `vanilla-proxy` (no ~8-9k wiring confound to subtract).
  - Unlike every other arm it rewrites the RECORDED ACTIONS. `rtk hook claude` returns
    `updatedInput.command`, so the transcript stores `rtk git status`, not `git status`.
    Every command-shaped metric therefore normalizes through `engine.unrtk` — without it
    `report.rework_count` scores the arm a flawless ZERO rework on identical behaviour,
    because its read-only pattern is ^-anchored and rtk renames cat/head/tail to `rtk read`.

This agent installs RTK the reproducible way — a PINNED release binary, no `install.sh` piped
from the network at run time, no mutation of the user's machine:
  1. download the pinned release asset for the container's arch (install phase);
  2. drop the binary on PATH and wire the PreToolUse hook into $CLAUDE_CONFIG_DIR/settings.json.

The hook command is the NATIVE `rtk hook claude` rather than upstream's hooks/claude/
rtk-rewrite.sh: the shell script shells out to `jq`, which is not guaranteed in the image, and
it degrades to a silent no-op when jq is missing ("WARNING ... Hook cannot rewrite commands"
on stderr, exit 0) — i.e. exactly the silent-inactivity failure that makes an arm measure
nothing while scoring a clean pass. The binary subcommand reads the same PreToolUse JSON from
stdin and has no external dependency.

Run via:  -a harbor_agents.rtk_claude_code:RtkClaudeCode
"""
from __future__ import annotations

import json
import os
import shlex

from harbor.agents.installed.claude_code import ClaudeCode

# Pinned so a run is reproducible and an upstream retag cannot silently change the
# intervention mid-experiment. Bump deliberately, and re-run every arm when you do: an rtk
# version change is a change to the METHOD under test, not a dependency update. A bump can
# also add a command rename (cat -> `rtk read` is the only one at this pin), which would skew
# rework and fidelity — tests/test_quality.py re-derives that table from the binary and fails.
RTK_REF = "v0.44.0"
RTK_SHA = "3fc407027589acef9579df5b2ad10b0f3042e030"
RTK_RELEASE = f"https://github.com/rtk-ai/rtk/releases/download/{RTK_REF}"

# Release assets, by `uname -m`. musl for x86_64 so the binary is static and cannot miss a
# glibc in a slim image; upstream ships aarch64 as gnu only.
RTK_ASSETS = {
    "x86_64": "rtk-x86_64-unknown-linux-musl.tar.gz",
    "aarch64": "rtk-aarch64-unknown-linux-gnu.tar.gz",
    "arm64": "rtk-aarch64-unknown-linux-gnu.tar.gz",
}

# Where the binary lands. $CLAUDE_CONFIG_DIR does not exist at install time (ClaudeCode.run()
# creates and exports it), so the binary goes somewhere stable and the hook path is written
# absolute during setup.
RTK_BIN_DIR = "$HOME/.local/bin"
RTK_BIN = f"{RTK_BIN_DIR}/rtk"


def _awareness_text() -> str:
    """rtk's model-facing instructions, frozen at the pin (data/rtk/awareness.md).

    This is what `rtk init -g` embeds into CLAUDE.md. Read from disk rather than inlined so a
    pin bump is a file swap, and so the exact bytes are reviewable next to data/rtk/PIN.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "data", "rtk", "awareness.md"), encoding="utf-8") as fh:
        text = fh.read()
    if "rtk" not in text.lower():
        raise RuntimeError("data/rtk/awareness.md looks empty/stale — regenerate it")
    return text


class RtkClaudeCode(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "rtk-claude-code"

    async def install(self, environment) -> None:
        # Base installs Claude Code, and with it curl — all rtk needs (a single static binary,
        # no toolchain, no npm, no python).
        await super().install(environment)

        # Fetch the pinned release asset for this container's architecture. `--fail` so a 404
        # from a deleted/retagged release is a hard error rather than an HTML error page
        # silently extracted as a "binary".
        cases = " ".join(
            f'{arch}) asset={shlex.quote(name)} ;;' for arch, name in RTK_ASSETS.items()
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f'arch=$(uname -m); case "$arch" in {cases} '
                f'*) echo "rtk: no {RTK_REF} release asset for arch $arch" >&2; exit 1 ;; esac; '
                f"mkdir -p {RTK_BIN_DIR}; tmp=$(mktemp -d); "
                f'curl -fsSL "{RTK_RELEASE}/$asset" | tar xz -C "$tmp"; '
                # the tarball layout is a bare `rtk` at the root, but find it either way so a
                # repackaging upstream fails loudly here instead of leaving a missing binary
                f'src=$(find "$tmp" -type f -name rtk -perm -u+x | head -1); '
                f'test -n "$src" || {{ echo "rtk: no rtk binary inside $asset at {RTK_REF}" '
                f'>&2; exit 1; }}; '
                f'install -m 0755 "$src" {RTK_BIN}; rm -rf "$tmp"; '
                # verify it runs AND is the version we pinned — a wrong binary here would
                # rewrite commands differently and silently change the method under test
                f'v=$({RTK_BIN} --version); '
                f'case "$v" in *"{RTK_REF.lstrip("v")}"*) ;; *) '
                f'echo "rtk: installed $v, expected {RTK_REF}" >&2; exit 1 ;; esac; '
                f'echo "rtk {RTK_REF} installed ($v)"'
            ),
        )

    def _build_register_skills_command(self) -> str | None:
        """Wire the PreToolUse hook, then smoke-test that rtk actually rewrites.

        Appended to ClaudeCode.run()'s setup_command, the first point at which
        $CLAUDE_CONFIG_DIR exists and is exported. The base class's own skills command is
        preserved — dropping it would strip skills the vanilla arm has and make the arms
        incomparable.
        """
        base = super()._build_register_skills_command()
        settings = self._settings_command()

        # `rtk init -g` installs the hook AND embeds hooks/claude/rtk-awareness.md into
        # CLAUDE.md, so a faithful install includes it. It is NOT a skill in caveman's sense —
        # ~10 lines advertising four ANALYTICS commands (gain/discover/proxy) plus "everything
        # else is rewritten automatically". It never redirects the model away from the native
        # Read tool, so it does not lift rtk's Bash-only ceiling. Installed anyway for
        # faithfulness, and because it is a real (small) context cost the arm should carry:
        # omitting it would quietly measure a lighter rtk than a user actually runs.
        # Frozen at data/rtk/awareness.md rather than re-fetched: this agent installs a release
        # BINARY, not the repo, so the file is captured on the host at the same pin.
        awareness = (
            f"printf '%s' {shlex.quote(_awareness_text())} > $CLAUDE_CONFIG_DIR/CLAUDE.md"
        )

        # Smoke test: a hook that never fires makes this arm identical to vanilla while
        # scoring a clean "trajectory preserved". Assert on the real rewrite engine (`rtk
        # rewrite`, the single source of truth the hook delegates to) AND on the hook
        # subcommand's own JSON, so either half breaking is caught here rather than becoming
        # a quiet null result. `rtk rewrite` exits 0 (auto-allow) or 3 (ask) on a hit and 1
        # when it has no equivalent — 1 for `git status` would mean the registry is gone.
        # Smoke test: a hook that never fires makes this arm identical to vanilla while
        # scoring a clean "trajectory preserved". Assert on the real rewrite engine (`rtk
        # rewrite`, the source of truth the hook delegates to) AND on the hook subcommand's
        # own JSON, so either half breaking is caught here rather than becoming a null result.
        #
        # NO top-level `||` anywhere in here. `&&` and `||` are equal precedence and
        # left-associative in sh, so `a && b && c || d` parses as `((a && b) && c) || d` —
        # an earlier failure in the chain this string is joined into would fall through to
        # THIS block's error branch and be misreported as an rtk problem. That is not
        # hypothetical: it is how a missing `node` got blamed on rtk's rewrite engine.
        # Exit codes are absorbed with `|| true` inside the $( ), and every assertion is a
        # `case` on the captured STRING — which is the thing worth asserting anyway.
        # Braces make the whole block ONE unit in the `&&` chain. Without them only the first
        # command is guarded — everything after the first `;` runs even when an earlier step
        # failed, which is how a failed settings write still reached these assertions and got
        # reported as "rtk's rewrite engine is broken".
        verify = (
            "{ "
            f'out=$({RTK_BIN} rewrite "git status" 2>/dev/null || true); '
            'case "$out" in "rtk git status") ;; *) '
            'echo "rtk: rewrite engine did not rewrite a known command (got: [$out]) — the '
            'arm would run WITHOUT the intervention" >&2; exit 1 ;; esac; '
            'hook=$(echo \'{"tool_name":"Bash","tool_input":{"command":"git status"}}\' '
            f'| {RTK_BIN} hook claude 2>/dev/null || true); '
            'case "$hook" in *\'"command":"rtk git status"\'*) ;; *) '
            'echo "rtk: hook claude did not rewrite the command (got: [$hook])" >&2; '
            'exit 1 ;; esac; '
            # prove the hook is actually WIRED, not merely functional — the settings write is
            # the step that silently did nothing when node was missing
            'grep -q "hook claude" $CLAUDE_CONFIG_DIR/settings.json || '
            '{ echo "rtk: settings.json has no PreToolUse hook — it was never wired" >&2; '
            'exit 1; }; '
            f'echo "rtk {RTK_REF} active (PreToolUse hook wired)"; '
            "}"
        )

        parts = [p for p in (base, settings, awareness, verify) if p]
        return " && ".join(parts)

    def _settings_command(self) -> str:
        """Shell command that writes the PreToolUse hook into settings.json.

        Built on the HOST and echoed in — the same approach headroom_ccr_claude_code.py uses
        for .claude.json — rather than shelling out to `node -e`. node is not on PATH during
        the agent SETUP phase (the base agent only exports ~/.local/bin for the agent RUN),
        so the node version silently wrote nothing and the hook was never wired.

        Writing fresh rather than merging is safe here: the base ClaudeCode never writes
        settings.json (its memory command targets projects/-app/memory/ and its MCP command
        .claude.json), so there is nothing of its to preserve.

        $HOME is expanded in the CONTAINER via a placeholder + sed, because the hook command
        must be an absolute path — assuming Claude Code expands environment variables inside
        hook commands would fail silently as "no intervention", the one failure this arm must
        never have.

        The matcher is "Bash": rtk only rewrites shell commands, and an unmatched hook would
        fire on every tool call for nothing.
        """
        cfg = {
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command",
                               "command": "__RTK_HOME__/.local/bin/rtk hook claude",
                               "timeout": 10}],
                }]
            }
        }
        blob = shlex.quote(json.dumps(cfg, indent=2))
        return (f'printf %s {blob} | sed "s|__RTK_HOME__|$HOME|g" '
                "> $CLAUDE_CONFIG_DIR/settings.json")
