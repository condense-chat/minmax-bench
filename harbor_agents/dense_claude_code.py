"""Custom Harbor agent: Claude Code launched THROUGH the `dense` CLI.

The condense arms previously reproduced `dense claude`'s wiring by hand: ANTHROPIC_BASE_URL
plus the two x-condense-* headers, passed as plain env. That under-measured the product —
`dense claude` also sets _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL / ENABLE_TOOL_SEARCH /
CLAUDE_CODE_AUTO_COMPACT_WINDOW on the child (without them Claude Code composes an
~8-11k-token-larger request behind a non-default base URL and silently drops to a 200k
window), mints a per-launch x-condense-session-id, and heartbeats the session. Measured
2026-08-11: the env vars alone move a first-request prompt from ~43.5k tokens back to
within ~200 of the api.anthropic.com baseline.

Rather than carbon-copying that list (and re-copying it every time the CLI grows a
feature), this agent installs the REAL pinned `dense` binary in the container and rewrites
harbor's launch line `claude …` -> `dense claude -- …` — the exact process tree a real user
runs. Whatever the pinned dense does to its child, the arm does too, by construction.

What deliberately does NOT change vs the plain condense arm:
  - the allowlist: `dense claude` talks only to the profile's api host (proxy leg,
    ensure_auth probe, session heartbeat all share it), which is the arm's allow_host
    already. No new network reach.
  - the container env: the condense token is written to the files dense reads
    (~/.config/dense/…) and nowhere else, so the agent's own Bash can't read it. The
    ANTHROPIC_CUSTOM_HEADERS --ae passing is gone — dense builds fresh headers itself.

Run via:  -a harbor_agents.dense_claude_code:DenseClaudeCode
with:     TMB_DENSE_PROFILE / TMB_DENSE_URL / TMB_DENSE_TOKEN / TMB_DENSE_USER in the
          HARBOR PROCESS environment (set by minmax_bench.quality.generate from the local
          dense profile; deliberately not --ae).
"""
from __future__ import annotations

import os
import re
import shlex

from harbor.agents.installed.claude_code import ClaudeCode

# Pinned so a run is reproducible and a CLI release cannot silently change the method
# under test mid-experiment. Bump deliberately, and start a fresh dated arm when you do —
# a dense version change is a change to the product under test, not a dependency update.
# (The asset names match what cli.condense.chat/unix itself downloads.)
DENSE_VERSION = "v0.6.0"
DENSE_RELEASE = "https://github.com/condense-chat/dense/releases/download"
DENSE_ASSETS = {
    "aarch64": "dense-linux-aarch64",
    "arm64": "dense-linux-aarch64",
    "x86_64": "dense-linux-x86_64",
    "amd64": "dense-linux-x86_64",
}

# Same slot the base bootstrap puts `claude` in; already exported onto PATH by the run
# command, and present in the setup chain via the explicit export below.
DENSE_BIN_DIR = "$HOME/.local/bin"
DENSE_BIN = f"{DENSE_BIN_DIR}/dense"

# Harbor's ClaudeCode.run() launches `claude --verbose --output-format=stream-json …`.
# Match that exact shape: the lookbehind keeps us off paths (`…/bin/claude`) and the
# lookahead pins us to the launch line, never a setup command that mentions claude.
_LAUNCH = re.compile(r"(?<![\w/.-])claude (?=--verbose --output-format=stream-json)")

# The `--` is load-bearing. `dense claude` passes ARGS through, but it carries dense's own
# global flags too — including `-v/--verbose` — and clap matches those before the
# passthrough starts. Without the separator dense EATS harbor's leading `--verbose`, the
# child sees `--output-format=stream-json --print` without it, and Claude Code exits
# immediately with "When using --print, --output-format=stream-json requires --verbose" —
# every trial of the arm dies at launch. Verified against the pinned dense v0.6.0.
_DENSE_LAUNCH = "dense claude -- "


class DenseClaudeCode(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "dense-claude-code"

    # -- wiring -------------------------------------------------------------------

    async def exec_as_agent(self, environment, command, env=None, cwd=None,
                            timeout_sec=None):
        """Rewrite the launch exec so Claude Code runs under `dense claude`.

        Harbor hardcodes its launch command inside run() with no seam for the binary, so
        the rewrite happens here, at the last point before the container. Guarded loudly:
        if harbor reshapes its launch line, refusing to run beats silently measuring the
        headers-only wiring while the results carry a dense-launched arm's name.
        """
        if command and "--output-format=stream-json" in command:
            rewritten = _LAUNCH.sub(_DENSE_LAUNCH, command, count=1)
            if rewritten == command:
                raise RuntimeError(
                    "dense-claude-code: harbor's launch command no longer matches "
                    "'claude --verbose --output-format=stream-json' — the dense wrapper "
                    "would be skipped and the arm would run WITHOUT the product under "
                    f"test. Update _LAUNCH for this harbor version. (got: {command[:200]})"
                )
            command = rewritten
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd,
                                           timeout_sec=timeout_sec)

    # -- provisioning -------------------------------------------------------------

    def _profile(self) -> tuple[str, str, str, str]:
        """(profile_name, api_url, token, user_id) for the container's dense config.

        Read from the HOST process environment, which minmax_bench.quality.generate fills
        from the user's own `dense` profile — the container authenticates as exactly the
        identity the arm name claims. Deliberately NOT passed through `--ae`: the token
        only needs to reach a file dense reads, and putting it in the container's
        environment would expose it to the agent's own Bash calls.
        """
        name = os.environ.get("TMB_DENSE_PROFILE", "").strip()
        url = os.environ.get("TMB_DENSE_URL", "").strip()
        token = os.environ.get("TMB_DENSE_TOKEN", "").strip()
        user = os.environ.get("TMB_DENSE_USER", "").strip()
        if not name or not url:
            raise RuntimeError(
                "dense-claude-code: TMB_DENSE_PROFILE / TMB_DENSE_URL are unset. They are "
                "set by `minmax-bench quality run` from your local dense profile; this "
                "agent cannot be run standalone without them."
            )
        if not token and not user:
            raise RuntimeError(
                "dense-claude-code: no dense credentials to install in the container — "
                "`dense claude` would fail auth on every launch and the arm would measure "
                "nothing. Run `dense login` (or set CONDENSE_API_KEY) and retry."
            )
        return name, url, token, user

    async def install(self, environment) -> None:
        # Base installs Claude Code. dense is a single static binary — nothing else.
        await super().install(environment)
        self._profile()  # fail at install time, not after the container is paid for

        arch = (await self._container_arch(environment)).strip()
        asset = DENSE_ASSETS.get(arch)
        if asset is None:
            raise RuntimeError(
                f"dense-claude-code: no dense release asset for container arch {arch!r} "
                f"(have: {', '.join(sorted(set(DENSE_ASSETS.values())))})"
            )
        url = f"{DENSE_RELEASE}/{DENSE_VERSION}/{asset}"
        want = DENSE_VERSION.lstrip("v")
        # `--fail` so a deleted/renamed release asset is a hard error rather than an HTML
        # error page saved as a "binary". The version assert catches the other silent
        # failure: a tag that was force-moved to a different build.
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f"mkdir -p {DENSE_BIN_DIR}; "
                f"curl -fsSL {shlex.quote(url)} -o {DENSE_BIN}; "
                f"chmod 0755 {DENSE_BIN}; "
                f"v=$({DENSE_BIN} --version); "
                f'case "$v" in *"{want}"*) ;; *) '
                f'echo "dense: downloaded {DENSE_VERSION} but binary reports [$v] — '
                f'release tag moved?" >&2; exit 1 ;; esac; '
                f'echo "dense installed ($v, pinned {DENSE_VERSION})"'
            ),
        )

    async def _container_arch(self, environment) -> str:
        """`uname -m` inside the container, via the same exec path install() uses."""
        result = await self.exec_as_agent(environment, command="uname -m")
        for attr in ("stdout", "output", "text"):
            val = getattr(result, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        return str(result or "").strip()

    def _build_register_skills_command(self) -> str | None:
        """Write dense's credentials, then rehearse the exact launch the trial will make.

        Appended to ClaudeCode.run()'s setup_command. The base class's own skills command
        is preserved — dropping it would strip skills the vanilla arm has and make the
        arms incomparable.
        """
        base = super()._build_register_skills_command()
        # setup execs — unlike the run command — do not export ~/.local/bin themselves.
        path = 'export PATH="$HOME/.local/bin:$PATH"'
        parts = [p for p in (path, base, self._creds_command(), self._verify_command())
                 if p]
        return " && ".join(parts)

    def _creds_command(self) -> str:
        """Shell command that reproduces the host's dense profile inside the container.

        Mirrors the on-disk layout dense itself expects (contrib/dense config.rs
        `cred_dir_for`): prod credentials sit bare in ~/.config/dense, a named profile in
        a subdir, with a `target` pointer naming the active one. Replicating the real
        layout — rather than exporting CONDENSE_URL into the run env — matters because
        `dense claude` must resolve the profile off `target` exactly as it does on the
        user's machine, and because the token stays out of the container's environment.
        """
        name, url, token, user = self._profile()
        cred_dir = "$HOME/.config/dense" if name == "prod" else f"$HOME/.config/dense/{name}"
        parts = [f"mkdir -p {cred_dir}"]
        if name != "prod":
            profile_toml = f'name = "{name}"\napi_url = "{url}"\nauth_required = true\n'
            parts.append(f"printf %s {shlex.quote(profile_toml)} > {cred_dir}/profile.toml")
            parts.append(f"printf %s {shlex.quote(name)} > $HOME/.config/dense/target")
        if token:
            parts.append(f"printf %s {shlex.quote(token)} > {cred_dir}/token")
        if user:
            parts.append(f"printf %s {shlex.quote(user)} > {cred_dir}/user")
        # 0600: the agent can read its own files either way, but a token sitting
        # world-readable in a trial artifact tree is worth not doing.
        parts.append(f"chmod 600 {cred_dir}/token {cred_dir}/user 2>/dev/null || true")
        return "{ " + "; ".join(parts) + "; }"

    def _verify_command(self) -> str:
        """Smoke test: `dense claude -- --version` — the full harness, on a throwaway child.

        Exercises everything the trial's launch will: profile resolution off `target`,
        ensure_auth's token probe, the session open/end round-trip, and the resolve of the
        real `claude` binary. A trial that would fail any of those dies HERE, legibly,
        instead of burning the container build on a doomed run.

        Wrapped in `timeout` where available: a rejected token makes dense fall into its
        interactive device-login flow, which would otherwise hang this setup exec until
        harbor's setup timeout with no message pointing at the token. Exit codes are
        absorbed into the captured string and asserted with `case` — no top-level `||`,
        which would bind wrongly against the `&&` chain this string is joined into.
        """
        return (
            "{ "
            "if command -v timeout >/dev/null 2>&1; then t='timeout 90'; else t=''; fi; "
            f"out=$($t {_DENSE_LAUNCH}--version 2>&1 </dev/null || true); "
            'case "$out" in '
            # the pass: dense resolved and ran the real claude, which printed its version
            '*"(Claude Code)"*) ;; '
            '*"dense login"*|*"log in"*|*"device"*|*"401"*|*"403"*) '
            'echo "dense-claude-code: dense could not authenticate with the installed '
            'credentials — every trial launch would fail or hang in the login flow. Check '
            '\\`dense login\\` on the host and the profile this arm pins. '
            '(got: [$out])" >&2; exit 1 ;; '
            '*"error sending request"*|*"dns error"*|*"Connection refused"*) '
            'echo "dense-claude-code: dense cannot reach its api host from inside the '
            'container — check --allow-agent-host covers the profile host. '
            '(got: [$out])" >&2; exit 1 ;; '
            "*) "
            'echo "dense-claude-code: \\`dense claude --version\\` did not produce a '
            'Claude Code version banner — the launch path is broken and every trial '
            'would fail the same way. (got: [$out])" >&2; exit 1 ;; '
            "esac; "
            f'echo "dense claude launch path verified ({DENSE_VERSION})"; '
            "}"
        )
