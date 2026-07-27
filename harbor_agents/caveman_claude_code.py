"""Custom Harbor agent: Claude Code + the caveman skill, wired fully self-contained.

caveman (https://github.com/JuliusBrussee/caveman) is a *generation-policy* method, not a
proxy: no ANTHROPIC_BASE_URL change, no rewriting of history. A SessionStart hook injects a
terse-output ruleset at session start, and the model writes fragments instead of prose for
the rest of the run. Its context savings are indirect — smaller assistant messages
accumulate into a smaller prefix on every later turn — against a fixed ~750-token cost for
the ruleset itself.

That makes it the first arm with NO proxy, so it reads against plain `vanilla` rather than
`vanilla-proxy` (it does not carry the ~8-9k non-default-base-URL wiring confound).

This agent installs it the reproducible way — no `install.sh` piped from the network at run
time, no mutation of the user's machine:
  1. fetch the repo tarball at a PINNED tag into the container (install phase);
  2. copy `src/hooks/` -> $CLAUDE_CONFIG_DIR/hooks/ and `skills/caveman/` ->
     $CLAUDE_CONFIG_DIR/skills/caveman/ (the standalone layout caveman-activate.js resolves
     SKILL.md from), and wire the SessionStart hook into $CLAUDE_CONFIG_DIR/settings.json —
     mirroring what src/hooks/install.sh does.

Two failure modes this guards against, because both would silently make the arm measure
NOTHING and report "caveman ~= vanilla" for the wrong reason:
  - a moved/renamed path upstream: caveman-activate.js tries three SKILL.md locations and
    then falls back to a hardcoded ruleset, so a bad layout degrades quietly. `install`
    smoke-tests the hook and aborts the trial if it does not emit the activation marker.
  - the hook never firing under `claude --print`: unverifiable from here, so the marker is
    left in the transcript for the report side to assert on
    (`minmax_bench.quality.report` surfaces it as `⚠ inactive`).

The statusline IS configured, deliberately: when settings.json has no `statusLine`,
caveman-activate.js appends "STATUSLINE SETUP NEEDED ... Proactively offer to set this up
for the user on first interaction" to its output, which would burn benchmark turns on
statusline configuration. install.sh wires a statusline too, so configuring it is both
faithful to a real install and removes the confound.

Run via:  -a harbor_agents.caveman_claude_code:CavemanClaudeCode
with:     --ae TMB_CAVEMAN_MODE=full   (lite | full | ultra; default full)
"""
from __future__ import annotations

import shlex

from harbor.agents.installed.claude_code import ClaudeCode

# Pinned so a run is reproducible and an upstream retag cannot silently change the
# intervention mid-experiment. Bump deliberately, and re-run every arm when you do:
# a caveman version change is a change to the method under test, not a dependency update.
CAVEMAN_REF = "v1.9.1"
CAVEMAN_SHA = "033f918602bd5319931256a537c4bd9ea7a48c25"
CAVEMAN_TARBALL = f"https://codeload.github.com/JuliusBrussee/caveman/tar.gz/{CAVEMAN_SHA}"

# Where the pinned checkout lands in the container (install phase). $CLAUDE_CONFIG_DIR does
# not exist yet at install time — it is created and exported by ClaudeCode.run() — so the
# source is staged here and copied into place during the run's setup command.
SRC_DIR = "$HOME/.caveman-src"

# caveman-activate.js prefixes its output with this. Absence of it anywhere downstream means
# the arm ran without the intervention.
ACTIVATION_MARKER = "CAVEMAN MODE ACTIVE"

VALID_MODES = ("lite", "full", "ultra")


class CavemanClaudeCode(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "caveman-claude-code"

    def _mode(self) -> str:
        mode = (self._get_env("TMB_CAVEMAN_MODE") or "full").strip().lower()
        if mode not in VALID_MODES:
            raise RuntimeError(
                f"TMB_CAVEMAN_MODE={mode!r} is not one of {', '.join(VALID_MODES)}. "
                "The wenyan-* levels are excluded on purpose: they switch the output "
                "language to classical Chinese, which confounds the trajectory metrics."
            )
        return mode

    async def install(self, environment) -> None:
        # Base installs Claude Code. On alpine that pulls in nodejs; on other images the
        # bootstrap installs a SELF-CONTAINED `claude` and leaves NO `node` binary. But
        # caveman's hooks ARE node scripts — Claude Code runs `node caveman-activate.js` on
        # SessionStart, and the setup below runs `node -e` to merge settings.json — so a `node`
        # on the system PATH is a hard requirement. Ensure it the same way the headroom agent
        # ensures python3, or the setup dies with exit 127 (command not found) and the trial
        # runs with NO agent at all. Skip if already present (alpine, or a node-bearing image).
        await super().install(environment)
        await self.exec_as_root(
            environment,
            command=(
                "if command -v node >/dev/null 2>&1; then echo 'node present, skipping'; "
                "elif command -v apk >/dev/null 2>&1; then apk add --no-cache nodejs; "
                "elif command -v apt-get >/dev/null 2>&1; then apt-get update && "
                "  apt-get install -y --no-install-recommends nodejs; "
                "elif command -v yum >/dev/null 2>&1; then yum install -y nodejs; fi; "
                "command -v node >/dev/null 2>&1 && node --version"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        # Fetch the pinned tree. `--fail` so a 404 from a deleted tag is a hard error
        # rather than a tarball containing an HTML error page.
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f"rm -rf {SRC_DIR}; mkdir -p {SRC_DIR}; "
                f"curl -fsSL {shlex.quote(CAVEMAN_TARBALL)} "
                f"| tar xz -C {SRC_DIR} --strip-components=1; "
                # Fail early and legibly if upstream moves any path we depend on. Every file
                # _settings_script wires is checked, not just the two the smoke test exercises:
                # the mode tracker and the statusline are referenced by COMMAND STRING, so a
                # missing one fails at hook-run time inside the trial (or, for the statusline,
                # not visibly at all) — the smoke test's statusline check only proves
                # settings.json has the key, never that the script resolves.
                + "".join(
                    f'test -f {SRC_DIR}/{p} '
                    f'|| {{ echo "caveman: {p} missing at {CAVEMAN_REF}" >&2; exit 1; }}; '
                    for p in ("src/hooks/caveman-activate.js",
                              "src/hooks/caveman-mode-tracker.js",
                              "src/hooks/caveman-statusline.sh",
                              "skills/caveman/SKILL.md")
                )
                + f'echo "caveman {CAVEMAN_REF} staged"'
            ),
        )

    def _build_register_skills_command(self) -> str | None:
        """Install caveman into $CLAUDE_CONFIG_DIR, then smoke-test that it activates.

        Appended to ClaudeCode.run()'s setup_command, which is the first point at which
        $CLAUDE_CONFIG_DIR exists and is exported. The base class's own skills command is
        preserved — dropping it would strip skills the vanilla arm has and make the arms
        incomparable.
        """
        mode = self._mode()
        base = super()._build_register_skills_command()

        # Layout note: caveman-activate.js resolves SKILL.md relative to its own __dirname,
        # and the candidate that matches this layout is '../skills/caveman/SKILL.md' — i.e.
        # hooks at $CLAUDE_CONFIG_DIR/hooks/ and the skill at
        # $CLAUDE_CONFIG_DIR/skills/caveman/. Keep the two in that relationship or the hook
        # silently falls back to its abridged hardcoded ruleset.
        install = (
            "mkdir -p $CLAUDE_CONFIG_DIR/hooks $CLAUDE_CONFIG_DIR/skills && "
            f"cp -r {SRC_DIR}/src/hooks/. $CLAUDE_CONFIG_DIR/hooks/ && "
            f"cp -r {SRC_DIR}/skills/caveman $CLAUDE_CONFIG_DIR/skills/caveman"
        )

        settings = f"node -e {shlex.quote(self._settings_script(mode))}"

        # Smoke test: run the hook exactly as Claude Code will, and require three things.
        # The intensity-table row is the discriminating one — caveman-activate.js falls back
        # to an abridged HARDCODED ruleset when it cannot resolve SKILL.md, and that
        # fallback shares the marker and most section headings with the real thing but
        # never emits a `| **<mode>** |` table row. Without this check an upstream layout
        # change would quietly downgrade the intervention.
        verify = (
            f"out=$(CAVEMAN_DEFAULT_MODE={mode} node "
            "$CLAUDE_CONFIG_DIR/hooks/caveman-activate.js </dev/null) && "
            f'case "$out" in *"{ACTIVATION_MARKER}"*) ;; *) '
            'echo "caveman: activation hook produced no marker — the arm would run '
            'WITHOUT the intervention" >&2; exit 1 ;; esac && '
            f'case "$out" in *"| **{mode}** |"*) ;; *) '
            'echo "caveman: hook fell back to its hardcoded ruleset (SKILL.md not '
            'resolved) — refusing to run a degraded intervention" >&2; exit 1 ;; esac && '
            'case "$out" in *"STATUSLINE SETUP NEEDED"*) '
            'echo "caveman: statusline nudge is live; the agent would be told to '
            'configure a statusline mid-task" >&2; exit 1 ;; *) ;; esac && '
            f'echo "caveman {CAVEMAN_REF} active (mode={mode})"'
        )

        # Put node on PATH for the whole chain (settings + verify both shell out to `node`):
        # install() ensures a SYSTEM node, but a bootstrap-installed one lands in ~/.local/bin,
        # which this setup exec — unlike ClaudeCode.run()'s agent command — does not export.
        path = 'export PATH="$HOME/.local/bin:$PATH"'
        parts = [p for p in (path, base, install, settings, verify) if p]
        return " && ".join(parts)

    def _settings_script(self, mode: str) -> str:
        """Node one-liner that merges caveman's hooks + statusline into settings.json.

        Merges rather than clobbers (and creates the file if absent) so any settings the
        base agent or the image already wrote survive — same contract as install.sh.

        The mode is baked into the hook COMMAND rather than passed via the environment:
        ClaudeCode.run() builds the subprocess env explicitly, so an `--ae` var is not
        guaranteed to reach the hook's own process. Hook paths are likewise baked ABSOLUTE
        (expanded here, at write time) rather than left as `$CLAUDE_CONFIG_DIR` — that would
        assume Claude Code expands environment variables inside hook commands, and a wrong
        guess there fails silently as "no intervention".
        """
        return f"""
const fs = require('fs');
const dir = process.env.CLAUDE_CONFIG_DIR;
const p = dir + '/settings.json';
let s = {{}};
try {{ s = JSON.parse(fs.readFileSync(p, 'utf8')); }} catch (e) {{}}
if (!s.hooks) s.hooks = {{}};
const hook = script =>
  'CAVEMAN_DEFAULT_MODE={mode} node "' + dir + '/hooks/' + script + '"';
const add = (event, command) => {{
  if (!s.hooks[event]) s.hooks[event] = [];
  const has = s.hooks[event].some(e =>
    e.hooks && e.hooks.some(h => h.command && h.command.includes('caveman')));
  if (!has) s.hooks[event].push({{ hooks: [{{ type: 'command', command, timeout: 5 }}] }});
}};
add('SessionStart', hook('caveman-activate.js'));
add('UserPromptSubmit', hook('caveman-mode-tracker.js'));
// Deliberate: an unset statusLine makes caveman-activate.js emit a "proactively offer to
// set this up" nudge into the agent's context, which would spend benchmark turns on
// statusline configuration. install.sh wires one too.
if (!s.statusLine) {{
  const sl = 'bash "' + process.env.CLAUDE_CONFIG_DIR + '/hooks/caveman-statusline.sh"';
  s.statusLine = {{ type: 'command', command: sl }};
}}
fs.writeFileSync(p, JSON.stringify(s, null, 2) + '\\n');
"""
