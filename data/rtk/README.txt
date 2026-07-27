Frozen copy of rtk's model-facing instructions (hooks/claude/rtk-awareness.md) at the pinned
release — the file `rtk init -g` embeds into CLAUDE.md. The harbor agent installs a release
BINARY, not the repo, so this is captured here rather than fetched a second time.

It is NOT a skill in caveman's sense: ~10 lines advertising four analytics commands
(gain/discover/proxy) plus "everything else is rewritten automatically by the hook". It never
redirects the model away from the native Read tool, so it does not lift rtk's Bash-only
ceiling. It is installed anyway because a real `rtk init -g` installs it, and because those
lines are a genuine (small) context cost the arm should carry.

Regenerate on a pin bump:
  gh api repos/rtk-ai/rtk/contents/hooks/claude/rtk-awareness.md?ref=<tag> --jq .content \
    | base64 -d > awareness.md
