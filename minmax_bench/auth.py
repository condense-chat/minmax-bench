"""Anthropic auth resolution, shared by the cost and quality benches.

Two ways an upstream Anthropic call can authenticate: an API key, or the user's
Claude Code subscription OAuth token — the same fallback order the quality bench
has always used, now available to the cost bench's executors too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_CC_TOKEN_CACHE: list = ["unset"]  # one keychain read per process, not per request


def _dotenv(key: str) -> str | None:
    try:
        from dotenv import dotenv_values

        return dotenv_values(".env").get(key)
    except Exception:  # noqa: BLE001 — no .env / unreadable: fall through
        return None


def cc_oauth_token():
    """The user's own Claude Code subscription credential, read locally.

    Most Claude Code users have no API key — their auth is the OAuth token the
    Claude Code login stores. The bench replays THEIR sessions in Claude Code's
    own request shape, so when no API key is configured we authenticate the same
    way Claude Code does. The token stays in-process and is only ever sent where
    the request points (api.anthropic.com or the user's chosen gateway).

    Sources, in order: CLAUDE_CODE_OAUTH_TOKEN env var, ``.env``,
    ~/.claude/.credentials.json, the macOS keychain item Claude Code maintains.
    Returns None when absent/expired.
    """
    if _CC_TOKEN_CACHE[0] != "unset":
        return _CC_TOKEN_CACHE[0]

    def parse(raw):
        try:
            oauth = json.loads(raw).get("claudeAiOauth") or {}
        except (json.JSONDecodeError, AttributeError):
            return None
        exp = oauth.get("expiresAt") or 0
        if exp and exp / 1000 < time.time():
            # the stored ACCESS token is short-lived and has expired. Claude Code refreshes it
            # on use via the refresh token, so your subscription is fine — but this reader does
            # NOT do the OAuth refresh, so it can't use the stale token and falls back to an API
            # key. `claude setup-token` mints a LONG-LIVED token for exactly this programmatic use.
            hint = ("run `claude setup-token` and `export CLAUDE_CODE_OAUTH_TOKEN=<it>` (a "
                    "long-lived token)" if oauth.get("refreshToken")
                    else "open `claude` once to refresh it")
            print("warn: the stored Claude Code access token has expired (Claude Code refreshes "
                  f"it on use; this bench does not) — {hint}, or set ANTHROPIC_API_KEY.",
                  file=sys.stderr)
            return None
        return oauth.get("accessToken")

    # os.environ first (an explicit export wins), then .env — the setup wizard writes the
    # token to .env, and not every entry point load_dotenv's it into os.environ, so read it
    # directly or a wizard-wired token stays invisible and we wrongly fall back to an API key.
    tok = (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
           or _dotenv("CLAUDE_CODE_OAUTH_TOKEN") or None)
    if not tok:
        cred = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(cred):
            tok = parse(open(cred).read())
    if not tok and sys.platform == "darwin":
        try:
            r = subprocess.run(["security", "find-generic-password",
                                "-s", "Claude Code-credentials", "-w"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                tok = parse(r.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            tok = None
    _CC_TOKEN_CACHE[0] = tok
    return tok


def anthropic_auth_headers(beta: str | None = None,
                           key_env: str = "ANTHROPIC_API_KEY") -> dict[str, str]:
    """Auth headers for an Anthropic-dialect call: the API key from ``key_env``
    when set, else the subscription OAuth token with its beta flag merged in."""
    key = os.environ.get(key_env) or _dotenv(key_env)
    if key:
        h = {"x-api-key": key}
        if beta:
            h["anthropic-beta"] = beta
        return h
    tok = cc_oauth_token()
    if not tok:
        raise RuntimeError(
            f"no {key_env} and no Claude Code OAuth token — set the key, or run "
            "`claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN"
        )
    merged = (beta + "," if beta else "") + "oauth-2025-04-20"
    return {"authorization": f"Bearer {tok}", "anthropic-beta": merged}
