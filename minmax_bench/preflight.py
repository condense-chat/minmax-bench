"""Dependency preflight for a run.

Checks whether each requested strategy's dependency is present (
headroom proxy, condense dense creds, provider keys) and whether the dataset needs
a HuggingFace token. Nothing here spends money or mutates state — it only probes,
so the TUI can show what will and won't run and let the user decide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from .catalog import is_gemini, is_openai
from .config import get_settings
from .dense import load_profile
from .strategies import get_entry, has_entry, tool_for


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    disables: list[str]  # strategy names this check gates (empty = informational)


def _kind(name: str) -> str | None:
    return get_entry(name).kind if has_entry(name) else None


def _provider(name: str) -> str | None:
    """The dialect a proxy strategy speaks ('anthropic' | 'openai'), or None."""
    if not has_entry(name):
        return None
    try:
        r = get_entry(name).resolve()
        return r.proxy.provider.value if r.proxy else None
    except Exception:
        return None


def _headroom_reachable(base_url: str, timeout: float = 1.5) -> bool:
    try:
        # any HTTP response (even 404/401) proves something is listening.
        httpx.get(base_url, timeout=timeout)
        return True
    except Exception:
        return False


def preflight(strategies: list[str], models: list[str], dataset: str) -> list[Check]:
    s = get_settings()
    checks: list[Check] = []
    any_anthropic = any(not is_openai(m) for m in models)
    any_gemini = any(is_gemini(m) for m in models)
    any_true_openai = any(is_openai(m) and not is_gemini(m) for m in models)

    # Group the requested strategies by what they depend on (derived from the
    # matrix, so new/renamed variants are covered automatically).
    order = list(strategies)
    proxy_wanted = [n for n in order if _kind(n) == "proxy"]
    anth_proxies = [n for n in proxy_wanted if _provider(n) == "anthropic"]
    gemini_proxies = [n for n in proxy_wanted if _provider(n) == "openai"]
    headroom_wanted = [n for n in order if tool_for(n) == "headroom"]
    condense_wanted = [n for n in order if tool_for(n) == "dense"]

    # Provider keys. The Anthropic key gates only the Anthropic-dialect proxies /
    # models; the Gemini executor uses its own key; true-OpenAI models are baseline.
    if anth_proxies or any_anthropic:
        anth = bool(os.environ.get("ANTHROPIC_API_KEY") or s.anthropic_api_key)
        checks.append(Check(
            "ANTHROPIC_API_KEY", anth,
            "set" if anth else "missing — Anthropic proxy strategies can't reach the upstream",
            anth_proxies if not anth else [],
        ))
    if gemini_proxies or any_gemini:
        gk = bool(os.environ.get("GEMINI_API_KEY") or s.gemini_api_key)
        checks.append(Check(
            "GEMINI_API_KEY", gk,
            "set" if gk else "missing — the Gemini executor can't reach Google",
            gemini_proxies if not gk else [],
        ))
    if any_true_openai:
        oai = bool(os.environ.get("OPENAI_API_KEY") or s.openai_api_key)
        checks.append(Check(
            "OPENAI_API_KEY", oai,
            "set" if oai else "missing — OpenAI models measure baseline tokens only",
            [],
        ))

    # headroom proxy reachability (gates every headroom variant together).
    if headroom_wanted:
        up = _headroom_reachable(s.headroom_base_url)
        checks.append(Check(
            "headroom proxy", up,
            s.headroom_base_url + ("" if up else " — not reachable"),
            [] if up else headroom_wanted,
        ))

    # condense dense creds — gates every condense variant on the active profile.
    if condense_wanted:
        prof = load_profile(s.condense_profile)
        ok = bool(prof.auth_token and prof.user_id)
        checks.append(Check(
            "condense (dense creds)", ok,
            f"profile={prof.name} url={prof.api_url}"
            + ("" if ok else " — token/user missing under ~/.config/dense"),
            [] if ok else condense_wanted,
        ))

    # HuggingFace token for the gated SWE-chat dataset (only if not already cached).
    if dataset.startswith("swe-chat"):
        from .data.loaders import swe_chat_cache_path

        _, _, arg = dataset.partition(":")
        limit = int(arg) if arg.strip() else None
        cached = swe_chat_cache_path(limit).exists()
        hf = bool(s.hf_token)
        if not cached:
            checks.append(Check(
                "HF_TOKEN", hf,
                "set" if hf else "missing — needed to stream the gated SWE-chat dataset",
                [],
            ))

    return checks


def disabled_strategies(checks: list[Check]) -> dict[str, str]:
    """strategy -> reason, for every strategy a failed check disables."""
    out: dict[str, str] = {}
    for c in checks:
        if not c.ok:
            for st in c.disables:
                out.setdefault(st, f"{c.name}: {c.detail}")
    return out
