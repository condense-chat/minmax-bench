"""Cache-aware cost model.

Rates are USD *per token* for the four billable tiers: full-rate input, output,
cache read, and cache write (5-minute cache-creation). We prefer ``tokencost``'s
live table when installed and fall back to a small hardcoded table for SKUs the
pinned table may lag.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Usage


@dataclass(frozen=True)
class Rates:
    """USD per token."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read * self.cache_read
            + usage.cache_write * self.cache_write
        )


# Hardcoded fallback (USD/token). Cache-write is the 5-minute creation rate.
_FALLBACK: dict[str, Rates] = {
    # Anthropic. sonnet-5 / opus-4-8 fallbacks approximate the prior generation;
    # tokencost overrides with real rates when its table carries the SKU.
    "claude-opus-4-8": Rates(5e-6, 25e-6, 0.5e-6, 6.25e-6),
    "claude-opus-4-5": Rates(5e-6, 25e-6, 0.5e-6, 6.25e-6),
    "claude-sonnet-5": Rates(3e-6, 15e-6, 0.3e-6, 3.75e-6),
    "claude-sonnet-4-5": Rates(3e-6, 15e-6, 0.3e-6, 3.75e-6),
    "claude-haiku-4-5": Rates(1e-6, 5e-6, 0.1e-6, 1.25e-6),
    "claude-fable-5": Rates(10e-6, 50e-6, 1e-6, 12.5e-6),
    # OpenAI (no separate cache-write tier; cached input billed at cache_read).
    "gpt-5": Rates(1.25e-6, 10e-6, 0.125e-6, 1.25e-6),
    "gpt-5-mini": Rates(0.25e-6, 2e-6, 0.025e-6, 0.25e-6),
    "gpt-4.1": Rates(2e-6, 8e-6, 0.5e-6, 2e-6),
    "o4-mini": Rates(1.1e-6, 4.4e-6, 0.275e-6, 1.1e-6),
    # Google Gemini (approx flash-lite rates; the OpenAI-compat endpoint reports no
    # cache split, so cache tiers are rarely exercised).
    "gemini-3.1-flash-lite": Rates(0.10e-6, 0.40e-6, 0.025e-6, 0.10e-6),
}

_DEFAULT = Rates(3e-6, 15e-6, 0.3e-6, 3.75e-6)  # sonnet-class default

# Long-context (>200k prompt tokens) premium tiers, where the provider bills
# large prompts at a higher rate. Anthropic's 1M-context Sonnet doubles input and
# cache rates and ~1.5x output above 200k.
LONG_CONTEXT_THRESHOLD = 200_000
_LONG: dict[str, Rates] = {
    "claude-sonnet-5": Rates(6e-6, 22.5e-6, 0.6e-6, 7.5e-6),
    "claude-sonnet-4-5": Rates(6e-6, 22.5e-6, 0.6e-6, 7.5e-6),
}


def _normalize(model: str) -> str:
    m = model.lower().strip()
    for prefix in ("anthropic/", "openai/", "us.anthropic.", "claude-3-5-"):
        m = m.replace(prefix, "")
    # Strip dated / -latest suffixes: claude-sonnet-4-5-20250929 -> claude-sonnet-4-5
    for base in _FALLBACK:
        if m.startswith(base):
            return base
    return m


def _from_tokencost(model: str) -> Rates | None:
    try:
        from tokencost import TOKEN_COSTS  # type: ignore
    except Exception:
        return None
    key = model.lower().strip()
    row = TOKEN_COSTS.get(key)
    if row is None:
        # tokencost keys are often provider-qualified; try a few variants.
        for cand in (f"anthropic/{key}", f"openai/{key}", _normalize(model)):
            row = TOKEN_COSTS.get(cand)
            if row is not None:
                break
    if row is None:
        return None

    def g(*names: str) -> float:
        for n in names:
            v = row.get(n)
            if v is not None:
                return float(v)
        return 0.0

    inp = g("input_cost_per_token")
    if not inp:
        return None
    return Rates(
        input=inp,
        output=g("output_cost_per_token"),
        cache_read=g("cache_read_input_token_cost", "input_cost_per_token_cache_hit") or inp * 0.1,
        cache_write=g("cache_creation_input_token_cost") or inp * 1.25,
    )


def rates_for(model: str, prompt_tokens: int = 0) -> Rates:
    # Long-context premium kicks in above the threshold, when we have a tier.
    if prompt_tokens > LONG_CONTEXT_THRESHOLD:
        lc = _LONG.get(_normalize(model))
        if lc is not None:
            return lc
    live = _from_tokencost(model)
    if live is not None:
        return live
    return _FALLBACK.get(_normalize(model), _DEFAULT)


def cost_usd(model: str, usage: Usage) -> float:
    # Bill by the total prompt size so >200k chains use the long-context tier.
    return rates_for(model, usage.total_input).cost(usage)


def cost_breakdown(model: str, usage: Usage) -> dict[str, float]:
    """Per-tier USD cost: input, cache_read, cache_write, output (+ total)."""
    r = rates_for(model, usage.total_input)
    b = {
        "input": usage.input_tokens * r.input,
        "cache_read": usage.cache_read * r.cache_read,
        "cache_write": usage.cache_write * r.cache_write,
        "output": usage.output_tokens * r.output,
    }
    b["total"] = sum(b.values())
    return b
