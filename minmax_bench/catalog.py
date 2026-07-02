"""Selectable model catalog for the guided TUI.

Each entry is ``(label, model_id, provider, default)``. A model runs the baseline,
local rewrites, and any proxy strategy that speaks its dialect: Anthropic models
run the Anthropic proxies (condense/headroom); OpenAI/Gemini models run the OpenAI-
dialect executor. A proxy whose dialect doesn't match the model is skipped (and the
skip is logged by the runner, never silent).

Model ids must resolve in :mod:`minmax_bench.pricing` (tokencost first, then the
fallback table). Edit here to add/relabel tiers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    label: str
    model: str
    provider: str  # "anthropic" | "openai" | "google"
    default: bool = False


CATALOG: list[ModelChoice] = [
    ModelChoice("haiku 4.5", "claude-haiku-4-5", "anthropic", default=True),
    ModelChoice("sonnet 5", "claude-sonnet-5", "anthropic"),
    ModelChoice("opus 4.8", "claude-opus-4-8", "anthropic"),
    ModelChoice("gpt-5-mini", "gpt-5-mini", "openai"),
    ModelChoice("gpt-4.1", "gpt-4.1", "openai"),
    ModelChoice("gpt-5", "gpt-5", "openai"),
    ModelChoice("gemini 3.1 flash-lite", "gemini-3.1-flash-lite", "google"),
]

DEFAULT_MODELS: list[str] = [c.model for c in CATALOG if c.default]


def is_gemini(model: str) -> bool:
    m = model.lower()
    return "gemini" in m or "google" in m


def is_openai(model: str) -> bool:
    """True for any model spoken over the OpenAI chat/completions dialect — the real
    OpenAI SKUs and Gemini's OpenAI-compatible endpoint (they differ only in which
    API key/base URL they use, handled by the strategy that targets them)."""
    m = model.lower()
    return m.startswith(("gpt", "o1", "o3", "o4")) or "openai" in m or is_gemini(m)


def provider_of(model: str) -> str:
    if is_gemini(model):
        return "google"
    return "openai" if is_openai(model) else "anthropic"
