"""Runtime settings, loaded from environment / ``.env``.

Copy ``.env.dist`` to ``.env`` and fill in your keys. Only the keys for the
providers/strategies you actually run are required.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ so components that read os.environ directly (the
# proxy executor's upstream key lookup) see the same values pydantic-settings
# reads from the file. pydantic-settings alone does NOT populate os.environ.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Upstream provider keys (the proxies forward with these).
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Direct-upstream base URLs (used by the "upstream" live-baseline strategy).
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com/v1"

    # Google Gemini via its OpenAI-compatible chat/completions endpoint. A cheap
    # direct executor for evaluating the anatomy without Anthropic spend.
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # headroom proxy endpoint.
    headroom_base_url: str = "http://127.0.0.1:8787"

    # condense: api_url + caller creds come from the local `dense` CLI config
    # (~/.config/dense). `condense_profile` picks the profile; None follows the
    # `target` pointer (prod default). No condense keys live in .env.
    condense_profile: str | None = None
    # Force inline compaction so the forwarded request is actually compressed.
    condense_auto_condense_mode: str = "sync"
    # `condense-async` think time: seconds of simulated user work per output token
    # of the previous turn, giving background compaction time to land before the
    # next request. Capped per turn so one large output can't stall the run.
    condense_async_secs_per_output_token: float = 0.0001
    condense_async_max_think_seconds: float = 45.0

    # Install commands used by the `run` auto-setup step when a tool is missing.
    # Each is the command that project's own README documents; override if needed.
    #   headroom -> PyPI package   (README: pip install "headroom-ai[proxy]")
    #   dense    -> curl | sh       (dense README: https://cli.condense.chat/unix)
    headroom_install_cmd: str = 'uv tool install "headroom-ai[proxy]"'
    dense_install_cmd: str = "curl -fsSL https://cli.condense.chat/unix | sh"

    # Anthropic 1M-context beta header, sent on Anthropic proxy requests so
    # chains up to 1M tokens are accepted. Blank to disable.
    anthropic_beta: str = "context-1m-2025-08-07"

    # HuggingFace token for the gated SALT-NLP/SWE-chat dataset.
    hf_token: str | None = None

    # Local cache dir for materialized datasets (avoids re-streaming from HF).
    # This is the one cache that is NOT scoped to a run — it is source data,
    # not measurement, so it is shared across every run.
    data_dir: str = "data/cache"

    # Root under which each benchmark run gets its own run-<uuid>/ directory
    # (baseline + per-strategy measurement caches; cost is recomputed from them).
    runs_dir: str = "runs"

    # Default request cap for proxy executor (Anthropic requires >= 1).
    proxy_max_tokens: int = 1

    # Parallelism. Worker threads run (model, strategy, session) units concurrently;
    # per-endpoint concurrency caps in-flight requests to any single destination.
    run_max_workers: int = 12
    endpoint_max_concurrency: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
