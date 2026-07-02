"""Dataset dispatcher.

A dataset *spec* is a short string ``<source>[:<arg>]`` resolved to a list of
normalized :class:`~minmax_bench.models.Session`:

    sample                     -> built-in offline sample
    swe-chat[:LIMIT]           -> SALT-NLP/SWE-chat (HuggingFace, gated)
    claude-code:PATH           -> a Claude Code .jsonl transcript (file or glob)
    codex:PATH                 -> a Codex rollout .jsonl (file or glob)
    opencode:PATH              -> an OpenCode session (best-effort)
    jsonl:PATH                 -> newline-delimited Session dumps (this project's)
"""

from __future__ import annotations

import glob
from pathlib import Path

from ..config import get_settings
from ..models import Session


def load_dataset(spec: str) -> list[Session]:
    source, _, arg = spec.partition(":")
    source = source.strip()

    if source == "sample":
        from .sample import sample_sessions

        return sample_sessions()

    if source == "swe-chat":
        limit = int(arg) if arg.strip() else None
        return _load_swe_chat_cached(limit)

    if source in {"claude-code", "codex", "opencode"}:
        from . import local

        loader = {
            "claude-code": local.load_claude_code,
            "codex": local.load_codex,
            "opencode": local.load_opencode,
        }[source]
        return [loader(p) for p in _expand(arg)]

    if source == "jsonl":
        return _load_jsonl(arg)

    raise ValueError(f"unknown dataset spec {spec!r}")


def swe_chat_cache_path(limit: int | None, config: str = "conversations") -> Path:
    return Path(get_settings().data_dir) / f"swe-chat-{config}-{limit if limit else 'all'}.jsonl"


def _load_swe_chat_cached(limit: int | None, force: bool = False) -> list[Session]:
    """Load SWE-chat from the local cache, streaming from HF only on a miss.

    The first `swe-chat:N` load streams from HF (slow/flaky) and writes a local
    jsonl; every later load reads that file instantly. Use ``fetch`` / ``force``
    to (re)materialize.
    """
    cache = swe_chat_cache_path(limit)
    if cache.exists() and not force:
        return _load_jsonl(str(cache))
    from .swe_chat import load_swe_chat

    sessions = load_swe_chat(limit=limit, hf_token=get_settings().hf_token)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(s.model_dump_json() + "\n")
    tmp.replace(cache)  # atomic — a crashed fetch never leaves a partial cache
    return sessions


def _expand(arg: str) -> list[str]:
    if not arg:
        raise ValueError("this dataset source needs a path, e.g. claude-code:/path/to/*.jsonl")
    paths = sorted(glob.glob(arg)) if any(c in arg for c in "*?[") else [arg]
    if not paths:
        raise FileNotFoundError(f"no files matched {arg!r}")
    return paths


def _load_jsonl(arg: str) -> list[Session]:
    out: list[Session] = []
    for path in _expand(arg):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Session.model_validate_json(line))
    return out
