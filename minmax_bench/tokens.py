"""Local token counting (tiktoken).

Used to count assistant-turn output tokens and, via :mod:`minmax_bench.chain`, to
count full/rewritten chains offline. Cache-aware costing lives in
:mod:`minmax_bench.chain` (a per-turn differencing model); exact numbers come from
the proxy's real usage or Anthropic ``count_tokens``.
"""

from __future__ import annotations

import hashlib
import json

from .models import Message


def parse_token_count(raw: str | None) -> int | None:
    """'200k' / '1.5m' / '50,000' -> int tokens; None/''/unparseable -> None.

    The one parser for every human-entered token amount (CLI --token-budget, the
    wizard's budget prompt) so they accept identical spellings.
    """
    if not raw:
        return None
    r = raw.strip().lower().replace(",", "")
    mult = 1
    if r.endswith("k"):
        mult, r = 1_000, r[:-1]
    elif r.endswith("m"):
        mult, r = 1_000_000, r[:-1]
    try:
        return int(float(r) * mult)
    except ValueError:
        return None


class TokenCounter:
    """Local token counter. Uses tiktoken (offline).

    tiktoken is an approximation for Claude, but it is free, deterministic, and
    good enough for *estimated*-savings offline runs; the proxy executor and
    Anthropic ``count_tokens`` supply exact numbers when you need them.
    """

    def __init__(self, encoding: str = "o200k_base"):
        self._enc = None
        self._encoding_name = encoding
        self._cache: dict[str, int] = {}

    def _enc_obj(self):
        if self._enc is None:
            import tiktoken

            try:
                self._enc = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")
        return self._enc

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()
        hit = self._cache.get(h)
        if hit is not None:
            return hit
        n = len(self._enc_obj().encode(text, disallowed_special=()))
        self._cache[h] = n
        return n

    def count_json(self, obj) -> int:
        return self.count_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))

    def count_message(self, msg: Message) -> int:
        total = 4  # rough per-message framing overhead
        for b in msg.blocks:
            if b.text:
                total += self.count_text(b.text)
            if b.content:
                total += self.count_text(b.content)
            if b.tool_input is not None:
                total += self.count_json(b.tool_input)
            if b.tool_name:
                total += self.count_text(b.tool_name)
        return total
