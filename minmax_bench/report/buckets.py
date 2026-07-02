"""Bucket per-point results by input-chain length and compute rich savings.

Each point is compared against the uncompressed baseline for the same
(session, turn), then bucketed by the *baseline* prompt size so every strategy is
bucketed identically. For each bucket we report token savings and USD savings
broken out by cache tier (cache-read, cache-write), plus totals including output,
and absolute base/post prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Usage
from ..pricing import cost_breakdown

# Upper edges (tokens) of the input-chain buckets; the last bucket is open-ended.
# Sub-16k chains are lumped together — too small to amortize compression overhead,
# so their savings are noise. Empty buckets are dropped at summarize time.
DEFAULT_EDGES = [16_000, 32_000, 100_000, 200_000]


@dataclass
class PairedRow:
    session_id: str
    index: int
    chain_tokens: int  # baseline total prompt tokens (bucket key)
    model: str
    base: Usage
    strat: Usage
    ok: bool = True


def bucket_label(chain_tokens: int, edges: list[int]) -> str:
    lo = 0
    for hi in edges:
        if chain_tokens < hi:
            return f"{_k(lo)}-{_k(hi)}"
        lo = hi
    return f"{_k(lo)}+"


def _k(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


def _pct(base: float, strat: float) -> float:
    return 100.0 * (base - strat) / base if base else 0.0


@dataclass
class BucketStats:
    label: str
    n: int = 0
    avg_chain_tokens: float = 0.0
    # token totals saved (base - strat), summed over the bucket
    tokens_saved_prompt: int = 0          # input + cache_read + cache_write
    tokens_saved_total: int = 0           # prompt + output
    cache_read_tokens_saved: int = 0
    cache_write_tokens_saved: int = 0
    # token savings percentages
    pct_tokens_saved_prompt: float = 0.0
    pct_tokens_saved_total: float = 0.0
    pct_cache_read_tokens_saved: float = 0.0
    pct_cache_write_tokens_saved: float = 0.0
    # cost (USD) — absolute prices and per-tier savings percentages
    base_price_usd: float = 0.0
    post_price_usd: float = 0.0
    cost_saved_usd: float = 0.0
    pct_cost_saved_total: float = 0.0
    pct_cost_saved_cache_read: float = 0.0
    pct_cost_saved_cache_write: float = 0.0
    pct_cost_saved_input: float = 0.0
    _rows: list[PairedRow] = field(default_factory=list, repr=False)

    def finalize(self) -> BucketStats:
        rows = [r for r in self._rows if r.ok]
        self.n = len(rows)
        if not rows:
            return self
        self.avg_chain_tokens = sum(r.chain_tokens for r in rows) / self.n

        def tok(u: Usage, tier: str) -> int:
            return {
                "input": u.input_tokens, "cache_read": u.cache_read,
                "cache_write": u.cache_write, "output": u.output_tokens,
                "prompt": u.total_input, "total": u.total_input + u.output_tokens,
            }[tier]

        sums: dict[str, list[int]] = {}  # tier -> [base_sum, strat_sum]
        for tier in ("input", "cache_read", "cache_write", "output", "prompt", "total"):
            b = sum(tok(r.base, tier) for r in rows)
            s = sum(tok(r.strat, tier) for r in rows)
            sums[tier] = [b, s]

        self.tokens_saved_prompt = sums["prompt"][0] - sums["prompt"][1]
        self.tokens_saved_total = sums["total"][0] - sums["total"][1]
        self.cache_read_tokens_saved = sums["cache_read"][0] - sums["cache_read"][1]
        self.cache_write_tokens_saved = sums["cache_write"][0] - sums["cache_write"][1]
        self.pct_tokens_saved_prompt = _pct(*sums["prompt"])
        self.pct_tokens_saved_total = _pct(*sums["total"])
        self.pct_cache_read_tokens_saved = _pct(*sums["cache_read"])
        self.pct_cache_write_tokens_saved = _pct(*sums["cache_write"])

        # cost tiers
        csum: dict[str, list[float]] = {
            t: [0.0, 0.0] for t in ("input", "cache_read", "cache_write", "output", "total")
        }
        for r in rows:
            bb = cost_breakdown(r.model, r.base)
            sb = cost_breakdown(r.model, r.strat)
            for t in csum:
                csum[t][0] += bb[t]
                csum[t][1] += sb[t]
        self.base_price_usd = csum["total"][0]
        self.post_price_usd = csum["total"][1]
        self.cost_saved_usd = csum["total"][0] - csum["total"][1]
        self.pct_cost_saved_total = _pct(*csum["total"])
        self.pct_cost_saved_cache_read = _pct(*csum["cache_read"])
        self.pct_cost_saved_cache_write = _pct(*csum["cache_write"])
        self.pct_cost_saved_input = _pct(*csum["input"])
        return self


def summarize(rows: list[PairedRow], edges: list[int] | None = None) -> list[BucketStats]:
    edges = edges or DEFAULT_EDGES
    order: list[str] = []
    lo = 0
    for hi in edges:
        order.append(f"{_k(lo)}-{_k(hi)}")
        lo = hi
    order.append(f"{_k(lo)}+")

    buckets: dict[str, BucketStats] = {lab: BucketStats(lab) for lab in order}
    for r in rows:
        buckets[bucket_label(r.chain_tokens, edges)]._rows.append(r)

    total = BucketStats("ALL")
    total._rows = list(rows)
    # Drop empty buckets — an all-zero row carries no signal and just adds noise.
    kept = [buckets[lab].finalize() for lab in order if buckets[lab]._rows]
    return kept + [total.finalize()]
