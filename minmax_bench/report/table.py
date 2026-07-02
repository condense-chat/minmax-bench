"""Render bucketed savings as rich tables, and (de)serialize results for
reproducible, inspectable result bundles."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..models import Usage
from .buckets import BucketStats, PairedRow, summarize

# Fields emitted per bucket in report.json (everything except the private rows).
_BUCKET_FIELDS = [
    "label", "n", "avg_chain_tokens",
    "tokens_saved_prompt", "tokens_saved_total",
    "cache_read_tokens_saved", "cache_write_tokens_saved",
    "pct_tokens_saved_prompt", "pct_tokens_saved_total",
    "pct_cache_read_tokens_saved", "pct_cache_write_tokens_saved",
    "base_price_usd", "post_price_usd", "cost_saved_usd",
    "pct_cost_saved_total", "pct_cost_saved_cache_read",
    "pct_cost_saved_cache_write", "pct_cost_saved_input",
]


def render_tables(
    per_strategy: dict[str, list[BucketStats]], console: Console | None = None
) -> None:
    console = console or Console()
    for name, buckets in per_strategy.items():
        _tokens_table(name, buckets, console)
        _cost_table(name, buckets, console)


def _tokens_table(name: str, buckets: list[BucketStats], console: Console) -> None:
    t = Table(title=f"[bold]{name}[/] — tokens saved by input-chain bucket")
    for col in ("chain bucket", "n", "avg chain", "prompt %", "total(+out) %",
                "cache-rd saved", "cache-wr saved"):
        t.add_column(col, justify="right" if col != "chain bucket" else "left")
    for b in buckets:
        style = "bold" if b.label == "ALL" else ""
        t.add_row(
            b.label, str(b.n), _i(b.avg_chain_tokens),
            f"{b.pct_tokens_saved_prompt:.1f}%", f"{b.pct_tokens_saved_total:.1f}%",
            _i(b.cache_read_tokens_saved), _i(b.cache_write_tokens_saved), style=style,
        )
    console.print(t)


def _cost_table(name: str, buckets: list[BucketStats], console: Console) -> None:
    t = Table(title=f"[bold]{name}[/] — cost saved by input-chain bucket (USD)")
    for col in ("chain bucket", "n", "base $", "post $", "total %",
                "cache-rd %", "cache-wr %"):
        t.add_column(col, justify="right" if col != "chain bucket" else "left")
    for b in buckets:
        style = "bold" if b.label == "ALL" else ""
        t.add_row(
            b.label, str(b.n), _usd(b.base_price_usd), _usd(b.post_price_usd),
            f"{b.pct_cost_saved_total:.1f}%", f"{b.pct_cost_saved_cache_read:.1f}%",
            f"{b.pct_cost_saved_cache_write:.1f}%", style=style,
        )
    console.print(t)


def _i(x: float) -> str:
    return f"{x:,.0f}"


def _usd(x: float) -> str:
    return f"${x:,.4f}"


def rows_to_json(per_strategy: dict[str, list[BucketStats]]) -> dict:
    return {
        name: [{f: getattr(b, f) for f in _BUCKET_FIELDS} for b in buckets]
        for name, buckets in per_strategy.items()
    }


def run_report_json(per_model: dict[str, dict[str, list[BucketStats]]]) -> dict:
    """Nested {model: {strategy: [bucket dicts]}} for a run's report.json."""
    return {model: rows_to_json(ps) for model, ps in per_model.items()}


def _all_bucket(buckets: list[BucketStats]) -> BucketStats | None:
    for b in buckets:
        if b.label == "ALL":
            return b
    return None


def render_model_summary(
    per_model: dict[str, dict[str, list[BucketStats]]], console: Console | None = None
) -> None:
    """One compact matrix: cost saved % (ALL bucket) per model x strategy."""
    console = console or Console()
    strat_names: list[str] = []
    for ps in per_model.values():
        for name in ps:
            if name not in strat_names:
                strat_names.append(name)
    t = Table(title="[bold]cost saved % — per model × strategy (all conversations)")
    t.add_column("model", justify="left")
    for name in strat_names:
        t.add_column(name, justify="right")
    for model, ps in per_model.items():
        cells = [model]
        for name in strat_names:
            b = _all_bucket(ps.get(name, []))
            cells.append(f"{b.pct_cost_saved_total:.1f}%" if b and b.n else "—")
        t.add_row(*cells)
    console.print(t)


def render_run(
    per_model: dict[str, dict[str, list[BucketStats]]], console: Console | None = None
) -> None:
    """Per-model headline summary, then per-model per-bucket token & cost tables."""
    console = console or Console()
    render_model_summary(per_model, console)
    for model, ps in per_model.items():
        console.rule(f"[bold]{model}")
        for name, buckets in ps.items():
            b = _all_bucket(buckets)
            if not b or not b.n:  # strategy not run for this model (e.g. proxy+OpenAI)
                continue
            _tokens_table(f"{model} · {name}", buckets, console)
            _cost_table(f"{model} · {name}", buckets, console)


def measurements_to_json(per_strategy_rows: dict[str, list[PairedRow]]) -> dict:
    """Raw per-point measurements so buckets can be recomputed offline (verify)."""
    def u(x: Usage) -> dict:
        return {"it": x.input_tokens, "ot": x.output_tokens,
                "cr": x.cache_read, "cw": x.cache_write}
    return {
        name: [
            {
                "session_id": r.session_id, "index": r.index, "model": r.model,
                "chain_tokens": r.chain_tokens, "ok": r.ok,
                "base": u(r.base), "strat": u(r.strat),
            }
            for r in rows
        ]
        for name, rows in per_strategy_rows.items()
    }


def measurements_from_json(data: dict) -> dict[str, list[PairedRow]]:
    """Inverse of :func:`measurements_to_json` — rebuild rows for recompute."""
    def usage(d: dict) -> Usage:
        return Usage(input_tokens=d["it"], output_tokens=d["ot"],
                     cache_read=d["cr"], cache_write=d["cw"])
    return {
        name: [
            PairedRow(
                session_id=r["session_id"], index=r["index"], model=r["model"],
                chain_tokens=r["chain_tokens"], base=usage(r["base"]),
                strat=usage(r["strat"]), ok=r["ok"],
            )
            for r in rows
        ]
        for name, rows in data.items()
    }


def recompute_buckets(
    per_strategy_rows: dict[str, list[PairedRow]], edges: list[int] | None = None
) -> dict[str, list[BucketStats]]:
    return {name: summarize(rows, edges) for name, rows in per_strategy_rows.items()}
