# minmax-bench run fa5ef49e-cea1-4ee3-8364-a87b68c57042

Generated: 2026-07-08T08:28:44+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report fa5ef49e-cea1-4ee3-8364-a87b68c57042

## Replay the animated evolution

    minmax-bench replay fa5ef49e-cea1-4ee3-8364-a87b68c57042

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench run --run fa5ef49e-cea1-4ee3-8364-a87b68c57042 -s condense
