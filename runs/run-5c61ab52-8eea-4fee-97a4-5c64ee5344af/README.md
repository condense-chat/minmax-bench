# minmax-bench run 5c61ab52-8eea-4fee-97a4-5c64ee5344af

Generated: 2026-07-23T11:10:48+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report 5c61ab52-8eea-4fee-97a4-5c64ee5344af

## Replay the animated evolution

    minmax-bench replay 5c61ab52-8eea-4fee-97a4-5c64ee5344af

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench cost run --run 5c61ab52-8eea-4fee-97a4-5c64ee5344af -s condense
