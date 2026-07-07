# minmax-bench run 12f48cf2-ddb0-4b9e-b4b9-1efcfbab3df4

Generated: 2026-07-07T17:49:15+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report 12f48cf2-ddb0-4b9e-b4b9-1efcfbab3df4

## Replay the animated evolution

    minmax-bench replay 12f48cf2-ddb0-4b9e-b4b9-1efcfbab3df4

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench run --run 12f48cf2-ddb0-4b9e-b4b9-1efcfbab3df4 -s condense
