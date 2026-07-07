# minmax-bench run dbd99004-525f-4973-9e70-ca0c4c26329b

Generated: 2026-07-07T17:52:53+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report dbd99004-525f-4973-9e70-ca0c4c26329b

## Replay the animated evolution

    minmax-bench replay dbd99004-525f-4973-9e70-ca0c4c26329b

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench run --run dbd99004-525f-4973-9e70-ca0c4c26329b -s condense
