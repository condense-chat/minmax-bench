# minmax-bench run 3064bdc2-ac51-4517-ba7a-a67fbee3e5b6

Generated: 2026-07-18T08:10:04+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    minmax-bench report 3064bdc2-ac51-4517-ba7a-a67fbee3e5b6

## Replay the animated evolution

    minmax-bench replay 3064bdc2-ac51-4517-ba7a-a67fbee3e5b6

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    minmax-bench run --run 3064bdc2-ac51-4517-ba7a-a67fbee3e5b6 -s condense
