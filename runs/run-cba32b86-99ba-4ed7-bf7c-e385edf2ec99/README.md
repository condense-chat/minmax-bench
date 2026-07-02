# cost-bench run cba32b86-99ba-4ed7-bf7c-e385edf2ec99

Generated: 2026-07-02T14:43:07+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    cost-bench report cba32b86-99ba-4ed7-bf7c-e385edf2ec99

## Replay the animated evolution

    cost-bench replay cba32b86-99ba-4ed7-bf7c-e385edf2ec99

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    cost-bench run --run cba32b86-99ba-4ed7-bf7c-e385edf2ec99 -s condense
