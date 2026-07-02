# cost-bench run 202f98bd-a2f1-4390-8307-658b451b7727

Generated: 2026-07-02T14:43:07+00:00

- `run.json` — the run manifest (dataset, models, strategies, per-conversation cache-bust ids).
- `models/<model>/baseline.json` — the uncompressed "original cost" per conversation, per model.
- `models/<model>/strategies/<name>.json` — each strategy's measurement per conversation, per model.
- `report.json` — bucketed metrics, fully derived from the caches above.

Only raw token usage is stored; cost is always recomputed from it, per model.

## Verify / re-price (no spend)

    cost-bench report 202f98bd-a2f1-4390-8307-658b451b7727

## Replay the animated evolution

    cost-bench replay 202f98bd-a2f1-4390-8307-658b451b7727

## Resume (reuse caches, add strategies/models, spend only on what's missing)

    cost-bench run --run 202f98bd-a2f1-4390-8307-658b451b7727 -s condense
