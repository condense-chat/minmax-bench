# Opus 4.8 · 64 recorded coding sessions

Condense saved **63.8% in model cost** on 11,712 matched turns, from **$861.28 to $312.20**. Rewrite mode, local token counts and simulated cache pricing. The model prices the recorded workload. No live Opus calls were needed to score the rewrites. These totals exclude optimizer fees.

[Run manifest](run.json) · [Machine-readable report](report.json) · [Dataset coverage](../../data/README.md)

## Results by strategy

| Strategy | Matched turns | Baseline | After | Input tokens saved | Cost saved |
| --- | --- | --- | --- | --- | --- |
| Condense | 11,712 | $861.28 | $312.20 | 72.7% | 63.8% |
| Headroom · cache mode | 11,712 | $861.28 | $878.31 | −2.2% | −2.0% |

The largest reduction came on longer inputs. Under 16k tokens the model cost fell 9.2%. At 400k–800k tokens it fell 73.1%. Headroom in cache mode cost 2.0% more than baseline over the same 11,712 turns.

## Condense by input length

| Baseline input tokens | Matched turns | Baseline | After | Model cost saved |
| --- | --- | --- | --- | --- |
| 0–16k | 1,018 | $11.68 | $10.61 | 9.2% |
| 16k–32k | 1,319 | $26.28 | $17.28 | 34.2% |
| 32k–100k | 3,803 | $145.72 | $70.34 | 51.7% |
| 100k–200k | 2,845 | $226.27 | $81.52 | 64.0% |
| 200k–400k | 2,105 | $297.93 | $91.12 | 69.4% |
| 400k–800k | 622 | $153.39 | $41.33 | 73.1% |

## Reproduce this report

From the repository root, with dependencies installed, run either command. Both read the saved measurements without new model calls.


```sh
uv run minmax-bench report 5c61ab52-8eea-4fee-97a4-5c64ee5344af
uv run minmax-bench replay 5c61ab52-8eea-4fee-97a4-5c64ee5344af
```


[Baseline usage](models/claude-opus-4-8/baseline.json) · [Headroom · cache mode usage](models/claude-opus-4-8/strategies/headroom.json) · [Condense usage](models/claude-opus-4-8/strategies/condense-sync.json)

The stored Condense identifier is `condense-sync`. The [cost guide](../../docs/cost.md) explains matching and cache pricing. For task completion, use the separate [quality benchmark](../../docs/quality.md).
