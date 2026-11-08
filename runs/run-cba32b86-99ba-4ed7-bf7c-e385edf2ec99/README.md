# Haiku 4.5 · the same six sessions without a cap

Condense saved **37.3% in model cost** on 2,909 matched turns, from **$56.69 to $35.52**. Historical Haiku run. Model cost below is recomputed from the saved usage on each arm’s matched turns. The stored manifest predates the explicit mode field. These totals exclude optimizer fees.

[Run manifest](run.json) · [Machine-readable report](report.json) · [Dataset coverage](../../data/README.md)

## Results by strategy

| Strategy | Matched turns | Baseline | After | Input tokens saved | Cost saved |
| --- | --- | --- | --- | --- | --- |
| Condense | 2,909 | $56.69 | $35.52 | 53.8% | 37.3% |

The 400k+ bucket saved 53.4% across 130 turns. Below 16k, the 81 matched turns cost 58.6% more. Across all buckets, the matched baseline is $56.69. Compare the capped run to see how removing later turns changes the result.

## Condense by input length

| Baseline input tokens | Matched turns | Baseline | After | Model cost saved |
| --- | --- | --- | --- | --- |
| 0–16k | 81 | $0.16 | $0.25 | −58.6% |
| 16k–32k | 102 | $0.46 | $0.40 | 13.4% |
| 32k–100k | 618 | $4.95 | $4.02 | 18.8% |
| 100k–200k | 886 | $14.45 | $9.76 | 32.4% |
| 200k–400k | 1,092 | $30.75 | $18.33 | 40.4% |
| 400k+ | 130 | $5.92 | $2.76 | 53.4% |

## Reproduce this report

From the repository root, with dependencies installed, run either command. Both read the saved measurements without new model calls.


```sh
uv run minmax-bench report cba32b86-99ba-4ed7-bf7c-e385edf2ec99
uv run minmax-bench replay cba32b86-99ba-4ed7-bf7c-e385edf2ec99
```


[Baseline usage](models/claude-haiku-4-5/baseline.json) · [Condense usage](models/claude-haiku-4-5/strategies/condense-async.json)

The stored Condense identifier is `condense-async`. The [cost guide](../../docs/cost.md) explains matching and cache pricing. For task completion, use the separate [quality benchmark](../../docs/quality.md).
