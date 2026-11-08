# Haiku 4.5 · three strategies under a 190k cap

Condense saved **28.0% in model cost** on 1,600 matched turns, from **$18.17 to $13.08**. Historical Haiku run. Model cost below is recomputed from the saved usage on each arm’s matched turns. The stored manifest predates the explicit mode field. These totals exclude optimizer fees.

[Run manifest](run.json) · [Machine-readable report](report.json) · [Dataset coverage](../../data/README.md)

## Results by strategy

| Strategy | Matched turns | Baseline | After | Input tokens saved | Cost saved |
| --- | --- | --- | --- | --- | --- |
| Condense | 1,600 | $18.17 | $13.08 | 50.7% | 28.0% |
| Headroom · cache mode | 1,573 | $17.66 | $15.22 | 15.2% | 13.8% |
| Headroom · Kompress | 1,571 | $17.63 | $17.31 | 11.8% | 1.8% |

The arm denominators differ. Condense has 1,600 matched turns, Headroom cache mode 1,573, and Kompress 1,571. Each percentage uses the baseline for that arm’s matched turns. An exact comparison across all three would require intersecting their turn IDs.

## Condense by input length

| Baseline input tokens | Matched turns | Baseline | After | Model cost saved |
| --- | --- | --- | --- | --- |
| 0–16k | 81 | $0.16 | $0.25 | −58.6% |
| 16k–32k | 102 | $0.46 | $0.40 | 13.4% |
| 32k–100k | 618 | $4.95 | $4.02 | 18.8% |
| 100k–200k | 799 | $12.60 | $8.40 | 33.3% |

## Reproduce this report

From the repository root, with dependencies installed, run either command. Both read the saved measurements without new model calls.


```sh
uv run minmax-bench report 202f98bd-a2f1-4390-8307-658b451b7727
uv run minmax-bench replay 202f98bd-a2f1-4390-8307-658b451b7727
```


[Baseline usage](models/claude-haiku-4-5/baseline.json) · [Headroom · cache mode usage](models/claude-haiku-4-5/strategies/headroom.json) · [Headroom · Kompress usage](models/claude-haiku-4-5/strategies/headroom-kompress.json) · [Condense usage](models/claude-haiku-4-5/strategies/condense-async.json)

The stored Condense identifier is `condense-async`. The [cost guide](../../docs/cost.md) explains matching and cache pricing. For task completion, use the separate [quality benchmark](../../docs/quality.md).
