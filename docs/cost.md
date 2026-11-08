# Price a recorded session

For each recorded turn, the harness rebuilds what the agent would send. It compares each strategy with the uncompressed request at the **same session and turn**, then groups the pair by the original input length.

## Why token savings and cost savings differ

In the 190k-capped Haiku run, Headroom Kompress removed **11.8% of input tokens**. Model cost fell only **1.8%**. Rewriting earlier messages forced new cache writes on later calls. [See all three strategies](../runs/run-202f98bd-a2f1-4390-8307-658b451b7727/README.md).

| Measurement | What the harness uses | What it costs to generate |
| --- | --- | --- |
| Proxy | Usage returned by the upstream model with the output capped | Provider usage or subscription quota |
| Rewrite | Transformed request, local token counts, simulated cache reads and writes | No upstream model calls for scoring. The rewrite service may require access. |
| Saved report | Usage already in a run directory, priced by model | No new model calls |

## Read a result

Start with matched turns, then compare **baseline dollars**, **strategy dollars** and **input tokens**. Savings are `(baseline − strategy) / baseline`. A negative value means model cost rose. The all-turns percentage uses summed costs, so expensive turns carry more weight. Offline Claude token counts are approximate.


```sh
uv run minmax-bench report 5c61ab52-8eea-4fee-97a4-5c64ee5344af
```

[Opus breakdown](../runs/run-5c61ab52-8eea-4fee-97a4-5c64ee5344af/README.md) · [Dataset coverage](../data/README.md) · [Check task quality](quality.md)

## Running

```bash
uv run minmax-bench cost run -d swe-chat:32 \
  -s headroom -s headroom-kompress -s condense-async \
  -m claude-haiku-4-5 --token-budget 190k
```

Bare `cost run` launches the guided wizard; `-y/--yes` skips it. The main knobs:

| flag | meaning |
|---|---|
| `-d/--dataset` | dataset spec (default `sample`) |
| `-s/--strategy` | strategy name(s), repeatable (default `headroom condense`) |
| `-m/--model` | model id(s), repeatable (default haiku) |
| `--mode` | run-wide `proxy` \| `rewrite` (see above) |
| `--transport` | run-wide `anthropic` \| `bedrock` |
| `--token-budget` | truncate each conversation at the first turn whose chain reaches N tokens |
| `-n/--limit` | max conversations to load |
| `--longest N` | keep only the N longest conversations |
| `--max-points` | cap turns per conversation (bounds cost) |
| `--edges` | comma-separated bucket upper edges (tokens) |
| `--count` | baseline counting: `local` (tiktoken) or `api` (exact, free) |
| `--run <uuid>` | resume an existing run, reusing its caches — spend only on what's missing |
| `--refresh` | comma list of strategies (or `all`/`baseline`) to recompute, ignoring cache |
| `--live/--no-live` | animate the live dashboard while measuring |
| `--setup` | auto-install/start the local tools the strategies need (`auto`/`none`/list) |
| `--runs-dir` | root for `run-<uuid>` dirs |

## Run store, resume, and re-scoring

Each run writes `runs/run-<uuid>/`, self-describing and replayable:

```
run.json                          manifest (dataset, strategies, mode/transport, models)
report.json                       bucketed metrics
models/<model>/baseline.json      raw per-turn baseline usage
models/<model>/strategies/<s>.json  raw per-turn strategy usage
README.md                         human summary
```

Because raw usage is stored, everything downstream is free and offline:

```bash
uv run minmax-bench report <uuid>            # recompute + print the bucket tables
uv run minmax-bench report <uuid> --edges …  # re-bucket without re-spending
uv run minmax-bench replay <uuid>            # animated context + cost evolution
uv run minmax-bench runs                     # list stored runs
```


<details>
<summary>Harness reconstruction, cache assumptions, and input limits</summary>

## Harness simulator

A recorded session is chopped into the exact sequence of model calls a real harness would
make: each assistant turn is one call whose request prefix is every preceding message
(user → tool_use → tool_result → assistant → …). The replay is deterministic — the same
transcript always yields the same request points (`minmax_bench/harness/`).

## Scoring and bucketing

Every strategy is compared to the **baseline** (no compression) for the same
`(session, turn)`, then **bucketed by the baseline input-chain length** — so every
strategy is bucketed identically and comparisons are apples-to-apples. Empty buckets are
dropped; the `ALL` row aggregates. Bucket edges are re-adjustable after the fact
(`report <uuid> --edges 16000,32000,100000,200000`) because raw per-turn usage is stored,
not just the tables.

Per strategy the report shows two tables:

- **tokens saved** — baseline vs strategy prompt tokens, per cache tier and percent.
- **cost saved (USD)** — the same after cache-aware pricing.

## Caching is modeled, on purpose

Savings that destroy the prompt cache aren't real savings. Proxy strategies rewrite
*historical* turns, which can invalidate the cache from the edit point on. The bench
therefore:

- reads the upstream's real, cache-aware `usage` (cache-read vs cache-write tiers) in
  proxy mode, or
- simulates the same incremental cache model locally in rewrite mode
  (`minmax_bench/tokens.py`, `minmax_bench/chain.py`),

and prices both with cache-aware per-model pricing (`minmax_bench/pricing.py`). That's why
the **cost** tables differ from the raw-token tables — a strategy can save tokens yet cost
*more* if it trades cheap cache-reads for dearer cache-writes.

## The 200k context cap

Models have a hard context ceiling (Haiku: 200k). A turn whose prompt exceeds it is
rejected, so runs can be **truncated to a token budget** (`--token-budget 190k`) to keep
every scored turn valid — otherwise the deepest turns of long sessions drop out and skew
the buckets.

## Measurement: mode × transport

How a strategy is *measured* is a run-wide choice, orthogonal to the strategy itself
(full mechanics in [architecture.md](architecture.md)):

- **`--mode proxy`** (default) — send the real request through the strategy's proxy to the
  real upstream, output capped to a minimal `max_tokens`, and read the provider's real
  cache-aware `usage`. Real usage, real money: you pay for the *input* tokens of every
  replayed turn.
- **`--mode rewrite`** — ask the strategy's rewrite function for the transformed request
  body and cost it **offline** from locally counted tokens plus the simulated cache model.
  Zero model spend — this is how a 64-session / ~11.7k-turn run costs nothing to measure.
- **`--transport anthropic|bedrock`** — where the bench's own direct model traffic lands
  (default from `UPSTREAM_VIA` in `.env`). With `--mode proxy`, `bedrock` *emulates* each
  proxy: the rewritten body is fetched locally, invoked on Bedrock, and Bedrock's real
  usage reported (headroom can proxy to Bedrock directly; condense is emulated).

The baseline is always the **noop** path: recorded usage from the transcript, or a local
token count (`--count local`, tiktoken) or the free exact Anthropic
`count_tokens` API (`--count api`).

</details>
