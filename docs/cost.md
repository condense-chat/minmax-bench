# Cost bench — methodology

**The minimize target.** How many tokens does a context-reduction strategy save on real
agent sessions — and how much of that survives as actual dollars once prompt caching is
accounted for?

```
session ─▶ harness simulator ─▶ [request points] ─▶ strategy (via executor) ─▶ usage ─▶ cost ─▶ bucketed report
```

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

## Datasets

`--dataset` accepts (`minmax_bench/data/loaders.py`):

| spec | source |
|---|---|
| `sample` | built-in offline sample (no keys, no network) |
| `swe-chat:N` | first N conversations of the SWE-chat dataset (needs `uv sync --extra hf`) |
| `claude-code:/path/*.jsonl` | your own Claude Code session files |
| `codex:…` / `opencode:…` | Codex / OpenCode session files |
| `jsonl:…` | pre-normalized session JSONL |

`minmax-bench fetch` downloads/prepares a dataset into the local shared cache up front.

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

## Reference runs (committed, zero spend)

Three reference runs ship in `runs/` — every number is recomputed from stored usage:

- **`run-202f98bd`** — `headroom` vs `headroom-kompress` vs `condense-async` on Haiku 4.5
  over `swe-chat:32` (6 long sessions), truncated to 190k. The clean head-to-head.
- **`run-cba32b86`** — baseline + `condense-async`, untruncated (baseline ~$73.6 on
  replay). Shows savings growing with chain length into the 200k–400k+ bands.
- **`run-5c61ab52`** — `headroom` vs `condense-sync` on Opus 4.8 over the full
  `swe-chat:64` (64 sessions, ~11.7k scored turns) in `--mode rewrite` — zero spend. At
  this scale `condense-sync` saves ~73% tokens / ~64% cost (~$549 off an ~$861 baseline)
  while `headroom` nets slightly negative overall.

### Illustrative findings

| strategy | cost saved (truncated 190k) | notes |
|---|---|---|
| `condense-async` | **28%** (37% untruncated) | scales *up* with chain length — 53% saved in the 400k+ band |
| `headroom` (cache mode) | ~14% | preserves the prefix cache |
| `headroom-kompress` (token mode) | ~2% | rewrites history, torching the cache — token savings barely survive to cost |

Illustrative of *this* dataset/model, not universal — rerun on your own sessions.

## Caveats

- Offline token counts use `tiktoken` (an approximation for Claude); `--count api` or
  proxy mode supply exact provider numbers when you need them.
- Cost tables answer "how much does it save"; "does it still work" is the
  [quality bench](quality.md) — read savings only where trajectory preservation is
  confirmed (and where the task actually cleared the compaction gate).
