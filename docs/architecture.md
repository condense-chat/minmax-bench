# Architecture — strategies × mode × transport

How a benchmark run is assembled: what a **strategy** is, how the run-wide **mode** and
**transport** axes change how it's measured, and which **executor** ends up doing the
work.

```
                       run-wide axes
                    ┌────────────────┐
                    │  --mode        │  proxy | rewrite
                    │  --transport   │  anthropic | bedrock
                    └───────┬────────┘
                            ▼
 Strategy matrix ──▶ resolve(mode, transport) ──▶ ResolvedStrategy(kind, endpoint, headers)
 (name = runner+config)                                   │
                                                          ▼
                                                  executor for `kind`
                                    proxy | rewrite | rewrite_capture | rewrite_invoke | noop
```

## The strategy matrix

The strategy set is a flat, ordered **matrix** (`minmax_bench/strategies/matrix.py`).
Each entry pairs a reusable **runner** (the implementation of a technique) with a
**config** (its variant — e.g. condense's dense profile + sync/async mode). Add a row to
benchmark a new variant; the matrix drives the run, the report rows, and the wizard.

| matrix entry | runner | config | notes |
|---|---|---|---|
| `baseline` | noop | — | mandatory; always runs; every strategy is scored against it |
| `upstream` | upstream | — | direct call to the provider — a live, cache-aware baseline |
| `headroom` | headroom | `mode=cache` | freezes prior turns to maximize prefix-cache hits |
| `headroom-kompress` | headroom | `mode=token` | Kompress rewrites prior turns for max compression |
| `condense-sync` | condense | `mode=sync` | blocks the request until compaction lands |
| `condense-async` | condense | `mode=async` | background compaction, paced by realistic think time (`think_secs_per_output_token`) so async savings are measured honestly |

A strategy is **only the compression technique**. *How* it is measured is resolved at
run time from the two run-wide axes.

## The mode axis — `proxy` | `rewrite`

- **proxy** — the real request goes through the strategy's proxy, which forwards to the
  upstream. Real cache-aware `usage`, real money.
- **rewrite** — the bench asks the strategy's **rewrite function** for the transformed
  request body and costs it offline: local token counts (`tokens.py`) + an exact
  incremental cache simulation over the reconstructed chain (`chain.py`) + cache-aware
  pricing (`pricing.py`). Zero model spend — large datasets become affordable.

A runner advertises `supports_rewrite`; combinations that can't work raise
`UnsupportedCombo` at preflight (e.g. `upstream` under rewrite — a direct call has
nothing to rewrite).

## The transport axis — `anthropic` | `bedrock`

Where the **bench's own direct model traffic** lands (`minmax_bench/transport.py`).
A transport only applies where the bench holds the request — the `upstream` strategy and
the rewrite-invoke emulation; strategy proxies normally make their own upstream calls.

- **anthropic** — `api.anthropic.com` (or `ANTHROPIC_BASE_URL`). Auth is resolved by
  `auth.py`: `ANTHROPIC_API_KEY` if set, otherwise the **Claude Code subscription**
  login (`CLAUDE_CODE_OAUTH_TOKEN` → `~/.claude/.credentials.json` → macOS keychain),
  forceable per-run with `--auth api-key|subscription`.
- **bedrock** — AWS Bedrock's Anthropic-compatible `/anthropic/v1/messages` endpoint
  (`bedrock.py`): bearer token minted from AWS creds, model ids mapped to
  inference-profile ids (pricing keeps the session's model).

## How the combination resolves — executor kinds

`Strategy.resolve(mode, transport)` yields the executor-facing `kind`
(`strategies/base.py`, `strategies/runners.py`):

| runner | `proxy` + `anthropic` | `rewrite` | `proxy` + `bedrock` |
|---|---|---|---|
| noop (`baseline`) | `noop` | `noop` | `noop` |
| `upstream` | `proxy` (direct) | ✗ `UnsupportedCombo` | `proxy` (direct to Bedrock) |
| headroom | `proxy` | `rewrite_capture` — headroom has no rewrite API, but honors `x-headroom-base-url` as a per-request upstream override, so the bench points it at a local **capture sink**: the forwarded body *is* the rewrite | `proxy` — the same override header pointed straight at Bedrock with bearer auth (true proxying) |
| condense | `proxy` | `rewrite` — `x-condense-function: rewrite` returns the rewritten body | `rewrite_invoke` — **proxy emulation**: fetch the rewritten body here, invoke Bedrock with it, report Bedrock's real usage (the proxy itself can't reach Bedrock) |

The executors (`minmax_bench/executors/`) then do the measuring:

- **`proxy.py`** — sends the (possibly proxied) request with a minimal output cap and
  reads the provider's real cache-aware `usage`; paces requests through the shared
  endpoint gate (`gate.py`) and honors per-strategy think time.
- **`rewrite.py`** — the offline path: rewritten body → local counting → simulated
  incremental cache → cost. Also hosts the capture sink (`rewrite_capture`) and the
  Bedrock invoker (`rewrite_invoke`).
- **`noop.py`** — the baseline: recorded usage from the transcript, or a local/API token
  count.

## Module map

```
minmax_bench/
  models.py         normalized session/message/usage model
  harness/          session → request points (harness simulator)
  strategies/       base.py (Strategy/Runner/Mode/Transport model) · matrix.py (the rows) · runners.py (noop, upstream, headroom, condense)
  executors/        proxy | rewrite | noop measurement
  providers.py      normalized → Anthropic/OpenAI request bodies (+ cache breakpoints)
  chain.py          exact cache-aware costing of a full reconstructed chain
  tokens.py         local token counting + incremental cache model
  pricing.py        cache-aware per-model cost
  transport.py      upstream transports (anthropic | bedrock) for the bench's own calls
  bedrock.py        AWS Bedrock endpoint, bearer minting, model-id mapping
  auth.py           Anthropic auth resolution: API key or Claude Code subscription OAuth
  dense.py          condense creds from the local `dense` CLI config (~/.config/dense)
  config.py         runtime settings from environment/.env
  catalog.py        selectable model catalog for the guided TUI
  runner.py         orchestration: load → simulate → measure (model × strategy × conversation)
  runstore.py       per-run/model/conversation measurement store (three cache tiers)
  cache.py          persistent measurement cache (proxy runs cost money; counting costs time)
  gate.py           shared self-adjusting per-endpoint rate gate
  dashboard.py      live animated terminal dashboard for a run
  interactive.py    guided setup/run wizards (TUI)
  preflight.py      dependency preflight for a run
  provision.py      auto-provision the local tools a run needs (headroom, dense)
  counterfactual.py rich front-end of `quality incremental` (picker, cost preview, tables)
  tls.py            SSL context for the raw urllib call sites
  report/           bucketing + rich tables + JSON
  data/             dataset loaders (sample, swe-chat, claude-code, codex, opencode, jsonl)
  cli.py            `minmax-bench` entrypoint
  quality/          quality bench (generate.py, engine.py, report.py, passthrough.py, paths.py)
harbor_agents/      custom Harbor agent (self-contained headroom-CCR wiring)
runs/               committed reference runs (replayable, no spend); quality runs auto-mint under runs/quality/
```

## Where the quality bench plugs in

The quality bench reuses the same auth (`auth.py`), condense creds (`dense.py`), pricing,
and provisioning, but drives a **real agent** (Claude Code via Harbor) instead of the
harness simulator — arms are wired via `ANTHROPIC_BASE_URL` (proxy arms), the passthrough
forwarder (`quality/passthrough.py`, the wiring-confound control), or the CCR Harbor agent
(`harbor_agents/`). Methodology: [quality.md](quality.md).
