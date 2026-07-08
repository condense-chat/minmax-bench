# minmax-bench

**A token- and cost-savings benchmark for context-reduction proxies.**

`minmax-bench` takes real coding-agent sessions, replays them turn-by-turn the way
a harness actually calls the model, drives each turn through a context-reduction
**strategy**, and reports how many **tokens** and how much **USD** each strategy
saves — bucketed by input-chain length and accounting for prompt caching.

It compares proxies like [`headroom`](https://pypi.org/project/headroom-ai/) and
`condense` on an equal footing, using only their **public** interfaces — no access
to any strategy's source is required.

> **Scope.** This repo measures both halves of the trade-off: **cost** (tokens and
> dollars, above) and **quality** — whether a compressed session stays *correct*, i.e.
> the agent still does the same work. The quality half is the
> [trajectory-preservation bench](#quality--trajectory-preservation-bench) below; the
> cost numbers are only trustworthy where it confirms the trajectory is preserved.

## Methodology

```
session ─▶ harness simulator ─▶ [request points] ─▶ strategy (via executor) ─▶ usage ─▶ cost ─▶ bucketed report
```

- **Harness simulator.** A recorded session is chopped into the exact sequence of
  model calls a real harness would make: each assistant turn is one call whose
  request prefix is every preceding message (user → tool_use → tool_result →
  assistant → …). Replayed deterministically from the transcript.
- **Scored against the uncompressed baseline.** Every strategy is compared to the
  *baseline* (no compression) for the same `(session, turn)`, then **bucketed by
  the baseline input-chain length** so every strategy is bucketed identically and
  comparisons are apples-to-apples.
- **Caching is modeled, on purpose.** Savings that destroy the prompt cache aren't
  real savings. Proxy strategies rewrite *historical* turns, which can invalidate
  the cache from the edit point on. The proxy executor reads the upstream's real,
  cache-aware `usage` (cache-read vs cache-write tiers); cost uses cache-aware
  per-model pricing. That's why the **cost** tables differ from the raw-token
  tables — a strategy can save tokens yet cost more if it trades cheap cache-reads
  for dearer cache-writes.
- **The 200k context cap.** Models have a hard context ceiling (Haiku: 200k). A
  turn whose prompt exceeds it is rejected, so runs can be **truncated to a token
  budget** (`--token-budget 190k`) to keep every scored turn valid — otherwise the
  deepest turns of long sessions drop out and skew the buckets.

Two measurement paths (executors):

- **proxy** — send the real request to the strategy's Anthropic/OpenAI-compatible
  proxy, capped to 1 output token, and read the upstream's real cache-aware
  `usage` (reflecting the proxy's server-side compression). Used for `headroom`,
  `condense`, `upstream`, `gemini`.
- **noop** — the uncompressed baseline (recorded usage, or a local token count).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync                 # core
uv sync --extra hf      # + HuggingFace loaders for the SWE-chat dataset
cp .env.dist .env       # then fill in only the keys for what you run
```

### What you need, per feature

| feature | credentials |
|---|---|
| `report` / `replay` reference runs, offline demo, `run --dataset sample` | **none** |
| your sessions' *recorded* cost (the counterfactual `recorded` row) | **none** |
| replays: counterfactual, incremental, cost-bench proxy runs | `ANTHROPIC_API_KEY`, **or a Claude Code login** (see below) |
| the condense arm | + `CONDENSE_API_KEY` |
| `--mode full` (Harbor) | Docker + `uv tool install harbor` + the above |

**No API key? Your Claude Code subscription works.** When `ANTHROPIC_API_KEY` is unset, the
bench authenticates the way Claude Code itself does, using your own login: it checks
`CLAUDE_CODE_OAUTH_TOKEN` (mint one with `claude setup-token`), then
`~/.claude/.credentials.json`, then the macOS keychain entry Claude Code maintains. Replays
are Claude Code-shaped requests over your own sessions; usage draws on your plan, the token
never leaves your machine except toward the endpoint an arm points at, and the run banner
says `auth: Claude Code subscription` so it's never implicit.

## Quick start (offline, no keys)

The richest no-spend demo is re-scoring and replaying the committed reference runs
(see [Checking the cached replays](#checking-the-cached-replays)):

```bash
uv run minmax-bench report 202f98bd-a2f1-4390-8307-658b451b7727   # per-bucket tables
uv run minmax-bench replay 202f98bd-a2f1-4390-8307-658b451b7727   # animated evolution
uv run minmax-bench strategies                                    # list strategies
uv run minmax-bench info                                          # configured keys/endpoints
```

A `run` on the built-in sample computes the baseline offline (every comparison
strategy is a proxy that needs keys — see below — so those turns are skipped
without them):

```bash
uv run minmax-bench run --dataset sample
```

## Running against real strategies

```bash
# headroom (proxy): pip install "headroom-ai[proxy]" then run the proxy
headroom proxy --port 8787 --mode cache      # or --mode token (max compression)
uv run minmax-bench run -d swe-chat:32 -s headroom -m claude-haiku-4-5

# condense (proxy): creds are read from the local `dense` CLI config (~/.config/dense)
uv run minmax-bench run -d swe-chat:32 -s condense-async -m claude-haiku-4-5

# gemini (direct, OpenAI-compatible chat/completions): set GEMINI_API_KEY in .env
uv run minmax-bench run -d swe-chat:32 -s gemini -m gemini-3.1-flash-lite
```

> **Proxy runs cost real money.** Each request hits the real upstream (with
> `max_tokens=1`), so you pay for the *input* tokens of every replayed turn. Use
> `--limit/-n` to cap sessions and `--token-budget` to cap chain length while
> iterating.

## Checking the cached replays

`runs/` ships two committed **reference runs** you can inspect with **zero spend** —
every number is recomputed from stored token usage, and the animated evolution is
replayed from the same cache. No API keys or network needed.

```bash
# re-score from the stored usage (prints the per-bucket tables):
uv run minmax-bench report 202f98bd-a2f1-4390-8307-658b451b7727
uv run minmax-bench report cba32b86-99ba-4ed7-bf7c-e385edf2ec99

# replay the animated context + cost evolution:
uv run minmax-bench replay 202f98bd-a2f1-4390-8307-658b451b7727
uv run minmax-bench replay cba32b86-99ba-4ed7-bf7c-e385edf2ec99
```

- **`run-202f98bd`** — `headroom` vs `headroom-kompress` vs `condense-async` on
  Haiku 4.5 over `swe-chat:32` (6 long sessions), **truncated to 190k** so every
  scored turn is under the 200k cap. This is the clean head-to-head.
- **`run-cba32b86`** — baseline + `condense-async`, **untruncated**, over the full
  long sessions (baseline totals **~$73.6** on replay). Shows how savings grow with
  chain length into the 200k–400k+ bands.

What to look at: each strategy gets a **tokens-saved** and a **cost-saved (USD)**
table, bucketed by input-chain length, with an `ALL` aggregate row. Cost-saved
trails token-saved because compression trades cheap cache-reads for dearer
cache-writes — that gap is the point of modeling the cache.

Each run directory is self-describing: `run.json` (manifest), `report.json`
(bucketed metrics), `models/<model>/baseline.json` + `.../strategies/<name>.json`
(raw per-turn usage), and a `README.md`.

## Retesting yourself

Produce your own run — each is written under `runs/run-<uuid>/` and is itself
replayable and re-scorable exactly like the reference runs above:

```bash
uv run minmax-bench run -d swe-chat:32 \
  -s headroom -s headroom-kompress -s condense-async \
  -m claude-haiku-4-5 --token-budget 190k --json out/run.json
```

- `--token-budget 190k` keeps every turn under Haiku's 200k cap so long sessions
  don't drop out of the buckets.
- `-n/--limit N` caps how many conversations load (cheaper iteration).
- `--run <uuid>` resumes an existing run, reusing its caches and spending only on
  what's missing.
- Re-price or re-bucket any finished run without re-spending:
  `uv run minmax-bench report <uuid> --edges 16000,32000,100000,200000`.

## Reading the output

Per strategy, two tables bucketed by input-chain length:

- **tokens saved** — baseline vs strategy prompt tokens, per-tier and percent.
- **cost saved (USD)** — the same after cache-aware pricing.

Empty buckets are dropped; the `ALL` row is the aggregate. `--json PATH` writes the
full bucket stats for further analysis.

### Illustrative findings (from the reference runs)

| strategy | cost saved (truncated 190k) | notes |
|---|---|---|
| `condense-async` | **28%** (37% untruncated) | scales *up* with chain length — 53% saved in the 400k+ band |
| `headroom` (cache mode) | ~14% | preserves the prefix cache |
| `headroom-kompress` (token mode) | ~2% | rewrites history, torching the cache — token savings barely survive to cost |

Take these as illustrative of *this* dataset/model, not universal — rerun on your
own sessions to compare.

## Quality / trajectory-preservation bench

The cost tables above assume compaction **preserves the trajectory** — the agent does the same
work in about the same number of steps. That assumption is load-bearing: if a method makes the
agent wander or take more turns, the "savings" is partly illusory (more turns = more cost). This
bench tests it by running the **full agent** (not a replay) under each method on real coding tasks
([Terminal-Bench](https://github.com/laude-institute/terminal-bench) via
[Harbor](https://github.com/laude-institute/harbor)) and comparing the trajectories to a
**vanilla-vs-vanilla noise floor**. The bar is not "identical to control" (two vanilla runs already
differ) — it's **"within the vanilla-vs-vanilla spread."**

The two modes test **two different claims** and are deliberately complementary:
**full** runs catch *behavioral* changes (turn-count inflation, induced planning, solve rate) but
can't attribute them; **incremental** (teacher-forced) runs catch *informational* loss (does
compressing this exact history change the next decision?) but are structurally blind to behavioral
effects on short sessions where threshold-gated compaction never fires. A method can pass one and
fail the other — both outcomes are informative.

Per task, vanilla `k≈3` sets the floor; each method's runs are tested against it, axis by axis:

| axis | question |
|---|---|
| **length** | does compaction change the # of steps? *(the load-bearing axis)* |
| **rework** | does it re-fetch info it already had? *(compaction amnesia; range-aware — post-edit re-inspection counts as verification, not rework)* |
| **milestone** | does it accomplish the same subgoals? *(approach-agnostic, LLM-judged, arm-blind; the reference run is excluded from vanilla's own coverage)* |
| **solve** | does it still pass the verifier? *(trials that crash or hit the wall timeout count as failures — `⚠ lost` in the table — not as missing data)* |
| **fid** | teacher-forced per-step action agreement, shown next to the **control replay's** agreement (the noise floor) — only the gap below the floor is signal |

**When does context-reduction actually trigger?** Both products are gated to act only where
reduction pays off, and a small task structurally cannot exercise them: condense compacts the
*whole conversation* only past an internal size threshold (its savings appear in the 100k+ bands),
and headroom's token mode compresses *individual tool outputs* only when they exceed ~200 tokens
(`min_tokens_to_crush`, v0.28 defaults) and score as stale/irrelevant. The report therefore shows
vanilla's **peak context** per task and marks tasks **⊘** when it stays under the compaction gate
(`--ctx-gate`, default 50k): on a ⊘ task no compaction/compression fired, so a length ✗ there
measures the arm's *wiring and behavioral* side-effects, not compaction damage. (One such effect
we measured: Claude Code composes a ~8-9k-token-larger request whenever `ANTHROPIC_BASE_URL` is
non-default, a flat confound shared by every proxy arm — isolating it cleanly needs a passthrough
vanilla-proxy control arm, which is on the roadmap.) Compaction *quality* claims must come from
tasks whose vanilla runs clear the gate — the long half of the curated list.

A verdict is **✓** if the method's band *overlaps* vanilla's, **✗** if disjoint — and needs
**≥ 2 finished runs per arm** (a single run can't be told from a fluke; this kills the k=1 mirage
where length and cost swing wildly). Read ✓ honestly: with k≈3, band overlap only detects *gross*
divergence — it means "no detectable divergence at this k", not statistical equivalence.
Arm names match the cost bench's strategy matrix so quality verdicts and cost numbers join
by name: `headroom` = cache-mode proxy; **`headroom-ccr`** = token-mode proxy *plus* the MCP
retrieve loop — the full Compress-Cache-Retrieve product, which is headroom's intended
token-mode deployment and therefore the fair token-mode arm; `headroom-kompress` = token-mode
compression *without* retrieval, kept only as an ablation (it matches the cost-bench strategy
of the same name, but judging headroom's quality by it would be a strawman).
Full tooling + reproduction: [`scripts/README.md`](scripts/README.md).

**Offline demo — no keys, no Docker (~30 s)** — a tiny sample of real recorded runs ships in
`runs/quality-sample/`:

```bash
python3 scripts/report.py --from runs/quality-sample --tasks kv-store-grpc --arms condense
```

→ `kv-store-grpc  condense  2/2 · 2/2  6[5-6]  12[11-14]  ✗ DIVERGES  ✓ OK` — on this task condense
~doubles the trajectory (both still solve). That's the whole point, visible from a fresh clone.

**Generate your own** (Docker + `uv tool install harbor` + `.env` keys):

```bash
# `quality run` is the analog of the cost bench's `run` — bare, it launches a
# guided wizard (arms, tasks, model, k, budget, preflight, cost ceiling → confirm):
uv run minmax-bench quality run
# or drive it with flags (skips the wizard); the default is the first 5 recommended
# tasks (--list-tasks shows all; --tasks 2, --tasks random:8 --seed 42, or a,b):
uv run minmax-bench quality run -m claude-haiku-4-5 --tasks 5 --milestones --out results/jobs/run1
uv run minmax-bench quality report --from results/jobs/run1 --tasks 5     # pure display, free
# also: `quality incremental --session <f> --task <t>` (teacher-forced per-step), `quality judge`.
# The scripts/ wrappers (`python3 scripts/generate.py …`) still work identically.
```

`generate` prints its plan up front (arms × tasks × k = N trials + the cost ceiling from
`--budget-usd`) before spending anything. Defaults: k=4 trials per arm and k+1 for the
vanilla baseline (the noise floor every arm is judged against — the extra run sharpens
every verdict). `--dataset` selects the harbor dataset; only
`terminal-bench/terminal-bench-2-1` is validated so far.

The quality bench is **pure standard library** — nothing to install to *analyze* runs; Docker + Harbor
are only needed to *generate* them.

### Counterfactual: replay *your own* Claude Code session

How would one of your real sessions have played out under condense? Pick any session from
`~/.claude/projects` and teacher-force it step-by-step through each arm, next to a control replay
(the noise floor). No Docker, no Harbor — just API keys. It shows a cost estimate and asks before
spending:

```bash
uv run minmax-bench counterfactual                       # interactive picker + confirm
uv run minmax-bench counterfactual ~/.claude/projects/<proj>/<id>.jsonl --arms condense -n 30
```

Per arm you get: **same-action agreement** (did it still make the same next move? read it against
the control floor, not against 100%), **avg context tokens** and **$ vs control** — plus a
**recorded** row showing what those turns *actually* consumed when the session ran, so the table
is both a counterfactual and a backtest. For the cost-only backtest across many sessions, the
`minmax-bench run` wizard's dataset step accepts a bare `claude-code` and opens the same picker
(replays your sessions turn-by-turn through each strategy's proxy). Scriptable
equivalent: `python3 scripts/generate.py --mode incremental --session <file> --task <name>`.
Note the condense arm sends your session content to `api.condense.chat`.

### Quality findings (so far)

Reported impartially, including results unfavorable to condense.

- **Preservation mostly holds** — on 8/9 tasks with enough runs, condense's trajectory length is
  within the vanilla-vs-vanilla spread (no detectable divergence at k≈3), redundant re-work is zero,
  and the same subgoals are reached. A per-turn cost claim is sound *where both arms solve reliably.*
- **One real exception — short tasks (`kv-store`):** condense consistently ~doubles the trajectory
  (5 → 12 steps) by **inducing todo-tool planning + verification** — behavioral, *not* amnesia. This
  *explains* that task's large full-run cost gap (which looked like noise at k=1).
- **Token savings ≠ dollar savings** — compaction busts the prompt cache (the same effect behind the
  `headroom-token` cost result above); verified two ways (teacher-forced replay + real runs).

## Layout

```
minmax_bench/
  models.py        normalized session/message/usage model
  harness/         session -> request points (harness simulator)
  strategies/      strategy registry: noop, upstream, headroom, condense, gemini
  executors/       proxy | noop measurement
  providers.py     normalized -> Anthropic/OpenAI request bodies (+ cache breakpoints)
  chain.py         exact cache-aware costing of a full reconstructed chain
  tokens.py        local token counting + incremental cache model
  pricing.py       cache-aware per-model cost
  report/          bucketing + rich tables + JSON
  data/            dataset loaders + offline sample
  cli.py           `minmax-bench` entrypoint
runs/              committed reference runs (replayable, no spend)

minmax_bench/quality/   quality / trajectory-preservation bench (pure stdlib; see scripts/README.md)
  generate.py      GENERATE data (spends): --mode full|incremental, --arms, --agent
  engine.py        teacher-forced replay engine (session I/O, scoring, pricing)
  report.py        DISPLAY (pure, never spends): reads artifacts -> html/md
scripts/           thin `python3 scripts/{generate,report}.py` wrappers around the above
harbor_agents/     custom Harbor agent (self-contained headroom-CCR wiring)
runs/quality-sample/  tiny bundled recorded runs for the offline quality demo
```

## Status / caveats

- Offline token counts use `tiktoken` (an approximation for Claude); the proxy
  executor supplies exact provider numbers when you need them.
- The cost tables answer "how much does it save"; "does it still work" is the
  [quality / trajectory-preservation bench](#quality--trajectory-preservation-bench)
  above — read cost savings only where the quality bench confirms preservation
  (and where the task actually cleared the compaction gate, ⊘).
