# Quality / trajectory-preservation bench

Does a context-reduction method **preserve the agent's trajectory**, or change what it does?
Two clearly separated phases — the same split the cost bench uses (`run` → store → `report`):

```
quality run | incremental  ──▶  results/<out>/**    quality report  ──▶  report.html | report.md
 (SPENDS: run agents / replay)      (artifacts)      (PURE DISPLAY: reads artifacts, never spends)
```

**`quality report` never calls a model.** Everything that spends lives in
`quality run` (full) / `quality incremental`.

## `quality run` / `quality incremental` — produce data (spends)

```
uv run minmax-bench quality run           # full = end-to-end trajectory (bare = guided wizard)
    --agent claude-code            # default; codex / opencode = TODO (errors, doesn't fake it)
    --arms condense,headroom       # DEFAULT; vanilla always included (cost-bench names):
                                   #   headroom          = the REGULAR/full product —
                                   #                       token proxy + retrieve loop (CCR)
                                   #   headroom-kompress = token mode, no retrieval (ablation)
    --tasks 5      # N = first N recommended; random:N = seeded sample of the
                   # whole dataset (--seed); a,b,c by name; omitted = 5.
                   # --list-tasks shows everything known locally.
    --dataset terminal-bench/terminal-bench-2-1   # the only validated dataset so far
    --out results/jobs/run
uv run minmax-bench quality incremental   # teacher-forced per-step replay of a recorded session
```

- **`quality run`** (full) — runs the agent end-to-end through [Harbor](https://github.com/laude-institute/terminal-bench)
  (Docker) for `vanilla` + each arm, `--k` repeats → `results/<out>/<arm>-<task>/**`. Needs Docker +
  `uv tool install harbor` + `.env` keys (validated up front — a missing condense key refuses to
  start instead of silently running unauthenticated). `--wall-timeout` is per *trial* (the cell gets
  `wall_timeout × k`); every cell writes `attempted.json` first, so killed trials show up as `⚠ lost`
  in the report instead of vanishing. `--dry-run` prints the Harbor commands without running.
- **`quality incremental`** — teacher-forces one `--session` (or `--swechat` conversation)
  step-by-step through control + each arm, arms in parallel →
  `results/<out>/incremental/<task>-<arm>.jsonl` (paired, cache-aware, no turn-count noise).
  `--task` is required and must match the name you pass `quality report --tasks`. Starts the
  headroom proxy if needed (`--headroom-mode cache|token`) and stops it on exit. Teacher-forced
  replay executes no tools, so CCR's retrieve loop can't engage here — the full headroom product
  (with CCR) belongs to `quality run` with the `headroom` arm.
- **`--milestones`** (full) — runs an LLM judge (temperature 0, arm-blind) over the runs →
  `results/<out>/milestones.json`. Milestones are grounded in a solved vanilla run, which is then
  excluded from vanilla's own coverage scoring.

## `quality report` — display (pure, offline, free)

```
uv run minmax-bench quality report --from results/jobs/run --tasks a,b,c --arms condense,headroom --format {html,md}
```

Reads whatever the generation commands wrote — full run dirs (**length / rework / solve**, each vs the vanilla
noise floor: **✓ overlap / ✗ disjoint**, ≥2 finished runs/arm, attempted-but-unfinished trials
surfaced as `⚠ lost` and counted as unsolved) plus `milestones.json` (found recursively) and
`incremental/*.jsonl` (rendered as **fid** — per-step action agreement, arm next to the control
noise floor — plus **comp** and **$Δ** over the common step set, cold-cache step 0 excluded) —
and renders. Deterministic, no network.

## Files

The code lives in `minmax_bench/quality/` (importable, unit-tested, still pure standard
library on the analysis path).

| file | side | role |
|---|---|---|
| `minmax_bench/quality/generate.py` | generate | the one generation command (full + incremental + milestone judge) |
| `minmax_bench/quality/engine.py` | generate | library: session I/O, request building, scoring, pricing — imported by generate, report (parser only) and `quality incremental` |
| `minmax_bench/quality/report.py` | display | reads artifacts → html/md; never spends |
| `harbor_agents/headroom_ccr_claude_code.py` | generate | self-contained CCR wiring for the `headroom` arm (preserves base MCP servers) |
| `tests/test_quality.py` | — | unit tests for the metric code (`uv run pytest`) |

## Counterfactual replay of a local session

`quality incremental` works on any Claude Code session file, including ones from your own
`~/.claude/projects` — that's the "what if I had used condense?" counterfactual. The ergonomic
front-end (interactive picker, cost preview, cross-model thinking handling, summary table) is:

```bash
uv run minmax-bench quality incremental   # rich replay of your own sessions (wraps this engine)
```

## End to end

```bash
# generate then display (needs Docker + harbor + .env keys)
cp .env.dist .env      # ANTHROPIC_API_KEY, CONDENSE_API_KEY
uv run minmax-bench quality run --out results/jobs/run1 --milestones   # 5 default tasks
uv run minmax-bench quality report --from results/jobs/run1 --tasks 5
```
