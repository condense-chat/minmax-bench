# Quality / trajectory-preservation bench

Does a context-reduction method **preserve the agent's trajectory**, or change what it does?
Two clearly separated phases — the same split the cost bench uses (`run` → store → `report`):

```
generate.py  ──▶  results/<out>/**            report.py  ──▶  report.html | report.md
 (SPENDS: run agents / replay)   (artifacts)   (PURE DISPLAY: reads artifacts, never spends)
```

**`report.py` never calls a model.** Everything that spends lives in `generate.py`.

## `generate.py` — produce data (spends)

```
generate.py --mode {full,incremental}     # full = end-to-end trajectory; incremental = teacher-forced per-step
            --agent claude-code            # default; codex / opencode = TODO (errors, doesn't fake it)
            --arms condense,headroom       # vanilla baseline always included; headroom = cache mode,
                                           # headroom-kompress = token mode (cost-bench names)
            --tasks a,b,c  --out results/jobs/run
```

- **`--mode full`** — runs the agent end-to-end through [Harbor](https://github.com/laude-institute/terminal-bench)
  (Docker) for `vanilla` + each arm, `--k` repeats → `results/<out>/<arm>-<task>/**`. Needs Docker +
  `uv tool install harbor` + `.env` keys (validated up front — a missing condense key refuses to
  start instead of silently running unauthenticated). `--wall-timeout` is per *trial* (the cell gets
  `wall_timeout × k`); every cell writes `attempted.json` first, so killed trials show up as `⚠ lost`
  in the report instead of vanishing. `--dry-run` prints the Harbor commands without running.
- **`--mode incremental`** — teacher-forces one `--session` (or `--swechat` conversation)
  step-by-step through control + each arm, arms in parallel →
  `results/<out>/incremental/<task>-<arm>.jsonl` (paired, cache-aware, no turn-count noise).
  `--task` is required and must match the name you pass `report.py --tasks`. Starts the headroom
  proxy if needed (`--headroom-mode cache|token`).
- **`--milestones`** (full) — runs an LLM judge (temperature 0, arm-blind) over the runs →
  `results/<out>/milestones.json`. Milestones are grounded in a solved vanilla run, which is then
  excluded from vanilla's own coverage scoring.

## `report.py` — display (pure, offline, free)

```
report.py --from results/jobs/run --tasks a,b,c --arms condense,headroom --format {html,md}
```

Reads whatever `generate.py` wrote — full run dirs (**length / rework / solve**, each vs the vanilla
noise floor: **✓ overlap / ✗ disjoint**, ≥2 finished runs/arm, attempted-but-unfinished trials
surfaced as `⚠ lost` and counted as unsolved) plus `milestones.json` (found recursively) and
`incremental/*.jsonl` (rendered as **fid** — per-step action agreement, arm next to the control
noise floor — plus **comp** and **$Δ** over the common step set, cold-cache step 0 excluded) —
and renders. Deterministic, no network.

## Files

The code lives in `minmax_bench/quality/` (importable, unit-tested, still pure standard
library); `scripts/generate.py` and `scripts/report.py` are thin wrappers so the commands
below keep working on a bare `python3` from a fresh clone.

| file | side | role |
|---|---|---|
| `minmax_bench/quality/generate.py` | generate | the one generation command (full + incremental + milestone judge) |
| `minmax_bench/quality/engine.py` | generate | library: session I/O, request building, scoring, pricing — imported by generate, report (parser only) and `minmax-bench counterfactual` |
| `minmax_bench/quality/report.py` | display | reads artifacts → html/md; never spends |
| `harbor_agents/headroom_ccr_claude_code.py` | generate | self-contained CCR wiring for the `headroom-ccr` arm (preserves base MCP servers) |
| `tests/test_quality.py` | — | unit tests for the metric code (`uv run pytest`) |

## Counterfactual replay of a local session

`--mode incremental` works on any Claude Code session file, including ones from your own
`~/.claude/projects` — that's the "what if I had used condense?" counterfactual. The ergonomic
front-end (interactive picker, cost preview, cross-model thinking handling, summary table) is:

```bash
uv run minmax-bench counterfactual        # wraps this engine; see minmax_bench/counterfactual.py
```

## End to end

```bash
# offline demo (bundled sample) — no keys, no Docker
python3 scripts/report.py --from runs/quality-sample --tasks kv-store-grpc --arms condense

# generate then display (needs Docker + harbor + .env keys)
cp .env.dist .env      # ANTHROPIC_API_KEY, CONDENSE_API_KEY
python3 scripts/generate.py --mode full --tasks kv-store-grpc,fix-code-vulnerability \
  --arms condense,headroom --out results/jobs/run1 --milestones
python3 scripts/report.py --from results/jobs/run1 --tasks kv-store-grpc,fix-code-vulnerability \
  --arms condense,headroom
```
