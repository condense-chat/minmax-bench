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
            --arms condense,headroom       # both (vanilla baseline always included)
            --tasks a,b,c  --out results/jobs/run
```

- **`--mode full`** — runs the agent end-to-end through [Harbor](https://github.com/laude-institute/terminal-bench)
  (Docker) for `vanilla` + each arm, `--k` repeats → `results/<out>/<arm>-<task>/**`. Needs Docker +
  `uv tool install harbor` + `.env` keys. `--dry-run` prints the Harbor commands without running.
- **`--mode incremental`** — teacher-forces one `--session` step-by-step through control + each arm →
  `results/<out>/incremental/<task>-<arm>.jsonl` (paired, cache-aware, no turn-count noise). Uses
  `incremental_engine.py`.
- **`--milestones`** (full) — runs an LLM judge over the runs → `results/<out>/milestones.json`.

## `report.py` — display (pure, offline, free)

```
report.py --from results/jobs/run --tasks a,b,c --arms condense,headroom --format {html,md}
```

Reads whatever `generate.py` wrote — full run dirs (**length / rework / solve**, each vs the vanilla
noise floor: **✓ overlap / ✗ disjoint**, ≥2 runs/arm) plus `milestones.json` and `incremental/*.jsonl`
if present — and renders. Deterministic, no network.

## Files

| file | side | role |
|---|---|---|
| `generate.py` | generate | the one generation command (full + incremental + milestone judge) |
| `incremental_engine.py` | generate | session I/O + teacher-forced replay engine `generate` imports |
| `report.py` | display | reads artifacts → html/md; never spends |
| `../harbor_agents/headroom_ccr_claude_code.py` | generate | self-contained CCR wiring for the `headroom-ccr` arm |

## End to end

```bash
# offline demo (bundled sample) — no keys, no Docker
python3 scripts/report.py --from results/sample --tasks kv-store-grpc --arms condense

# generate then display (needs Docker + harbor + .env keys)
cp .env.dist .env      # ANTHROPIC_API_KEY, CONDENSE_API_KEY
python3 scripts/generate.py --mode full --tasks kv-store-grpc,fix-code-vulnerability \
  --arms condense,headroom --out results/jobs/run1 --milestones
python3 scripts/report.py --from results/jobs/run1 --tasks kv-store-grpc,fix-code-vulnerability \
  --arms condense,headroom
```
