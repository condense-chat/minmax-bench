# Quality bench — trajectory preservation

**The maximize target.** Does a context-reduction method **preserve the agent's
trajectory**, or change what it does? The cost tables assume compaction is behaviorally
free — that assumption is load-bearing: if a method makes the agent wander or take more
turns, the "savings" is partly illusory (more turns = more cost).

The bench runs the **full agent** (not a replay) under each method on real coding tasks
([Terminal-Bench](https://github.com/laude-institute/terminal-bench) via
[Harbor](https://github.com/laude-institute/harbor)) and compares trajectories to a
**vanilla-vs-vanilla noise floor**. The bar is not "identical to control" (two vanilla
runs already differ) — it's **"within the vanilla-vs-vanilla spread."**

Two clearly separated phases — the same split the cost bench uses (`run` → store →
`report`):

```
quality run | incremental  ──▶  results root/**     quality report  ──▶  report.html | report.md
 (SPENDS: run agents / replay)     (artifacts)      (PURE DISPLAY: reads artifacts, never spends)
```

**`quality report` never calls a model.** Everything that spends lives in `quality run`
(full) / `quality incremental` / the judges. The analysis path is pure standard library —
Docker + Harbor are only needed to *generate* runs.

## Two modes, two claims

The modes are deliberately complementary — a method can pass one and fail the other, and
both outcomes are informative:

- **full** — end-to-end trajectories. Catches *behavioral* changes (turn-count inflation,
  induced planning, solve rate) but can't attribute them, and needs several repeats to
  beat the variance.
- **incremental** (teacher-forced) — replays a recorded session step-by-step: does
  compressing this exact history change the next decision? Deterministic and paired, so it
  catches *informational* loss — but it's structurally blind to behavioral effects, and on
  short sessions threshold-gated compaction never fires. Teacher-forced replay executes no
  tools, so CCR's retrieve loop can't engage here; the full headroom product belongs to
  full mode.

## Axes

Per task, vanilla `k` runs set the floor; each method's runs are tested against it,
axis by axis:

| axis | question |
|---|---|
| **length** | does compaction change the # of steps? *(the load-bearing axis)* |
| **rework** | does it re-fetch info it already had? *(compaction amnesia; range-aware — post-edit re-inspection counts as verification, not rework)* |
| **milestone** | does it accomplish the same subgoals? *(approach-agnostic, LLM-judged at temperature 0, arm-blind; the reference run is excluded from vanilla's own coverage)* |
| **solve** | does it still pass the verifier? *(trials that crash or hit the wall timeout count as failures — `⚠ lost` — not as missing data)* |
| **fid** | teacher-forced per-step action agreement, shown next to the **control incremental run's** agreement (the noise floor) — only the gap below the floor is signal |

A verdict is **✓** if the method's band *overlaps* vanilla's, **✗** if disjoint — and
needs **≥ 2 finished runs per arm** (a single run can't be told from a fluke; this kills
the k=1 mirage where length and cost swing wildly). Read ✓ honestly: with small k, band
overlap only detects *gross* divergence — "no detectable divergence at this k", not
statistical equivalence.

## The compaction gate (⊘)

Both products act only where reduction pays off, and a small task structurally cannot
exercise them: condense compacts the *whole conversation* only past an internal size
threshold (its savings appear in the 100k+ bands), and headroom's token mode compresses
*individual tool outputs* only when they exceed ~200 tokens (`min_tokens_to_crush`, v0.28
defaults) and score as stale/irrelevant. The report shows vanilla's **peak context** per
task and marks tasks **⊘** when it stays under the gate (`--ctx-gate`, default 50k): on a
⊘ task no compaction fired, so a length ✗ there measures the arm's *wiring and behavioral*
side-effects, not compaction damage.

One such wiring effect, measured: Claude Code composes a ~8-9k-token-larger request
whenever `ANTHROPIC_BASE_URL` is non-default — a flat confound shared by every proxy arm.
The **`vanilla-proxy` arm** isolates it: vanilla routed through a do-nothing local
forwarder (`minmax_bench/quality/passthrough.py`) — same wiring, zero content change — so
vanilla-proxy vs vanilla is the wiring effect alone, and a proxy arm read against
vanilla-proxy has the confound subtracted. Add it with
`--arms condense,headroom,vanilla-proxy` or tick it in the wizard.

Compaction *quality* claims must come from tasks whose vanilla runs clear the gate — the
long half of the curated list (`--tasks long`).

## Arms — naming, carefully

- `condense` — the condense proxy (whole-conversation compaction).
- `headroom` — the token-mode proxy **plus** the MCP retrieve loop: the full
  Compress-Cache-Retrieve (CCR) product.
- `headroom-kompress` — token-mode compression *without* retrieval, kept only as an
  ablation (judging headroom's quality by it would be a strawman).
- `vanilla-proxy` — the passthrough control above.

⚠ The same names carry different meanings across the two benches: in the **cost** bench
`headroom` is the cache-mode strategy and `headroom-kompress` is token-mode; only
`headroom-kompress` (token, no retrieval) means the same thing in both. The quality
bench's `headroom` (token + CCR) has no cost-bench counterpart, because CCR needs a
running agent.

## `quality run` — full trajectories (SPENDS)

```bash
uv run minmax-bench quality run                     # bare = guided wizard
uv run minmax-bench quality run -m claude-haiku-4-5 --tasks 5 --milestones
```

Runs the agent end-to-end through Harbor (Docker) for `vanilla` + each arm, `--k` repeats
→ `<out>/<arm>-<task>/**`. Needs Docker + `uv tool install harbor` + creds (validated up
front — a missing condense credential refuses to start instead of silently running
unauthenticated). It prints its plan (arms × tasks × k = N trials + the cost ceiling)
before spending anything.

| flag | meaning |
|---|---|
| `--tasks` | `N` = first N recommended \| `random:N` (with `--seed`) \| a group: `all`/`long`/`short`/`hard`/`medium` (`long` = author timeout ≥ 30m, biasing toward sessions long enough to compact) \| `a,b,c` by name \| omitted = 5. `--list-tasks` shows everything known. |
| `--arms` | default `condense,headroom`; vanilla always included; also `headroom-kompress`, `vanilla-proxy` |
| `-m/--model` | default `claude-sonnet-4-6` |
| `-d/--dataset` | Harbor dataset; only `terminal-bench/terminal-bench-2-1` is validated so far |
| `--k` | trials per arm/task (default 4); `--k-vanilla` defaults to k+1 — the extra noise-floor run sharpens every verdict |
| `--budget-usd` | per-trial spend cap (default 5.0) |
| `--wall-timeout` | per-trial wall-clock **floor** (default 2400s); the effective cap **auto-sizes** up to each task's own author budget (× the arm's exec multiplier) + build/setup/verify overhead, so long tasks aren't guillotined |
| `--retries N` | re-attempt a cell that *crashed or timed out* (no verifier result) until every trial resolves or attempts run out; a trial that ran and scored (even 0) is a real result and is **not** retried |
| `--force` | full retry: re-run everything. Default is **automatic resume** — re-run the same command/`--out` and finished cells are skipped |
| `--milestones` | also run the LLM milestone judge → `milestones.json` (grounded in a solved vanilla run, which is then excluded from vanilla's own coverage scoring) |
| `--out` | results root (default: fresh auto-minted dir under `settings.quality_runs_dir`, `runs/quality/…` — never clobbers) |
| `--concurrency` | parallel trials per cell (harbor `-n`) |
| `--agent-timeout-mult` / `--setup-timeout-mult` | Harbor exec/setup timeout multipliers (headroom auto-3; slow container installs) |
| `--auth` | `auto` \| `api-key` \| `subscription` (force the Claude Code login — no API key needed) |
| `--dry-run` | print the Harbor commands without running |
| `--agent` | `claude-code` (default); `codex` / `opencode` = TODO (errors, doesn't fake it) |

Every cell writes `attempted.json` first, so killed trials show up as `⚠ lost` in the
report (counted as unsolved) instead of vanishing.

## `quality incremental` — teacher-forced replay of your own session (SPENDS)

How would one of your real sessions have played out under condense? Pick any session from
`~/.claude/projects` and teacher-force it step-by-step through control + each arm. No
Docker, no Harbor — just auth (API key **or** Claude Code login). It auto-detects the
session's model (auto-falling back if an arm can't serve it), shows a cost estimate, and
asks before spending.

```bash
uv run minmax-bench quality incremental                       # interactive picker + confirm
uv run minmax-bench quality incremental ~/.claude/projects/<proj>/<id>.jsonl --arms condense -n 30
```

Per arm you get **same-action agreement** (read against the control floor, not 100%) or,
with `--judge goal`, a per-step good/degraded/bad rating; plus **avg context tokens**,
**$ vs control** (over the common step set, cold-cache step 0 excluded), and a
**recorded** row — what those turns *actually* consumed when the session ran, making the
table both a comparison and a backtest. Note the condense arm sends your session content
to `api.condense.chat`.

| flag | meaning |
|---|---|
| `--arms` | default `condense`; also `headroom` (`--headroom-mode token|cache`, auto-starts/stops the proxy; `--ccr/--no-ccr` injects the retrieve loop via `headroom mcp serve` — `--no-ccr` = kompress) |
| `-n/--limit` | max decision points, contiguous from the start (strided sampling was removed — it distorted cost/compaction numbers) |
| `--budget-usd` | per-arm spend cap, control included (default 2.0) |
| `--judge` | `off` \| `goal` (rate each action toward the task — robust, recommended) \| `equivalence` (upgrade grep-vs-rg near-misses to "agrees") |
| `--ctx-gate` | skip sessions whose peak context stays below this (default 50k; 0 = run anyway) |
| `--capture` | run your version-matched Claude Code binary once, locally, to capture the exact system prompt + tools instead of a stored template |
| `--independent-budgets` | default (`--cap-to-control`) caps every arm at the steps control reached within budget — the paired comparison window, no wasted spend; this flag lets each arm run to its own budget instead ("how far can each arm get"), at the cost of ragged step counts |
| `--resume/--no-resume` | re-running to the same `--out` skips arms that finished cleanly (`.done` sentinel) — a cancel mid-run picks up at the next arm instead of re-running control |
| `--task` | task label for the report join (default `session`); must match what you pass `quality report --tasks` |
| `--max-tokens` | per-step output cap (default 6000) |
| `--auth` | as in `quality run` |

Output: `<out>/incremental/<task>-<arm>.jsonl` (paired, cache-aware, no turn-count noise).

## `quality report` — display (pure, offline, free)

```bash
uv run minmax-bench quality report --from results/jobs/run1 --tasks 5 --format md
```

| flag | meaning |
|---|---|
| `--from` | results root produced by `quality run` (default `results/jobs`) |
| `--tasks` / `--arms` | what to display (must cover what was run) |
| `--format` | `html` (default) \| `md`; `--out` overrides the output path |
| `--ctx-gate` | ⊘ threshold for the display (default 50k) |

Reads whatever the generation commands wrote — full run dirs (**length / rework /
solve**, each vs the vanilla noise floor: ✓ overlap / ✗ disjoint, ≥2 finished runs/arm,
`⚠ lost` surfaced), `milestones.json` (found recursively), and `incremental/*.jsonl`
(rendered as **fid** next to the control floor, plus **comp** and **$Δ** over the common
step set). Deterministic, no network.

## The rest of the toolbox

```bash
uv run minmax-bench quality runs      # list every stored quality run (full + incremental) — free
uv run minmax-bench quality judge     # run the LLM milestone judge over existing full runs (SPENDS)
uv run minmax-bench quality rejudge   # re-score an incremental run with the current judge —
                                      # no re-replay, spends only on judge calls; control's
                                      # good-rate is the calibration check (should be ≥90%)
```

## Files

The code lives in `minmax_bench/quality/` (importable, unit-tested, pure standard library
on the analysis path).

| file | side | role |
|---|---|---|
| `minmax_bench/quality/generate.py` | generate | the generation engine (full + incremental + milestone judge) |
| `minmax_bench/quality/engine.py` | generate | library: session I/O, request building, scoring, pricing |
| `minmax_bench/quality/report.py` | display | reads artifacts → html/md; never spends |
| `minmax_bench/quality/passthrough.py` | generate | the do-nothing forwarder behind the `vanilla-proxy` arm |
| `minmax_bench/quality/paths.py` | both | auto-minted run dirs under `settings.quality_runs_dir` |
| `minmax_bench/counterfactual.py` | generate | the rich `quality incremental` front-end (picker, cost preview, summary table) |
| `harbor_agents/headroom_ccr_claude_code.py` | generate | self-contained CCR wiring for the `headroom` arm (preserves base MCP servers) |
| `tests/test_quality.py` | — | unit tests for the metric code (`uv run pytest`) |

## Findings so far

Reported impartially, including results unfavorable to condense.

- **Preservation mostly holds** — on 8/9 tasks with enough runs, condense's trajectory
  length is within the vanilla-vs-vanilla spread (no detectable divergence at k≈3),
  redundant re-work is zero, and the same subgoals are reached. A per-turn cost claim is
  sound *where both arms solve reliably.*
- **One real exception — short tasks (`kv-store`):** condense consistently ~doubles the
  trajectory (5 → 12 steps) by **inducing todo-tool planning + verification** —
  behavioral, *not* amnesia. This *explains* that task's large full-run cost gap (which
  looked like noise at k=1).
- **Token savings ≠ dollar savings** — compaction busts the prompt cache (the same effect
  behind the cost bench's `headroom-kompress` result); verified two ways (teacher-forced
  incremental + real runs).

## End to end

```bash
# generate then display (needs Docker + harbor + creds)
uv run minmax-bench setup                                              # or: cp .env.dist .env
uv run minmax-bench quality run --out results/jobs/run1 --milestones   # 5 default tasks
uv run minmax-bench quality report --from results/jobs/run1 --tasks 5
```
