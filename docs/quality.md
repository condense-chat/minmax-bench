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
| **length** | does compaction change the # of steps? *(a **cost** axis, and asymmetric: longer than vanilla is a real signal — more turns means more money, and often a wandering agent — while **shorter is the point of the method**, never a mark against it)* |
| **rework** | does it re-fetch info it already had? *(compaction amnesia; range-aware — post-edit re-inspection counts as verification, not rework)* |
| **milestone** | does it accomplish the same subgoals? *(approach-agnostic, LLM-judged at temperature 0, arm-blind; the reference run is excluded from vanilla's own coverage)* |
| **solve** | does it still pass the verifier? *(trials that crash or hit the wall timeout count as failures — `⚠ lost` — not as missing data)* |
| **fid** | teacher-forced per-step action agreement, shown next to the **control incremental run's** agreement (the noise floor) — only the gap below the floor is signal |

**The per-task verdict is about quality, not length.** `✓ quality held` / `✗ quality lost`
is decided by **solve** and **milestone** together — quality has to survive both, because
they fail differently and neither excuses the other. The verifier is ground truth for "did
it do the task", so a solve regression is a loss however good the subgoal coverage looks;
the milestone judge is finer and approach-agnostic, so it catches an arm that still passes
having done materially less. The verifier arm of the test is deliberately blunt: at k≈4 one
trial is worth 20-25%, so a loss must exceed one trial's worth of the arm's own denominator
— a 3/5 → 2/4 wobble is one trial landing differently, not a regression.

Length rides along only in the expensive direction, as `✓ held · ↑ longer`. An arm that
reached the same result in 14 steps instead of 27 has not diverged — that is what these
methods are *for* — so shorter is never marked against it. (Earlier versions decided the
verdict on length alone and labelled that case "drifted", which read as a failure.)

A band comparison is **✓** if the method's band *overlaps* vanilla's, **✗** if disjoint — and
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

**⊘ applies to compaction methods only** (`report.COMPACTION_GATED`). It is an excuse for a
*history transform*: no compaction fired, so the method never got to act. It is no excuse for
a method that acts from the first step regardless of context size — that method's length,
tokens and cost on a small task are measured and comparable, and blanking its verdict turns a
whole run into a column of shrugs. The list is an **allowlist**: an arm nobody classified is
un-gated, so a new method shows a real verdict rather than quietly vanishing from the table.

## Arms — naming, carefully

- `condense` — the condense proxy (whole-conversation compaction), against whichever
  deployment `CONDENSE_PROFILE` / `dense target` selects (prod by default).
  Nothing in a trial's artifacts records which deployment served it, so the arm name is the
  only durable provenance — give each deployment its own arm, never separate runs of the same
  one.
- `headroom` — the token-mode proxy **plus** the MCP retrieve loop: the full
  Compress-Cache-Retrieve (CCR) product.
- `headroom-kompress` — token-mode compression *without* retrieval, kept only as an
  ablation (judging headroom's quality by it would be a strawman).
- `vanilla-proxy` — the passthrough control above.
- `caveman` — the [caveman](https://github.com/JuliusBrussee/caveman) skill (full mode
  only), pinned at `v1.9.1`. A different *class* of method, and the distinction matters
  for how you read it — see below.

### caveman is a generation policy, not a history transform

Every other arm is a **history transform**: hand it the conversation, it hands back a
smaller one. caveman never touches history. A SessionStart hook injects a terse-output
ruleset, and the agent writes fragments instead of prose for the rest of the run. Three
consequences:

- **It reads against `vanilla`, not `vanilla-proxy`.** It's the only non-vanilla arm with
  no proxy — `ANTHROPIC_BASE_URL` stays default — so it doesn't carry the ~8-9k wiring
  confound. Reading it against `vanilla-proxy` would credit it with that entire difference.
- **Read it on `comp`, exactly like condense.** caveman's terse prose replaces the verbose
  prose *span by span* and accrues into the context — the same shape as condense's condensed
  blocks (see the incremental section below), just at prose granularity. So the fair token
  measure is the same one every arm uses: **`comp`**, the accrued-context change vs control.
  It only touches prose (tool calls, tool results, code and errors stay verbatim), so `comp`
  is **small — often ~0 or net-negative** once the flat ~750-token ruleset cost per turn is
  counted, because tool I/O dominates the prefix and caveman never touches it. That small/
  negative number *is* the honest answer, not a metric to work around. (The ⊘ gate note above
  applies in reverse: a ⊘ task doesn't mean "nothing fired", it means "probably net negative
  on context".) It still counts as **engaged** even at `comp≈0` — its terse prose can move the
  next decision — so its `fid` stays a real verdict rather than being dimmed as a passthrough.
- **The interesting risk is different.** condense's failure mode is amnesia; caveman's is
  that an agent's prior messages are its own working scaffold, and terser notes may reason
  worse. `length` and `milestone` are the axes to watch, not `rework`.

```bash
uv run minmax-bench quality run --arms caveman --tasks long                  # level: full
uv run minmax-bench quality run --arms caveman --caveman-mode ultra --tasks long
```

`--caveman-mode` is `lite` | `full` (default) | `ultra` — the same knob for `quality run`
and `quality incremental` (it sets `TMB_CAVEMAN_MODE` for the container). Upstream's
`wenyan-*` levels are rejected on purpose: they switch the output language to classical
Chinese, which confounds every trajectory metric.

**Silent-inactivity guard.** An arm that installs but never activates is indistinguishable
from vanilla, and would score a clean ✓ "trajectory preserved" while measuring nothing.
Two checks prevent that in full mode: agent setup smoke-tests the hook and refuses to run if
it produces no ruleset (or falls back to its abridged hardcoded one), and the report scans
each transcript for the activation marker and labels trials that lack it `⚠ n inactive`.
Incremental has the same hazard in a different shape — a step where caveman returns no usable
prose keeps the recorded *verbose* narration, so the replayed history is control's — and the
same answer: reverted steps are counted, the end-of-run readout names the mixture instead of
claiming a "fully-terse history", a run with zero terse steps reads `caveman: inactive`, and
the report treats it as `⊘ passthrough` rather than scoring its fidelity.

**In `quality incremental`, caveman freezes the action rails and keeps its terse prose.** A
naive teacher-forced replay would pin the recorded *verbose* assistant messages — exactly what
caveman changes — and read as "no effect". But letting caveman *free-run* its actions is also
wrong: **caveman doesn't change actions**, only prose, and a diverged action has no recorded
`tool_result` to continue from. So the arm keeps the recorded trajectory's **rails frozen** —
every `tool_use`, `tool_result`, and thinking block stays verbatim (no id remap, always valid,
perfectly paired) — and swaps in caveman's own terse narration on **every** step, *including*
steps where its proposed action diverged. That's safe because the prose in an agent turn is
dominantly a **reflection on the prior (frozen) tool_result** — which both caveman and control
saw identically — not a commitment to the next action (caveman drops tool-call narration by
rule). So the history stays coherent with the frozen action regardless, and it's **genuinely
terse throughout** rather than half-reverted to verbose.

The per-step **action-fidelity** — would caveman have *proposed* the recorded action given that
terse history — is the **drift / alignment** signal: high = the terse history didn't change the
decision, low = it drifted. Measuring it against a genuinely-terse history (rather than a
corrected one) is the whole point — the run reports `action-aligned %` and the `caveman_diverged`
count alongside it. Only an actual API error keeps the recorded verbose prose (there's no
caveman prose to use). The history is append-only — caveman never rewrites an earlier turn —
so each step's prefix nests the last and the incremental cache is preserved by construction,
the same as control; a compaction method instead rewrites earlier spans *when it compacts*,
invalidating the cache at that point (the cost bench's documented cache-bust). We don't
separately measure caveman's cache here — it's a structural property of an append-only prefix,
not a result.

`--caveman-mode lite|full|ultra` picks the intensity; the injected ruleset is the exact text
the pinned hook emits (`data/caveman/ruleset-<mode>.txt`), so incremental and full present the
same guidance. What this mode still *can't* see is behavioral **compounding** — a mis-proposed
action never actually executes, because the rails stay recorded, so its downstream effect can't
propagate. That question is full mode's, where the trajectory is just genuinely different (real
execution + verifier). The residual artifact here is the turn whose terse prose *does* lead
forward to a diverged action; caveman's no-narration rule keeps it rare, and a future rewriter
pass (terse prose written *about* the recorded action) would remove it entirely.

That artifact only exists on turns carrying a **frozen tool_use** — about 60% of real decision
points are prose *and* a tool call, ~31% are tool-only (no prose to shrink, so they can neither
be terse nor drift), and ~9% are prose-only. A divergence on a prose-only turn replaces the
whole message and stays coherent; a divergence next to a frozen action does not. So the run
splits its drift into **`n beside a different frozen action`** (the artifact, and the exact turn
set a rewriter would target) and **`n on a prose-only turn (clean)`** — a run can drift a lot
with no artifact, or a little with all of it stapled, and one number cannot tell them apart.
Note the alignment rate's denominator is **prose steps**, while the table's `vs original` is
every successful step, so the two legitimately differ; the readout states both.

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
| `--arms` | default `condense,headroom`; vanilla always included; also `headroom-kompress`, `vanilla-proxy`, `caveman` |
| `-m/--model` | default `claude-sonnet-4-6` |
| `-d/--dataset` | Harbor dataset; only `terminal-bench/terminal-bench-2-1` is validated so far |
| `--k` | trials per arm/task (default 4); `--k-vanilla` defaults to k+1 — the extra noise-floor run sharpens every verdict |
| `--budget-usd` | per-trial spend cap (default 5.0) |
| `--wall-timeout` | per-trial wall-clock **floor** (default 2400s); the effective cap **auto-sizes** up to each task's own author budget (× the arm's exec multiplier) + build/setup/verify overhead, so long tasks aren't guillotined |
| `--retries N` | re-attempt a cell that *crashed or timed out* (no verifier result) until every trial resolves or attempts run out; a trial that ran and scored (even 0) is a real result and is **not** retried |
| `--force` | full retry: re-run everything. Default is **automatic resume** — re-run the same command/`--out` and finished cells are skipped |
| `--milestones` | also run the LLM milestone judge → `milestones.json` (grounded in a solved vanilla run, which is then excluded from vanilla's own coverage scoring) |
| `--out` | results root (default: fresh auto-minted dir under `settings.quality_runs_dir`, `runs/quality/…` — never clobbers) |
| `--concurrency` | trials of one cell run at once (harbor `-n`); **default 1 = sequential**, and the wizard asks. Cells run one at a time so it is capped by `k`. Same total spend, N× the burn rate — but N containers contending for CPU/RAM/disk can slow the agent's commands and shift the trajectories this bench measures, so sequential is the clean setting |
| `--agent-timeout-mult` / `--setup-timeout-mult` | Harbor exec/setup timeout multipliers (headroom auto-3; slow container installs) |
| `--auth` | `auto` \| `api-key` \| `subscription` (force the Claude Code login — no API key needed) |
| `--dry-run` | print the Harbor commands without running |
| `--agent` | `claude-code` (default); `codex` / `opencode` = TODO (errors, doesn't fake it) |

Every cell writes `attempted.json` first, so killed trials show up as `⚠ lost` in the
report (counted as unsolved) instead of vanishing.

On a Claude Code subscription (no API key), every upstream call — replay and judges alike —
carries the CLI's identity system block. Subscription traffic is classified by whether it
looks like Claude Code, and a request with no system prompt is throttled far harder: without
this the milestone judge could `429` on three small calls right after the same credentials had
happily served a whole 6-cell run. A `429` from the judge is transient, not a quota failure —
re-run it, and `milestones.json` caches per task so you only pay for what's still missing.

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
table both a comparison and a backtest. It closes with a plain-English **bottom line** per
arm — *context saved X% · $ saved Y% · <quality>* — where the quality metric is the one you
chose (goal-quality under `--judge goal`, same-action fidelity otherwise; the structural
`exact` column is dimmed under `--judge goal` so it isn't mistaken for the verdict). Note the
condense arm sends your session content to `api.condense.chat`.

| flag | meaning |
|---|---|
| `--arms` | default `condense`; also `headroom` (`--headroom-mode token|cache`, auto-starts/stops the proxy; `--ccr/--no-ccr` injects the retrieve loop via `headroom mcp serve` — `--no-ccr` = kompress); `caveman` (frozen rails, terse-prose drift, `--caveman-mode lite|full|ultra`) |
| `-n/--limit` | max decision points, contiguous from the start (strided sampling was removed — it distorted cost/compaction numbers) |
| `--budget-usd` | per-arm spend cap, control included (default 2.0) |
| `--judge` | `off` \| `goal` (rate each action toward the task — robust, recommended) \| `equivalence` (upgrade grep-vs-rg near-misses to "agrees") |
| `--ctx-gate` | skip sessions whose peak context stays below this (default 50k; 0 = run anyway) |
| `--caveman-mode` | caveman intensity: `lite` \| `full` (default) \| `ultra`. Injects the pinned hook's ruleset and keeps caveman's terse prose over frozen tool rails; read on `comp` like condense, plus `action-aligned %` (drift) |
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

### the full-run table — one row per arm, every metric against vanilla

The per-task tables answer *"what happened on this task"*; every cell there is one or a few
trials, so nothing in them carries an error bar and reading them means holding a column of
small numbers in your head. The **full runs** table, printed first, answers the other question
— *"across everything that was run, what did this arm cost, and is the difference bigger than
the noise?"*

```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                        ┃      solve ┃      turns ┃   peak ctx ┃     tokens ┃          $ ┃         $ ┃    cache wr ┃      compact┃
┃arm                     ┃  rate, pts ┃  per trial ┃     tokens ┃  per trial ┃  per trial ┃  per Mtok ┃       share ┃    per trial┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│vanilla                 │        80% │       21.9 │     59,308 │    989,160 │      $1.44 │     $1.45 │        4.0% │         0.00│
│14 tasks · ground       │            │            │            │            │            │           │             │             │
├────────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼───────────┼─────────────┼─────────────┤
│condense                │   -1.4 pts │     +12.7% │      -3.8% │     +13.7% │     +27.9% │    +12.4% │     +103.3% │         2.14│
│14 tasks · 56v70 trials │ [-17, +14] │  [-6, +35] │  [-16, +9] │ [-12, +48] │  [+8, +53] │ [-7, +33] │ [+37, +183] │ [1.39, 2.89]│
└────────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────┴─────────────┴─────────────┘
```

The `vanilla` row is the absolute **ground**; every arm row is that arm against it. (The real
table also carries a `$ per step` column, dropped here for width.)

- **solve** — verifier pass rate, in percentage **points**. Not a ratio: a ratio is undefined
  when vanilla is 0% and meaningless when it is 100%, which is 8 of 14 tasks on this suite.
  Lost trials count as failures.
- **turns** — billed requests, deduped by `requestId`. **Steps** (`$ per step`) are `tool_use`
  blocks — near 1:1 in practice but not the same unit, so a $-per-unit column must say which.
- **cache wr** — cache-write share of cached tokens. Writes bill at **12.5×** reads, so this
  is the one cache number that explains a `$/Mtok` move.
- **compact** — billed turns whose context came back *smaller* than the turn before. Measured
  on the outcome, not inferred from cache traffic: the old cache-side heuristic (a large
  `cache_creation` without matching context growth) also fires on ordinary prefix rewrites and
  over-counts by ~1.6×. Shown **absolute**, because vanilla is exactly 0 and every ratio would
  be infinite — and validated by exactly that: vanilla and caveman measure `0.00` across 72 and
  56 trials.

Three rules keep it from over-claiming:

1. **Every % is a geometric mean of per-task ratios** — `exp(mean(log(arm/vanilla))) − 1` — so
   each task votes once. The pooled ratio-of-sums this replaced was *dollar-weighted in
   disguise*: on the opus-5 suite three of fourteen tasks carried 41% of the spend, so "cost
   +81%" was really "+81% on the three tasks that happen to be expensive".
2. **The interval is a two-level bootstrap**, seeded, so the same artifacts always render the
   same bars: resample *tasks*, then resample *trials* within each resampled cell. Resampling
   tasks alone treats a 4-trial cell mean as exact and produces intervals far too narrow —
   two levels is what makes vanilla's own spread visible. Tasks are shared across arms (the
   comparison is paired); trial draws are independent per arm.
3. **Bold marks an interval clear of zero.** Everything else is indistinguishable from vanilla
   at this n — the common case, and not a pass. Only tasks where *both* the arm and vanilla
   produced a usable trial are counted, so every number down a column rests on the same tasks.

### the incremental table

Printed below it, and about something else entirely — the teacher-forced replay, whose control
is a no-compression ceiling rather than a competitor.

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃arm          ┃     quality ┃   redundant ┃     context┃
┃             ┃ incremental ┃  /100 steps ┃     removed┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│control      │  78.1 ± 4.7 │   5.0 ± 2.5 │           —│
│             │ 317 steps/4s│             │            │
│condense     │  73.5 ± 4.7 │  11.0 ± 3.5 │      +39.1%│
│⊘6 passthru  │  Δ-4.6 ± 5.9│  Δ+6.0 ± 3.2│            │
└─────────────┴─────────────┴─────────────┴────────────┘
```

- **quality (incremental)** — the share of replayed steps whose action the goal judge rated
  *good*; without `--judge goal` it falls back to structural agreement with the recording,
  a much noisier floor.
- **redundant** — steps that re-fetched information the agent already had, per 100 steps.
- **context removed** — not a quality axis. It is what the quality columns are the *price
  of*, and the number without which they can't be read: an arm at `+3%` next to one at
  `+39%` isn't gentler, it barely fired. `⊘` marks <2%.

Here `±` resamples the paired *steps*, and `Δ` under an arm value is its **paired** difference
vs control with its own CI — the ± on the values themselves are marginal, since arm and control
move together step-for-step, so only the Δ answers "is this a real difference?". Steps within a
session are correlated and are resampled as independent, so those bars are if anything
optimistic. Sessions the arm passed through are dropped and counted (`⊘6 passthrough`) —
nothing is dropped quietly.
   A Δ bar straddling zero means *indistinguishable from control at this n*, which is the
   common outcome and not a pass. Nothing is coloured or marked better/worse: the table
   prints the numbers and you draw the line.

Each arm carries its own control reference, paired over that arm's material. When the arms
ran over the same tasks and sessions one control row heads the table; when they didn't,
control is repeated per arm, because a single baseline row would be comparing an arm against
material it never ran.

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
| `minmax_bench/quality/engine.py` | generate | library: session I/O, request building, scoring, pricing, the caveman frozen-rails replay state (`CavemanState`) |
| `minmax_bench/quality/report.py` | display | reads artifacts → html/md; never spends |
| `minmax_bench/quality/passthrough.py` | generate | the do-nothing forwarder behind the `vanilla-proxy` arm |
| `minmax_bench/quality/paths.py` | both | auto-minted run dirs under `settings.quality_runs_dir` |
| `minmax_bench/counterfactual.py` | generate | the rich `quality incremental` front-end (picker, cost preview, summary table) |
| `harbor_agents/headroom_ccr_claude_code.py` | generate | self-contained CCR wiring for the `headroom` arm (preserves base MCP servers) |
| `harbor_agents/caveman_claude_code.py` | generate | pinned caveman install + activation smoke test for the full-mode `caveman` arm |
| `data/caveman/ruleset-<mode>.txt` | generate | frozen SessionStart-hook rulesets injected by incremental caveman (pinned; see `data/caveman/PIN`) |
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
