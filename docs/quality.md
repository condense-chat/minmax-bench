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

**⊘ applies to compaction methods only** (`report.COMPACTION_GATED`). It is an excuse for a
*history transform*: no compaction fired, so the method never got to act. It is no excuse for
a method that acts from the first step regardless of context size — that method's length,
tokens and cost on a small task are measured and comparable, and blanking its verdict turns a
whole run into a column of shrugs. The list is an **allowlist**: an arm nobody classified is
un-gated, so a new method shows a real verdict rather than quietly vanishing from the table.

## Arms — naming, carefully

- `condense` — the condense proxy (whole-conversation compaction).
- `headroom` — the token-mode proxy **plus** the MCP retrieve loop: the full
  Compress-Cache-Retrieve (CCR) product.
- `headroom-kompress` — token-mode compression *without* retrieval, kept only as an
  ablation (judging headroom's quality by it would be a strawman).
- `vanilla-proxy` — the passthrough control above.
- `caveman` — the [caveman](https://github.com/JuliusBrussee/caveman) skill (full mode
  only), pinned at `v1.9.1`. A different *class* of method, and the distinction matters
  for how you read it — see below.
- `rtk` — [RTK](https://github.com/rtk-ai/rtk) ("Rust Token Killer"), pinned at `v0.44.0`.
  A **third** class: it transforms neither the history nor the agent's output but its
  **observations** — see below.

### three classes of method, and why the class changes how you read the arm

| class | transforms | arms |
|---|---|---|
| history transform | the conversation you resend | `condense`, `headroom` |
| generation policy | what the agent **writes** | `caveman` |
| observation transform | what the agent **reads back** | `rtk` |

The first hands back a smaller conversation. The second makes the agent's own messages
smaller as they accrue. The third leaves both alone and shrinks the *tool output* — which
is what actually dominates a coding session's prefix, and the exact gap the caveman section
below names ("tool I/O dominates the prefix and caveman never touches it").

Neither `caveman` nor `rtk` is a proxy, so both read against plain **`vanilla`**, not
`vanilla-proxy` — they don't carry the ~8-9k non-default-base-URL wiring confound, and
comparing them to `vanilla-proxy` would credit them with that whole difference.

### rtk is an observation transform

RTK is a single Rust binary and is **deterministic** — rule-based filters, no LLM, nothing
sampled. A `PreToolUse` hook rewrites a Bash command to its rtk equivalent (`git status` →
`rtk git status`), rtk runs it, and only the filtered output reaches the model. Consequences:

- **It rewrites the recorded actions.** `rtk hook claude` returns `updatedInput.command`, so
  the transcript stores `rtk git status`, not `git status`. This is unique among the arms and
  it is load-bearing: every command-shaped metric un-wraps first (`engine.unrtk`). Without
  that, `rework_count` scores the arm a flawless **zero** on identical behaviour — its
  read-only pattern is `^`-anchored so `rtk grep …` never matches, and it looks for
  `cat`/`head`/`tail` by name while rtk renames all three to `rtk read`. That would be a
  strawman *in RTK's favour*, the mirror image of the `headroom-kompress` warning above.
- **In full mode `comp` should be substantial**, unlike caveman's ~0 — it is attacking the
  part of the prefix that is actually large. Incremental is a different story; see the
  coverage numbers below.
- **The risk to watch is information loss, not amnesia.** RTK's claim is "smaller context,
  same signal" — it keeps failures and drops passing boilerplate. So `milestone` and solve
  rate are the axes: a filter that ate the one line the agent needed shows up as a failed
  task, not as a longer trajectory.

```bash
uv run minmax-bench quality run --arms rtk --tasks long           # full: pinned build per trial
uv run minmax-bench quality incremental --arms rtk                # incremental: filters locally
```

**What rtk does and does not activate.** The hook fires on *every* Bash call, but
`rtk rewrite` has no equivalent for most commands and passes them through — measured, **348 of
2286 Bash commands (15%)** are actually rewritten on real sessions. rtk also ships a
model-facing file (`hooks/claude/rtk-awareness.md`, embedded into CLAUDE.md by `rtk init -g`
and frozen here at `data/rtk/awareness.md`). It is **not** a skill in caveman's sense: ~10
lines advertising four analytics commands (`gain`/`discover`/`proxy`) plus "everything else is
rewritten automatically". It never redirects the model off the native Read tool, so it does not
lift the Bash-only ceiling — but it is installed, because a real `rtk init -g` installs it and
those lines are a genuine (small) context cost the arm should carry.

**Aggressiveness is not configurable through the hook.** Unlike caveman's `--caveman-mode`,
rtk has no level knob the bench can set: `--level`/`--ultra-compact` are per-invocation flags,
`src/core/config.rs` has no corresponding config key, and the hook emits bare commands. Setting
one would require shimming `rtk` on PATH — i.e. benchmarking a wrapper the user would have to
build — and it could not move the result anyway: the bytes that reach rtk at all total ~17kB
against multi-million-token transcripts, so even deleting them outright is ~0.05%.

**Silent-inactivity guard.** Structurally stronger than caveman's marker scan: an *active*
trial stores the `rtk` prefix in the recorded `tool_use` itself, so the report flags any trial
that issued Bash commands with none carrying it (`⚠ n inactive`). A trial that ran no Bash at
all is not counted — having nothing to rewrite is a property of the task. Container-side, the
agent smoke-tests both `rtk rewrite` and `rtk hook claude` and refuses to run if either stops
rewriting. It deliberately wires the **native** `rtk hook claude` rather than upstream's
`hooks/claude/rtk-rewrite.sh`, which shells out to `jq` and degrades to a silent no-op when jq
is absent — precisely the failure that makes an arm measure nothing while scoring a clean pass.

**In `quality incremental`, rtk filters the recorded output.** This is far simpler than
caveman's replay, because the transform is deterministic and independent of anything the model
says: the whole session is filtered **once** up front through `rtk pipe --filter <name>`, and
the arm then teacher-forces its transformed history exactly like control teacher-forces the
original. Same actions, same prose, same step set — only the observations are smaller — so the
arm is paired with control step-for-step by construction. There is no free-running, no state
machine, and none of caveman's prose/action stapling.

Which commands rtk touches is decided by `rtk rewrite`, the same registry the hook delegates
to, so incremental and full agree. The filter set is queried from the binary rather than
hardcoded, so a pin bump can't silently rot it. A command with no rtk *pipe* filter (`rtk ls`
has none) passes through verbatim rather than being faked, and a failed filter returns the text
**unchanged** — dropping an observation would read as a spectacular saving while destroying the
trajectory.

Two caveats to read it honestly. Both are measured, not asserted — the numbers below come
from 2286 Bash commands across six real recorded sessions, at the pinned rtk:

- **Incremental reaches about half of what full mode does.** rtk *rewrites* 15% of those
  Bash commands, but only 7% have a **pipe** filter — the pipe set (25 filters) is much
  narrower than the rewrite set (100+ commands): `rtk ls`, `rtk read`, `rtk wc` and friends
  have no pipe equivalent. So a low incremental `comp` is the arm's *coverage ceiling*, not
  evidence that rtk doesn't work; full mode is where its real reach shows. The run prints
  how many observations it filtered and how many passed through, and says so outright when
  it filtered nothing.
- **Replay runs the REAL rtk command wherever its input can be reconstructed.** For a file
  read the recorded output *is* the file's content, so it is materialized and the actual
  `rtk read …` runs on it — exact, and reaching commands no pipe filter covers. Pipe-filtering
  is the fallback, used only where post-processing stdout is genuinely rtk's mechanism. Where
  neither applies (`rtk ls`, `rtk wc` need a real filesystem) the observation is left verbatim,
  never faked. Note `cat x` rewrites to a *bare* `rtk read x`, whose default level is full
  content — so it returns unchanged, and that is the faithful answer; passing
  `--level aggressive` would cut ~94% but would measure a method rtk's hook does not implement.
- **Where the pipe fallback is used, it is EXACT for pure post-processors.**
  Verified byte-identical for `pytest` and `grep`. It diverges only where rtk re-invokes the
  underlying tool with its own format: `git status` (hook 35B vs pipe 55B — it emits
  `* branch / clean`, not a trimmed `git status`), `git log -5` (hook **1856B** vs pipe
  227B), `git diff <ref>` (34517B vs 13544B). Note `git log` goes the *wrong way*: the real
  hook produces more output than the pipe filter, so a git-heavy session could show a saving
  that the real thing does not deliver. On the sessions measured this affects **19 of 170**
  transformed observations (11%) — the other 89% are exact.

  This gap is intrinsic, not an implementation shortcut: replay only holds the output of the
  *original* command, and `rtk git log` runs git with its own `--format`, so those bytes were
  never recorded. Re-executing locally would be worse — the repo has moved on since the
  recording, so the rtk arm would see different underlying *facts* than control rather than
  different formatting, breaking the pairing far more badly. **Full mode is the mode that
  runs rtk for real**; that is the division of labour the two modes exist for.
- **Incremental uses your LOCAL rtk; full mode uses the pin.** The transform runs on this
  machine (like condense's `dense` CLI), so if `rtk --version` differs from `v0.44.0` the two
  legs ran different versions of the method under test. The run warns, `minmax-bench setup`
  reports the alignment and offers to install/upgrade, and comparing rtk-vs-control *within*
  one run is unaffected either way.

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
| `--arms` | default `condense,headroom`; vanilla always included; also `headroom-kompress`, `vanilla-proxy`, `caveman`, `rtk` |
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
| `--arms` | default `condense`; also `headroom` (`--headroom-mode token|cache`, auto-starts/stops the proxy; `--ccr/--no-ccr` injects the retrieve loop via `headroom mcp serve` — `--no-ccr` = kompress); `caveman` (frozen rails, terse-prose drift, `--caveman-mode lite|full|ultra`); `rtk` (filters the recorded tool output through `rtk pipe` — needs rtk installed locally) |
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

### the overall table — one row per arm, with error bars

The per-task tables answer *"what happened on this task"*; every cell there is one or a few
trials, so nothing in them carries an error bar and reading them means holding a column of
small numbers in your head. The **overall** table, printed first, answers the other question
— *"across everything that was run, does this arm cost quality, and is the difference bigger
than the noise?"*

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃arm          ┃     quality ┃     quality ┃   redundant ┃     context┃
┃             ┃   full runs ┃ incremental ┃  /100 steps ┃     removed┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│control      │  52.8 ±25.0 │  78.1 ± 4.7 │   5.0 ± 2.5 │           —│
│             │    12 tasks │ 317 steps/4s│             │            │
│condense     │  75.0 ±25.0 │  73.5 ± 4.7 │  11.0 ± 3.5 │      +39.1%│
│⊘2 too short │ Δ+22.2 ±30.6│  Δ-4.6 ± 5.9│  Δ+6.0 ± 3.2│            │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────┘
```

- **quality (full runs)** — verifier pass rate, macro-averaged per task so a task with many
  trials can't outvote one with few. Lost trials count as failures.
- **quality (incremental)** — the share of replayed steps whose action the goal judge rated
  *good*; without `--judge goal` it falls back to structural agreement with the recording,
  a much noisier floor.
- **redundant** — steps that re-fetched information the agent already had, per 100 steps.
- **context removed** — not a quality axis. It is what the quality columns are the *price
  of*, and the number without which they can't be read: an arm at `+3%` next to one at
  `+39%` isn't gentler, it barely fired. `⊘` marks <2%.

Three rules keep it from over-claiming:

1. **Each column is pooled over material where the method could act.** Incremental drops
   sessions the arm passed through; full drops ⊘ tasks whose peak context never reached the
   compaction gate. Both counts are printed on the arm row (`⊘2 too short`, `⊘6
   passthrough`) — nothing is dropped quietly.
2. **± is a 95% bootstrap CI**, seeded, so the same artifacts always render the same bars.
   Full-run quality resamples *tasks*; the incremental columns resample the paired *steps*.
   Steps within a session are correlated and are resampled as independent, so those bars are
   if anything optimistic.
3. **Δ under an arm value is its *paired* difference vs control**, with its own CI. The ± on
   the values themselves are marginal — arm and control move together step-for-step, so they
   can't be eyeballed against each other, and only the Δ answers "is this a real difference?"
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
| `harbor_agents/rtk_claude_code.py` | generate | pinned rtk binary install + PreToolUse hook wiring + rewrite smoke test for the `rtk` arm |
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
