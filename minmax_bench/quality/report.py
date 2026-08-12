#!/usr/bin/env python3
"""report.py — DISPLAY only. Reads what `generate.py` produced and renders it. Never spends.

Mirrors the sibling cost bench's `report` ("recompute from stored data, verify without
re-spending"):
generation writes artifacts; this reads them and computes the offline, deterministic views.

Reads from a results root (`--from`):
  - full run dirs      <root>/**/<arm>-<task>/*/*/{verifier/reward.txt, agent/sessions/.../*.jsonl}
                       plus <arm>-<task>/attempted.json (trials REQUESTED — killed/crashed trials
                       write no reward.txt; counting them as failures avoids survivorship bias)
  - milestones.json    anywhere under <root> (merged; produced by `generate --milestones`)
  - incremental jsonl  <root>/**/incremental/<task>-<arm>.jsonl  (`generate --mode incremental`)

Offline axes (always, no model calls): length / rework / solve, each vs the vanilla noise floor
(OK = bands overlap, DIVERGES = disjoint; needs ≥2 runs/arm). Milestone + incremental axes appear
only if their artifacts exist. Verdicts are deliberately coarse: with k≈3 runs, band overlap can
only catch GROSS divergence — "OK" means "no detectable divergence at this k", not equivalence.

The per-task verdict is QUALITY divergence (solve + milestone, both must hold), not length.
Length is a cost axis and asymmetric: longer than vanilla rides along as "✓ held · ↑ longer",
shorter is what the method is for and is never marked against an arm.

  minmax-bench quality report --from results/jobs/run1 --tasks kv-store-grpc
  minmax-bench quality report --from results/jobs --tasks a,b --arms condense,headroom --format md
"""
import argparse
import glob
import json
import math
import os
import random
import re
import statistics

try:  # rich renders the terminal tables, but analysis must work on a bare python3
    from rich.console import Console
    from rich.table import Table
    HAVE_RICH = True
except ImportError:  # fresh clone, no install — fall back to the plain-text views
    HAVE_RICH = False

# one session parser for the spend side (generate/engine) and the display side.
from minmax_bench.quality.engine import (
    SESSION_GLOB,
    cost_usd,
    ctx_tokens,
    extract_action,
    parse_session,
    peak_ctx,
    recorded_usage,
    resolve_tasks,
    unrtk,
)

AGENT_SESSION_GLOB = {  # only claude-code is wired; others are TODO
    "claude-code": SESSION_GLOB,
}

# longest names first — cell dir names are '<arm>-<task>' and arm names contain hyphens,
# so splitting is a longest-prefix match against the arms the bench knows about.
#
# EVERY name starting with 'condense-' must appear ABOVE plain 'condense', or the prefix match
# eats it: 'condense-recover-dna-assembly' read back as arm 'condense', task
# 'recover-dna-assembly' — 13 real cells filed under the wrong arm as phantom tasks that no
# --tasks list matches. Add the arm here in the SAME commit that creates the cells.
#
# 'condense-prod-08-02' and 'condense-recover' are ARCHIVED arms: past runs renamed off
# 'condense' so a re-run starts from zero trials instead of resuming into the old cells, and so
# the generations show up side by side in one report. Archived arms are deliberately absent
# from generate.py's _arm_wiring — they are readable, not runnable.
KNOWN_ARMS = ("condense-prod-08-02", "condense-recover",
              "headroom-kompress", "vanilla-proxy", "headroom",
              "condense", "caveman", "ponytail", "rtk", "vanilla", "control")

# Arms whose intervention only EXISTS once the harness compacts — the history transforms. ⊘
# (vanilla's peak context never reached --ctx-gate) means nothing compacted, so for these the
# method never got to act, and a verdict there would be about wiring, not compaction.
#
# Every OTHER arm acts regardless of context size: a passthrough proxy pays its wiring cost on
# the first request, and so do the arms that transform what the agent writes or reads. Blanking
# their verdict as "⊘ too short" reports "not comparable" about the one thing that WAS
# comparable — a whole small-context run renders as a column of shrugs while its length, tokens
# and cost sit right there, measured and comparable.
#
# An ALLOWLIST, not a blocklist: an arm nobody classified is un-gated, so a new method shows a
# real verdict instead of quietly vanishing from the table. The direction of the default matters
# more than the default being right — a wrong verdict gets argued with, a blank one doesn't.
COMPACTION_GATED = ("condense", "condense-prod-08-02", "condense-recover",
                    "headroom", "headroom-kompress")


def gated(arm, sub_gate):
    """Does the ⊘ compaction gate apply to THIS arm on a sub-gate task?"""
    return bool(sub_gate) and arm in COMPACTION_GATED


def split_cell(name):
    """'<arm>-<task>' -> (arm, task) by longest known-arm prefix; (None, name) if no match."""
    for arm in KNOWN_ARMS:
        if name.startswith(arm + "-"):
            return arm, name[len(arm) + 1:]
    return None, name


# legacy static default; the settings-aware default (incl. quality_runs_dir) is
# paths.default_run_roots(), used when discover_runs is called with roots=None
DEFAULT_RUN_ROOTS = ("results", "runs/quality-sample")


def _run_info(d):
    """What one results dir holds: modes (full/incremental), arms, tasks, model."""
    modes, arms, tasks, model = set(), set(), set(), None
    for mf in ("run-manifest.json", "summary.json"):  # full manifest first, then incremental
        try:
            model = model or json.load(open(os.path.join(d, mf))).get("model")
        except (OSError, json.JSONDecodeError):
            continue
    cells = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(os.path.join(d, "*", "attempted.json"))}
    for p in glob.glob(os.path.join(d, "*", "*", "*", "verifier", "reward.txt")):
        cell = p
        for _ in range(4):
            cell = os.path.dirname(cell)
        cells.add(os.path.basename(cell))
    for c in cells:
        arm, task = split_cell(c)
        if arm:
            modes.add("full")
            arms.add(arm)
            tasks.add(task)
    for p in glob.glob(os.path.join(d, "incremental", "*.jsonl")):
        label, _, arm = os.path.basename(p)[:-len(".jsonl")].rpartition("-")
        modes.add("incremental")
        arms.add(arm or label)
        tasks.add(label or arm)
    return {"dir": d, "modes": sorted(modes), "arms": sorted(arms),
            "tasks": sorted(tasks), "model": model}


def discover_runs(roots=None):
    """Quality result dirs under `roots`, newest first — pure filesystem walk, never spends.

    A run dir is any directory holding full-mode artifacts (run-manifest.json /
    attempted.json cells / verifier rewards) or incremental artifacts
    (incremental/*.jsonl, summary.json). Used by `quality runs` and the wizard's
    view mode so stored results are discoverable without remembering paths. `roots=None`
    uses the settings-aware default (the configured quality_runs_dir first, then the
    legacy results tree) so freshly-saved runs are found without passing --roots.
    """
    if roots is None:
        from .paths import default_run_roots
        roots = default_run_roots()
    dirs = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat, up in (("run-manifest.json", 1), ("summary.json", 1), ("attempted.json", 2),
                        (os.path.join("incremental", "*.jsonl"), 2),
                        (os.path.join("verifier", "reward.txt"), 5)):
            for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
                d = p
                for _ in range(up):
                    d = os.path.dirname(d)
                if d:
                    dirs.add(d)
    infos = [i for i in (_run_info(d) for d in dirs) if i["modes"]]
    infos.sort(key=lambda i: os.path.getmtime(i["dir"]), reverse=True)
    return infos


def actions(path):
    msgs, points = parse_session(path)
    return [extract_action(msgs[i]["content"]) for i in points]


# ---------------------------------------------------------------- offline metrics
def _read_span(inp):
    off = inp.get("offset")
    start = off if isinstance(off, int) and off > 0 else 1
    lim = inp.get("limit")
    return start, (start + lim if isinstance(lim, int) and lim > 0 else float("inf"))


def _covered(spans, s, e):
    cur = s
    for a, b in sorted(spans):
        if a > cur:
            break
        cur = max(cur, b)
        if cur >= e:
            return True
    return cur >= e


def rework_count(acts):
    """Redundant re-fetches: re-read of an already-seen file span (range-aware), re-cat, re-run.

    A Write/Edit invalidates everything known about that file — re-reading, re-catting, or
    re-running a read-only command that touches it afterwards is VERIFICATION, not rework.
    Counting it would penalize verify-heavy behavior (which some compaction methods induce).

    Bash commands are un-rtk'd first. The rtk arm records WRAPPED commands, and both patterns
    below would miss them — RO is ^-anchored so `rtk grep …` never matches, and CAT looks for
    cat/head/tail by name while rtk renames all three to `rtk read`. Left unnormalized the arm
    scores a flawless ZERO rework on identical behaviour, flattering it on a headline verdict.
    """
    CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
    RO = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc)\b")
    read_spans, last_read, seen, hits = {}, {}, {}, 0
    for a in acts:
        if a.get("type") != "tool_use":
            continue
        name, inp = a.get("name"), a.get("input", {})
        if name in ("Write", "Edit"):
            fp = inp.get("file_path")
            if fp:
                read_spans[fp] = []
                last_read.pop(fp, None)
                seen = {c: 1 for c in seen if fp not in c}
        elif name == "Read":
            fp = inp.get("file_path")
            s, e = _read_span(inp)
            if fp and read_spans.get(fp) and _covered(read_spans[fp], s, e):
                hits += 1
            if fp:
                read_spans.setdefault(fp, []).append((s, e))
                last_read[fp] = 1
        elif name == "Bash":
            cmd = unrtk(inp.get("command", ""))
            if CAT.search(cmd) and any(f and f in cmd for f in last_read):
                hits += 1
            c = " ".join(cmd.split())
            if RO.match(cmd) and c in seen:
                hits += 1
            seen[c] = 1
    return hits


def band(xs):
    return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None


def overlaps(a, b):
    if not a or not b:
        return None
    return a[0] <= b[2] and b[0] <= a[2]


def _is_sidechain(path):
    """A sub-agent transcript: its records carry isSidechain=true from the first lines."""
    try:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n >= 20:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("isSidechain"):
                    return True
                if rec.get("type") in ("user", "assistant"):
                    return False
    except OSError:
        return True
    return False


def _trial_metrics(trial_dir, session_path=None):
    """cost_usd + total tokens + wall-clock seconds from a trial's result.json ({} if absent).
    Tokens = input + output, which IS the whole trajectory's spend: harbor's n_input_tokens
    already contains the cache tiers (verified against the transcripts — n_input_tokens equals
    input + cache_read + cache_creation on 342 of 399 opus-5 trials, the rest being transcripts
    truncated by a kill). n_cache_tokens is a SUBSET of it, reported separately for visibility,
    so the old input + cache + output sum double-counted every cache read — which on this suite
    inflated tokens by ~1.8x and halved every $-per-Mtok figure derived from them.

    Cost is priced from the transcript, NOT read from harbor's `cost_usd`. Harbor leaves that
    field null on a trial killed by the agent wall timeout — 7-25% of trials depending on the
    arm, and unevenly so, since an arm with a longer wall gets killed less often. Mixing
    harbor's number with our own on the trials it skipped made the $ columns a couple of
    percent light in an arm-dependent way. One formula over every trial keeps a column
    comparable across arms, and it agrees with harbor to ~2% in aggregate on the trials where
    both exist. Tokens still come from harbor's counters, so both columns cover the same
    trial set: a tokens mean over more trials than its $ mean is not comparable down the row.
    """
    try:
        r = json.load(open(os.path.join(trial_dir, "result.json")))
    except (OSError, json.JSONDecodeError):
        return {}
    ar = r.get("agent_result") or {}
    m = {}
    tok = sum(ar.get(k) or 0 for k in ("n_input_tokens", "n_output_tokens"))
    if tok:
        m["tok"] = tok
    if session_path:
        model = ((ar.get("model_info") or {}).get("name")
                 or ((r.get("agent_info") or {}).get("model_info") or {}).get("name"))
        try:
            m["cost"] = sum(cost_usd(u, model) for u in recorded_usage(session_path))
        except (OSError, ValueError, KeyError):
            pass
    if "cost" not in m and ar.get("cost_usd") is not None:
        m["cost"] = ar["cost_usd"]  # unreadable transcript — harbor's number is all there is
    try:
        from datetime import datetime
        s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
        f = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00"))
        m["lat"] = (f - s).total_seconds()
    except (KeyError, ValueError, TypeError):
        pass
    return m


# A compaction is a billed turn whose context came back SMALLER than the turn before it.
# Context only grows when nothing removes from it, so a drop is the removal itself — measured
# on the outcome rather than inferred from cache traffic. The earlier cache-side heuristic
# (a large cache_creation without matching context growth) also fires on ordinary prefix
# rewrites and over-counts by ~1.6x. The floor is jitter insurance only: at a 0 threshold
# vanilla and caveman still measure exactly 0.00 across 72 and 56 trials, so nothing real
# is being filtered here.
_COMP_DROP = 1_000


def _econ(path):
    """Per-trial economics, one pass over the transcript, keyed by BILLED TURN.

    recorded_usage() dedupes by requestId, so a response split across several records counts
    once — the unit the bill is actually computed on. Steps (tool_use blocks) are a different
    unit and are counted separately by actions(); the two are ~1:1 here but must not be
    conflated in a $-per-unit column.

    Returns None for a transcript too short to have a before/after (nothing to compare).
    """
    try:
        us = recorded_usage(path)
    except OSError:
        return None
    ctx = [ctx_tokens(u) for u in us]
    if len(ctx) < 2:
        return None
    return {
        "turns": len(us),
        "peak": max(ctx),
        "cache_w": sum(u.get("cache_creation_input_tokens", 0) or 0 for u in us),
        "cache_r": sum(u.get("cache_read_input_tokens", 0) or 0 for u in us),
        "comps": sum(1 for i in range(1, len(ctx)) if ctx[i - 1] - ctx[i] > _COMP_DROP),
        # billed tokens straight from the transcript, so tokens / cache share / turns all come
        # from one pass and cannot disagree on a trial harbor recorded oddly
        "tok": sum(c + (u.get("output_tokens", 0) or 0) for c, u in zip(ctx, us, strict=True)),
    }


def index_runs(root, agent):
    """One walk of the results tree -> {'<arm>-<task>': {'runs': [...], 'attempted': n}}.

    Keyed by the literal cell dir name (arm names can contain hyphens — headroom-kompress —
    so the name is not splittable; build() looks up f"{arm}-{task}" directly).
    Discovery anchors on verifier/reward.txt (a finished trial). attempted.json records
    how many trials generate.py REQUESTED, so trials that crashed or were killed by the
    wall timeout surface as missing instead of silently shrinking n.
    """
    idx = {}
    for rt in sorted(glob.glob(f"{root}/**/verifier/reward.txt", recursive=True)):
        inst = os.path.dirname(os.path.dirname(rt))                        # .../<trial>/<inst>
        key = os.path.basename(os.path.dirname(os.path.dirname(inst)))    # <arm>-<task>
        cands = [s for s in sorted(glob.glob(os.path.join(inst, AGENT_SESSION_GLOB[agent])))
                 if not _is_sidechain(s)]
        if not key or not cands:
            continue
        # >1 session file per trial (Task-tool sub-agents, resumes): the main trajectory
        # is the largest remaining transcript, not glob order
        s = max(cands, key=os.path.getsize)
        cell = idx.setdefault(key, {"runs": [], "attempted": None, "seen": set()})
        if s not in cell["seen"]:
            cell["seen"].add(s)
            cell["runs"].append((s, open(rt).read().strip(), _trial_metrics(inst, s)))
    for ap in glob.glob(f"{root}/**/attempted.json", recursive=True):
        key = os.path.basename(os.path.dirname(ap))
        try:
            k = int(json.load(open(ap)).get("k", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        cell = idx.setdefault(key, {"runs": [], "attempted": None, "seen": set()})
        cell["attempted"] = (cell["attempted"] or 0) + k
        cell["dir"] = os.path.dirname(ap)  # to tell "never ran" from "ran and crashed"
    return idx


# ---------------------------------------------------------------- assemble (from artifacts only)
# caveman-activate.js prefixes the ruleset it injects at SessionStart with this. Its absence
# from a transcript means the trial ran WITHOUT the intervention (the hook never fired, or
# Claude Code skipped hooks in --print mode). That is the one failure this arm cannot be
# allowed to have silently: an inactive caveman run is indistinguishable from vanilla, so it
# would score as a clean ✓ "no divergence" while measuring nothing at all.
CAVEMAN_MARKER = "CAVEMAN MODE ACTIVE"


def _caveman_inactive(runs):
    """How many of these trials have no caveman activation marker in their transcript.

    Substring scan of the raw jsonl rather than the parsed actions: the ruleset arrives as
    SessionStart context, which is not an assistant action and so never reaches actions().
    """
    n = 0
    for path, _reward, _m in runs:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                if not any(CAVEMAN_MARKER in line for line in fh):
                    n += 1
        except OSError:
            n += 1
    return n


def _bash_cmds(path):
    """Recorded Bash commands of one trial. Raises OSError/ValueError on an unreadable
    transcript — the caller treats that as inactive rather than as evidence of activity."""
    return [(a.get("input") or {}).get("command", "") for a in actions(path)
            if a.get("type") == "tool_use" and a.get("name") == "Bash"]


def _rtk_gain(path):
    """rtk's OWN execution stats for this trial, or None when the run predates them.

    `agent/rtk-gain.json` is dumped by harbor_agents/rtk_claude_code.py after the agent exits
    (`rtk gain -f json`), and it counts commands rtk ACTUALLY RAN. That is the only artifact
    that proves the intervention was HONORED rather than merely wired: the hook can emit a
    perfect rewrite that Claude Code then ignores, and every transcript-side signal below
    would still look active.
    """
    agent_dir = os.path.normpath(os.path.join(os.path.dirname(path), "..", "..", ".."))
    try:
        with open(os.path.join(agent_dir, "rtk-gain.json")) as fh:
            return json.load(fh).get("summary") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _rtk_hook_rewrote(path):
    """Did rtk's PreToolUse hook return an rtk-prefixed command anywhere in this transcript?

    Claude Code MOVED the rewrite out of the recorded action. Through 2.0.x the hook's
    `updatedInput.command` replaced the tool_use input, so an active trial literally stored
    `rtk git status`; by 2.1.228 the tool_use keeps the ORIGINAL command and the rewrite is
    recorded next to it, in a `hook_success` attachment carrying the hook's stdout. Since the
    container installs Claude Code unpinned, which shape a run has depends on the day it ran.

    Weaker than _rtk_gain: this proves the hook fired and produced a rewrite, not that the
    rewritten command is what executed. Used only for runs with no gain artifact.
    """
    try:
        with open(path) as fh:
            for line in fh:
                if '"hook_success"' not in line or "updatedInput" not in line:
                    continue     # cheap reject: hook attachments are a fraction of a transcript
                try:
                    att = (json.loads(line).get("attachment") or {})
                    out = json.loads(att.get("stdout") or "{}")
                except json.JSONDecodeError:
                    continue
                cmd = (((out.get("hookSpecificOutput") or {}).get("updatedInput")
                        or {}).get("command") or "")
                if re.match(r"\s*rtk\s", cmd):
                    return True
    except OSError:
        return False
    return False


def _rtk_inactive(runs):
    """How many of these trials ran WITHOUT rtk actually rewriting anything.

    The guard is STRUCTURAL rather than a text-marker scan, and so stronger than caveman's —
    but WHERE the structure lives has moved, so it reads three signals, strongest first:

      1. `rtk gain` stats dumped from the container: rtk counts what it RAN. >0 commands is
         proof the rewrite was honored end to end. Present, it is the only signal consulted —
         a transcript full of rewrites that rtk never executed is exactly the failure the
         other two cannot see.
      2. a `hook_success` attachment whose stdout carries an rtk-prefixed `updatedInput`
         (Claude Code ≥2.1.x records the rewrite beside the unchanged tool_use).
      3. the `rtk ` prefix in the recorded tool_use itself (Claude Code ≤2.0.x).

    None of the three, on a trial that issued Bash, means the hook never fired (missing
    binary, hook not wired, Claude Code skipping hooks under --print) — vanilla in disguise,
    which would otherwise score a clean ✓ "trajectory preserved" while measuring nothing.

    A trial that ran NO Bash commands is not counted: rtk only wraps shell commands, so having
    nothing to rewrite is a property of the task, not a failed intervention.
    """
    n = 0
    for path, _reward, _m in runs:
        gain = _rtk_gain(path)
        try:
            cmds = _bash_cmds(path)
        except (OSError, ValueError):
            n += 1
            continue
        if not cmds:
            continue
        if gain is not None:
            n += 0 if (gain.get("total_commands") or 0) > 0 else 1
            continue
        if not (any(re.match(r"\s*rtk\s", c or "") for c in cmds) or _rtk_hook_rewrote(path)):
            n += 1
    return n


def _inactive_for(arm, runs):
    """Trials of `arm` that ran without their intervention, or None for arms where the notion
    doesn't apply (a proxy arm either routed or it didn't — the run would have failed loudly)."""
    if arm == "caveman":
        return _caveman_inactive(runs)
    if arm == "rtk":
        return _rtk_inactive(runs)
    return None


def inactive_total(arm_rows, arm):
    """Trials of `arm` across these rows that ran WITHOUT their intervention (caveman / rtk).

    Surfaced in every renderer's per-arm heading, not just one: an inactive trial is vanilla
    in disguise and scores a clean ✓, so a reader who only sees the HTML must not be told
    "trajectory preserved" about a run where the method never loaded.
    """
    return sum((r["arms"][arm].get("inactive") or 0) for r in arm_rows)


def _cell_stats(cell):
    runs = cell["runs"] if cell else []
    lens, rws, peaks = [], [], []
    costs, toks, lats = [], [], []
    trials = []
    for p, rw, m in runs:
        acts = actions(p)
        lens.append(len(acts))
        rws.append(rework_count(acts))
        e = _econ(p)
        # _econ's peak IS peak_ctx (both are max ctx_tokens over recorded_usage); reuse it
        # rather than re-walking the transcript, and fall back when the trial was too short
        peaks.append(e["peak"] if e else peak_ctx(p))
        if m.get("cost") is not None:
            costs.append(m["cost"])
        if m.get("tok") is not None:
            toks.append(m["tok"])
        if m.get("lat") is not None:
            lats.append(m["lat"])
        # one record per trial, fields aligned — the economics summary resamples TRIALS, so
        # it needs cost/tokens/turns to travel together rather than as separate filtered lists
        if e:
            trials.append({**e, "solve": 1.0 if rw == "1" else 0.0, "steps": len(acts),
                           "cost": m.get("cost"), "tok": e["tok"] or m.get("tok")})
    n = len(runs)
    attempted = cell["attempted"] if cell and cell["attempted"] else n
    started = n  # trial dirs that actually opened (reward or not); >n means some crashed
    cdir = cell.get("dir") if cell else None
    if cdir:
        try:
            started = max(started, sum(1 for e in os.scandir(cdir) if e.is_dir()))
        except OSError:
            pass
    return {
        "n": n, "attempted": max(attempted, n), "lost": max(attempted, n) - n,
        "started": started,
        # a trial that never finished is a failure on the solve axis, not missing data
        "solve": sum(r == "1" for _, r, _m in runs),
        "length": band(lens), "rework": band(rws), "_lens": lens,
        "peak_ctx": max(peaks, default=0),
        # cost/token/latency for the $ and tokens columns; lists so a band can be shown
        "_costs": costs, "_toks": toks, "_lats": lats,
        "_trials": trials,
    }


def build(args, include_incremental_only=True):
    root = args.__dict__["from"]
    arms = [a for a in args.arms.split(",") if a]
    idx = index_runs(root, args.agent)
    milestones = {}
    for mp in sorted(glob.glob(f"{root}/**/milestones.json", recursive=True)):
        try:
            milestones.update(json.load(open(mp)))  # {task: {arm: [min,mean,max]}}
        except (OSError, json.JSONDecodeError):
            continue
    incr = _load_incremental(root, arms)
    rows = []
    gate = getattr(args, "ctx_gate", 50_000)
    # curated full-run tasks first, then (unless scoped off — e.g. the inline render after a
    # full run) any task that only has incremental-replay data, so session-labelled replays
    # (swe-long, a uuid) still get a row when the report is pointed at an incremental dir
    curated = list(resolve_tasks(args.tasks))
    extra = sorted({t for t, _a in incr} - set(curated)) if include_incremental_only else []
    for task in curated + extra:
        v = _cell_stats(idx.get(f"vanilla-{task}"))
        row = {"task": task, "vanilla": v, "arms": {},
               "sub_gate": bool(v["peak_ctx"]) and v["peak_ctx"] < gate}
        for arm in arms:
            a = _cell_stats(idx.get(f"{arm}-{task}"))
            enough = len(v["_lens"]) >= 2 and len(a["_lens"]) >= 2
            mv = (milestones.get(task, {}) or {}).get("vanilla")
            mc = (milestones.get(task, {}) or {}).get(arm)
            a.update(
                length_ok=overlaps(v["length"], a["length"]) if enough else None,
                rework_ok=overlaps(v["rework"], a["rework"]) if enough else None,
                milestone=mc, milestone_ok=(overlaps(mv, mc) if (mv and mc) else None),
                incr=incr.get((task, arm)),
                # only meaningful for the in-agent arms; None elsewhere so the column stays
                # quiet. caveman is detected by its injected marker, rtk structurally by the
                # rtk prefix its hook writes into the recorded command.
                inactive=_inactive_for(arm, (idx.get(f"{arm}-{task}") or {}).get("runs", [])),
            )
            row["arms"][arm] = a
        rows.append(row)
    model = None
    try:  # the run's model, for the report title (written by generate.full)
        model = json.load(open(f"{root}/run-manifest.json")).get("model")
    except (OSError, json.JSONDecodeError):
        pass
    return {"arms": arms, "rows": rows, "model": model,
            "has_milestone": bool(milestones), "has_incr": bool(incr)}


def _load_incremental(root, arms):
    """generate --mode incremental jsonl -> per (task,arm) comp% / costΔ / fidelity VS CONTROL.

    Everything is computed on the common step set (both arms answered, step > 0 — the
    cold-cache first step is excluded consistently for tokens, cost AND fidelity), and
    the arm's action-fidelity is reported next to control's: the control incremental run is
    the noise floor (sampling + reconstruction error); only the gap below it is signal.
    """
    def rows(p):
        d = {}
        if not os.path.exists(p):
            return d
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in r and "usage" in r:
                d[r["step"]] = r
        return d

    ctx = ctx_tokens  # the shared context-size definition (engine.ctx_tokens)

    out = {}
    for cf in sorted(glob.glob(f"{root}/**/incremental/*-control.jsonl", recursive=True)):
        task = os.path.basename(cf)[: -len("-control.jsonl")]
        C = rows(cf)
        for arm in arms:
            A = rows(os.path.join(os.path.dirname(cf), f"{task}-{arm}.jsonl"))
            if not A or not C:
                continue
            common = sorted(s for s in (set(A) & set(C)) if s > 0)
            if not common:
                continue
            cc = sum(ctx(C[s]["usage"]) for s in common) or 1
            ac = sum(ctx(A[s]["usage"]) for s in common)
            oc = sum(C[s].get("cost_usd", 0) for s in common) or 1e-9
            zc = sum(A[s].get("cost_usd", 0) for s in common)
            out[(task, arm)] = {
                "comp": round(1 - ac / cc, 4), "costd": round(1 - zc / oc, 4),
                # caveman applies its transform (ruleset + terse prose) on every step, so it is
                # "engaged" even when the net context change is ~0 — its terse prose can still
                # move the next decision. This flag keeps fid meaningful (not dimmed as a
                # passthrough) at comp≈0, exactly as `retrieves` does for headroom CCR.
                # Read the VALUE, not the key: counterfactual writes caveman_native on every
                # caveman step (True or False), so a presence test would mark a run whose every
                # step reverted to the recorded verbose prose as engaged — the one case where
                # fid really is measuring nothing.
                "native": any(r.get("caveman_native") for r in A.values()),
                # rtk: did its filters actually shrink any recorded observation? A session-level
                # fact (the transform is deterministic and runs once up front) stamped onto
                # every step so it survives here without depending on the .done sentinel.
                # 0/absent = no recorded command had an rtk pipe filter, i.e. a genuine
                # passthrough — and then ⊘ below is the correct label, not a bug.
                "rtk_filtered": max((r.get("rtk_filtered") or 0) for r in A.values()) or 0,
                # per-step wall-clock (only on newer artifacts) — mean over the common steps
                "latency": _mean_latency(A, common), "latency_ctrl": _mean_latency(C, common),
                "fid": sum(_faithful_step(A[s]) for s in common) / len(common),
                "fid_ctrl": sum(_faithful_step(C[s]) for s in common) / len(common),
                "redund": sum(bool(A[s].get("redundant")) for s in common),
                # PAIRED per-step vectors (arm and control aligned index-for-index over the
                # common steps) — the overall summary pools these across sessions and
                # bootstraps them for error bars. Kept private: they are raw material for
                # summarize(), not a displayed field.
                "_good": [int(_good_step(A[s])) for s in common],
                "_good_ctrl": [int(_good_step(C[s])) for s in common],
                "_red": [int(bool(A[s].get("redundant"))) for s in common],
                "_red_ctrl": [int(bool(C[s].get("redundant"))) for s in common],
                # only steps where BOTH legs recorded the field — older artifacts predate
                # redundancy scoring, and a missing flag is unknown, not "not redundant"
                "_red_ok": [int("redundant" in A[s] and "redundant" in C[s]) for s in common],
                # token totals behind `comp`, so pooling across sessions can weight by size
                # instead of averaging percentages of wildly different denominators
                "_ctx": ac, "_ctx_ctrl": cc,
                # CCR engagement (headroom): retrieves the agent made, and whether the arm ran
                # with the retrieve loop wired at all (the field is only written when it is)
                "retrieves": sum(r.get("ccr_retrieves", 0) for r in A.values()),
                "ccr": any("ccr_retrieves" in r for r in A.values()),
                "scoring": _scoring(A), "steps": len(common),
            }
    return out


def _mean_latency(rowset, common):
    """Mean per-step wall-clock over the common steps; None when nothing was timed."""
    v = [rowset[s].get("latency_s") for s in common]
    v = [x for x in v if isinstance(x, (int, float))]
    return (sum(v) / len(v)) if v else None


def _scoring(recset):
    """Infer per-step scoring from the stored records: an LLM goal judge (each action rated
    good/degraded/bad toward the task), an LLM equivalence judge (structural near-misses
    upgraded), or plain structural agreement (exact/action match, no LLM)."""
    vals = list(recset.values())
    if any("quality" in r for r in vals):
        return "llm:goal"
    if any(r.get("agree_semantic") not in (None, r.get("agree_action")) for r in vals):
        return "llm:equiv"
    return "struct"


def _good_step(r):
    """Was this step's action a valid one, ignoring redundancy?

    'Valid' depends on how the run was scored. With a GOAL judge (each action rated on its own
    merit toward the task), a good step is a good step — control replaying itself scores ~100%,
    so an arm only loses by taking WORSE steps: a trustworthy floor. Without a judge, we fall
    back to STRUCTURAL agreement with the original recorded action, whose ceiling is low even
    for control (the model rarely reproduces its own past sampling) — noisy, not a true 100%.
    """
    q = r.get("quality")
    if q is not None:
        return q == "good"                                   # goal judge
    return bool(r.get("agree_semantic", r.get("agree_action")))  # structural / equiv near-miss


def _faithful_step(r):
    """A faithful step took a valid action AND didn't redundantly re-fetch info it already had.

    The two halves are separable and the overall summary reports them separately (quality vs
    redundant steps) — this conjunction is the single per-step number the per-task fidelity
    column is built from."""
    return _good_step(r) and not r.get("redundant")


# ---------------------------------------------------------------- overall summary (pooled)
# The per-task tables answer "what happened on THIS task"; every cell there is one or a few
# trials, so nothing in them carries an error bar and a reader is left eyeballing a column of
# small numbers. This section answers the other question — "across everything that was run,
# how does each arm compare to the no-compaction baseline, and is the difference bigger than
# the noise?" — by pooling and attaching a 95% CI to every number.
#
# Deliberately still just arithmetic over stored artifacts: no model call, no scipy. The CIs
# are a seeded bootstrap so the same artifacts always render the same report.
_BOOT_N = 2000
_BOOT_SEED = 0xC0FFEE


def _ci(draws, mean):
    """(mean, lo, hi) at 95% from sorted bootstrap draws — percentile method."""
    if not draws:
        return (mean, mean, mean)
    d = sorted(draws)
    lo = d[int(0.025 * len(d))]
    hi = d[min(len(d) - 1, int(0.975 * len(d)))]
    return (mean, lo, hi)


def _boot_pair(arm_vec, ctrl_vec, weights=None):
    """Bootstrap a paired rate comparison.

    arm_vec / ctrl_vec are aligned 0/1 vectors over the SAME units (steps, or tasks). Resamples
    unit INDICES with replacement, so the arm and its control always move together — the
    comparison is paired, which is the whole point of teacher-forcing the same steps through
    both legs. `weights` (per-unit denominators, for task-level rates) makes each unit's
    contribution its own rate rather than a count.

    Returns (arm(mean,lo,hi), ctrl(mean,lo,hi), delta(mean,lo,hi)) in RATE units (0-1).
    Caveat kept honest in the legend: steps within one session are correlated and this
    resamples them as if independent, so step-level bars are, if anything, optimistic.
    """
    n = len(arm_vec)
    if not n:
        return None, None, None
    w = weights or [1.0] * n
    tot = sum(w) or 1.0
    am = sum(a * x for a, x in zip(arm_vec, w, strict=True)) / tot
    cm = sum(c * x for c, x in zip(ctrl_vec, w, strict=True)) / tot
    rng = random.Random(_BOOT_SEED)
    da, dc, dd = [], [], []
    idx = range(n)
    for _ in range(_BOOT_N):
        s = [rng.choice(idx) for _ in idx]
        t = sum(w[i] for i in s) or 1.0
        a = sum(arm_vec[i] * w[i] for i in s) / t
        c = sum(ctrl_vec[i] * w[i] for i in s) / t
        da.append(a)
        dc.append(c)
        dd.append(a - c)
    return _ci(da, am), _ci(dc, cm), _ci(dd, am - cm)


def _solve_rate(c):
    """A cell's solve rate, with lost trials counted as failures (the report's convention)."""
    n = c["attempted"] or c["n"]
    return (c["solve"] / n) if n else None


# ------------------------------------------------- full-run economics (geometric, vs vanilla)
# The pooled numbers this replaced were dollar-weighted in disguise. A ratio of macro-means
# (Σ arm / Σ vanilla) lets the costliest tasks carry the result — on the opus-5 suite three of
# fourteen tasks carried 41% of the spend, so "cost +81%" was really "cost +81% on the three
# tasks that happen to be expensive". Each task gets ONE vote here instead:
#
#     delta = exp( mean over tasks of log( arm_task / vanilla_task ) ) - 1
#
# and the interval is a TWO-level bootstrap — resample tasks, then resample trials within each
# resampled cell. Resampling tasks alone treats a 4-trial cell mean as if it were exact and
# produces intervals far too narrow; two levels is what makes vanilla's own spread visible.
# Tasks are shared across arms (the comparison is paired), trial draws are independent per arm.
#
# Three metrics cannot be ratios and say so in their own units:
#   solve rate     percentage POINTS — a ratio is undefined when vanilla is 0% and meaningless
#                  when it is 100%, which is 8 of 14 tasks on this suite.
#   compactions    ABSOLUTE per-trial count — vanilla is exactly 0, so every ratio is infinite.
#   cache write    a share already; its ratio is still meaningful, so it stays a ratio.
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _ratio(num, den):
    n, d = sum(x for x in num if x is not None), sum(x for x in den if x is not None)
    return (n / d) if d else None


_FULL_METRICS = (
    ("solve",  "solve",   "rate, pts",  "pts",
     lambda c: _mean([t["solve"] for t in c]) and _mean([t["solve"] for t in c]) * 100,
     lambda v: f"{v:.0f}%"),
    ("turns",  "turns",   "per trial",  "ratio",
     lambda c: _mean([t["turns"] for t in c]), lambda v: f"{v:.1f}"),
    ("peak",   "peak ctx", "tokens",    "ratio",
     lambda c: _mean([t["peak"] for t in c]), lambda v: f"{v:,.0f}"),
    ("tok",    "tokens",  "per trial",  "ratio",
     lambda c: _mean([t["tok"] for t in c]), lambda v: f"{v:,.0f}"),
    ("cost",   "$",       "per trial",  "ratio",
     lambda c: _mean([t["cost"] for t in c]), lambda v: f"${v:.2f}"),
    ("cstep",  "$",       "per step",   "ratio",
     lambda c: _ratio([t["cost"] for t in c], [t["steps"] for t in c]),
     lambda v: f"${v:.4f}"),
    ("rate",   "$",       "per Mtok",   "ratio",
     lambda c: (lambda r: r and r * 1e6)(_ratio([t["cost"] for t in c], [t["tok"] for t in c])),
     lambda v: f"${v:.2f}"),
    ("cwsh",   "cache wr", "share",     "ratio",
     lambda c: (lambda r: r and r * 100)(_ratio([t["cache_w"] for t in c],
                                                [t["cache_w"] + t["cache_r"] for t in c])),
     lambda v: f"{v:.1f}%"),
    ("comp",   "compact", "per trial",  "abs",
     lambda c: _mean([t["comps"] for t in c]), lambda v: f"{v:.2f}"),
)


def _agg(trials, fn):
    """A cell's value for one metric, or None when the trials can't support it."""
    if not trials:
        return None
    try:
        v = fn(trials)
    except (ZeroDivisionError, TypeError, statistics.StatisticsError):
        return None
    return v if isinstance(v, (int, float)) else None


def _full_effect(arm_cells, van_cells, tasks, fn, mode, rng=None):
    """One task-equal effect estimate. With `rng`, trials inside each cell are resampled too.

    ratio -> geometric mean of per-task ratios, as a percentage change.
    pts   -> arithmetic mean of per-task differences, in the metric's own units.
    abs   -> arithmetic mean of the arm's own per-task values (no vanilla reference).
    """
    acc = []
    for t in tasks:
        a, v = arm_cells.get(t), van_cells.get(t)
        if not a:
            continue
        if rng is not None:
            a = [a[rng.randrange(len(a))] for _ in a]
            if v:
                v = [v[rng.randrange(len(v))] for _ in v]
        av = _agg(a, fn)
        if av is None:
            continue
        if mode == "abs":
            acc.append(av)
            continue
        vv = _agg(v, fn)
        if vv is None:
            continue
        if mode == "pts":
            acc.append(av - vv)
        elif av > 0 and vv > 0:            # a log needs both legs strictly positive
            acc.append(math.log(av / vv))
    if not acc:
        return None
    if mode == "ratio":
        return 100 * (math.exp(statistics.fmean(acc)) - 1)
    return statistics.fmean(acc)


def _full_ci(arm_cells, van_cells, tasks, fn, mode):
    """(point, lo, hi) at 95% from the two-level bootstrap. Seeded: same artifacts, same bars."""
    point = _full_effect(arm_cells, van_cells, tasks, fn, mode)
    if point is None or not tasks:
        return None
    rng = random.Random(_BOOT_SEED)
    draws = []
    for _ in range(_BOOT_N):
        ts = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        v = _full_effect(arm_cells, van_cells, ts, fn, mode, rng)
        if v is not None:
            draws.append(v)
    return _ci(draws, point)


def _ground(van_cells, tasks, fn, mode):
    """Vanilla's own level for a metric — the number the percentages are percentages OF.

    Geometric over tasks for ratio metrics so that ground x (1 + delta) lands on the arm's
    own geometric mean; arithmetic for the others, where that identity doesn't apply.
    """
    vals = [x for x in (_agg(van_cells.get(t), fn) for t in tasks) if x is not None]
    if not vals:
        return None
    if mode == "ratio":
        pos = [v for v in vals if v > 0]
        return math.exp(statistics.fmean([math.log(v) for v in pos])) if pos else None
    return statistics.fmean(vals)


def summarize_full(d):
    """Per-arm economics of the FULL runs, every metric as a task-equal effect vs vanilla.

    Only tasks where BOTH the arm and vanilla produced a usable trial are counted, so every
    number down a column rests on the same task set. Arms may cover different tasks; each is
    reported over its own, and the caller is told when they diverge.
    """
    van = {r["task"]: r["vanilla"].get("_trials") or [] for r in d["rows"]}
    van = {t: v for t, v in van.items() if v}
    out = {"arms": [], "tasks": {}, "metrics": {}, "ground": {}, "ntrials": {}}
    for arm in d["arms"]:
        cells = {r["task"]: r["arms"][arm].get("_trials") or [] for r in d["rows"]}
        cells = {t: v for t, v in cells.items() if v and t in van}
        tasks = sorted(cells)
        if not tasks:
            continue
        out["arms"].append(arm)
        out["tasks"][arm] = tasks
        out["ntrials"][arm] = (sum(len(cells[t]) for t in tasks),
                               sum(len(van[t]) for t in tasks))
        out["metrics"][arm] = {k: _full_ci(cells, van, tasks, fn, mode)
                               for k, _h, _s, mode, fn, _f in _FULL_METRICS}
        out["ground"][arm] = {k: _ground(van, tasks, fn, mode)
                              for k, _h, _s, mode, fn, _f in _FULL_METRICS}
    return out


def _ntask(n):
    return f"{n} task{'s' * (n != 1)}"


def full_summary_rows(d):
    """The economics summary as renderer-agnostic rows — console, md and HTML share it."""
    s = summarize_full(d)
    if not s["arms"]:
        return None
    ref = max(s["arms"], key=lambda a: len(s["tasks"][a]))
    shared = all(s["tasks"][a] == s["tasks"][ref] for a in s["arms"])

    def gcell(arm, k, mode, fmt):
        g = s["ground"][arm][k]
        return {"txt": fmt(g) if g is not None else "—", "sub": ""}

    def acell(arm, k, mode):
        tri = s["metrics"][arm][k]
        if not tri:
            return {"txt": "—", "sub": "", "sig": False}
        m, lo, hi = tri
        sig = mode != "abs" and (lo > 0 or hi < 0)
        if mode == "abs":
            return {"txt": f"{m:.2f}", "sub": f"[{lo:.2f}, {hi:.2f}]", "sig": False}
        unit = "%" if mode == "ratio" else " pts"
        return {"txt": f"{m:+.1f}{unit}", "sub": f"[{lo:+.0f}, {hi:+.0f}]", "sig": sig}

    rows = []
    if shared:
        rows.append({"arm": "vanilla", "note": f"{_ntask(len(s['tasks'][ref]))} · ground",
                     "control": True,
                     "cells": [gcell(ref, k, mode, fmt)
                               for k, _h, _sb, mode, _fn, fmt in _FULL_METRICS]})
    for arm in s["arms"]:
        na, nv = s["ntrials"][arm]
        if not shared:
            rows.append({"arm": "vanilla", "note": f"{_ntask(len(s['tasks'][arm]))} · ground",
                         "control": True,
                         "cells": [gcell(arm, k, mode, fmt)
                                   for k, _h, _sb, mode, _fn, fmt in _FULL_METRICS]})
        rows.append({"arm": arm, "note": f"{_ntask(len(s['tasks'][arm]))} · {na}v{nv} trials",
                     "control": False,
                     "cells": [acell(arm, k, mode)
                               for k, _h, _sb, mode, _fn, _f in _FULL_METRICS]})
    return {"rows": rows, "shared": shared, "summary": s}


def summarize(d):
    """Pool every task/session into ONE row per arm + a baseline row, each number with a CI.

    Three quality axes, from the two modes that measure different things (kept labelled, never
    blended into a single score):
      - full runs      solve rate (verifier passed), macro-averaged over tasks so a task with
                       many trials doesn't outvote one with few; CI bootstraps TASKS.
      - incremental    per-step action quality (goal judge's `good`, or structural agreement
                       when the run wasn't goal-judged) and the redundant-step rate; both
                       paired against control step-for-step, CI bootstraps STEPS.
      - context        tokens actually removed vs control — not a quality axis, but the one
                       number without which the quality columns cannot be read (an arm that
                       compressed nothing has no quality result, only an absence of one).

    Both modes count only material where the method could ACT, and the two exclusions are the
    same rule wearing different clothes:
      - incremental drops sessions the arm passed through (`_engaged`: it never compressed,
        retrieved, or applied its transform);
      - full drops ⊘ tasks, where vanilla's peak context never reached the compaction gate, so
        nothing could compact.
    Including either would pad every arm toward control and let "was never applied" render as
    "trajectory preserved". Both counts are carried out and shown, so nothing is dropped quietly.
    """
    out = {"arms": [], "n_arms": {}, "sessions_skipped": {}}
    for arm in d["arms"]:
        # ---- full mode: one solve rate per task, paired with vanilla's on the same task
        a_rates, v_rates, tasks, short = [], [], [], []
        for r in d["rows"]:
            a, v = r["arms"][arm], r["vanilla"]
            ar, vr = _solve_rate(a), _solve_rate(v)
            if ar is None or vr is None:
                continue
            if gated(arm, r["sub_gate"]):
                short.append(r["task"])
                continue
            a_rates.append(ar)
            v_rates.append(vr)
            tasks.append(r["task"])
        full_a, full_c, full_d = _boot_pair(a_rates, v_rates) if a_rates else (None, None, None)

        # ---- incremental: pool the paired per-step vectors of every ENGAGED session
        g_a, g_c, r_a, r_c, ctx_a, ctx_c = [], [], [], [], 0, 0
        sessions, skipped = [], []
        for r in d["rows"]:
            inc = r["arms"][arm].get("incr") or {}
            if not inc.get("_good"):
                continue
            if not _engaged(inc):
                skipped.append(r["task"])
                continue
            sessions.append(r["task"])
            g_a += inc["_good"]
            g_c += inc["_good_ctrl"]
            r_a += [x for x, ok in zip(inc["_red"], inc["_red_ok"], strict=True) if ok]
            r_c += [x for x, ok in zip(inc["_red_ctrl"], inc["_red_ok"], strict=True) if ok]
            ctx_a += inc.get("_ctx", 0)
            ctx_c += inc.get("_ctx_ctrl", 0)
        good_a, good_c, good_d = _boot_pair(g_a, g_c) if g_a else (None, None, None)
        red_a, red_c, red_d = _boot_pair(r_a, r_c) if r_a else (None, None, None)
        out["arms"].append(arm)
        out[arm] = {
            "full": full_a, "full_ctrl": full_c, "full_d": full_d, "full_tasks": tasks,
            "full_short": short,
            "good": good_a, "good_ctrl": good_c, "good_d": good_d,
            "red": red_a, "red_ctrl": red_c, "red_d": red_d,
            "comp": (1 - ctx_a / ctx_c) if ctx_c else None,
            "sessions": sessions, "skipped": skipped,
            "steps": len(g_a), "red_steps": len(r_a),
            "scoring": next((r["arms"][arm]["incr"]["scoring"] for r in d["rows"]
                             if (r["arms"][arm].get("incr") or {}).get("scoring")), None),
        }
    return out


def _delta(delta, scale=100.0, dec=1):
    """The paired arm-vs-control difference, printed as a number rather than a verdict.

    The ± on a value is a MARGINAL interval; it can't be eyeballed against control's, because
    the two legs are paired step-for-step and move together. This is that paired difference
    with its own CI — a bar that clears zero is a real difference, one that straddles it isn't.
    The table shows the number and leaves that call to the reader.
    """
    if not delta:
        return ""
    m, lo, hi = delta
    return f"Δ{m * scale:+.{dec}f} ±{max(hi - m, m - lo) * scale:.{dec}f}"


# Full-run solve rate used to head this table as a pooled rate. It now lives in the economics
# table above, as a task-equal percentage-POINT delta with a two-level interval — the pooled
# version answered a different (dollar-weighted) question and the two disagreed on screen.
# What stays here is the incremental replay, which measures something else entirely.
SUMMARY_COLS = (("quality", "incremental"),
                ("redundant", "/100 steps"), ("context", "removed"))


def _ref_arm(s):
    """The arm with the widest coverage — its control values head the table when all the arms
    share them."""
    return max(s["arms"], key=lambda a: (len(s[a]["full_tasks"]), s[a]["steps"]), default=None)


def _refs_agree(s, ref):
    """Do all arms share the reference arm's control values (to within a rounding step)?

    They do whenever the arms ran over the same tasks and sessions — the normal case, and then
    one control row is enough. When they don't, a single baseline row would be comparing each
    arm against material it never ran, so the table repeats control per arm instead.
    """
    for a in s["arms"]:
        for k in ("full_ctrl", "good_ctrl", "red_ctrl"):
            x, y = s[a][k], s[ref][k]
            if (x is None) != (y is None):
                return False
            if x and y and abs(x[0] - y[0]) > 0.005:
                return False
    return True


def _arm_note(a):
    """Why an arm's columns might be thinner than they look: material where the method could
    not act was dropped, or the artifacts are too old to carry redundancy scoring."""
    note = []
    if a["full_short"]:
        note.append(f"⊘{len(a['full_short'])} too short")
    if a["skipped"]:
        note.append(f"⊘{len(a['skipped'])} passthrough")
    if a["steps"] and not a["red_steps"]:
        note.append("no redundancy data")
    return " · ".join(note)


def _cell(tri, delta=None, sub="", scale=100.0, dec=1):
    """One summary cell, renderer-agnostic: a value ±CI, with a second line underneath.

    That second line is the denominator on a control row and the paired delta on an arm row.
    No cell carries a good/bad verdict: at bench-scale k most differences are indistinguishable
    from control, and marking cells by which side of the mean they landed would invent a result
    the numbers don't support. The delta and its CI say it all, and the reader reads them.
    """
    if not tri:
        return {"txt": "—", "sub": sub}
    m, lo, hi = tri
    hw = max(hi - m, m - lo)
    return {"txt": f"{m * scale:.{dec}f} ±{hw * scale:.{dec}f}", "sub": sub or _delta(delta)}


def summary_rows(d):
    """The overall summary as renderer-agnostic rows — console, md and HTML all derive from
    this one function so the three views cannot drift apart (the same contract `_cmp` and
    `_verdict` hold for the per-task tables).

    Each arm carries its OWN control reference, paired over that arm's material. When the arms
    ran over the same tasks and sessions those references coincide and one shared control row
    heads the table; when they don't, control is repeated per arm — the only way the absolute
    numbers stay comparable down a column.
    """
    s = summarize(d)
    ref = _ref_arm(s)
    # incremental-only now: a full-mode run has nothing to say here and renders no table
    if not ref or not any(s[a]["good"] for a in s["arms"]):
        return None
    shared = _refs_agree(s, ref)

    def control(a):
        ns = f"{a['steps']} steps/{len(a['sessions'])}s" if a["sessions"] else ""
        nr = f"{a['red_steps']} steps" if a["red_steps"] and a["red_steps"] != a["steps"] else ""
        return {"arm": "control", "note": "", "control": True, "cells": [
            _cell(a["good_ctrl"], sub=ns),
            _cell(a["red_ctrl"], sub=nr), {"txt": "—", "sub": ""}]}

    def arm(name):
        a = s[name]
        comp = a["comp"]
        # an arm that removed <2% never really fired: its flat quality columns are an absence
        # of measurement, and saying so here is the difference between "preserved" and "untested"
        ccell = {"txt": "—", "sub": ""} if comp is None else {
            "txt": f"{comp * 100:+.1f}%", "sub": "⊘ barely fired" if abs(comp) < 0.02 else ""}
        return {"arm": name, "note": _arm_note(a), "control": False, "cells": [
            _cell(a["good"], a["good_d"]),
            _cell(a["red"], a["red_d"]), ccell]}

    rows = []
    if shared:
        rows.append(control(s[ref]))
    for name in s["arms"]:
        if not shared:
            rows.append(control(s[name]))
        rows.append(arm(name))
    return {"rows": rows, "shared": shared, "summary": s}


# ---------------------------------------------------------------- rendering
def _b(x):
    return "—" if not x else (f"{x[1]:.0f}[{x[0]}-{x[2]}]" if x[0] != x[2] else f"{x[1]:.0f}")


def _v(ok):
    return "✓ OK" if ok else ("✗ DIVERGES" if ok is False else "—")


def _pct(x):
    return "—" if x is None else f"{x * 100:+.0f}%"


def _solve(c):
    # requested but no trial dir ever opened -> aborted / never ran, not a real 0/k
    if c["n"] == 0 and c.get("started", 0) == 0:
        return "—"
    s = f"{c['solve']}/{c['attempted']}"
    return s + (f" (⚠{c['lost']} lost)" if c["lost"] else "")


def table(d):
    """One (header, rows) build shared by console, md and html renderers.

    Each row is [(cell_text, ok_flag_or_None), ...]; ok drives ✓/✗ colouring in html.
    """
    head = ["task", "arm", "solve v·arm", "ctx peak (v)", "length (v)", "length (arm)",
            "len", "rework"]
    if d["has_milestone"]:
        head.append("milestone")
    if d["has_incr"]:
        head += ["fid (arm·ctrl)", "comp", "$Δ"]
    rows = []
    for r in d["rows"]:
        for arm in d["arms"]:
            a = r["arms"][arm]
            pk = r["vanilla"]["peak_ctx"]
            pk_txt = ("—" if not pk else f"{pk / 1000:.0f}k"
                      + (" ⊘" if gated(arm, r["sub_gate"]) else ""))
            sv, sa = _solve(r["vanilla"]), _solve(a)
            solve_txt = "—" if sv == "—" and sa == "—" else f"{sv} · {sa}"
            # an arm that never activated measured nothing — say so on the arm itself, or a
            # ✓ here reads as "preserved the trajectory" when it means "was never applied"
            arm_txt = arm if not a.get("inactive") else f"{arm} ⚠{a['inactive']} inactive"
            cells = [(r["task"], None), (arm_txt, None),
                     (solve_txt, None),
                     (pk_txt, None),
                     (_b(r["vanilla"]["length"]), None), (_b(a["length"]), None),
                     (_v(a["length_ok"]), a["length_ok"]),
                     (_v(a["rework_ok"]), a["rework_ok"])]
            if d["has_milestone"]:
                cells.append((_v(a["milestone_ok"]), a["milestone_ok"]))
            if d["has_incr"]:
                inc = a.get("incr") or {}
                # "passthrough" = the arm changed nothing measurable → fid deltas are noise.
                # comp≈0 usually means that, EXCEPT caveman still applied its transform (native),
                # so its fid stays meaningful even when the net context change is ~0.
                moved = (abs(inc.get("comp") or 0) >= 0.02 or inc.get("native")
                         or inc.get("rtk_filtered"))
                if inc.get("fid") is None:
                    fid = "—"
                elif not moved:
                    fid = "⊘ passthrough"
                else:
                    fid = f"{inc['fid']:.0%} · {inc['fid_ctrl']:.0%}"
                cells += [(fid, None), (_pct(inc.get("comp")), None),
                          (_pct(inc.get("costd")), None)]
            rows.append(cells)
    return head, rows


SUB = ("vanilla = the noise floor; ✓ = the arm's band overlaps vanilla's, ✗ = disjoint "
       "(needs ≥2 finished runs/arm). In solve, ⚠ lost = trials that opened but crashed / "
       "timed out (counted as unsolved), while — = the cell was never run in this pass "
       "(no trial ever opened — aborted or out of scope, not a failure). "
       "⊘ = vanilla's peak context stayed below the compaction gate "
       "(--ctx-gate, default 50k): condense's whole-conversation compaction cannot have "
       "triggered and headroom only compresses individual tool outputs >200 tokens, so a "
       "len ✗ on a ⊘ task is a BEHAVIORAL effect of the arm's wiring, not compaction damage. "
       "fid = per-step action agreement vs the control noise floor; it is only shown when "
       "the arm actually compressed (|comp| ≥ 2%) — ⊘ passthrough means the incremental run proved "
       "no compaction happened, so agreement deltas would be noise. "
       "⚠ n inactive = n trials that ran WITHOUT their intervention — vanilla in disguise, so "
       "read their ✓ as 'not measured', not 'preserved'. For caveman that means no activation "
       "marker in the transcript (the skill never loaded); for rtk it means the trial issued "
       "Bash commands and none carried the rtk prefix, so the hook never fired. "
       "rtk, like caveman, is not a proxy and should be read against vanilla, NOT vanilla-proxy. "
       "It is the only arm that transforms OBSERVATIONS rather than history or output: it "
       "shrinks what the agent reads back from Bash, so unlike caveman its comp is expected to "
       "be substantial (tool I/O dominates the prefix). Its recorded commands are rtk-wrapped, "
       "and every command-shaped metric here un-wraps them first — without that, rework would "
       "score a flawless 0 for the arm on identical behaviour. "
       "caveman is not a proxy, so it should be read against vanilla, NOT vanilla-proxy: it "
       "does not carry the non-default-base-URL wiring cost, and comparing it to "
       "vanilla-proxy would credit it with that whole ~8-9k/request difference. "
       "caveman is read like condense — comp is the accrued-context token measure (its terse "
       "prose replaces the verbose prose span by span, the same shape as condense's condensed "
       "blocks). It only touches prose (tool calls/results/code/errors stay verbatim), so its "
       "comp is small — often ~0 or net-negative once the per-turn ruleset cost is counted — "
       "but it still counts as engaged (fid stays meaningful), because the terse prose can move "
       "the next decision even at comp≈0. "
       "No model was called to produce this.")


def _plain(text):
    """Strip rich markup so a legend written for the console reads in md/html too."""
    return re.sub(r"\[/(?:[a-z][a-z ]*)?\]|\[[a-z][a-z ]*\]", "", text)


def _full_summary_md(d):
    """The full-run economics as a markdown table — the same rows the console renders."""
    sr = full_summary_rows(d)
    if not sr:
        return []
    head = ["arm"] + [f"{h} ({s})" for _k, h, s, _m, _fn, _f in _FULL_METRICS]
    o = ["## Full runs — vs vanilla\n", _plain(_FULL_LEGEND) + "\n",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in sr["rows"]:
        label = row["arm"] + (f" ({row['note']})" if row["note"] else "")
        cells = []
        for c in row["cells"]:
            txt = f"**{c['txt']}**" if c.get("sig") else c["txt"]
            cells.append(txt + (f" {c['sub']}" if c["sub"] else ""))
        o.append("| " + " | ".join([label] + cells) + " |")
    if not sr["shared"]:
        o.append("\nArms covered different tasks, so each carries its own vanilla ground row — "
                 "read down an arm's pair, not across the table.")
    return o + [""]


def _summary_md(d):
    """The incremental summary as a markdown table — the same rows the console renders."""
    sr = summary_rows(d)
    if not sr:
        return []
    head = ["arm"] + [f"{h} ({s})" for h, s in SUMMARY_COLS]
    o = ["## Incremental replay\n", _plain(_SUMMARY_LEGEND) + "\n",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in sr["rows"]:
        label = row["arm"] + (f" ({row['note']})" if row["note"] else "")
        cells = [c["txt"] + (f" [{c['sub']}]" if c["sub"] else "") for c in row["cells"]]
        o.append("| " + " | ".join([label] + cells) + " |")
    if not sr["shared"]:
        o.append("\nArms ran over different tasks/sessions, so each is shown against its own "
                 "control — read down an arm's pair, not across the table.")
    return o + [""]


def render_md(d):
    head, rows = table(d)
    o = ["# Trajectory preservation\n", SUB + "\n"] + _full_summary_md(d) + _summary_md(d)
    o += ["## Per task\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    o += ["| " + " | ".join(c for c, _ in row) + " |" for row in rows]
    return "\n".join(o) + "\n"


_HTML_STYLE = """
:root{--bg:#f7f9fb;--panel:#fff;--panel2:#f1f4f8;--line:#e2e7ee;--ink:#17202e;--ink2:#586274;
--ink3:#8a94a5;--good:#1f9d5b;--bad:#d1453a;--warn:#b9812a;--mut:#8791a2;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#0d1219;--panel:#151d27;--panel2:#1b2530;
--line:#28323f;--ink:#e2e8f1;--ink2:#9aa6b6;--ink3:#6b7686;--good:#33c37e;--bad:#f26a5c;
--warn:#e0a93f;--mut:#7d8798}}
:root[data-theme=dark]{--bg:#0d1219;--panel:#151d27;--panel2:#1b2530;--line:#28323f;--ink:#e2e8f1;
--ink2:#9aa6b6;--ink3:#6b7686;--good:#33c37e;--bad:#f26a5c;--warn:#e0a93f;--mut:#7d8798}
:root[data-theme=light]{--bg:#f7f9fb;--panel:#fff;--panel2:#f1f4f8;--line:#e2e7ee;--ink:#17202e;
--ink2:#586274;--ink3:#8a94a5;--good:#1f9d5b;--bad:#d1453a;--warn:#b9812a;--mut:#8791a2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
font-size:14px;line-height:1.5}.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-family:var(--mono);font-size:19px;margin:0 0 4px}h2{font-family:var(--mono);font-size:15px;
margin:30px 0 8px}.dim{color:var(--ink3);font-weight:400}.sub{color:var(--ink2);max-width:70ch;font-size:13px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink3);font-weight:600;
text-align:right;padding:9px 11px;background:var(--panel2);border-bottom:1px solid var(--line)}
th.l{text-align:left}td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line);
font-family:var(--mono);font-size:12.5px}td.l{text-align:left;font-family:var(--sans)}
tr.row{cursor:pointer}tr.row:hover td{background:var(--panel2)}tr.row.open td{background:var(--panel2)}
.task b{font-weight:600}.task .s{display:block;font-family:var(--mono);font-size:10.5px;color:var(--ink3)}
.v{color:var(--ink3)}.d{font-size:11px;display:block}.good{color:var(--good)}.bad{color:var(--bad)}
.warn{color:var(--warn)}.mut{color:var(--ink2)}
.pill{font-family:var(--sans);font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:20px;
white-space:nowrap;display:inline-block}.pill.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.pill.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.pill.warn{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.pill.na{background:var(--panel2);color:var(--ink2)}
tr.det td{background:var(--panel2);font-family:var(--sans);text-align:left;font-size:12.5px;color:var(--ink2)}
.det .grid{display:flex;flex-wrap:wrap;gap:8px 22px;padding:4px 2px}.det b{color:var(--ink)}
.det .k{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.caret{color:var(--ink3);display:inline-block;width:12px}tr.row.open .caret{transform:rotate(90deg)}
.foot{margin-top:22px;font-size:12px;color:var(--ink3)}
"""


def _html_report_body(d):
    """Interactive per-arm HTML derived from the SAME _cmp/_verdict the console uses."""
    import html as H

    def fL(x):
        return f"{x:.0f}"

    def fT(x):
        return f"{x / 1e6:.1f}M"

    def fU(x):
        return f"${x:.2f}"

    ST = {"good": "good", "bad": "bad", "within": "mut", "warn": "warn", "na": "mut"}

    def mcell(ctrl, arm, fmt):
        c = _cmp(ctrl, arm)
        if not c:
            return '<td>—</td>'
        st = ST[c["state"]]
        return (f'<td><span class="v">{fmt(c["van"])}→</span><span class="{st}">{fmt(c["arm"])}</span>'
                f'<span class="d {st}">{c["saved"]:+.0f}%</span></td>')

    def detail(v, a):
        bits = [f'<span><span class="k">length</span> vanilla [{", ".join(map(str, v["_lens"]))}] '
                f'· arm [{", ".join(map(str, a["_lens"]))}]</span>',
                f'<span><span class="k">$/trial</span> vanilla [{", ".join(f"{x:.2f}" for x in v["_costs"])}] '
                f'· arm [{", ".join(f"{x:.2f}" for x in a["_costs"])}]</span>']
        if a.get("_lats") and v.get("_lats"):
            bits.append(f'<span><span class="k">latency</span> vanilla '
                        f'{sum(v["_lats"]) / len(v["_lats"]):.0f}s · arm '
                        f'{sum(a["_lats"]) / len(a["_lats"]):.0f}s</span>')
        inc = a.get("incr") or {}
        if inc.get("fid") is not None:
            bits.append(f'<span><span class="k">incremental</span> compaction '
                        f'<b>{_pct(inc.get("comp"))}</b> · faithful <b>{inc["fid"]:.0%}</b> '
                        f'(ctrl {inc.get("fid_ctrl", 0):.0%}) · $ savings <b>{_pct(inc.get("costd"))}</b></span>')
        return '<div class="grid">' + "".join(bits) + '</div>'

    secs = []
    for arm in d["arms"]:
        arm_rows = [r for r in d["rows"] if r["arms"][arm]["_lens"] or r["arms"][arm]["n"]]
        if not arm_rows:
            continue
        msh = '<th>milestone</th>' if d["has_milestone"] else ''
        span = 6 if d["has_milestone"] else 5
        trs = []
        for r in arm_rows:
            v, a, sub = r["vanilla"], r["arms"][arm], r["sub_gate"]
            label, state = _verdict(v, a, arm, sub)
            pk = v["peak_ctx"] // 1000
            ms = ""
            if d["has_milestone"]:
                m = a.get("milestone")
                mst = ("good" if a.get("milestone_ok") else "bad"
                       if a.get("milestone_ok") is False else "mut")
                ms = f'<td><span class="{mst}">{m[1] * 100:.0f}%</span></td>' if m else '<td>—</td>'
            trs.append(
                f'<tr class="row" onclick="tg(this)"><td class="l task"><span class="caret">▸</span> '
                f'<b>{H.escape(r["task"])}</b><span class="s">peak {pk}k'
                f'{" ⊘" if gated(arm, sub) else ""} · '
                f'solve {a["solve"]}/{a["attempted"]}</span></td>'
                + mcell(v["_lens"], a["_lens"], fL) + mcell(v["_toks"], a["_toks"], fT)
                + mcell(v["_costs"], a["_costs"], fU)
                + f'<td class="l"><span class="pill {state}">{H.escape(label)}</span></td>{ms}</tr>'
                + f'<tr class="det" style="display:none"><td class="l" colspan="{span}">{detail(v, a)}</td></tr>')
        dead = inactive_total(arm_rows, arm)
        warn = (f' <span class="pill bad">⚠ {dead} trial{"s" if dead > 1 else ""} ran without '
                f'the skill — read those verdicts as "not measured"</span>') if dead else ""
        secs.append(f'<h2>{H.escape(arm)} <span class="dim">vs vanilla</span>{warn}</h2>'
                    f'<table><thead><tr><th class="l">task ▸</th><th>length</th><th>tokens</th>'
                    f'<th>$</th><th class="l">verdict</th>{msh}</tr></thead><tbody>'
                    + "".join(trs) + '</tbody></table>')
    return "".join(secs)


def _html_full_summary(d):
    """The full-run economics section — first thing in the page, same rows as the console."""
    import html as H
    sr = full_summary_rows(d)
    if not sr:
        return ""
    ths = "".join(f'<th>{H.escape(h)}<span class="d">{H.escape(s)}</span></th>'
                  for _k, h, s, _m, _fn, _f in _FULL_METRICS)
    trs = []
    for row in sr["rows"]:
        note = f'<span class="s">{H.escape(row["note"])}</span>' if row["note"] else ""
        tds = []
        for c in row["cells"]:
            v = H.escape(c["txt"])
            if c.get("sig"):
                v = f"<b>{v}</b>"
            sub = f'<span class="d">{H.escape(c["sub"])}</span>' if c["sub"] else ""
            tds.append(f"<td>{v}{sub}</td>")
        cls = " v" if row["control"] else ""
        trs.append(f'<tr><td class="l task{cls}"><b>{H.escape(row["arm"])}</b>{note}</td>'
                   + "".join(tds) + "</tr>")
    foot = "" if sr["shared"] else (
        '<div class="foot">Arms covered different tasks, so each carries its own vanilla ground '
        "row — read down an arm's pair, not across the table.</div>")
    return ('<h2>full runs — vs vanilla</h2>'
            f'<div class="sub">{H.escape(_plain(_FULL_LEGEND))}</div>'
            f'<table><thead><tr><th class="l">arm</th>{ths}</tr></thead><tbody>'
            + "".join(trs) + f'</tbody></table>{foot}')


def _html_summary(d):
    """The incremental summary section — same rows as the console."""
    import html as H
    sr = summary_rows(d)
    if not sr:
        return ""
    ths = "".join(f'<th>{H.escape(h)}<span class="d">{H.escape(s)}</span></th>'
                  for h, s in SUMMARY_COLS)
    trs = []
    for row in sr["rows"]:
        note = f'<span class="s">{H.escape(row["note"])}</span>' if row["note"] else ""
        tds = []
        for c in row["cells"]:
            sub = f'<span class="d">{H.escape(c["sub"])}</span>' if c["sub"] else ""
            tds.append(f'<td>{H.escape(c["txt"])}{sub}</td>')
        cls = " v" if row["control"] else ""
        trs.append(f'<tr><td class="l task{cls}"><b>{H.escape(row["arm"])}</b>{note}</td>'
                   + "".join(tds) + "</tr>")
    foot = "" if sr["shared"] else (
        '<div class="foot">Arms ran over different tasks/sessions, so each is shown against its '
        'own control — read down an arm\'s pair, not across the table.</div>')
    return ('<h2>incremental replay</h2>'
            f'<div class="sub">{H.escape(_plain(_SUMMARY_LEGEND))}</div>'
            f'<table><thead><tr><th class="l">arm</th>{ths}</tr></thead><tbody>'
            + "".join(trs) + f'</tbody></table>{foot}')


def render_html(d):
    import html as H
    model = d.get("model") or ""
    title = "Trajectory preservation" + (f" · {model}" if model else "")
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{H.escape(title)}</title>'
            f'<style>{_HTML_STYLE}</style></head><body><div class="wrap">'
            f'<h1>trajectory preservation <span class="dim">{H.escape("· " + model) if model else ""}</span></h1>'
            f'<div class="sub">{H.escape(SUB)}</div>'
            + _html_full_summary(d)
            + _html_summary(d)
            + _html_report_body(d)
            + '<div class="foot">length/tokens/$ show vanilla→arm with the SAVING below '
            '(+ = better: fewer steps / fewer tokens / less money — same sign convention as '
            'the incremental table), coloured green (saved) / red (cost more) / grey (within '
            'vanilla\'s band) — independently, so token-savings that cost more show green '
            'next to red. '
            'Click a row for per-trial values and incremental compaction / faithful / $ savings.</div>'
            '</div><script>function tg(r){var d=r.nextElementSibling;'
            'var o=d.style.display!=="none";d.style.display=o?"none":"";'
            'r.classList.toggle("open",!o);}</script></body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="display the quality bench (reads generate.py's artifacts)")
    ap.add_argument("--from", default="results/jobs", help="results root produced by generate.py")
    ap.add_argument("--tasks", default=None,
                    help="comma list, a number N for the first N curated defaults, "
                         "or omitted = 5 (must cover what generate.py ran)")
    ap.add_argument("--arms", default="condense,headroom")
    ap.add_argument("--agent", default="claude-code", choices=list(AGENT_SESSION_GLOB))
    ap.add_argument("--ctx-gate", type=int, default=50_000,
                    help="peak-context threshold below which compaction cannot have "
                         "triggered (marks the task ⊘)")
    ap.add_argument("--format", default="html", choices=["html", "md"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    # An unreadable or empty --from used to render a complete-looking report with every cell
    # blank and exit 0 — indistinguishable from "the arm genuinely produced nothing", and the
    # easiest way to hit it is a relative path from the wrong cwd. Fail loudly instead: this
    # command is pure display, so having nothing to display is always a mistake worth stopping on.
    root = args.__dict__["from"]
    if not os.path.isdir(root):
        ap.error(f"--from {root!r} is not a directory (relative to cwd {os.getcwd()!r})")
    if not glob.glob(f"{root}/**/verifier/reward.txt", recursive=True) \
            and not glob.glob(f"{root}/**/incremental/*-control.jsonl", recursive=True):
        ap.error(f"--from {root!r} holds no finished trials and no incremental replays — "
                 "nothing to report. Check the path, or point it at the run's output dir.")
    d = build(args)
    render_console(d)
    out = args.out or f"report.{args.format}"
    open(out, "w").write(render_md(d) if args.format == "md" else render_html(d))
    if HAVE_RICH:
        Console().print(f"[dim]wrote {out}[/]")
    else:
        print(f"wrote {out}")


# The wide table() view (all axes as separate columns) is the HTML/md layout. The terminal
# shows two focused tables so the modes aren't confounded — FULL trajectories (solve, length)
# and INCREMENTAL teacher-forced per-step (scoring, compression, faithfulness, cost) — each
# with control on its own row. rework/milestone/ctx detail stays in report.html.
_FULL_LEGEND = (
    "one table per arm; each row a task. length / tokens / $ each show [dim]vanilla[/]→arm (mean "
    "steps, total tokens, USD/trial) with the SAVING below — signed like every other savings "
    "column in this report, [bold]+ = BETTER[/] (fewer steps / fewer tokens / less money), − = "
    "worse. Coloured INDEPENDENTLY: green below vanilla's spread, red above, dim within — so an "
    "arm that cuts tokens yet costs more, the cache-bust tax, shows green next to red. verdict = "
    "QUALITY preservation, not length: ✓ quality held / ✗ quality lost, judged on milestone "
    "coverage when the judge ran and otherwise on the verifier (a loss must exceed one trial's "
    "worth of the arm's own denominator, since k≈4 makes a single trial worth 20-25%). Doing "
    "the same work in FEWER steps is the point of these methods and is never marked against an "
    "arm; running LONGER is a real cost signal, so it rides along as ✓ held · ↑ longer rather "
    "than as a failure. ⊘ too short = "
    "vanilla never crossed the compaction gate, so a method that only acts ON a compaction "
    "never acted — shown for those arms only; a method that acts from step 1 regardless of "
    "context size gets a real verdict here. n1 = single run, a trend not a verdict (≥2/arm). "
    "milestone = mean % of the task's subgoals the arm reached (LLM-judged, approach-agnostic), "
    "green when it matches vanilla's band / red when below (only shown with --milestones). "
    "Under each task: peak context + solve rate (⚠lost trials count as fails). No model was called.")
_INCR_LEGEND = (
    "teacher-forced per-step run of a recorded session. control = the baseline (its own row); "
    "its faithful is the noise floor, its s/step the latency baseline. Every delta is vs "
    "control and signed so [bold]+ = BETTER[/], − = worse. scoring = how each step was judged: "
    "llm:goal rates each action on its own merit (control ≈100% — trustworthy floor; an arm "
    "only loses by taking WORSE steps), whereas struct / llm:equiv match the ORIGINAL recorded "
    "action, whose ceiling is sampling-driven and low even for control (a noisy floor — prefer "
    "--judge goal). compaction % / $ savings / speed up = context compressed / cost saved / "
    "per-step wall-clock faster vs control (+ = better = less context / cheaper / faster). ↻N = "
    "CCR retrieve calls a headroom arm made (net compaction can be ~0 yet still engaged). "
    "caveman is read here like condense — compaction % is its accrued-context change (small, "
    "often ~0/net-negative since it only terse-ifies prose); it still counts as engaged, so its "
    "faithful stays a verdict rather than dimmed. "
    "faithful = the % of control's own faithfulness the arm keeps (control = 100% by "
    "construction; its absolute good-rate is shown in parens) — normalising divides out the "
    "sampling/judge floor, so 100% = no measurable loss. Coloured green ≥95% / yellow 85–95% / "
    "red <85% when the arm engaged (compressed or CCR-retrieved); else dim (≈100% by "
    "construction — no verdict). — = no incremental data. speed up is wall-clock — read it as a "
    "trend, not exact.")
_FULL_LEGEND = (
    "the vanilla row is the absolute ground; every arm row is that arm against it. % figures "
    "are GEOMETRIC means of per-task ratios — exp(mean(log(arm/vanilla))) − 1 — so each task "
    "votes once. The pooled ratio-of-sums this replaced was dollar-weighted in disguise: the "
    "costliest few tasks carried the answer. Under each value is a 95% TWO-LEVEL bootstrap "
    "interval (seeded): tasks are resampled, then trials within each resampled cell. Tasks are "
    "shared across arms so the comparison is paired; trial draws are independent per arm. "
    "[bold]Bold[/] marks an interval clear of zero — everything else is indistinguishable from "
    "vanilla at this n, which is the common case and not a pass. Three columns are not ratios: "
    "solve is percentage POINTS (a ratio is undefined at vanilla 0% and meaningless at 100%), "
    "compact is the arm's ABSOLUTE count (vanilla is exactly 0, so every ratio is infinite) and "
    "carries no significance mark. compact counts billed turns whose context came back smaller "
    "than the turn before — measured on the outcome, not inferred from cache traffic. cache wr "
    "= cache-write share of cached tokens; writes bill at 12.5x reads, so it is the cache number "
    "that moves $/Mtok. turns are billed requests (deduped by requestId); steps are tool_use "
    "blocks — near 1:1 here but not the same unit. Lost trials count as solve failures. Only "
    "tasks where BOTH the arm and vanilla produced a usable trial are counted. No model was "
    "called.")
_SUMMARY_LEGEND = (
    "the incremental replay, pooled into one row per arm. ± is a 95% bootstrap CI (seeded, so "
    "the same artifacts always render the same bars), resampling the paired STEPS. Full-run "
    "solve rate is NOT here — it moved to the economics table above, as a task-equal points "
    "delta. quality (incremental) = the share of replayed steps whose action the "
    "goal judge rated good (or, unjudged, that structurally matched the recording — a much "
    "noisier floor). redundant = steps that re-fetched information the agent already had. Under "
    "each arm value, Δ is its PAIRED difference vs control with its own CI — the ± on the values "
    "are marginal and can't be eyeballed against each other. A Δ bar that straddles zero means "
    "indistinguishable from control at this n, which is the common case and not a pass. No cell "
    "is marked better or worse: that reading is yours. context removed is not a quality axis: "
    "it is what the quality columns are the price of, and the number without which they can't "
    "be read. ⊘ marks an arm that removed <2% — its flat quality numbers are an "
    "absence of measurement, not a preserved trajectory. Both modes count only material where "
    "the method could act, and say what they dropped: ⊘n too short = tasks whose peak context "
    "never reached the compaction gate (counted only for methods that act ON a compaction — a "
    "method that acts from the first step keeps those tasks), ⊘n passthrough = sessions where "
    "the arm never engaged. "
    "Steps within a session are correlated and the bootstrap resamples them as independent, so "
    "those bars are if anything optimistic. No model was called.")
_DASH = "[dim]—[/]"


def _colok(text, ok):
    return f"[green]{text}[/]" if ok is True else (f"[red]{text}[/]" if ok is False else text)


def _solve_pct(c):
    """Solve rate as a %, yellow with ⚠lost when some trials crashed; — when never run."""
    if c["n"] == 0 and c.get("started", 0) == 0:
        return _DASH
    pct = f"{c['solve'] / c['attempted'] * 100:.0f}%" if c["attempted"] else "0%"
    return f"[yellow]{pct} ⚠{c['lost']}[/]" if c["lost"] else pct


def _runs(c):
    """Finished-trial count (0× when trials opened but all crashed; — when never run)."""
    if c["n"] == 0 and c.get("started", 0) == 0:
        return _DASH
    return f"{c['n']}×"


def _cmp(ctrl_list, arm_list):
    """The ONE metric comparison both the console and HTML derive from, so they can't drift.
    Returns {van, arm, saved, state} or None. state: good = arm below vanilla's spread
    (shorter/saved), bad = above (longer/costlier), within = inside the band.

    `saved` is signed the way every other savings column in this bench is signed — + is BETTER,
    i.e. less of the thing. It is deliberately NOT the raw arm/vanilla delta: this table sat
    next to the incremental one, whose compaction % / $ savings / speed up all read + for good,
    and printed −18% in green for an arm that got 18% CHEAPER. Two savings conventions in one
    report is one too many, and the green was doing the work the sign should have done.
    """
    if not ctrl_list or not arm_list:
        return None
    lo, hi = min(ctrl_list), max(ctrl_list)
    cm, am = sum(ctrl_list) / len(ctrl_list), sum(arm_list) / len(arm_list)
    state = "good" if am < lo else "bad" if am > hi else "within"
    saved = (1 - am / cm) * 100 if cm else 0.0
    return {"van": cm, "arm": am, "saved": saved or 0.0, "state": state}  # `or` kills -0.0


def _cmp_cell(ctrl_list, arm_list, fmt):
    """Console cell: vanilla→arm on line 1, signed SAVING below, coloured by _cmp's state."""
    c = _cmp(ctrl_list, arm_list)
    if not c:
        return _DASH
    color = {"good": "green", "bad": "red", "within": "dim"}[c["state"]]
    return f"[dim]{fmt(c['van'])}[/]→[{color}]{fmt(c['arm'])}[/]\n[{color}]{c['saved']:+.0f}%[/]"


def _milestone_cell(a):
    """Arm's milestone coverage (mean % of the task's subgoals reached), green when it overlaps
    vanilla's band (same subgoals) / red when it falls below / plain when ungraded."""
    m = a.get("milestone")
    if not m:
        return _DASH
    return _colok(f"{m[1] * 100:.0f}%", a.get("milestone_ok"))


def _TOKFMT(x):
    return f"{x / 1e6:.1f}M"


def _USDFMT(x):
    return f"${x:.2f}"


def _engaged(inc):
    """Did the arm actually do something? Compression that moved the context (condense,
    headroom-kompress), a CCR retrieve (headroom fetched a compressed output back), OR caveman
    having applied its transform (native) — its terse prose can move the next decision even when
    the net context change is ~0. If none, faithfulness is reconstruction noise."""
    return bool(abs(inc.get("comp") or 0) >= 0.02 or inc.get("retrieves", 0) > 0
                or inc.get("native") or inc.get("rtk_filtered"))


def _comp_cell(inc):
    """The 'compaction %' cell: CONTEXT reduction vs control (+ the CCR retrieve count ↻ for
    headroom). caveman is read here too — its terse prose replaces the verbose prose span by
    span, so this is its accrued-context change (small, often ~0/net-negative once the ruleset
    cost counts)."""
    txt = _pct(inc.get("comp"))
    if inc.get("ccr"):
        txt += f" [magenta]↻{inc.get('retrieves', 0)}[/]"
    return txt


def _latency_cell(inc):
    """Per-step wall-clock SAVED vs control, signed % (+ = faster, − = slower); — when not
    timed (old artifact). Same sign convention as ctx/$ saved: positive is better."""
    la, lc = inc.get("latency"), inc.get("latency_ctrl")
    return _DASH if la is None or not lc else _pct(1 - la / lc)


def _incr_field(r, arms, key):
    """The first arm's incremental value for `key` — used for control-side fields
    (fid_ctrl / latency_ctrl / scoring), which are ≈equal across a task's arms."""
    return next((r["arms"][a]["incr"][key] for a in arms
                 if (r["arms"][a].get("incr") or {}).get(key) is not None), None)


def _latency_floor(r, arms):
    """Control's own per-step wall-clock for a task (the latency baseline)."""
    return _incr_field(r, arms, "latency_ctrl")


def _faithful_norm(fid, floor):
    """The arm's faithfulness as a fraction of control's — control replaying itself is 1.0
    (100%) by construction, so this is 'how much of control's faithfulness the arm keeps',
    with the absolute sampling/judge floor divided out. None when there's no floor to divide by.
    Capped at 1.0: an arm can't be MORE faithful than the no-compaction reference (excess is
    noise)."""
    if fid is None or not floor:
        return None
    return min(fid / floor, 1.0)


def _faithful_cost(a, floor):
    """(faithfulness, cost) — incremental metrics. Faithfulness is normalised to control (=100%):
    the % of control's own faithfulness the arm retains, coloured green ≥95% (no measurable loss)
    / yellow 85–95% / red <85% when the arm engaged (compressed or made a CCR retrieve). When it
    barely touched the context (comp <2%, no CCR) the value is shown DIM (≈100% by construction —
    no verdict). — when there's no incremental data or no control floor to normalise against."""
    inc = a.get("incr") or {}
    cost = _pct(inc.get("costd"))
    norm = _faithful_norm(inc.get("fid"), floor)
    if norm is None:
        return _DASH, cost
    txt = f"{norm:.0%}"
    if not _engaged(inc):
        return f"[dim]{txt}[/]", cost
    return (f"[green]{txt}[/]" if norm >= 0.95
            else f"[red]{txt}[/]" if norm < 0.85 else f"[yellow]{txt}[/]"), cost


def _has_full(r, arms):
    """A task has full-run data if control or any arm opened at least one trial."""
    return any(c["n"] or c.get("started", 0)
               for c in [r["vanilla"]] + [r["arms"][a] for a in arms])


def _has_incr(r, arms):
    return any((r["arms"][a].get("incr") or {}).get("fid") is not None for a in arms)


def _scoring_for(r, arms):
    """The scoring method used for a task's incremental run (uniform across its arms)."""
    return _incr_field(r, arms, "scoring")


def _grouped(t, rows, render_row):
    """Add task-grouped rows to a table: a section divider between tasks, the task name on
    the control row, blank on the arm rows (render_row yields cell tuples per arm)."""
    for ri, r in enumerate(rows):
        if ri:
            t.add_section()
        for first, arm, cells in render_row(r):
            t.add_row(r["task"] if first else "", arm, *cells)


def _LENFMT(x):
    return f"{x:.0f}"


def _quality_held(v, a):
    """Did the arm keep vanilla's QUALITY on this task? -> True / False / None (can't tell).

    Two signals, and quality has to survive BOTH — they fail in different ways and neither
    excuses the other. The verifier is ground truth for "did it do the task", so a solve
    regression is a loss however good the subgoal coverage looks; the milestone judge is finer
    and approach-agnostic, so it can catch an arm that still passes but got there having done
    materially less. Milestone is only consulted when the judge actually ran.

    The verifier arm of the test is deliberately blunt: at k≈4 one trial is worth 20-25%, so a
    loss has to exceed one trial's worth of the arm's own denominator before it counts. A
    3/5 -> 2/4 wobble is one trial landing differently, not a regression.
    """
    va, aa = v["attempted"] or v["n"], a["attempted"] or a["n"]
    if not va or not aa:
        return a["milestone_ok"]           # no verifier denominator; milestone or None
    solve_held = (a["solve"] / aa) >= (v["solve"] / va) - (1.0 / aa)
    return solve_held and (a["milestone_ok"] is not False)


def _verdict(v, a, arm, sub_gate):
    """Shared verdict → (label, state). state ∈ good|bad|warn|na.

    The verdict is about QUALITY divergence, not trajectory length. Length used to decide it,
    which mislabelled the good case: an arm that reaches the same result in 14 steps instead of
    27 was reported as having "drifted", when doing the same work in fewer steps is the outcome
    these methods are FOR. Length still matters in one direction — running LONGER than vanilla
    costs real money and can mean the agent wandered — so it rides along as a cost note on the
    verdict rather than as the verdict itself. Shorter is never penalised.

    Needs ≥2 runs per arm for a real verdict; a single run reads as a trend (n1).
    `arm` is what decides whether ⊘ applies at all — see COMPACTION_GATED."""
    if not a["_lens"] or not v["length"]:
        return ("—", "na")
    if gated(arm, sub_gate):
        return ("⊘ too short", "na")
    held = _quality_held(v, a)
    n1 = " n1" if a["length_ok"] is None else ""      # length_ok is None exactly when k<2
    longer = a["length"][1] > v["length"][2]          # arm's mean above vanilla's spread
    if held is None:
        return (f"—{n1}", "na")
    if not held:
        return (f"✗ quality lost{n1}", "bad")
    # quality held; flag only the expensive direction of a length change
    return (f"✓ held · ↑ longer{n1}", "warn") if longer else (f"✓ quality held{n1}", "good")


def _verdict_cell(v, a, arm, sub_gate):
    """Console verdict cell — rich-markup wrapper around the shared _verdict."""
    label, state = _verdict(v, a, arm, sub_gate)
    color = {"good": "green", "bad": "red", "warn": "yellow", "na": "dim"}[state]
    return f"[{color}]{label}[/]"


def _under(cell, note):
    """A value with its own denominator on a dim second line. Each column is pooled over a
    DIFFERENT n — tasks for full runs, steps for the incremental pair — so one combined
    coverage string next to the arm name would be wrong for at least one column."""
    return f"{cell}\n[dim]{note}[/]" if note else cell


def _summary_cell(c):
    """Console cell from a shared summary_rows cell: value ±CI, with its denominator (control
    rows) or paired delta (arm rows) dim underneath. Deliberately monochrome — see `_cell`."""
    txt = c["txt"]
    if " ±" in txt:                              # dim the error bar, keep the value bright
        v, _, hw = txt.partition(" ±")
        txt = f"{v} [dim]±{hw}[/]"
    return _under(txt, c["sub"])


def _full_summary_table(console, d, model):
    """The full-run economics: one row per arm, every metric task-equal against vanilla.

    Sits first because it answers the question a reader arrives with — what did this arm cost,
    and is the difference bigger than the noise. Percentages are geometric means of per-task
    ratios, so each task votes once; the interval under each is a two-level bootstrap over
    tasks AND trials. A bar clear of zero is marked; everything else is left as a number,
    because at bench-scale k most differences genuinely aren't distinguishable.
    """
    sr = full_summary_rows(d)
    if not sr:
        return
    title = "[bold]quality — full runs · vs vanilla" + (f" · {model}" if model else "") + "[/]"
    t = Table(title=title, caption=_FULL_LEGEND, caption_justify="left", caption_style="dim",
              pad_edge=False)
    t.add_column("arm", no_wrap=True)
    for _k, head, sub, _m, _fn, _f in _FULL_METRICS:
        t.add_column(f"{head}\n[dim]{sub}[/]", justify="right", min_width=9)
    for i, row in enumerate(sr["rows"]):
        if i and row["control"]:
            t.add_section()
        label = row["arm"] + (f"\n[dim]{row['note']}[/]" if row["note"] else "")
        cells = []
        for c in row["cells"]:
            txt = f"[bold]{c['txt']}[/]" if c.get("sig") else c["txt"]
            cells.append(_under(txt, c["sub"]))
        t.add_row(label, *cells)
        if i == 0 and sr["shared"]:
            t.add_section()
    console.print(t)
    if not sr["shared"]:
        console.print("[dim]  arms covered different tasks, so each carries its own vanilla "
                      "ground row — read down an arm's pair, not across the table.[/]")


def _summary_table(console, d, model):
    """The overall read: one row per arm, pooled across every task and session, with error bars.

    Sits ABOVE the per-task tables on purpose. Those answer "what happened on this task"; a
    reader scanning them has to hold a dozen small numbers in their head to get to the question
    they actually came with, which is whether an arm costs quality overall. This answers that
    first, and the tables below are then the detail behind it.
    """
    sr = summary_rows(d)
    if not sr:
        return
    title = "[bold]quality — incremental replay" + (f" · {model}" if model else "") + "[/]"
    t = Table(title=title, caption=_SUMMARY_LEGEND, caption_justify="left", caption_style="dim",
              pad_edge=False)
    t.add_column("arm", no_wrap=True)
    for head, sub in SUMMARY_COLS:
        t.add_column(f"{head}\n[dim]{sub}[/]", justify="right", min_width=11)
    for i, row in enumerate(sr["rows"]):
        if i and row["control"]:                 # a divider before each repeated control block
            t.add_section()
        label = row["arm"] + (f"\n[dim]{row['note']}[/]" if row["note"] else "")
        t.add_row(label, *[_summary_cell(c) for c in row["cells"]])
        if i == 0 and sr["shared"]:              # ... or once, under a single shared control
            t.add_section()
    console.print(t)
    if not sr["shared"]:
        console.print("[dim]  arms ran over different tasks/sessions, so each is shown against "
                      "its own control — read down an arm's pair, not across the table.[/]")


def _full_table(console, d, model):
    """One table PER ARM: each row a task, vanilla's length/tokens/$ then the arm's (colored
    vs vanilla), verdict last. Vanilla is the shared reference, repeated in each arm's table."""
    rows = [r for r in d["rows"] if _has_full(r, d["arms"])]
    if not rows:
        return
    for ai, arm in enumerate(d["arms"]):
        arm_rows = [r for r in rows if r["arms"][arm]["_lens"] or r["arms"][arm]["n"]]
        if not arm_rows:
            continue
        dead = inactive_total(arm_rows, arm)
        title = (f"[bold]{arm} vs vanilla" + (f" · {model}" if model else "") + "[/]"
                 + (f" [red]⚠ {dead} trial(s) ran without the skill — "
                    f"those verdicts measure nothing[/]" if dead else ""))
        t = Table(title=title, caption=_FULL_LEGEND if ai == len(d["arms"]) - 1 else None,
                  caption_justify="left", caption_style="dim", pad_edge=False)
        has_ms = d.get("has_milestone")
        t.add_column("task", no_wrap=True)
        # one column per metric, each showing vanilla→arm + delta (see _cmp_cell) — far less
        # crammed than separate vanilla/arm columns, and the arm's absolute value is still there
        # the second header line is load-bearing: without it a reader has to infer from
        # colour whether +18% means the arm spent more or less
        for c in ("length", "tokens", "$"):
            t.add_column(f"{c}\n[dim]saved[/]", justify="right")
        t.add_column("verdict", justify="left", no_wrap=True)
        if has_ms:
            t.add_column("milestone", justify="right")  # subgoals reached vs vanilla
        for r in arm_rows:
            v, a, sub = r["vanilla"], r["arms"][arm], r["sub_gate"]
            peak = v["peak_ctx"] // 1000
            short = ' ⊘' if gated(arm, sub) else ''
            taskcell = (f"{r['task']}\n[dim]peak {peak}k{short} · "
                        f"solve {a['solve']}/{a['attempted']}[/]")
            cells = [
                taskcell,
                _cmp_cell(v["_lens"], a["_lens"], _LENFMT),
                _cmp_cell(v["_toks"], a["_toks"], _TOKFMT),
                _cmp_cell(v["_costs"], a["_costs"], _USDFMT),
                _verdict_cell(v, a, arm, sub),
            ]
            if has_ms:
                cells.append(_milestone_cell(a))
            t.add_row(*cells)
        console.print(t)


def _incremental_table(console, d, model):
    rows = [r for r in d["rows"] if _has_incr(r, d["arms"])]
    if not rows:
        return
    title = ("[bold]quality — incremental (teacher-forced per-step)"
             + (f" · {model}" if model else "") + "[/]")
    t = Table(title=title, caption=_INCR_LEGEND, caption_justify="left", caption_style="dim")
    for col in ("task", "arm", "scoring"):
        t.add_column(col, no_wrap=True)
    for col in ("compaction %", "faithful", "$ savings", "speed up"):
        t.add_column(col, justify="right" if col != "faithful" else "center")

    def render_row(r):
        floor = _floor_for(r, d["arms"])
        scoring = _scoring_for(r, d["arms"]) or _DASH
        # control is the 100% reference (arms are normalised to it); show its own absolute
        # good-rate in parens so a harsh judge / low sampling floor is visible
        ctrl_faith = f"[dim]100% ({floor:.0%})[/]" if floor is not None else _DASH
        lat0 = _latency_floor(r, d["arms"])
        ctrl_lat = f"[dim]{lat0:.1f}s[/]" if lat0 is not None else _DASH
        yield True, "control", (scoring, _DASH, ctrl_faith, _DASH, ctrl_lat)
        for arm in d["arms"]:
            inc = r["arms"][arm].get("incr") or {}
            if inc.get("fid") is None:
                yield False, arm, ("", _DASH, _DASH, _DASH, _DASH)
                continue
            faith, cost = _faithful_cost(r["arms"][arm], floor)
            yield False, arm, ("", _comp_cell(inc), faith, cost, _latency_cell(inc))

    _grouped(t, rows, render_row)
    console.print(t)


def render_console(d):
    """Terminal view: the pooled overall summary first (the question most readers arrive with),
    then two focused tables (full trajectories, then incremental) so the two measurement modes
    aren't confounded, followed by a one-glance takeaway."""
    if not HAVE_RICH:
        # bare-python3 fallback (fresh clone, nothing installed): the md view carries
        # every axis in one wide table — less pretty, same numbers
        print(render_md(d))
        return
    console = Console()
    model = d.get("model")
    _full_summary_table(console, d, model)
    _summary_table(console, d, model)
    _full_table(console, d, model)
    _incremental_table(console, d, model)
    _takeaway(console, d)


def _floor_for(r, arms):
    """The control incremental fidelity floor for a task (the noise floor)."""
    return _incr_field(r, arms, "fid_ctrl")


def _takeaway(console, d):
    """One-glance summary across both measurement modes: full-run comparability (graded AND
    compaction could fire) and incremental coverage, plus divergences — length (full) or
    fidelity below the control floor (incremental)."""
    comparable, tooshort = set(), set()
    for r in d["rows"]:
        graded = (len(r["vanilla"]["_lens"]) >= 2
                  and any(len(r["arms"][arm]["_lens"]) >= 2 for arm in d["arms"]))
        if graded:
            short = all(gated(arm, r["sub_gate"]) for arm in d["arms"])
            (tooshort if short else comparable).add(r["task"])
    replayed = {r["task"] for r in d["rows"] for arm in d["arms"]
                if (r["arms"][arm].get("incr") or {}).get("fid") is not None}
    # divergence means QUALITY divergence — an arm that did the same work in fewer steps has
    # not diverged, it has won. A longer trajectory is a cost signal and is called out
    # separately below, not as a failure.
    diverge = [f"{r['task']}/{arm}" for r in d["rows"] for arm in d["arms"]
               if not gated(arm, r["sub_gate"])
               and r["arms"][arm]["_lens"] and r["vanilla"]["length"]
               and _quality_held(r["vanilla"], r["arms"][arm]) is False]
    longer = [f"{r['task']}/{arm}" for r in d["rows"] for arm in d["arms"]
              if not gated(arm, r["sub_gate"])
              and r["arms"][arm]["_lens"] and r["vanilla"]["length"]
              and r["arms"][arm]["length"][1] > r["vanilla"]["length"][2]]
    for r in d["rows"]:                                   # fidelity meaningfully below floor
        floor = _floor_for(r, d["arms"])
        if floor is None or floor < 0.9:  # control should score ≥90% under a calibrated judge;
            continue                      # below that the baseline is noisy (warned below)
        for arm in d["arms"]:
            inc = r["arms"][arm].get("incr") or {}
            norm = _faithful_norm(inc.get("fid"), floor)
            if norm is not None and _engaged(inc) and norm < 0.85:  # kept <85% of control
                diverge.append(f"{r['task']}/{arm} fidelity ({norm:.0%} of control)")
    parts = []
    if comparable or tooshort:
        full = f"full: {len(comparable)} comparable"
        if tooshort:
            full += f", {len(tooshort)} too short"
        parts.append(full)
    if replayed:
        parts.append(f"incremental: {len(replayed)} tasks")
    console.print("[bold]takeaway[/]  " + " · ".join(parts or ["nothing comparable yet"]))
    if diverge:
        console.print("[red]  ✗ quality diverges vs control:[/] " + ", ".join(diverge))
    elif comparable or replayed:
        console.print("[green]  ✓ no quality divergence detected[/]")
    if longer:
        console.print("[yellow]  ↑ longer than control (costs more, quality unaffected):[/] "
                      + ", ".join(longer))
    # a low control floor means faithfulness was scored by structural match (no goal judge),
    # whose ceiling is sampling-driven — the comparison is noise-dominated, not trustworthy
    lowfloor = sum(1 for r in d["rows"]
                   if (_floor_for(r, d["arms"]) or 1) < 0.9 and _has_incr(r, d["arms"]))
    if lowfloor:
        console.print(f"[yellow]  ⚠ {lowfloor} incremental task(s) have control <90% faithful — "
                      "the judge is miscalibrated (or the run wasn't goal-judged), so the "
                      "baseline is noisy. Re-score with `minmax-bench quality rejudge --from "
                      "<dir>` (cheap) or re-run with --judge goal.[/]")


if __name__ == "__main__":
    main()
