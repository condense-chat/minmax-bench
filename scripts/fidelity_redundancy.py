#!/usr/bin/env python3
"""Redundant-work rate per trajectory — the compression quality signal that survives free-running
variance and doesn't saturate the way milestone coverage does.

Two runs can both hit every milestone (both solve the task) yet differ hugely in efficiency: a
compression method that drops context makes the agent RE-DO work it already did — re-Read a file
whose contents are still valid, re-cat it, or re-run an identical read-only probe it already ran.
That redundant re-work is a real quality cost of compression, visible even among solved runs.

Self-contained per trajectory (no alignment to a reference → no path-divergence noise):
  redundant = an info-gathering action that re-fetches content already obtained earlier in THIS
              run and not invalidated since (a Read of an un-modified file; a re-cat; an identical
              read-only Bash probe).
Compare vanilla (no compaction = natural baseline) vs condense/headroom: the EXCESS over vanilla
is compression-induced amnesia.

Usage:
  python3 scripts/fidelity_redundancy.py --run vanilla=<session.jsonl> --run condense=<...> ...
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity_replay import parse_session, extract_action  # noqa: E402
from fidelity_trajectory import _read_span, _covered  # noqa: E402

_TOK = re.compile(r"[A-Za-z0-9_./-]+")
_CAT = re.compile(r"\b(cat|head|tail|less|more|bat|sed -n|nl)\b")
_READONLY = re.compile(r"^\s*(grep|rg|find|ls|cat|head|tail|nm|ldd|which|file|stat|wc|"
                       r"python3? -c \"?import|pip show|pip list)\b")


def norm(s):
    return " ".join(str(s).split())


def redundancy(path):
    """-> (redundant_hits, n_actions). Hits: re-Read/-cat unmodified file, identical read-only re-run."""
    msgs, points = parse_session(path)
    read_spans, last_read, last_mod, seen_cmd = {}, {}, {}, {}
    hits, n = [], 0
    for i in points:
        a = extract_action(msgs[i]["content"])
        if a.get("type") != "tool_use":
            continue
        n += 1
        name, inp = a["name"], a.get("input", {})
        if name in ("Write", "Edit"):
            fp = inp.get("file_path")
            if fp:
                last_mod[fp] = i
                read_spans[fp] = []          # content changed: prior reads are stale
        elif name == "Read":
            fp = inp.get("file_path")
            s, e = _read_span(inp)
            if fp and read_spans.get(fp) and _covered(read_spans[fp], s, e):
                hits.append((i, "re-Read", os.path.basename(fp)))
            if fp:
                read_spans.setdefault(fp, []).append((s, e))
                last_read[fp] = i
        elif name == "Bash":
            cmd = inp.get("command", "")
            hit = next((f for f in last_read if f and f in cmd and _CAT.search(cmd)
                        and last_mod.get(f, -1) <= last_read[f]), None)
            if hit:
                hits.append((i, "re-cat", os.path.basename(hit)))
            c = norm(cmd)
            if _READONLY.match(cmd) and c in seen_cmd:
                hits.append((i, "re-run", (_TOK.findall(cmd) or ["?"])[0]))
            seen_cmd[c] = i
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=session.jsonl")
    args = ap.parse_args()
    base = None
    rows = []
    print(f"{'run':>14} {'actions':>8} {'redundant':>10} {'rate':>6}  detail")
    for i, spec in enumerate(args.run):
        name, path = spec.split("=", 1)
        hits, n = redundancy(path)
        rate = len(hits) / n if n else 0
        if i == 0:
            base = len(hits)
        rows.append((name, n, len(hits), rate))
        from collections import Counter
        kinds = Counter(k for _, k, _ in hits)
        print(f"{name:>14} {n:>8} {len(hits):>10} {rate:>5.0%}  {dict(kinds)}"
              + (f"  excess vs {args.run[0].split('=')[0]}: {len(hits)-base:+d}" if i else "  (baseline)"))


if __name__ == "__main__":
    main()
