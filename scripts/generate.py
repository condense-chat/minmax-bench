#!/usr/bin/env python3
"""Thin wrapper — the quality bench lives in minmax_bench/quality (importable, unit-tested).

Analysis (`scripts/report.py`) runs on a bare `python3` from a fresh clone; GENERATION
additionally needs `rich` for its progress display (part of the normal `uv sync` install;
Docker + harbor are only needed for --mode full itself).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minmax_bench.quality.generate import main  # noqa: E402

if __name__ == "__main__":
    main()
