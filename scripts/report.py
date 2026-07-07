#!/usr/bin/env python3
"""Thin wrapper — the quality bench lives in minmax_bench/quality (importable, unit-tested).

Still runs on a bare `python3` from a fresh clone: the quality subpackage is pure
standard library, so nothing needs to be installed to analyze runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minmax_bench.quality.report import main  # noqa: E402

if __name__ == "__main__":
    main()
