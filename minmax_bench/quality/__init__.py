"""Quality / trajectory-preservation bench.

The ANALYSIS path is dependency-free on purpose: `python3 scripts/report.py` must
work on a fresh clone with no install — report.py degrades to plain text when rich
is absent, and engine.py stays pure stdlib. generate.py may use rich for its
progress display (generation already requires an installed environment). Keep this
__init__ empty of imports — the cost bench's heavier deps live outside this package.
"""
