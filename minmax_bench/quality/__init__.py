"""Quality / trajectory-preservation bench.

The ANALYSIS path is dependency-free on purpose: `minmax-bench quality report`
must work without the cost bench's heavier deps — report.py degrades to plain
text when rich is absent, and engine.py stays pure stdlib. generate.py may use rich for its
progress display (generation already requires an installed environment). Keep this
__init__ empty of imports — the cost bench's heavier deps live outside this package.
"""
