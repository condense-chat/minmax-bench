"""Quality / trajectory-preservation bench (pure standard library).

Kept dependency-free on purpose: `python3 scripts/report.py` must work on a
fresh clone with no install. Keep this __init__ empty of imports and keep the
modules stdlib-only — the cost bench's heavier deps live outside this package.
"""
