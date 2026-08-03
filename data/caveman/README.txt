Frozen caveman SessionStart-hook rulesets, one per intensity level, captured from the
pinned release (see PIN). Incremental replay (minmax_bench/quality/engine.py) injects the
matching file as session-start context, so it presents the SAME ruleset the container
emits in full mode. Regenerate on a pin bump:

  CAVEMAN_DEFAULT_MODE=<mode> node <checkout>/src/hooks/caveman-activate.js </dev/null

with a settings.json that already has a statusLine (else the hook appends a setup nudge).
