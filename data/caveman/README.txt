Caveman replay fixtures

These files freeze the SessionStart instructions that Caveman adds to Claude Code.
The full-task benchmark receives its instructions from the installed hook. The
incremental benchmark injects the corresponding saved ruleset at session start.
Keeping them in sync lets the two modes test the same instructions.

Files
  PIN                 Caveman release and commit used for this snapshot
  ruleset-lite.txt    Lite instructions
  ruleset-full.txt    Full instructions
  ruleset-ultra.txt   Ultra instructions

Refresh a ruleset
  1. Check out the release and commit recorded in PIN.
  2. Use a Claude settings.json that already has a statusLine. Otherwise the
     hook appends a setup reminder that would change the replay.
  3. Run the command below for lite, full, and ultra. Replace MODE and
     CHECKOUT with the intensity and the Caveman checkout path.

     CAVEMAN_DEFAULT_MODE=MODE node CHECKOUT/src/hooks/caveman-activate.js </dev/null

  4. Compare the output with ruleset-MODE.txt. On a version bump, update PIN
     and all affected fixtures together.

See ../../docs/quality.md for how incremental replay handles Caveman prose.
