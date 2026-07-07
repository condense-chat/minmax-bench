# Quality / trajectory-preservation bench

One command — [`report.py`](report.py) — for the
[trajectory-preservation bench](../README.md#quality--trajectory-preservation-bench): does a
context-reduction method **preserve the agent's trajectory**, or change what it does?

**Pure standard library.** Analyzing recorded runs needs no pip install. Harbor + Docker are only
needed to *generate* new runs (`run_cc_matrix.sh`).

## `report.py` — the one report command

```
report.py --mode {full,replay}      # full = preservation vs vanilla floor; replay = teacher-forced per-step
          --arms condense[,headroom] # which methods to test against the floor
          --agent {cc,codex}         # session format (default cc; codex = replay only for now)
          --format {html,md}         # default html
          --tasks t1,t2,...          # full mode
          --out report.html          # default
```

- **`--mode full`** (offline, free): pools vanilla/method runs from `--root`, builds per-task
  distributions for **length / rework / milestone / solve**, and renders a per-axis
  **✓ overlap / ✗ disjoint** verdict (a verdict needs ≥2 runs/arm; `--milestones` adds the
  LLM-judged subgoal axis, which calls the model). All the trajectory/rework/milestone logic is
  folded into `report.py`.
- **`--mode replay`** (online, costs money): teacher-forced replay of one `--session` through
  control + each arm — per-step action agreement + cache-aware compaction, paired (no turn-count
  noise). Uses the `fidelity_replay.py` engine.

## Files

| file | role |
|---|---|
| **`report.py`** | the one command (full + replay modes, both formats) |
| `fidelity_replay.py` | shared session I/O + the teacher-forced replay engine `report.py` imports |
| `run_cc_matrix.sh` | drive Harbor to **generate** runs per arm (`vanilla`, `condense`, `headroom`, `headroom-ccr`) |
| `capture_cc_template.py` | (utility) re-capture the version-matched request template for replay; a sanitized one ships in `data/cc_request_template.json` |
| `../harbor_agents/headroom_ccr_claude_code.py` | self-contained CCR wiring for the `headroom-ccr` arm (no user/global config mutation) |

## Reproduce

```bash
# offline preservation report — no keys, no Docker
python3 scripts/report.py --mode full --tasks kv-store-grpc --root results/sample

# generate your own, then report (needs Docker + `uv tool install harbor` + .env keys)
cp .env.dist .env      # ANTHROPIC_API_KEY, CONDENSE_API_KEY
TASKS="kv-store-grpc,fix-code-vulnerability"
OUT=results/jobs/run1 WALL_TIMEOUT=2400 AGENT_TIMEOUT_MULT=3 \
  bash scripts/run_cc_matrix.sh "$TASKS" 3 vanilla,condense
python3 scripts/report.py --mode full --tasks "$TASKS" --root results/jobs --milestones
```
