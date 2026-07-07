# Quality / trajectory-preservation bench

Tooling for the [trajectory-preservation bench](../README.md#quality--trajectory-preservation-bench):
does a context-reduction method **preserve the agent's trajectory**, or change what it does? These
scripts drive full agent runs through [Harbor](https://github.com/laude-institute/terminal-bench)
(Terminal-Bench tasks, Docker) and score the resulting trajectories against a vanilla-vs-vanilla
noise floor.

**Pure standard library.** The analysis tools run on already-recorded `results/**` sessions with no
pip install. Harbor + Docker are only needed to *generate* new runs.

## Pipeline

```
run_cc_matrix.sh  ──▶  results/jobs/<batch>/<arm>-<task>/.../agent/sessions/*.jsonl
   (drive Harbor)                     │
                                      ▼
preservation_index.py ──▶ results/fidelity/index.html   (per-axis overlap verdict)
```

## Files

| file | what it does |
|---|---|
| **`preservation_index.py`** | **Main entry.** Pools vanilla/condense runs across all `results/**` dirs, builds per-task distributions for **length / rework / milestone / solve**, and renders an HTML report with a per-axis **✓ overlap / ✗ disjoint** verdict (needs ≥2 samples/arm). |
| `fidelity_trajectory.py` | Node-by-node trajectory explorer; also the **range-aware** re-work detector (`redundant_indices` — paginating a file is *not* redundant, re-reading a covered span is). |
| `fidelity_redundancy.py` | Standalone re-work metric (`redundancy()`). |
| `fidelity_milestones.py` | Approach-agnostic milestone extraction + coverage (LLM-judged) — the "same subgoals?" axis. |
| `fidelity_replay.py` | Teacher-forced per-step replay: replays a fixed prefix through each arm's endpoint, measuring per-step action agreement and **cache-aware** compaction with no turn-count noise. (Also the shared session/API utilities the others import.) |
| `capture_cc_template.py` | Capture a version-matched Claude Code request template for `fidelity_replay.py`. A sanitized one ships in `data/cc_request_template.json`. |
| `run_cc_matrix.sh` | Drive Harbor per (arm, task): `vanilla`, `condense`, `headroom` (proxy), `headroom-ccr`. Each method injected via its public interface, never by mutating traffic in a proxy we control. |
| `run_ccr_batch.sh` | Sequential `headroom-ccr` runs with per-task `/stats` snapshots (clean compression attribution). |
| `../harbor_agents/headroom_ccr_claude_code.py` | Self-contained CCR wiring: installs `headroom-ai` + `mcp` and registers headroom's MCP server in the task container's own config — no user/global config mutation. |

## Reproduce

```bash
# offline demo — no keys, no Docker
python3 scripts/preservation_index.py --root results/sample --tasks kv-store-grpc \
  --out results/sample/report.html

# generate your own (needs Docker + `uv tool install harbor` + .env keys)
cp .env.dist .env      # ANTHROPIC_API_KEY, CONDENSE_API_KEY
TASKS="kv-store-grpc,fix-code-vulnerability"
OUT=results/jobs/run1 WALL_TIMEOUT=2400 AGENT_TIMEOUT_MULT=3 \
  bash scripts/run_cc_matrix.sh "$TASKS" 3 vanilla,condense
python3 scripts/preservation_index.py --root results/jobs --tasks "$TASKS" \
  --out results/fidelity/index.html --milestones
```

A verdict needs ≥2 runs per arm per task; anything thinner is marked "— inconclusive" rather than
passing/failing on n=1.
