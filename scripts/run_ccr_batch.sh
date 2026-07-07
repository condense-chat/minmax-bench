#!/usr/bin/env bash
# Run the self-contained headroom-CCR arm across tasks, SEQUENTIALLY, snapshotting proxy /stats
# before/after each task so compression + retrievals are cleanly attributed per task.
# Default CCR thresholds (fair, representative) — no aggressive HEADROOM_* overrides.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH"
export ANTHROPIC_API_KEY PYTHONPATH="$PWD:${PYTHONPATH:-}"

TASKS="${1:?usage: run_ccr_batch.sh task1,task2,...}"
OUT="${OUT:-results/jobs/ccr}"; HRPORT="${HRPORT:-8787}"; MODEL="${MODEL:-claude-sonnet-4-6}"
mkdir -p "$OUT"; : > "$OUT/ccr_stats.txt"

pkill -f 'headroom proxy' 2>/dev/null; sleep 1
headroom proxy --port "$HRPORT" --mode token > "$OUT/proxy.log" 2>&1 &
HRPID=$!; trap 'kill $HRPID 2>/dev/null' EXIT
for _ in $(seq 1 20); do (exec 3<>"/dev/tcp/127.0.0.1/$HRPORT") 2>/dev/null && break; sleep 1; done
echo "[ccr] proxy up on :$HRPORT (token, default CCR thresholds)"

snap() {  # -> "saved before compressed retrievals" (cumulative)
  curl -s "http://127.0.0.1:$HRPORT/stats" | python3 -c "
import sys,json
d=json.load(sys.stdin)
def f(o,k):
  if isinstance(o,dict):
    if k in o and not isinstance(o[k],dict): return o[k]
    for v in o.values():
      r=f(v,k)
      if r is not None: return r
  return None
print(f(d,'proxy_compression_saved') or 0, f(d,'proxy_total_before_compression') or 0,
      f(d,'requests_compressed') or 0, f(d,'retrievals') or 0)"
}

IFS=',' read -ra T <<< "$TASKS"
for task in "${T[@]}"; do
  read -r s0 b0 c0 r0 <<< "$(snap)"
  echo "[ccr] === $task (start; cum saved=$s0) ==="
  ANTHROPIC_BASE_URL="http://host.docker.internal:$HRPORT" timeout "${WALL_TIMEOUT:-1800}" harbor run \
    -d terminal-bench/terminal-bench-2-1 \
    -a harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode \
    -m "$MODEL" -i "terminal-bench/$task" -k 1 -n 1 -o "$OUT/$task" \
    --ak max_budget_usd="${BUDGET:-5}" --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT:-2}" \
    --allow-agent-host host.docker.internal \
    --ae "TMB_HEADROOM_PROXY_URL=http://host.docker.internal:$HRPORT" >> "$OUT/_runlog.txt" 2>&1 || true
  read -r s1 b1 c1 r1 <<< "$(snap)"
  rt=$(find "$OUT/$task" -name reward.txt 2>/dev/null | head -1)
  echo "$task reward=$([ -n "$rt" ] && cat "$rt" || echo '?') saved=$((s1-s0)) before=$((b1-b0)) compressed=$((c1-c0)) retrievals=$((r1-r0))" | tee -a "$OUT/ccr_stats.txt"
done
touch "$OUT/.ccr_done"
echo "[ccr] batch complete"
