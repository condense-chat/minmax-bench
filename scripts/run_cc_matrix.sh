#!/usr/bin/env bash
# Drives real Claude Code sessions through Harbor per arm. Each method is injected the
# way a real user deploys it — never by mutating traffic in a proxy we control:
#
#   vanilla      : ANTHROPIC_BASE_URL=api.anthropic.com             (unmodified) — baseline + noise floor
#   condense     : ANTHROPIC_BASE_URL=api.condense.chat/anthropic   + X-Condense-Auth-Token
#   headroom     : ANTHROPIC_BASE_URL=<local headroom proxy>        headroom proxy --mode token
#   headroom-ccr : self-contained CCR (harbor_agents.headroom_ccr_claude_code) — installs headroom-ai + MCP in-container
#
# Verdict = Harbor verifier reward. Runs solo (n=1, sequential) for reliability.
#
# Usage: scripts/run_cc_matrix.sh "<task1,task2,...>" <k> "<arm1,arm2,...>"
#   e.g.: scripts/run_cc_matrix.sh kv-store-grpc,fix-code-vulnerability 3 vanilla,condense
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH"

TASKS="${1:-count-dataset-tokens,largest-eigenval,pypi-server}"
K="${2:-3}"
ARMS="${3:-vanilla,condense}"
MODEL="${MODEL:-claude-sonnet-4-6}"
CONC="${CONC:-1}"               # concurrent repeats per (arm,task); raise to k for speed
TASK_CONC="${TASK_CONC:-1}"     # concurrent TASKS within an arm (parallelize the probe)
BUDGET="${BUDGET:-5}"            # per-run native spend cap (USD)
HRPORT="${HRPORT:-8787}"
HRMODE="${HRMODE:-cache}"        # headroom: cache-preserving mode (right for a cache-heavy agent)
OUT="${OUT:-results/jobs/cc-matrix}"
mkdir -p "$OUT"
export ANTHROPIC_API_KEY        # headroom proxy forwards upstream with this
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # let Harbor import harbor_agents.*

HRPID=""
start_hr() {
  .venv/bin/headroom proxy --port "$HRPORT" --mode "$HRMODE" ${HR_EXTRA:-} > "$OUT/headroom.log" 2>&1 &
  HRPID=$!
  for _ in $(seq 1 15); do
    (exec 3<>"/dev/tcp/127.0.0.1/$HRPORT") 2>/dev/null && { exec 3>&- ; echo "[headroom] up on :$HRPORT ($HRMODE)"; return 0; }
    sleep 1
  done
  echo "[headroom] FAILED to start — see $OUT/headroom.log"; return 1
}
stop_hr() { [ -n "$HRPID" ] && kill "$HRPID" 2>/dev/null; HRPID=""; }
trap stop_hr EXIT

run_arm() {
  local arm="$1" task="$2" base allow agent; local -a extra=()
  agent="claude-code"
  case "$arm" in
    vanilla)  base="https://api.anthropic.com";          allow="api.anthropic.com" ;;
    condense) base="https://api.condense.chat/anthropic"; allow="api.condense.chat"
              extra+=(--ae "ANTHROPIC_CUSTOM_HEADERS=X-Condense-Auth-Token: $CONDENSE_API_KEY") ;;
    headroom) base="http://host.docker.internal:$HRPORT"; allow="host.docker.internal" ;;
    headroom-ccr) base="http://host.docker.internal:$HRPORT"; allow="host.docker.internal"
              agent="harbor_agents.headroom_ccr_claude_code:HeadroomCcrClaudeCode"
              extra+=(--ae "TMB_HEADROOM_PROXY_URL=http://host.docker.internal:$HRPORT") ;;
    *) echo "unknown arm $arm"; return 1 ;;
  esac
  echo "### $arm / $task (k=$K, base=$base, agent=$agent) ###"
  # optional longer agent timeout for tasks that run very long (e.g. largest-eigenval)
  local -a tmo=()
  [ -n "${AGENT_TIMEOUT_MULT:-}" ] && tmo+=(--agent-timeout-multiplier "$AGENT_TIMEOUT_MULT")
  # ${extra[@]+...} guards against bash 3.2 "unbound variable" on an empty array under set -u
  ANTHROPIC_BASE_URL="$base" timeout "${WALL_TIMEOUT:-2400}" harbor run \
    -d terminal-bench/terminal-bench-2-1 -a "$agent" -m "$MODEL" \
    -i "terminal-bench/$task" -k "$K" -n "$CONC" -o "$OUT/$arm-$task" \
    --ak "max_budget_usd=$BUDGET" ${tmo[@]+"${tmo[@]}"} \
    --allow-agent-host "$allow" ${extra[@]+"${extra[@]}"} >> "$OUT/_runlog.txt" 2>&1 || true
}

IFS=',' read -ra TASK_ARR <<< "$TASKS"
IFS=',' read -ra ARM_ARR  <<< "$ARMS"
for arm in "${ARM_ARR[@]}"; do
  case "$arm" in headroom|headroom-ccr) start_hr || { echo "skipping $arm arm"; continue; } ;; esac
  if [ "${TASK_CONC:-1}" -le 1 ]; then
    # sequential foreground — no backgrounding, so the headroom proxy (a bg job) can't
    # jam the jobs-based gate. This is the study path.
    for task in "${TASK_ARR[@]}"; do run_arm "$arm" "$task"; done
  else
    # parallel path (probe only; vanilla arm, no proxy job to miscount)
    for task in "${TASK_ARR[@]}"; do
      run_arm "$arm" "$task" &
      while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$TASK_CONC" ]; do sleep 10; done
    done
    wait
  fi
  case "$arm" in headroom|headroom-ccr) stop_hr ;; esac
done
