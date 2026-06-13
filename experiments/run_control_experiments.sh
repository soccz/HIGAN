#!/bin/bash
# Reproducible main-control experiment queue.
#
# The source of truth for hyperparameters is:
#   experiments/protocols/control_v1.json
set -uo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"

LOGDIR="$PAPER_DIR/logs/control"
mkdir -p "$LOGDIR"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/wrapper.log"; }

run_track() {
    local name="$1"; shift
    local logfile="$1"; shift
    log "=== $name ==="
    local start_ts=$(date +%s)
    "$@" > "$LOGDIR/$logfile" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - start_ts))
    if [ $rc -ne 0 ]; then
        log "    !! $name FAILED rc=$rc ${elapsed}s"
    else
        log "    $name OK in ${elapsed}s"
    fi
}

log "=== control experiment queue start (commit $GIT_COMMIT) ==="

run_track "Protocol-defined control experiments" "control_protocol_runner.log" \
    python3 -u experiments/control/run_control_protocol.py \
        --protocol experiments/protocols/control_v1.json \
        --logdir logs/control

run_track "Aggregate control results" "c_aggregate.log" \
    bash -c 'python3 experiments/aggregate_results.py > experiments/out/_aggregate_summary.txt'

log "=== control experiment queue COMPLETE ==="
