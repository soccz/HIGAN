#!/bin/bash
# Watcher: poll Round 4 bash (PID 659921) until it dies, then run
# inject_meta + aggregator. Compensates for the "(deleted)" FD issue
# where the in-flight bash holds the pre-edit script inode.
set -uo pipefail
PAPER="/mnt/20t/study/HIGAN/paper"
cd "$PAPER"
LOGDIR="$PAPER/logs/round4"
mkdir -p "$LOGDIR"

WATCH_PID=659921

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/watcher.log"; }

log "watcher start; monitoring bash PID=$WATCH_PID"
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 60
done
log "Round 4 bash $WATCH_PID has ended; running post-completion steps"

log "  inject_meta..."
python3 experiments/lib/inject_meta.py >> "$LOGDIR/watcher.log" 2>&1
log "  aggregate_results..."
python3 experiments/aggregate_results.py > experiments/out/_aggregate_summary.txt 2>> "$LOGDIR/watcher.log"
NEW_LINES=$(wc -l < experiments/out/_aggregate_summary.txt)
NEW_TRACKS=$(grep -c "^\[Track" experiments/out/_aggregate_summary.txt)
log "  summary: $NEW_LINES lines, $NEW_TRACKS track entries"

log "watcher COMPLETE — Round 4 fully post-processed"
