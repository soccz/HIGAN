#!/bin/bash
# Track 4 FFHQ encoder eval — re-run after fixing OOM bug.
# Original eval failed at FFHQ 1024² × N=128 OOM.
# Fixed: eval_c5.py now streams per-sample synthesis instead of batched.
#
# Run AFTER the main sequencer (run_remaining.sh) completes.
# Usage:  nohup bash experiments/run_track4_eval_fixup.sh > logs/track4_eval_fixup.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR="$(realpath logs)"
mkdir -p "$LOGDIR"

echo "[$(date +%T)] === Track 4 FFHQ encoder eval (fixed) ==="

# Use the actual checkpoints saved during the previous train run
# (we have enc_001000.pt … enc_080000.pt available)
python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
    --ckpts experiments/out/ffhq_c5/ckpt/enc_001000.pt \
            experiments/out/ffhq_c5/ckpt/enc_005000.pt \
            experiments/out/ffhq_c5/ckpt/enc_010000.pt \
            experiments/out/ffhq_c5/ckpt/enc_020000.pt \
            experiments/out/ffhq_c5/ckpt/enc_040000.pt \
            experiments/out/ffhq_c5/ckpt/enc_060000.pt \
            experiments/out/ffhq_c5/ckpt/enc_080000.pt \
    --num-test 32 \
    --out experiments/out/ffhq_c5/eval \
    > "$LOGDIR/track4_ffhq_eval_fixed.log" 2>&1

rc=$?
if [ $rc -eq 0 ]; then
    echo "[$(date +%T)] ✓ Track 4 eval (fixed) OK"
else
    echo "[$(date +%T)] ✗ Track 4 eval still FAILED (rc=$rc) — see $LOGDIR/track4_ffhq_eval_fixed.log"
fi
