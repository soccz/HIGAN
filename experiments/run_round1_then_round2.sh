#!/bin/bash
# Wrapper: wait for Round 1 sequencer to finish, then smoke-test
# K=100+checkpointing, then launch Round 2 with safest K that fits.
#
# Run with:
#   nohup bash experiments/run_round1_then_round2.sh \
#       > logs/round1_then_round2.log 2>&1 &
#
# This script is launched WHILE Round 1 is still running.
# It blocks until Round 1's sequencer dies, then smoke-tests, then
# launches Round 2.
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs/round2_wrapper"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/wrapper.log"; }

log "=== wrapper start; waiting for Round 1 sequencer to finish ==="

# Detect Round 1 sequencer
ROUND1_PID=$(pgrep -f "bash experiments/run_final.sh" | head -1 || true)
if [ -z "$ROUND1_PID" ]; then
    log "  no Round 1 sequencer running — proceed to smoke immediately"
else
    log "  Round 1 sequencer PID=$ROUND1_PID, waiting..."
    while kill -0 "$ROUND1_PID" 2>/dev/null; do
        sleep 60
    done
    log "  Round 1 sequencer ended"
fi

# Wait until GPU is truly idle (no python with our experiments)
while pgrep -f "python.*experiments/" >/dev/null 2>&1; do
    log "  waiting for GPU python to clear..."
    sleep 60
done
log "  GPU clear"

# ─── Smoke test: K=100 + gradient checkpointing fit in 8GB? ───
log ""
log "=== smoke: K=100 + checkpointing 8GB fit test ==="
SMOKE_OUT=/tmp/round2_smoke
rm -rf "$SMOKE_OUT"
timeout 180 python3 -u experiments/baselines/latentclr/train.py \
    --K 100 --B 2 --chunk 10 --epochs 1 --iters-per-epoch 3 \
    --lod 2 --out "$SMOKE_OUT" --seed 2027 \
    > "$LOGDIR/smoke_k100.log" 2>&1
smoke_rc=$?

if [ $smoke_rc -eq 0 ]; then
    log "  ✓ K=100 + checkpointing fits — Round 2 will use K=100"
    K_VALUE=100
elif [ $smoke_rc -eq 124 ]; then
    log "  ⊘ K=100 timed out (>180s for 3 iters = slow but no OOM)"
    log "  using K=100 anyway (timeout, not OOM)"
    K_VALUE=100
else
    log "  ✗ K=100 failed (probably OOM) — fallback to K=50"
    # smoke K=50
    timeout 180 python3 -u experiments/baselines/latentclr/train.py \
        --K 50 --B 2 --chunk 5 --epochs 1 --iters-per-epoch 3 \
        --lod 2 --out "$SMOKE_OUT" --seed 2027 \
        > "$LOGDIR/smoke_k50.log" 2>&1
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then
        log "  ✓ K=50 fits — Round 2 uses K=50"
        K_VALUE=50
    else
        log "  ✗ K=50 ALSO fails — keep original K=20 result, abort baseline retrain"
        K_VALUE=0   # signal: skip LatentCLR/DisCo
    fi
fi

# ─── Launch Round 2 with the chosen K ───
log ""
log "=== Round 2 launch with K=$K_VALUE ==="

# T3 × 3 domains (mean-of-ratios)
run_track_round2() {
    local name="$1"; shift; local lf="$1"; shift
    log "=== $name ==="
    local s=$(date +%s)
    "$@" > "$LOGDIR/$lf" 2>&1
    local rc=$?
    local e=$(($(date +%s) - s))
    if [ $rc -ne 0 ]; then
        log "    !! $name FAILED rc=$rc ${e}s"
    else
        log "    $name OK in ${e}s"
    fi
}

run_track_round2 "T3 bedroom v2" "t3_bedroom_v2.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain bedroom --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_bedroom_v2

run_track_round2 "T3 ffhq v2" "t3_ffhq_v2.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain ffhq --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_ffhq_v2

run_track_round2 "T3 church v2" "t3_church_v2.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain church --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_church_v2

# T4 FFHQ encoder v2 (longer + higher w_mse weight)
run_track_round2 "T4 FFHQ encoder train v2" "t4_train_v2.log" \
    python3 -u experiments/domains/ffhq/encoder/train.py \
        --num-iters 160000 --batch 1 --w-mse-weight 2.0 \
        --ckpt-iters 5000 20000 40000 80000 120000 160000 \
        --out experiments/out/ffhq_c5_v2 --seed 2027

run_track_round2 "T4 FFHQ encoder eval v2" "t4_eval_v2.log" \
    python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
        --ckpts experiments/out/ffhq_c5_v2/ckpt/enc_005000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_020000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_040000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_080000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_120000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_160000.pt \
        --num-test 32 \
        --out experiments/out/ffhq_c5_v2/eval --seed 2027

# T5 LatentCLR + DisCo with adaptive K
if [ "$K_VALUE" -gt 0 ]; then
    chunk_val=$((K_VALUE / 10))
    if [ "$chunk_val" -lt 1 ]; then chunk_val=1; fi
    run_track_round2 "T5 LatentCLR (K=$K_VALUE + checkpointing)" "t5_latentclr_v2.log" \
        python3 -u experiments/baselines/latentclr/train.py \
            --K $K_VALUE --B 2 --chunk $chunk_val \
            --epochs 50 --iters-per-epoch 500 \
            --lod 2 --seed 2027 \
            --out experiments/out/latentclr_ffhq_v2
    run_track_round2 "T5 DisCo (K=$K_VALUE + checkpointing)" "t5_disco_v2.log" \
        python3 -u experiments/baselines/disco/train.py \
            --K $K_VALUE --B 2 --chunk $chunk_val \
            --epochs 50 --iters-per-epoch 500 \
            --lod 2 --seed 2027 \
            --out experiments/out/disco_ffhq_v2
else
    log "  skipping LatentCLR/DisCo — keep K=20 result + honest limit"
fi

# Final: inject _meta into ALL new metrics.json
log ""
log "=== inject _meta into all metrics.json ==="
python3 experiments/lib/inject_meta.py > "$LOGDIR/inject_meta.log" 2>&1

log "=== Round 2 wrapper COMPLETE ==="
