#!/bin/bash
# Round 4 sequencer — all remaining 8GB-fittable experiments.
# Order: quick wins first, then heavy compute.
#
# Run with:
#   nohup setsid bash experiments/run_round4.sh \
#     > logs/round4/main.log 2>&1 < /dev/null &
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs/round4"
mkdir -p "$LOGDIR"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/wrapper.log"; }
run_track() {
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

log "=== Round 4 start (commit $GIT_COMMIT) ==="

# ─── Quick wins (~30 min total) ─────────────────────────────────────

# T42: multi-CLIP extension to church (currently bedroom+ffhq)
run_track "T42 multi-CLIP add church" "t42_multi_clip_church.log" \
    python3 -u experiments/metrics/run_multi_clip_c2.py \
        --domains bedroom ffhq church \
        --seed 2027 \
        --out experiments/out/multi_clip_c2_v2

# T43: 3rd-order curvature (∂³I/∂α³)
run_track "T43 3rd-order curvature FFHQ" "t43_third_order.log" \
    python3 -u experiments/metrics/run_third_order.py \
        --num-samples 8 --seed 2027 \
        --out experiments/out/ffhq_third_order

# ─── Sample scaling N=256 then N=512 (~1.5h total) ──────────────────

run_track "T44 sample scaling N=256 bedroom" "t44_bedroom_n256.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain bedroom --n-max 256 --Ns 8 16 32 64 128 256 \
        --seed 2027 \
        --out experiments/out/sample_scaling_bedroom_n256

run_track "T44 sample scaling N=256 ffhq" "t44_ffhq_n256.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain ffhq --n-max 256 --Ns 8 16 32 64 128 256 \
        --seed 2027 \
        --out experiments/out/sample_scaling_ffhq_n256

run_track "T44 sample scaling N=256 church" "t44_church_n256.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain church --n-max 256 --Ns 8 16 32 64 128 256 \
        --seed 2027 \
        --out experiments/out/sample_scaling_church_n256

# Push to N=512 only for ffhq (most attr-rich) — bedroom/church
# already converge at N=256 by the variance pattern
run_track "T45 sample scaling N=512 ffhq" "t45_ffhq_n512.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain ffhq --n-max 512 --Ns 64 128 256 512 \
        --seed 2027 \
        --out experiments/out/sample_scaling_ffhq_n512

# ─── SD timesteps full sweep (~2.5h) ────────────────────────────────

# Original Track 1 used t_frac ≈ 0.3, 0.5, 0.7. Extend to 9 timesteps.
run_track "T46 SD C1/C2 9-timesteps" "t46_sd_9t.log" \
    python3 -u experiments/diffusion/run_c1_c2.py \
        --timestep-fracs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --n-train 16 --n-test 16 \
        --seed 2027 \
        --out experiments/out/sd_c1_c2_full_t

# ─── Encoder backbone ablation FFHQ C5 (~10h total — biggest item) ───

# ResNet-18 (smaller; faster forward; potentially better generalization)
run_track "T47 FFHQ encoder ResNet-18 (40k iter)" "t47_resnet18_train.log" \
    python3 -u experiments/domains/ffhq/encoder/train.py \
        --num-iters 40000 --batch 1 --w-mse-weight 2.0 \
        --backbone resnet18 \
        --ckpt-iters 5000 10000 20000 40000 \
        --out experiments/out/ffhq_c5_resnet18 \
        --seed 2027

run_track "T47 FFHQ encoder ResNet-18 eval" "t47_resnet18_eval.log" \
    python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
        --ckpts experiments/out/ffhq_c5_resnet18/ckpt/enc_005000.pt \
                experiments/out/ffhq_c5_resnet18/ckpt/enc_010000.pt \
                experiments/out/ffhq_c5_resnet18/ckpt/enc_020000.pt \
                experiments/out/ffhq_c5_resnet18/ckpt/enc_040000.pt \
        --num-test 32 \
        --out experiments/out/ffhq_c5_resnet18/eval --seed 2027

run_track "T48 FFHQ encoder ResNet-34 (40k iter)" "t48_resnet34_train.log" \
    python3 -u experiments/domains/ffhq/encoder/train.py \
        --num-iters 40000 --batch 1 --w-mse-weight 2.0 \
        --backbone resnet34 \
        --ckpt-iters 5000 10000 20000 40000 \
        --out experiments/out/ffhq_c5_resnet34 \
        --seed 2027

run_track "T48 FFHQ encoder ResNet-34 eval" "t48_resnet34_eval.log" \
    python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
        --ckpts experiments/out/ffhq_c5_resnet34/ckpt/enc_005000.pt \
                experiments/out/ffhq_c5_resnet34/ckpt/enc_010000.pt \
                experiments/out/ffhq_c5_resnet34/ckpt/enc_020000.pt \
                experiments/out/ffhq_c5_resnet34/ckpt/enc_040000.pt \
        --num-test 32 \
        --out experiments/out/ffhq_c5_resnet34/eval --seed 2027

# ─── Final ──────────────────────────────────────────────────────────

log "=== inject _meta into all new metrics.json ==="
python3 experiments/lib/inject_meta.py > "$LOGDIR/inject_meta.log" 2>&1

log "=== auto-run aggregator to refresh _aggregate_summary.txt ==="
python3 experiments/aggregate_results.py > experiments/out/_aggregate_summary.txt 2>&1
NEW_LINES=$(wc -l < experiments/out/_aggregate_summary.txt)
NEW_TRACKS=$(grep -c "^\[Track" experiments/out/_aggregate_summary.txt)
log "  aggregate summary: $NEW_LINES lines, $NEW_TRACKS track entries"

log "=== Round 4 COMPLETE ==="
log "  Total wall-time: $(($(date +%s) - $(stat -c %Y "$LOGDIR/wrapper.log")))s estimated"
log "  All outputs in experiments/out/, log files in $LOGDIR/"
