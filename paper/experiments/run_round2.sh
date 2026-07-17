#!/bin/bash
# Round 2: re-run only the tracks affected by the 3 critical issues.
# T3 scaling (3 domains) — mean-of-ratios convention now
# T4 FFHQ encoder train (w_mse_weight 0.1 → 2.0) + eval
# T5 LatentCLR + DisCo with gradient checkpointing → K=100 possible
#
# Other tracks keep their current results from final run.
# Launch ONLY after run_final.sh completes (wait for sequencer to die).
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs/round2"
mkdir -p "$LOGDIR"
GIT_COMMIT=$(git rev-parse --short HEAD)

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/00_runner.log"; }
run_track() {
    local name="$1"; shift; local logfile="$1"; shift
    log "=== $name ==="
    local start=$(date +%s)
    "$@" > "$LOGDIR/$logfile" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - start))
    if [ $rc -ne 0 ]; then
        log "    !! $name FAILED rc=$rc  ${elapsed}s"
    else
        log "    $name OK in ${elapsed}s"
    fi
}

log "=== Round 2 start (commit $GIT_COMMIT) ==="

# T3 scaling re-run with mean-of-ratios convention
run_track "T3 bedroom (mean-of-ratios)" "t3_bedroom.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain bedroom --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_bedroom_v2

run_track "T3 ffhq (mean-of-ratios)" "t3_ffhq.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain ffhq --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_ffhq_v2

run_track "T3 church (mean-of-ratios)" "t3_church.log" \
    python3 -u experiments/metrics/run_sample_scaling.py \
        --domain church --n-max 128 --Ns 8 16 32 64 128 --seed 2027 \
        --out experiments/out/sample_scaling_church_v2

# T4 FFHQ encoder re-train with bumped w_mse weight + longer
run_track "T4 FFHQ encoder train v2" "t4_train.log" \
    python3 -u experiments/domains/ffhq/encoder/train.py \
        --num-iters 160000 --batch 1 --w-mse-weight 2.0 \
        --ckpt-iters 5000 20000 40000 80000 120000 160000 \
        --out experiments/out/ffhq_c5_v2 \
        --seed 2027

run_track "T4 FFHQ encoder eval v2" "t4_eval.log" \
    python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
        --ckpts experiments/out/ffhq_c5_v2/ckpt/enc_005000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_020000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_040000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_080000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_120000.pt \
                experiments/out/ffhq_c5_v2/ckpt/enc_160000.pt \
        --num-test 32 \
        --out experiments/out/ffhq_c5_v2/eval

# T5 LatentCLR + DisCo with gradient checkpointing (K=100 fits in 8GB now)
run_track "T5 LatentCLR (K=100 + checkpointing)" "t5_latentclr.log" \
    python3 -u experiments/baselines/latentclr/train.py \
        --K 100 --B 2 --chunk 10 --epochs 50 --iters-per-epoch 500 \
        --lod 2 --seed 2027 \
        --out experiments/out/latentclr_ffhq_v2

run_track "T5 DisCo (K=100 + checkpointing)" "t5_disco.log" \
    python3 -u experiments/baselines/disco/train.py \
        --K 100 --B 2 --chunk 10 --epochs 50 --iters-per-epoch 500 \
        --lod 2 --seed 2027 \
        --out experiments/out/disco_ffhq_v2

log "=== Round 2 COMPLETE ==="
