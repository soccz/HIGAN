#!/bin/bash
# Remaining-tracks runner — simplified, no bash -c wrapping.
# Each track runs as a plain python call with stdout/stderr to its log.
# Set -uo pipefail keeps failures local; track failures don't abort queue.
#
# Usage:  nohup bash experiments/run_remaining.sh > logs/run_remaining.log 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%T)] $*"; }

# Run one track: print start, time it, continue past failures.
run_track() {
    local name="$1"; shift
    local logfile="$1"; shift
    log "=== $name ==="
    local start=$(date +%s)
    "$@" > "$LOGDIR/$logfile" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - start))
    if [ $rc -ne 0 ]; then
        log "    !! $name FAILED rc=$rc  ${elapsed}s — see $LOGDIR/$logfile"
    else
        log "    $name OK in ${elapsed}s"
    fi
}

log "=== runner starting; cwd=$(pwd) ==="

# ---- Track 1B: SD N=64 follow-up ----
run_track "Track 1B SD N=64" "track1b_sd_n64.log" \
    python3 -u experiments/diffusion/run_c1_c2.py \
        --attrs smile age gender eyeglasses pose \
        --n-train 64 --n-test 64 \
        --timestep-fracs 0.7 0.5 0.3 \
        --out experiments/out/sd_c1_c2_n64 \
        --seed 2027

# ---- Track 4: FFHQ C5 encoder ----
run_track "Track 4 FFHQ encoder train" "track4_ffhq_encoder.log" \
    python3 -u experiments/domains/ffhq/encoder/train.py \
        --num-iters 80000 --batch 1 \
        --ckpt-iters 1000 5000 10000 20000 40000 60000 80000 \
        --seed 2027

run_track "Track 4 FFHQ encoder eval" "track4_ffhq_eval.log" \
    python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
        --ckpts experiments/out/ffhq_c5/ckpt/enc_001000.pt \
                experiments/out/ffhq_c5/ckpt/enc_005000.pt \
                experiments/out/ffhq_c5/ckpt/enc_010000.pt \
                experiments/out/ffhq_c5/ckpt/enc_020000.pt \
                experiments/out/ffhq_c5/ckpt/enc_040000.pt \
                experiments/out/ffhq_c5/ckpt/enc_060000.pt \
                experiments/out/ffhq_c5/ckpt/enc_080000.pt \
        --num-test 128

# ---- Track 5: LatentCLR + DisCo (K=20, lod=2 → 256² rendering) ----
run_track "Track 5 LatentCLR" "track5_latentclr.log" \
    python3 -u experiments/baselines/latentclr/train.py \
        --K 20 --B 2 --chunk 1 --epochs 50 --iters-per-epoch 500 \
        --lod 2 --seed 2027

run_track "Track 5 DisCo" "track5_disco.log" \
    python3 -u experiments/baselines/disco/train.py \
        --K 20 --B 2 --chunk 1 --epochs 50 --iters-per-epoch 500 \
        --lod 2 --seed 2027

# ---- Wave 2: Tracks 6-13 ----
run_track "Track 6 DAAM SD" "track6_daam.log" \
    python3 -u experiments/diffusion/run_daam_comparison.py \
        --n-seeds 32 --seed 2027

run_track "Track 7 Park-NeurIPS23 repro" "track7_park_repro.log" \
    python3 -u experiments/diffusion/run_park_repro.py \
        --n-seeds 24 --K-probes 64 --top-k 6 --seed 2027

run_track "Track 8 FFHQ truncation-ψ" "track8_truncation.log" \
    python3 -u experiments/domains/ffhq/run_truncation_ablation.py \
        --num-samples 64 --seed 2027

run_track "Track 9 multi-CLIP C2" "track9_multi_clip.log" \
    python3 -u experiments/metrics/run_multi_clip_c2.py \
        --num-samples 16 --seed 2027

run_track "Track 10 per-layer C1 bedroom" "track10_per_layer_bedroom.log" \
    python3 -u experiments/metrics/run_per_layer_c1.py \
        --domain bedroom --n-samples 32 --seed 2027

run_track "Track 10 per-layer C1 ffhq" "track10_per_layer_ffhq.log" \
    python3 -u experiments/metrics/run_per_layer_c1.py \
        --domain ffhq --n-samples 32 --seed 2027

run_track "Track 11 walltime benchmark" "track11_walltime.log" \
    python3 -u experiments/method/run_walltime_benchmark.py \
        --domain bedroom --seed 2027

run_track "Track 12 C6 N-scaling bedroom" "track12_c6_scaling.log" \
    python3 -u experiments/metrics/run_c6_scaling.py \
        --Ns 128 192 256 384 512 768 --seed 2027

run_track "Track 13 FFHQ resolution invariance" "track13_resolution.log" \
    python3 -u experiments/domains/ffhq/run_resolution_invariance.py \
        --num-samples 32 --seed 2027

# ---- Wave 3: Tracks 14, 17, 18 ----
run_track "Track 14 noise robustness bedroom" "track14_noise_bedroom.log" \
    python3 -u experiments/metrics/run_noise_robustness.py \
        --domain bedroom --num-samples 64 --seed 2027

run_track "Track 14 noise robustness ffhq" "track14_noise_ffhq.log" \
    python3 -u experiments/metrics/run_noise_robustness.py \
        --domain ffhq --num-samples 64 --seed 2027

run_track "Track 17 intrinsic dim bedroom" "track17_intrinsic_bedroom.log" \
    python3 -u experiments/method/run_intrinsic_dim.py \
        --domain bedroom --n-latents 32 --K-probes 128 --seed 2027

run_track "Track 17 intrinsic dim ffhq" "track17_intrinsic_ffhq.log" \
    python3 -u experiments/method/run_intrinsic_dim.py \
        --domain ffhq --n-latents 32 --K-probes 128 --seed 2027

run_track "Track 18 FD validation" "track18_fd_validation.log" \
    python3 -u experiments/method/run_fd_validation.py \
        --domain bedroom --n-pairs 16 --seed 2027

# ---- Wave 4: Tracks 19, 20 ----
run_track "Track 19 DINOv2 path curvature" "track19_dino.log" \
    python3 -u experiments/metrics/run_dino_path_curvature.py \
        --num-samples 16 --seed 2027

run_track "Track 20 FFHQ alpha-magnitude scan" "track20_alpha_scan.log" \
    python3 -u experiments/domains/ffhq/run_alpha_magnitude_scan.py \
        --n-samples 32 --seed 2027

log "=== ALL REMAINING TRACKS COMPLETE ==="
