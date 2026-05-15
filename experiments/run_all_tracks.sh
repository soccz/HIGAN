#!/bin/bash
# Sequential GPU runner for Tracks 1-5. Track 1 is assumed to already be
# running (or completed); this script does Tracks 3, 5, 4, 2 after waiting
# for any current run_c1_c2.py / sample_scaling / latentclr / disco / encoder
# to finish.
#
# Run with:  nohup bash experiments/run_all_tracks.sh > logs/run_all.log 2>&1 &
#
# Each track logs to logs/track<N>_*.log and saves outputs under
# experiments/out/*.
set -uo pipefail

cd "$(dirname "$0")/.."
LOGDIR="$(realpath logs)"
mkdir -p "$LOGDIR"

wait_gpu_free() {
    # Wait until no python process with "experiments/" in its cmd is running.
    while pgrep -af 'python.*experiments/' | grep -v 'run_all_tracks.sh' \
            | grep -v "$$" >/dev/null; do
        sleep 30
    done
}

echo "[$(date +%T)] === Track-runner starting; waiting for any current GPU job ==="
wait_gpu_free

echo "[$(date +%T)] === Track 3: sample scaling — bedroom ==="
python3 -u experiments/metrics/run_sample_scaling.py --domain bedroom \
    > "$LOGDIR/track3_scaling_bedroom.log" 2>&1
echo "[$(date +%T)] Track 3 bedroom done"

echo "[$(date +%T)] === Track 3: sample scaling — ffhq (N_max=64) ==="
python3 -u experiments/metrics/run_sample_scaling.py --domain ffhq --n-max 64 \
    --Ns 8 16 32 64 \
    > "$LOGDIR/track3_scaling_ffhq.log" 2>&1
echo "[$(date +%T)] Track 3 ffhq done"

echo "[$(date +%T)] === Track 3: sample scaling — church ==="
python3 -u experiments/metrics/run_sample_scaling.py --domain church \
    > "$LOGDIR/track3_scaling_church.log" 2>&1
echo "[$(date +%T)] Track 3 church done"

echo "[$(date +%T)] === Track 5: LatentCLR training ==="
python3 -u experiments/baselines/latentclr/train.py \
    --K 100 --B 8 --chunk 10 --epochs 100 --iters-per-epoch 500 \
    > "$LOGDIR/track5_latentclr.log" 2>&1
echo "[$(date +%T)] Track 5 LatentCLR done"

echo "[$(date +%T)] === Track 5: DisCo training ==="
python3 -u experiments/baselines/disco/train.py \
    --K 100 --B 8 --chunk 10 --epochs 100 --iters-per-epoch 500 \
    > "$LOGDIR/track5_disco.log" 2>&1
echo "[$(date +%T)] Track 5 DisCo done"

echo "[$(date +%T)] === Track 4: FFHQ C5 encoder training ==="
python3 -u experiments/domains/ffhq/encoder/train.py \
    --num-iters 40000 --batch 1 \
    > "$LOGDIR/track4_ffhq_encoder.log" 2>&1
echo "[$(date +%T)] Track 4 training done; running eval..."
python3 -u experiments/domains/ffhq/encoder/eval_c5.py \
    --ckpts experiments/out/ffhq_c5/ckpt/enc_001000.pt \
            experiments/out/ffhq_c5/ckpt/enc_005000.pt \
            experiments/out/ffhq_c5/ckpt/enc_010000.pt \
            experiments/out/ffhq_c5/ckpt/enc_020000.pt \
            experiments/out/ffhq_c5/ckpt/enc_040000.pt \
    --num-test 64 \
    > "$LOGDIR/track4_ffhq_eval.log" 2>&1
echo "[$(date +%T)] Track 4 done"

echo "[$(date +%T)] === Track 2: Editing head-to-head N=1000 ==="
python3 -u experiments/baselines/run_editing_head_to_head.py \
    --n-test 1000 \
    > "$LOGDIR/track2_editing.log" 2>&1
echo "[$(date +%T)] === Wave 1 complete; starting Wave 2 ==="

# ---- Wave 2 ----

echo "[$(date +%T)] === Track 6: DAAM SD baseline ==="
python3 -u experiments/diffusion/run_daam_comparison.py \
    --n-seeds 16 \
    > "$LOGDIR/track6_daam.log" 2>&1

echo "[$(date +%T)] === Track 7: Park-NeurIPS23 Riemannian repro ==="
python3 -u experiments/diffusion/run_park_repro.py \
    --n-seeds 8 --K-probes 32 --top-k 4 \
    > "$LOGDIR/track7_park_repro.log" 2>&1

echo "[$(date +%T)] === Track 8: FFHQ truncation-ψ ablation ==="
python3 -u experiments/domains/ffhq/run_truncation_ablation.py \
    --num-samples 16 \
    > "$LOGDIR/track8_truncation.log" 2>&1

echo "[$(date +%T)] === Track 9: Multi-CLIP-encoder C2 robustness ==="
python3 -u experiments/metrics/run_multi_clip_c2.py \
    --num-samples 8 \
    > "$LOGDIR/track9_multi_clip.log" 2>&1

echo "[$(date +%T)] === Track 10: Per-layer C1 (bedroom + ffhq) ==="
python3 -u experiments/metrics/run_per_layer_c1.py --domain bedroom \
    > "$LOGDIR/track10_per_layer_bedroom.log" 2>&1
python3 -u experiments/metrics/run_per_layer_c1.py --domain ffhq \
    > "$LOGDIR/track10_per_layer_ffhq.log" 2>&1

echo "[$(date +%T)] === Track 11: Wall-clock benchmark ==="
python3 -u experiments/method/run_walltime_benchmark.py --domain bedroom \
    > "$LOGDIR/track11_walltime.log" 2>&1

echo "[$(date +%T)] === Track 12: C6 N-scaling (bedroom) ==="
python3 -u experiments/metrics/run_c6_scaling.py \
    --Ns 128 192 256 384 512 \
    > "$LOGDIR/track12_c6_scaling.log" 2>&1

echo "[$(date +%T)] === Track 13: FFHQ resolution invariance ==="
python3 -u experiments/domains/ffhq/run_resolution_invariance.py \
    --num-samples 16 \
    > "$LOGDIR/track13_resolution.log" 2>&1

echo "[$(date +%T)] === ALL TRACKS COMPLETE (Wave 1 + Wave 2) ==="
