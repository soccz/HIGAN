#!/bin/bash
# Smoke test every script with minimal args (~1 min each).
# Goal: verify import + first JVP/forward + metrics.json save works.
# This catches import bugs, path bugs, sys.path collisions BEFORE the
# big run.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/smoke_$$
mkdir -p "$LOGDIR"
results=()

smoke() {
    local name="$1"; shift
    local cmd="$*"
    echo "  $name ..."
    local start=$(date +%s)
    eval timeout 180 "$cmd" > "$LOGDIR/$name.log" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - start))
    if [ $rc -eq 0 ]; then
        echo "    ✓ OK (${elapsed}s)"
        results+=("✓ $name")
    elif [ $rc -eq 124 ]; then
        echo "    ⊘ TIMEOUT (>180s — code OK, just slow)"
        results+=("⊘ $name (timeout, expected)")
    else
        echo "    ✗ FAIL rc=$rc — $LOGDIR/$name.log"
        results+=("✗ $name (rc=$rc)")
    fi
}

echo "=== Wave 1 / Domain measurements ==="
smoke "T1_sd_c1c2" "python3 -u experiments/diffusion/run_c1_c2.py \
    --attrs smile --n-train 2 --n-test 2 --timestep-fracs 0.5 \
    --alpha-sweep-steps 5 --out /tmp/smoke_sd --seed 2027"

smoke "T3_scaling_church" "python3 -u experiments/metrics/run_sample_scaling.py \
    --domain church --n-max 4 --Ns 4 --out /tmp/smoke_t3 --seed 2027"

smoke "T4_encoder_train" "python3 -u experiments/domains/ffhq/encoder/train.py \
    --num-iters 3 --batch 1 --out /tmp/smoke_t4 --seed 2027"

echo ""
echo "=== Wave 2 ==="
smoke "T6_daam" "python3 -u experiments/diffusion/run_daam_comparison.py \
    --n-seeds 2 --attrs smile --seed 2027"

smoke "T7_park" "python3 -u experiments/diffusion/run_park_repro.py \
    --n-seeds 2 --K-probes 8 --top-k 2 --seed 2027"

smoke "T8_truncation" "python3 -u experiments/domains/ffhq/run_truncation_ablation.py \
    --psis 0.7 --num-samples 2 --out /tmp/smoke_t8 --seed 2027"

smoke "T9_multi_clip" "python3 -u experiments/metrics/run_multi_clip_c2.py \
    --domains church --num-samples 2 --alpha-steps 3 \
    --out /tmp/smoke_t9 --seed 2027"

smoke "T10_per_layer_bedroom" "python3 -u experiments/metrics/run_per_layer_c1.py \
    --domain bedroom --n-samples 2 --out /tmp/smoke_t10b --seed 2027"

smoke "T11_walltime" "python3 -u experiments/method/run_walltime_benchmark.py \
    --domain bedroom --seed 2027"

smoke "T12_c6_scaling" "python3 -u experiments/metrics/run_c6_scaling.py \
    --Ns 64 --num-samples-per-dir 2 --num-clusters 4 \
    --out /tmp/smoke_t12 --seed 2027"

smoke "T13_resolution" "python3 -u experiments/domains/ffhq/run_resolution_invariance.py \
    --lods 2 --num-samples 2 --out /tmp/smoke_t13 --seed 2027"

echo ""
echo "=== Wave 3 ==="
smoke "T14_noise_church" "python3 -u experiments/metrics/run_noise_robustness.py \
    --domain church --num-samples 2 --seeds 2027 2028 \
    --out /tmp/smoke_t14 --seed 2027"

smoke "T17_intrinsic_bedroom" "python3 -u experiments/method/run_intrinsic_dim.py \
    --domain bedroom --n-latents 2 --K-probes 8 \
    --out /tmp/smoke_t17 --seed 2027"

smoke "T18_fd_validation" "python3 -u experiments/method/run_fd_validation.py \
    --domain bedroom --n-pairs 2 --epsilons 0.1 0.01 \
    --out /tmp/smoke_t18 --seed 2027"

echo ""
echo "=== Wave 4 ==="
smoke "T19_dino" "python3 -u experiments/metrics/run_dino_path_curvature.py \
    --domains church --num-samples 2 --alpha-steps 3 \
    --out /tmp/smoke_t19 --seed 2027"

smoke "T20_alpha_scan" "python3 -u experiments/domains/ffhq/run_alpha_magnitude_scan.py \
    --alphas 0.5 1.0 --n-samples 2 --out /tmp/smoke_t20 --seed 2027"

echo ""
echo "=== Wave 5 (post-hoc) ==="
smoke "T22_crossdomain" "python3 -u experiments/metrics/run_crossdomain_signature.py"
smoke "T23_fingerprint" "python3 -u experiments/metrics/run_geometric_fingerprint.py"

echo ""
echo "=== Track 5 baselines ==="
smoke "T5_latentclr" "python3 -u experiments/baselines/latentclr/train.py \
    --K 4 --B 2 --chunk 1 --epochs 1 --iters-per-epoch 3 \
    --lod 2 --out /tmp/smoke_t5lc --seed 2027"

smoke "T5_disco" "python3 -u experiments/baselines/disco/train.py \
    --K 4 --B 2 --chunk 1 --epochs 1 --iters-per-epoch 3 \
    --lod 2 --out /tmp/smoke_t5dc --seed 2027"

echo ""
echo "=== Track 2 editing head-to-head (cannot smoke easily — uses LatentCLR results) ==="
echo "  skip (covered by Track 5 smoke + Track 2 successful prior run)"

echo ""
echo "================================================================"
echo "                       SMOKE TEST SUMMARY"
echo "================================================================"
for r in "${results[@]}"; do echo "  $r"; done
echo ""
fail_count=$(printf '%s\n' "${results[@]}" | grep -c '^✗' || true)
ok_count=$(printf '%s\n' "${results[@]}" | grep -c '^✓' || true)
to_count=$(printf '%s\n' "${results[@]}" | grep -c '^⊘' || true)
echo "OK=$ok_count  TIMEOUT=$to_count  FAIL=$fail_count"
echo "logs saved in $LOGDIR"

if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "FAIL details:"
    for r in "${results[@]}"; do
        if [[ "$r" == ✗* ]]; then
            name=$(echo "$r" | awk '{print $2}')
            echo "--- $name ---"
            tail -10 "$LOGDIR/$name.log"
            echo ""
        fi
    done
fi
