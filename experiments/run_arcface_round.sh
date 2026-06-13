#!/bin/bash
# ArcFace robustness round — re-test prediction & control halves on FFHQ
# with a real VGGFace2 face-recognition identity metric (not CLIP-cosine).
# 20 seeds. If both halves match the CLIP-cosine result, the
# prediction-strong / control-flat finding is metric-robust.
#
# nohup setsid bash experiments/run_arcface_round.sh \
#   > logs/arcface/main.log 2>&1 < /dev/null &
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
export TORCH_HOME=/home/soccz/22tb/.cache/torch
LOGDIR="$PAPER_DIR/logs/arcface"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%T)] $*" | tee -a "$LOGDIR/wrapper.log"; }
run_track() {
    local name="$1"; shift; local lf="$1"; shift
    log "=== $name ==="
    local s=$(date +%s)
    "$@" > "$LOGDIR/$lf" 2>&1
    local rc=$?; local e=$(($(date +%s) - s))
    if [ $rc -ne 0 ]; then log "    !! $name FAILED rc=$rc ${e}s"
    else log "    $name OK in ${e}s"; fi
}

log "=== ArcFace robustness round start ==="

for seed in $(seq 2037 2056); do
    run_track "ArcFace FFHQ seed=$seed" "arcface_${seed}.log" \
        python3 -u experiments/control/run_arcface_robustness_ffhq.py \
            --attrs smile age pose gender eyeglasses \
            --candidate-k 6 --ganspace-samples 2048 \
            --methods ganspace sefa \
            --n-calib 16 --n-test 64 \
            --alpha-steps 7 --max-alpha 6 \
            --target-quantile 0.25 --min-gain-quantile 0.5 \
            --batch 4 --seed "$seed" \
            --out "experiments/out/arcface_robustness_ffhq/seed_${seed}"
done

# ---- aggregate across 20 seeds ----
log "=== aggregate ArcFace robustness ==="
python3 - <<'PYEOF' > "$LOGDIR/aggregate.log" 2>&1
import json
from pathlib import Path
import numpy as np

OUT = Path("experiments/out/arcface_robustness_ffhq")
runs = [json.load(open(p)) for p in sorted(OUT.glob("seed_*/metrics.json"))]
if not runs:
    print("no runs"); raise SystemExit

def col(path):
    out = []
    for r in runs:
        v = r
        for k in path: v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)): out.append(v)
    return np.array(out)

summ = {
    "n_seeds": len(runs),
    "prediction": {
        "spearman_rho_vs_clip_damage": {
            "mean": float(col(["prediction","spearman_rho_vs_clip_damage"]).mean()),
            "std": float(col(["prediction","spearman_rho_vs_clip_damage"]).std())},
        "spearman_rho_vs_arcface_damage": {
            "mean": float(col(["prediction","spearman_rho_vs_arcface_damage"]).mean()),
            "std": float(col(["prediction","spearman_rho_vs_arcface_damage"]).std())},
    },
    "control": {
        "clip_id_low_rho_win_rate": {
            "mean": float(col(["control","clip_id_low_rho_win_rate"]).mean()),
            "std": float(col(["control","clip_id_low_rho_win_rate"]).std())},
        "arcface_id_low_rho_win_rate": {
            "mean": float(col(["control","arcface_id_low_rho_win_rate"]).mean()),
            "std": float(col(["control","arcface_id_low_rho_win_rate"]).std())},
    },
}
json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
print()
p = summ["prediction"]; c = summ["control"]
print("PREDICTION rho->damage Spearman:")
print(f"  CLIP    {p['spearman_rho_vs_clip_damage']['mean']:+.3f} ± {p['spearman_rho_vs_clip_damage']['std']:.3f}")
print(f"  ArcFace {p['spearman_rho_vs_arcface_damage']['mean']:+.3f} ± {p['spearman_rho_vs_arcface_damage']['std']:.3f}")
print("CONTROL low-rho win-rate:")
print(f"  CLIP    {c['clip_id_low_rho_win_rate']['mean']:.3f} ± {c['clip_id_low_rho_win_rate']['std']:.3f}")
print(f"  ArcFace {c['arcface_id_low_rho_win_rate']['mean']:.3f} ± {c['arcface_id_low_rho_win_rate']['std']:.3f}")
PYEOF
cat "$LOGDIR/aggregate.log" | tee -a "$LOGDIR/wrapper.log"

log "=== inject_meta ==="
python3 experiments/lib/inject_meta.py > "$LOGDIR/inject_meta.log" 2>&1

log "=== ArcFace robustness round COMPLETE ==="
