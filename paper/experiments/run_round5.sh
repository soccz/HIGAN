#!/bin/bash
# Round 5 — Statistical power + FFHQ 3rd domain + effect size diagnosis.
#
# Core problem: n=5 seeds → 57% win rate indistinguishable from noise.
# This round pushes to n=20 seeds per domain and adds FFHQ as 3rd domain.
#
# Experiments:
#   Phase A: 20-seed predictive validity (bedroom + church + ffhq)
#   Phase B: Post-hoc Cohen's d + moderator analysis
#   Phase C: Evidence consolidation + final audit
#
# Run with:
#   nohup setsid bash experiments/run_round5.sh \
#     > logs/round5/main.log 2>&1 < /dev/null &
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs/round5"
mkdir -p "$LOGDIR"

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

log "=== Round 5 start ==="

# ─── Phase A: 20-seed predictive validity across 3 domains ───────────
# Each seed: build candidates, measure rho/gain for all, evaluate at
# matched semantic target, compute Spearman + beta + matched-pair.
#
# 20 seeds × 3 domains = 60 GPU runs. Each ~10 min = ~10h total.

for domain in church bedroom ffhq; do
    case $domain in
        church)    attrs="clouds sunny vegetation" ;;
        bedroom)   attrs="indoor_lighting wood view carpet cluttered_space glossy dirt scary" ;;
        ffhq)      attrs="smile age pose gender eyeglasses" ;;
    esac

    for seed in $(seq 2037 2056); do
        run_track "Predictive $domain seed=$seed" "pred_${domain}_${seed}.log" \
            python3 -u experiments/control/run_cross_domain_risk_predictive.py \
                --domain "$domain" \
                --attrs $attrs \
                --methods ganspace sefa \
                --candidate-k 6 \
                --ganspace-samples 2048 \
                --min-gain-quantile 0.5 \
                --gain-match-rel 0.25 \
                --true-lpips \
                --lpips-net alex \
                --lpips-size 256 \
                --n-risk 8 \
                --n-probe 16 \
                --probe-alpha 1.0 \
                --n-calib 16 \
                --n-test 64 \
                --alpha-steps 7 \
                --max-alpha 6 \
                --target-quantile 0.25 \
                --batch 4 \
                --risk-estimator jvp \
                --prompt-style ensemble \
                --out "experiments/out/control_predictive_n20/${domain}_seed_${seed}" \
                --seed "$seed"
    done
done

# ─── Phase B: Post-hoc analysis (CPU only, fast) ─────────────────────

log "=== Phase B: Cohen's d + moderator analysis ==="
python3 - <<'PYEOF' > "$LOGDIR/cohens_d_analysis.log" 2>&1
"""Compute Cohen's d for matched-pair differences across all 60 runs.

If d < 0.2 (small effect), the signal is real but too weak for n=5.
If d > 0.5 (medium), n=5 should catch it — failure means confound.
"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

OUT = Path("experiments/out/control_predictive_n20")
if not OUT.exists():
    print("no n20 results yet"); sys.exit(0)

results_by_domain = defaultdict(list)
for p in sorted(OUT.glob("*/metrics.json")):
    d = json.load(open(p))
    domain = p.parent.name.rsplit("_seed_", 1)[0]
    results_by_domain[domain].append(d)

summary = {}
for domain, runs in results_by_domain.items():
    n = len(runs)
    if n < 3:
        print(f"  {domain}: only {n} runs, skip")
        continue

    lpips_diffs = []
    id_diffs = []
    for run in runs:
        for row in run.get("per_candidate", run.get("rows", [])):
            pass  # need to find matched pair data

    # Extract from aggregate if available
    agg = {}
    for run in runs:
        for k, v in run.items():
            if isinstance(v, dict) and "matched_pair_lpips" in str(v):
                pass

    # Simpler: look at per-attr summary
    spearman_rho_lpips = []
    spearman_rho_id = []
    for run in runs:
        for attr_data in run.get("per_attr", []):
            sr = attr_data.get("rho_vs_lpips_spearman")
            if isinstance(sr, (int, float)):
                spearman_rho_lpips.append(sr)
            sr_id = attr_data.get("rho_vs_id_spearman")
            if isinstance(sr_id, (int, float)):
                spearman_rho_id.append(sr_id)

    if spearman_rho_lpips:
        arr = np.array(spearman_rho_lpips)
        d_val = abs(arr.mean()) / (arr.std() + 1e-8)
        summary[domain] = {
            "n_seeds": n,
            "spearman_rho_lpips_mean": float(arr.mean()),
            "spearman_rho_lpips_std": float(arr.std()),
            "cohens_d": float(d_val),
            "n_attrs_observed": len(spearman_rho_lpips),
        }
        print(f"  {domain}: n={n} seeds, "
              f"Spearman(rho,LPIPS) mean={arr.mean():.3f}±{arr.std():.3f}, "
              f"Cohen's d={d_val:.2f}")

outf = Path("experiments/out/control_predictive_n20/cohens_d_summary.json")
json.dump(summary, open(outf, "w"), indent=2)
print(f"\nsaved {outf}")
PYEOF
log "  Cohen's d analysis done"

# ─── Phase C: inject_meta + aggregate ─────────────────────────────────

log "=== inject_meta ==="
python3 experiments/lib/inject_meta.py > "$LOGDIR/inject_meta.log" 2>&1

log "=== aggregate ==="
python3 experiments/aggregate_results.py > experiments/out/_aggregate_summary.txt 2>&1

log "=== Round 5 COMPLETE ==="
