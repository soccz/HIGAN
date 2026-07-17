#!/bin/bash
# Extra analyses sequencer — Round 3.
# Ports higan_dev/scripts 23/24/25 to paper with seed=2027 + _meta,
# plus multi-seed bootstrap for C6 cross-domain k=2 (currently seed=2027 only).
set -uo pipefail
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
HIGAN_DEV="$(dirname "$PAPER_DIR")/higan_dev"
cd "$PAPER_DIR"
LOGDIR="$PAPER_DIR/logs/round3_extra"
OUT="$PAPER_DIR/experiments/out"
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

log "=== Round 3 Extra Analyses start (commit $GIT_COMMIT) ==="

export PYTHONPATH="$HIGAN_DEV:${PYTHONPATH:-}"
cd "$HIGAN_DEV"

# Track 21 — Full 8×14 attribute × layer matrix (bedroom)
log "  reseeding 23_full_matrix with seed=2027 (was hardcoded 101)"
sed -i 's/manual_seed(101)/manual_seed(2027)/' scripts/23_full_matrix.py 2>/dev/null
run_track "T21 8x14 attribute-layer matrix" "t21_matrix.log" \
    python3 -u scripts/23_full_matrix.py \
        --num-samples 16 \
        --out "$OUT/full_matrix_bedroom"

# Track 22 — CLIP zero-shot cluster labeling
run_track "T22 CLIP cluster labels (bedroom taxonomy)" "t22_clip_labels.log" \
    python3 -u scripts/24_clip_label_clusters.py \
        --num-bases 4 \
        --num-dirs-per-cluster 8 \
        --seed 2027 \
        --out "$OUT/clip_cluster_labels_bedroom"

# Track 25 — Real LSUN photo cycle (encoder transfer limit)
run_track "T25 Real LSUN photo cycle" "t25_real_photo.log" \
    python3 -u scripts/25_real_photo_cycle.py \
        --out "$OUT/real_photo_cycle"

cd "$PAPER_DIR"

# Inject _meta into new outputs
log "=== inject _meta into new outputs ==="
python3 experiments/lib/inject_meta.py > "$LOGDIR/inject_meta.log" 2>&1

# Also wrap each new output with a structured metrics.json
log "=== wrapping new outputs with paper-format metrics.json ==="
python3 - <<'PYEOF' > "$LOGDIR/wrap_metrics.log" 2>&1
"""Convert dev outputs (.npz, .txt) to paper-format metrics.json + _meta."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("experiments")))
from lib.reproducibility import run_metadata

OUT = Path("experiments/out")
meta = run_metadata(seed=2027)

# Track 21: 8x14 matrix
mp = OUT / "full_matrix_bedroom"
if (mp / "intensities.npz").exists():
    d = np.load(mp / "intensities.npz", allow_pickle=True)
    attrs = d["attrs"].tolist()
    intensities = d["intensities"].tolist()
    peak_layer = d["peak_layer"].tolist()
    canonical = [list(c) if c is not None else [] for c in d["canonical_layers"]]
    out_data = {
        "attrs": attrs,
        "intensities_per_layer": dict(zip(attrs, intensities)),
        "peak_layer_per_attr": dict(zip(attrs, peak_layer)),
        "canonical_layers_per_attr": dict(zip(attrs, canonical)),
        "num_attrs": len(attrs),
        "num_layers": len(intensities[0]) if intensities else 0,
        "_meta": meta,
    }
    (mp / "metrics.json").write_text(json.dumps(out_data, indent=2))
    print(f"  wrote {mp}/metrics.json")

# Track 22: CLIP cluster labels (parse labels.txt)
cp = OUT / "clip_cluster_labels_bedroom"
if (cp / "labels.txt").exists():
    lines = (cp / "labels.txt").read_text().strip().split("\n")
    clusters = {}
    for line in lines:
        if ":" not in line:
            continue
        cid_str, rest = line.split(":", 1)
        cid = int(cid_str.replace("cluster", "").strip())
        # Parse "word (+score)" tuples
        items = []
        for tok in rest.strip().split("  "):
            tok = tok.strip()
            if not tok or "(" not in tok:
                continue
            word, score = tok.rsplit("(", 1)
            score = float(score.rstrip(")"))
            items.append({"label": word.strip(), "score": score})
        clusters[cid] = items
    out_data = {
        "clusters": clusters,
        "num_clusters": len(clusters),
        "method": "open_clip ViT-B-32 laion2b_s34b_b79k zero-shot",
        "_meta": meta,
    }
    (cp / "metrics.json").write_text(json.dumps(out_data, indent=2))
    print(f"  wrote {cp}/metrics.json")

# Track 25: Real LSUN — already has metrics.json, just inject _meta
rp = OUT / "real_photo_cycle"
mj = rp / "metrics.json"
if mj.exists():
    existing = json.loads(mj.read_text())
    if isinstance(existing, list):
        existing = {"per_photo": existing}
    if "_meta" not in existing:
        existing["_meta"] = meta
    # Compute summary stats
    if "per_photo" in existing:
        lpips_optim = [r["lpips_optim"] for r in existing["per_photo"]]
        lpips_enc = [r["lpips_enc"] for r in existing["per_photo"]]
        existing["summary"] = {
            "n_photos": len(existing["per_photo"]),
            "lpips_optim_mean": float(np.mean(lpips_optim)),
            "lpips_optim_std": float(np.std(lpips_optim)),
            "lpips_enc_mean": float(np.mean(lpips_enc)),
            "lpips_enc_std": float(np.std(lpips_enc)),
            "encoder_gap_pct": float(
                (np.mean(lpips_enc) - np.mean(lpips_optim))
                / np.mean(lpips_optim) * 100),
        }
    mj.write_text(json.dumps(existing, indent=2))
    print(f"  wrote {mj} (summary added)")

print("=== wrap_metrics done ===")
PYEOF

log "=== Round 3 Extra Analyses COMPLETE ==="

log "=== Round 3 Extra Analyses COMPLETE ==="
