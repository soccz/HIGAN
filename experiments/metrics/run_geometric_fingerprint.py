"""Track 23 — geometric fingerprint per attribute (multi-axis meta-analysis).

Compiles every per-attribute geometric measurement we have:
- C1 pixel ρ
- C2 CLIP path / direct ratio
- C3 per-layer-IoU score (when available)
- argmax canonical layer fraction (Track 10, when available)
- log-slope of ρ(α) (Track 20, when available)

Then hierarchical clustering (Ward linkage) on z-normalised vectors
with k=3, expecting structural / mid / textural separation.

Graceful with missing inputs — uses only what's on disk.

See designs/23_geometric_fingerprint.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
OUT = PAPER / "experiments" / "out"


def safe_load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    out = OUT / "geometric_fingerprint"
    out.mkdir(parents=True, exist_ok=True)

    BEDROOM_PIXEL = {
        "indoor_lighting": 0.495, "wood": 0.624, "carpet": 0.95,
        "cluttered_space": 0.85, "glossy": 0.92, "dirt": 0.93,
        "scary": 1.10, "view": 23.22,
    }
    FFHQ_PIXEL = {
        "smile": 1.75, "age": 7.62, "gender": 8.71,
        "eyeglasses": 22.82, "pose": 49.87,
    }

    bed_path = safe_load(OUT / "bedroom_c2_path" / "metrics.json")
    ffhq_path = safe_load(OUT / "ffhq_c2_path" / "metrics.json")
    bed_c3 = safe_load(OUT / "bedroom_c3_iou" / "metrics.json")
    ffhq_c3 = safe_load(OUT / "ffhq_c3_iou" / "metrics.json")
    bed_layer = safe_load(OUT / "per_layer_c1_bedroom" / "metrics.json")
    ffhq_layer = safe_load(OUT / "per_layer_c1_ffhq" / "metrics.json")
    ffhq_alpha = safe_load(OUT / "ffhq_alpha_scan" / "metrics.json")

    points = []
    names = []
    pri_labels = []

    def push(name, domain_apriori, vec):
        names.append(name)
        points.append(vec)
        pri_labels.append(domain_apriori)

    # Bedroom
    for a, pix in BEDROOM_PIXEL.items():
        clip = next((e["mean_ratio"] for e in
                     (bed_path["per_attr"] if bed_path else [])
                     if e.get("attr") == a), None)
        c3 = (bed_c3["c3_scores"][a]["c3_score"]
              if bed_c3 and a in bed_c3.get("c3_scores", {}) else None)
        layer_frac = None
        if bed_layer and a in bed_layer.get("per_attr", {}):
            argmax = bed_layer["per_attr"][a]["argmax_layer"]
            L = bed_layer.get("L", 14)
            layer_frac = argmax / max(L - 1, 1)
        # alpha-slope only on FFHQ
        alpha = None
        # mark structural
        struct = a == "view"
        push(f"bed-{a}", "structural" if struct else "textural",
             [pix, clip, c3, layer_frac, alpha])

    # FFHQ
    for a, pix in FFHQ_PIXEL.items():
        clip = next((e["mean_ratio"] for e in
                     (ffhq_path["per_attr"] if ffhq_path else [])
                     if e.get("attr") == a), None)
        c3 = (ffhq_c3["c3_scores"][a]["c3_score"]
              if ffhq_c3 and a in ffhq_c3.get("c3_scores", {}) else None)
        layer_frac = None
        if ffhq_layer and a in ffhq_layer.get("per_attr", {}):
            argmax = ffhq_layer["per_attr"][a]["argmax_layer"]
            L = ffhq_layer.get("L", 18)
            layer_frac = argmax / max(L - 1, 1)
        alpha = (ffhq_alpha["results"][a].get("log_slope")
                 if ffhq_alpha and a in ffhq_alpha.get("results", {}) else None)
        struct = a in {"pose", "eyeglasses"}
        push(f"ffhq-{a}", "structural" if struct else "textural",
             [pix, clip, c3, layer_frac, alpha])

    points = np.array(points, dtype=object)
    print(f"=== geometric fingerprint, {len(names)} entries ===")
    feat_names = ["pixel_ρ", "CLIP_path", "C3_score",
                   "layer_argmax_frac", "alpha_slope"]
    print(f"  features: {feat_names}")

    # drop fully-missing columns
    M = np.array([[np.nan if v is None else float(v)
                    for v in row] for row in points])
    print("  per-feature coverage: " + ", ".join(
        f"{feat_names[i]}={int(np.sum(~np.isnan(M[:, i])))}/{len(names)}"
        for i in range(M.shape[1])
    ))

    # only use features that have ≥ 80% coverage
    valid = [i for i in range(M.shape[1])
              if np.sum(~np.isnan(M[:, i])) / len(names) >= 0.8]
    print(f"  using features at >=80% coverage: "
          f"{[feat_names[i] for i in valid]}")
    M_valid = M[:, valid]

    # mean-impute remaining NaNs
    col_means = np.nanmean(M_valid, axis=0)
    nan_mask = np.isnan(M_valid)
    for i in range(M_valid.shape[1]):
        M_valid[nan_mask[:, i], i] = col_means[i]

    # log-scale pixel_ρ for clustering
    if "pixel_ρ" in [feat_names[i] for i in valid]:
        k = [feat_names[i] for i in valid].index("pixel_ρ")
        M_valid[:, k] = np.log10(M_valid[:, k].clip(min=1e-3))

    # z-normalise
    M_norm = (M_valid - M_valid.mean(0)) / (M_valid.std(0) + 1e-8)

    # hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(M_norm, method="ward")
    for k_clus in [2, 3]:
        labels = fcluster(Z, t=k_clus, criterion="maxclust")
        print(f"\nk={k_clus} clusters:")
        for n, l, pl in zip(names, labels, pri_labels):
            print(f"  cluster {l}  |  {pl:11s}  |  {n}")

        # ARI vs a-priori binary labels
        from sklearn.metrics import adjusted_rand_score
        pri_int = [1 if p == "structural" else 0 for p in pri_labels]
        ari = adjusted_rand_score(pri_int, labels)
        print(f"  k={k_clus}  ARI vs (structural / textural) labels = "
              f"{ari:+.3f}")

    payload = {
        "names": names,
        "apriori_labels": pri_labels,
        "feature_names": feat_names,
        "coverage_used": [feat_names[i] for i in valid],
        "matrix": M_valid.tolist(),
        "z_normalized": M_norm.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
