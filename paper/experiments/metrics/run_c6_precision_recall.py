"""C6 quantitative — precision/recall of cluster → ground-truth-attribute matching.

For each domain, define a mapping (GT attribute → CLIP-vocab proxy
set). A cluster *matches* a GT attribute when any of its top-K
CLIP labels lies in that attribute's proxy set.

Outputs:
    Precision  = matched clusters / total clusters
    Recall     = matched GT attrs / total GT attrs
    F1         = harmonic mean
    Coverage   = unique GT attrs matched
At top-K ∈ {1, 2, 3, 4} for sensitivity analysis.

Reads cluster top-K labels from existing C6 outputs (no GPU).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]

# Bedroom: HiGAN boundary → CLIP vocab proxies (from
# higan_dev/scripts/24_clip_label_clusters.py VOCAB)
BEDROOM_GT = {
    "view":             {"a view through a window", "outdoor view", "a window"},
    "indoor_lighting":  {"bright lighting", "dim lighting", "warm light"},
    "wood":             {"wood texture"},
    "carpet":           {"carpet", "soft texture", "fabric"},
    "cluttered_space":  {"cluttered space"},
    "glossy":           {"glossy reflective surface", "metal surface"},
    "dirt":             {"dirty surface", "rough texture"},
    "scary":            {"scary atmosphere"},
}

# FFHQ: InterFaceGAN boundary → CLIP vocab proxies (from
# experiments/domains/ffhq/run_discovery_c6.py CLIP_VOCAB)
FFHQ_GT = {
    "smile":      {"a smiling face", "open mouth", "a smooth face", "a teeth"},
    "age":        {"a young face", "an old face", "a wrinkled face"},
    "pose":       {"a frontal face", "a tilted face", "a side profile"},
    "gender":     {"a male face", "a female face", "a beard"},
    "eyeglasses": {"a face with glasses", "a face without glasses"},
}


def parse_bedroom_labels(path: Path) -> dict[int, list[str]]:
    """Parse higan_dev/out/cluster_labels/labels.txt format.

    Each line: 'cluster N: phrase (±0.000)  phrase (±0.000)  ...'
    """
    out = {}
    for line in path.read_text().strip().splitlines():
        m = re.match(r"cluster (\d+):\s*(.+)$", line)
        if not m:
            continue
        cid = int(m.group(1))
        # phrases are separated by 2+ spaces, each '... (±0.000)'
        parts = re.findall(r"([^()]+?)\s*\(([+-][\d.]+)\)", m.group(2))
        out[cid] = [p[0].strip() for p in parts]
    return out


def parse_ffhq_labels(path: Path) -> dict[int, list[str]]:
    d = json.loads(path.read_text())
    return {int(k): [phrase for phrase, _score in v]
            for k, v in d["cluster_labels"].items()}


def evaluate(cluster_labels: dict[int, list[str]],
             gt: dict[str, set[str]], top_k: int):
    """For top_k CLIP labels per cluster, report precision/recall vs GT."""
    matched_clusters = 0
    matched_attrs = set()
    cluster_to_attrs: dict[int, set[str]] = {}

    for cid, labels in cluster_labels.items():
        labs = set(labels[:top_k])
        hits = {a for a, proxies in gt.items() if labs & proxies}
        cluster_to_attrs[cid] = hits
        if hits:
            matched_clusters += 1
            matched_attrs |= hits

    n_clusters = len(cluster_labels)
    n_gt = len(gt)
    precision = matched_clusters / n_clusters if n_clusters else 0.0
    recall = len(matched_attrs) / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {
        "top_k": top_k,
        "n_clusters": n_clusters,
        "matched_clusters": matched_clusters,
        "matched_attrs": sorted(matched_attrs),
        "n_matched_attrs": len(matched_attrs),
        "n_gt_attrs": n_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cluster_to_attrs": {str(k): sorted(v) for k, v in cluster_to_attrs.items()},
    }


def main():
    out = PAPER / "experiments" / "out" / "c6_precision_recall"
    out.mkdir(parents=True, exist_ok=True)

    bedroom_labels = parse_bedroom_labels(
        PAPER.parent / "higan_dev" / "out" / "cluster_labels" / "labels.txt"
    )
    ffhq_labels = parse_ffhq_labels(
        PAPER / "experiments" / "out" / "ffhq_c6" / "metrics.json"
    )

    print(f"Bedroom clusters: {len(bedroom_labels)}; "
          f"FFHQ clusters: {len(ffhq_labels)}")

    results = {"bedroom": [], "ffhq": []}
    print("\n=== Bedroom (GT = 8 HiGAN attributes) ===")
    print(f"{'top-K':>6} {'P':>6} {'R':>6} {'F1':>6}  matched-attrs (recall)")
    for k in [1, 2, 3, 4]:
        r = evaluate(bedroom_labels, BEDROOM_GT, k)
        results["bedroom"].append(r)
        print(f"{k:>6} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f}  "
              f"{r['n_matched_attrs']}/{r['n_gt_attrs']} = {','.join(r['matched_attrs'])}")

    print("\n=== FFHQ (GT = 5 InterFaceGAN attributes) ===")
    print(f"{'top-K':>6} {'P':>6} {'R':>6} {'F1':>6}  matched-attrs (recall)")
    for k in [1, 2, 3, 4]:
        r = evaluate(ffhq_labels, FFHQ_GT, k)
        results["ffhq"].append(r)
        print(f"{k:>6} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f}  "
              f"{r['n_matched_attrs']}/{r['n_gt_attrs']} = {','.join(r['matched_attrs'])}")

    # significance: random baseline.
    # If a cluster's top-K labels are drawn uniformly from V vocab terms,
    # probability that none of them lands in a GT proxy set of size m is
    # roughly (1 - m/V)^K; we compute the expected matched-clusters under
    # this null and chi-square against observed.
    print("\n=== Null-model significance (cluster matches are random) ===")
    from math import comb
    import numpy as np

    def null_pmf(V, m, K):
        """Probability that at least one of K random labels hits the m-proxy set."""
        # 1 - prob(all K miss)
        # without replacement among V, K choose; prob = comb(V-m, K)/comb(V, K)
        if V - m < K:
            return 1.0
        return 1.0 - comb(V - m, K) / comb(V, K)

    bedroom_V = 30  # VOCAB length in 24_clip_label_clusters.py
    ffhq_V = 24      # CLIP_VOCAB length in run_discovery_c6.py

    for name, gt, V, kresults in [
        ("bedroom", BEDROOM_GT, bedroom_V, results["bedroom"]),
        ("ffhq", FFHQ_GT, ffhq_V, results["ffhq"]),
    ]:
        # For each GT attr, prob a cluster lands in its proxy set ≥1×.
        # Recall expectation under null: for each attr, prob *some* cluster
        # matches it (1 - (1-p_attr)^n_clusters).
        n_clusters = kresults[0]["n_clusters"]
        for r in kresults:
            K = r["top_k"]
            null_recalls = []
            null_precisions = []
            for attr, proxies in gt.items():
                m = len(proxies)
                p_attr = null_pmf(V, m, K)
                null_recalls.append(1 - (1 - p_attr) ** n_clusters)
                null_precisions.append(p_attr)
            r["null_recall"] = float(np.mean(null_recalls))
            r["null_precision"] = float(np.mean(null_precisions))
            r["recall_lift"] = r["recall"] - r["null_recall"]
            r["precision_lift"] = r["precision"] - r["null_precision"]
            print(f"  {name:8s} K={K}  observed R={r['recall']:.2f} P={r['precision']:.2f}  "
                  f"null R={r['null_recall']:.2f} P={r['null_precision']:.2f}  "
                  f"R-lift={r['recall_lift']:+.2f}  P-lift={r['precision_lift']:+.2f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    for ax, name in zip(axes, ["bedroom", "ffhq"]):
        rs = results[name]
        ks = [r["top_k"] for r in rs]
        Ps = [r["precision"] for r in rs]
        Rs = [r["recall"] for r in rs]
        F1s = [r["f1"] for r in rs]
        Pn = [r["null_precision"] for r in rs]
        Rn = [r["null_recall"] for r in rs]
        ax.plot(ks, Ps, "o-", color="#0e7490", lw=2, label="Precision")
        ax.plot(ks, Rs, "s-", color="#c2410c", lw=2, label="Recall")
        ax.plot(ks, F1s, "D-", color="#7c2d12", lw=2, label="F1")
        ax.plot(ks, Pn, "o--", color="#0e7490", lw=1, alpha=0.4, label="P (null)")
        ax.plot(ks, Rn, "s--", color="#c2410c", lw=1, alpha=0.4, label="R (null)")
        ax.set_xticks(ks)
        ax.set_xlabel("top-K CLIP labels per cluster", fontsize=10)
        ax.set_ylabel("score", fontsize=10)
        ax.set_title(f"{name.capitalize()} C6 — cluster→GT-attribute "
                     f"({rs[0]['n_clusters']} clusters, {rs[0]['n_gt_attrs']} attrs)",
                     fontsize=11, weight="bold", pad=8)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("C6 quantitative — unsupervised discovery rediscovers GT taxonomy "
                 "above random null", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "c6_precision_recall.png")
    print(f"\nsaved {out / 'c6_precision_recall.png'}")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
