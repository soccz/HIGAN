"""Track 22 — cross-domain curvature signature 2D clustering.

For every (domain, attribute) pair, plot (pixel ρ, CLIP-path ratio)
and check that "structural" attributes (bedroom view, FFHQ pose,
FFHQ eyeglasses) cluster together separate from texture attributes.

Pure post-processing of existing metrics.json files.

See designs/22_crossdomain_curvature_signature.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[2]


def main():
    out = PAPER / "experiments" / "out" / "crossdomain_signature"
    out.mkdir(parents=True, exist_ok=True)

    # bedroom: load existing higher-order + CLIP path
    bed_h = json.loads((PAPER / "experiments" / "out" / "bedroom_c2_path"
                        / "metrics.json").read_text())["per_attr"]
    # bedroom pixel ratios are hardcoded in scripts; use them
    BEDROOM_PIXEL = {
        "indoor_lighting": 0.495, "wood": 0.624, "carpet": 0.95,
        "cluttered_space": 0.85, "glossy": 0.92, "dirt": 0.93,
        "scary": 1.10, "view": 23.22,
    }

    # FFHQ
    ffhq_path = json.loads((PAPER / "experiments" / "out" / "ffhq_c2_path"
                            / "metrics.json").read_text())["per_attr"]
    FFHQ_PIXEL = {
        "smile": 1.75, "age": 7.62, "gender": 8.71,
        "eyeglasses": 22.82, "pose": 49.87,
    }

    # collect points
    points = []
    labels = []        # "structural" or "textural" (a priori)
    names = []
    for e in bed_h:
        attr = e["attr"]
        pixel = BEDROOM_PIXEL[attr]
        clip = e["mean_ratio"]
        struct = attr == "view"
        points.append([pixel, clip])
        labels.append("structural" if struct else "textural")
        names.append(f"bed-{attr}")
    for e in ffhq_path:
        attr = e["attr"]
        pixel = FFHQ_PIXEL[attr]
        clip = e["mean_ratio"]
        struct = attr in {"pose", "eyeglasses"}
        points.append([pixel, clip])
        labels.append("structural" if struct else "textural")
        names.append(f"ffhq-{attr}")

    # If SD data exists, add it too
    sd_path = PAPER / "experiments" / "out" / "sd_c1_c2" / "metrics.json"
    if sd_path.exists():
        sd = json.loads(sd_path.read_text())
        for e in sd.get("per_attr", []):
            attr = e["attr"]
            # average across timesteps
            t_data = e.get("per_t", {})
            if not t_data:
                continue
            rhos = [v["rho_mean"] for v in t_data.values()]
            clips = [v["clip_path_mean"] for v in t_data.values()]
            pixel = float(np.mean(rhos))
            clip = float(np.mean(clips))
            struct = attr in {"pose", "eyeglasses"}
            points.append([pixel, clip])
            labels.append("structural" if struct else "textural")
            names.append(f"sd-{attr}")

    points_np = np.array(points)
    print("=== cross-domain curvature signature ===")
    for i, (n, p, l) in enumerate(zip(names, points, labels)):
        print(f"  {n:25s} pixel ρ={p[0]:6.2f}  CLIP-path={p[1]:5.2f}  [{l}]")

    # k-means k=2 on log-space (since ρ spans orders of magnitude)
    from sklearn.cluster import KMeans
    X = np.column_stack([np.log10(points_np[:, 0].clip(1e-3)),
                          points_np[:, 1]])
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    cluster_labels = km.labels_

    # check agreement with a priori labels
    struct_idxs = [i for i, l in enumerate(labels) if l == "structural"]
    if struct_idxs:
        struct_cluster_freq = np.bincount(cluster_labels[struct_idxs])
        majority_struct = int(np.argmax(struct_cluster_freq))
        n_struct_majority = int(struct_cluster_freq.max())
        textural_idxs = [i for i, l in enumerate(labels)
                          if l == "textural"]
        n_textural_other = int(
            sum(1 for i in textural_idxs
                if cluster_labels[i] != majority_struct)
        )
        agreement = (n_struct_majority + n_textural_other) / len(labels)
        print(f"\nk=2 clustering agreement with a-priori labels: "
              f"{agreement:.2%}")

    # save + plot
    payload = {
        "points": points,
        "labels_apriori": labels,
        "names": names,
        "cluster_labels": cluster_labels.tolist(),
        "agreement_rate": float(agreement) if struct_idxs else None,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    colors = ["#dc2626" if l == "structural" else "#0e7490"
              for l in labels]
    for i, (n, p) in enumerate(zip(names, points)):
        ax.scatter(p[0], p[1], s=120, c=colors[i], alpha=0.85,
                   edgecolors="white", linewidths=1.6)
        ax.annotate(n, p, fontsize=8, alpha=0.85,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(r"pixel curvature ratio $\bar\rho$", fontsize=10)
    ax.set_ylabel("CLIP-feature path / direct ratio", fontsize=10)
    ax.set_title("Cross-domain curvature signature\n"
                  "red = structural, blue = textural (a-priori labels)",
                  fontsize=11, weight="bold", pad=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "crossdomain_signature.png")
    print(f"saved {out / 'crossdomain_signature.png'}")


if __name__ == "__main__":
    main()
