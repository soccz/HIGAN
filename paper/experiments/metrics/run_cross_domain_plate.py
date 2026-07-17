"""Cross-domain headline plate for the paper.

Aggregates existing per-domain results into 3 panels:
  (a) C2 non-linearity ratio per attribute, grouped by domain.
  (b) C3 layer-IoU score per attribute (bedroom only — others pending).
  (c) C4 mixed-Hessian Spearman correlation per domain.

Inputs are the metrics.json files produced by earlier experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]


def load_json(path: Path):
    return json.loads(path.read_text())


def latest_pixel_rhos(path: Path) -> dict[str, float]:
    metrics = load_json(path)
    table = metrics["bootstrap_ci"]
    latest_n = max(int(k) for k in table)
    return {attr: float(row["mean"]) for attr, row in table[str(latest_n)].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/cross_domain_plate")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PAPER / "experiments" / out
    out.mkdir(parents=True, exist_ok=True)
    base = PAPER / "experiments" / "out"

    # --- pull metrics ---
    bedroom_c2 = latest_pixel_rhos(base / "sample_scaling_bedroom_n256" / "metrics.json")
    ffhq_c2 = latest_pixel_rhos(base / "sample_scaling_ffhq_n512" / "metrics.json")
    church_c2 = latest_pixel_rhos(base / "sample_scaling_church_n256" / "metrics.json")
    bedroom_c3 = load_json(base / "bedroom_c3_iou" / "metrics.json")["c3_scores"]
    bedroom_c4 = load_json(base / "bedroom_c4" / "metrics.json")
    ffhq_c4 = load_json(base / "ffhq_c4" / "metrics.json")
    church_c4_path = base / "church_c4" / "metrics.json"
    church_c4 = load_json(church_c4_path) if church_c4_path.exists() else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 11), dpi=130)
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)

    # ---- (a) C2 ratios per domain ----
    ax = fig.add_subplot(gs[0, :])
    color_dom = {"bedroom": "#44403c", "ffhq": "#6d28d9", "church": "#0e7490"}
    all_attrs_bd = list(bedroom_c2.keys())
    all_attrs_ffhq = list(ffhq_c2.keys())
    all_attrs_ch = list(church_c2.keys())
    y_bd = [bedroom_c2[a] for a in all_attrs_bd]
    y_ffhq = [ffhq_c2[a] for a in all_attrs_ffhq]
    y_ch = [church_c2[a] for a in all_attrs_ch]

    # plot grouped bar-like scatter
    xs_bd = np.arange(len(all_attrs_bd))
    xs_ffhq = np.arange(len(all_attrs_bd), len(all_attrs_bd) + len(all_attrs_ffhq))
    xs_ch = np.arange(len(all_attrs_bd) + len(all_attrs_ffhq),
                       len(all_attrs_bd) + len(all_attrs_ffhq) + len(all_attrs_ch))
    ax.bar(xs_bd, y_bd, color=color_dom["bedroom"], label="bedroom (StyleGAN1)")
    ax.bar(xs_ffhq, y_ffhq, color=color_dom["ffhq"], label="FFHQ (StyleGAN1)")
    ax.bar(xs_ch, y_ch, color=color_dom["church"], label="church (StyleGAN2)")

    all_x = list(xs_bd) + list(xs_ffhq) + list(xs_ch)
    all_labels = all_attrs_bd + all_attrs_ffhq + all_attrs_ch
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel(r"non-linearity ratio  $\bar{\rho}(b)$", fontsize=11)
    ax.set_title(
        "C2 — second-order saliency ratio per attribute, across 3 domains",
        fontsize=11, weight="bold", pad=8,
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    # ---- (b) C3 per attribute, bedroom ----
    ax = fig.add_subplot(gs[1, 0])
    attrs = list(bedroom_c3.keys())
    cc = [bedroom_c3[a]["iou_canonical_canonical"] for a in attrs]
    cn = [bedroom_c3[a]["iou_canonical_noncanonical"] for a in attrs]
    nn = [bedroom_c3[a]["iou_noncanonical_noncanonical"] for a in attrs]
    x = np.arange(len(attrs))
    width = 0.27
    ax.bar(x - width, cc, width, label="cc (canon ↔ canon)", color="#15803d")
    ax.bar(x, cn, width, label="cn (canon ↔ non-canon)", color="#b45309")
    ax.bar(x + width, nn, width, label="nn (non ↔ non)", color="#a8a29e")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("pairwise IoU at top-20%", fontsize=10)
    ax.set_title(
        "C3 — pairwise saliency IoU per layer-pair class (bedroom)\n"
        "computed from the bedroom layer-IoU metrics file",
        fontsize=10, weight="bold", pad=8,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    # ---- (b') C3 score ----
    ax = fig.add_subplot(gs[1, 1])
    c3_vals = [bedroom_c3[a]["c3_score"] for a in attrs]
    ax.bar(x, c3_vals, color="#7c2d12")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("C3 score = IoU_cc − IoU_cn", fontsize=10)
    ax.set_title(
        "C3 — layer-localisation score (bedroom)\n"
        "score is IoU_cc minus IoU_cn",
        fontsize=10, weight="bold", pad=8,
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.25, axis="y")

    # ---- (c) C4 scatter, both bedroom + FFHQ ----
    ax = fig.add_subplot(gs[2, :])
    bd_P = [p["predictor"] for p in bedroom_c4["pairs"]]
    bd_Y = [p["nonlinearity"] for p in bedroom_c4["pairs"]]
    ff_P = [p["predictor"] for p in ffhq_c4["pairs"]]
    ff_Y = [p["nonlinearity"] for p in ffhq_c4["pairs"]]
    # normalise each per-domain so they share x-axis range (just rescale)
    bd_Pn = np.array(bd_P) / max(bd_P)
    ff_Pn = np.array(ff_P) / max(ff_P)
    ax.scatter(bd_Pn, bd_Y, s=70, c=color_dom["bedroom"], alpha=0.85,
               edgecolors="white", linewidths=1.5,
               label=f"bedroom (n={len(bd_P)}, Spearman r={bedroom_c4['spearman']['r']:.2f}, "
                     f"p={bedroom_c4['spearman']['p']:.3f})")
    ax.scatter(ff_Pn, ff_Y, s=70, c=color_dom["ffhq"], alpha=0.85,
               edgecolors="white", linewidths=1.5,
               label=f"FFHQ (n={len(ff_P)}, Spearman r={ffhq_c4['spearman']['r']:.2f}, "
                     f"p={ffhq_c4['spearman']['p']:.3f})")
    if church_c4 is not None:
        ch_P = [p["predictor"] for p in church_c4["pairs"]]
        ch_Y = [p["nonlinearity"] for p in church_c4["pairs"]]
        ch_Pn = np.array(ch_P) / max(ch_P)
        ax.scatter(ch_Pn, ch_Y, s=70, c=color_dom["church"], alpha=0.85,
                   edgecolors="white", linewidths=1.5,
                   label=f"church (n={len(ch_P)}, Spearman r={church_c4['spearman']['r']:.2f}, "
                         f"p={church_c4['spearman']['p']:.3f})")
    ax.set_xlabel(r"normalised mixed-Hessian predictor  $\hat P(a,b) = P / \max P$ per domain",
                  fontsize=11)
    ax.set_ylabel(r"compositional non-linearity  $1 - \mathrm{corr}(\mathrm{sal}(a+b),\, \mathrm{sal}(a)+\mathrm{sal}(b))$",
                  fontsize=10)
    ax.set_title(
        "C4 — mixed-Hessian predictor of compositional failure",
        fontsize=11, weight="bold", pad=8,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Cross-domain curvature and composition evidence",
        fontsize=13, weight="bold", y=1.005,
    )
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    import matplotlib.pyplot as plt
    plt.close(fig)
    Image.fromarray(arr).save(out / "cross_domain_plate.png")
    print(f"saved {out / 'cross_domain_plate.png'}  ({arr.shape[1]} x {arr.shape[0]})")


if __name__ == "__main__":
    main()
