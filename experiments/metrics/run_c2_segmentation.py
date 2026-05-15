"""C2 quantitative — segmentation-label-count variability along boundary sweep.

For each attribute, render a sweep at α ∈ [-3, +3]. Apply pretrained
DeepLabV3-ResNet50 (COCO 21-class) to each frame, count distinct labels.
High curvature attributes (view) should show more variability in the
label count along the sweep than low-curvature ones (texture).

Uses torchvision's pretrained DeepLabV3 to avoid the transformers
torch-version conflict.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                  # noqa: E402
from higan_dev.manipulate import load_boundary                  # noqa: E402
from higan_dev.cam.grad_saliency import _layered_direction      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["view", "indoor_lighting", "wood", "glossy",
                             "carpet", "dirt", "scary", "cluttered_space"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--alpha-range", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("--num-steps", type=int, default=7)
    ap.add_argument("--out", default="out/bedroom_c2_seg")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading generator ===")
    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
    L, D = G.num_layers, G.w_dim
    bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    boundaries = {a: load_boundary(str(bdir), a, num_layers=L).to(G.device)
                  for a in args.attrs}
    b_layered = {a: _layered_direction(boundaries[a], L, D, G.device)
                 for a in args.attrs}

    print("=== loading DeepLabV3 (torchvision) ===")
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    seg_model = deeplabv3_resnet50(weights=weights).eval().to(G.device)
    seg_transforms = weights.transforms()
    n_classes = len(weights.meta["categories"])
    print(f"  {n_classes} class segmentation")

    def segment_count(image_neg1_pos1: torch.Tensor) -> int:
        """image: (1, 3, H, W) in [-1, 1]; returns count of distinct labels."""
        x = (image_neg1_pos1.clamp(-1, 1) + 1) / 2.0
        x = seg_transforms(x)
        with torch.no_grad():
            logits = seg_model(x)["out"]
            pred = logits.argmax(dim=1)
        return int(pred.unique().numel())

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    alphas = np.linspace(args.alpha_range[0], args.alpha_range[1], args.num_steps)

    print(f"=== sweep over {len(args.attrs)} attrs × {args.num_steps} steps × {args.num_samples} samples ===")
    results = []
    for attr in args.attrs:
        b_la = b_layered[attr]
        counts_per_sample = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            counts = []
            for alpha in alphas:
                with torch.no_grad():
                    img = G.synthesize(wp + float(alpha) * b_la.unsqueeze(0))
                counts.append(segment_count(img))
            counts_per_sample.append(np.asarray(counts))
            torch.cuda.empty_cache()
        counts_arr = np.stack(counts_per_sample)
        per_sample_range = counts_arr.max(axis=1) - counts_arr.min(axis=1)
        per_sample_std = counts_arr.std(axis=1)
        mean_range = float(per_sample_range.mean())
        mean_std = float(per_sample_std.mean())
        mean_count = float(counts_arr.mean())
        # fraction of samples with non-trivial variability
        frac_with_change = float((per_sample_range > 0).mean())
        results.append({
            "attr": attr,
            "alphas": alphas.tolist(),
            "counts": counts_arr.tolist(),
            "mean_range": mean_range,
            "mean_std": mean_std,
            "mean_count": mean_count,
            "frac_samples_with_change": frac_with_change,
        })
        print(f"  {attr:18s}  avg_count={mean_count:.2f}  "
              f"avg_range={mean_range:.2f}  avg_std={mean_std:.2f}  "
              f"frac>0={frac_with_change:.2f}")

    # Compare to non-linearity ratios (saliency C2 from §19)
    ratios_known = {
        "indoor_lighting": 0.495, "wood": 0.624, "carpet": 0.95,
        "cluttered_space": 0.85, "glossy": 0.92, "dirt": 0.93,
        "scary": 1.10, "view": 23.22,
    }
    rs = [ratios_known[r["attr"]] for r in results if r["attr"] in ratios_known]
    cs = [r["mean_range"] for r in results if r["attr"] in ratios_known]
    if len(rs) > 2:
        from scipy.stats import spearmanr
        sp_r, sp_p = spearmanr(rs, cs)
        print(f"\nSpearman(non-linearity ratio, segmentation-range) = {sp_r:+.3f}  (p={sp_p:.3g})")
    else:
        sp_r, sp_p = None, None

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=140)

    attrs_sorted = sorted(results, key=lambda r: -r["mean_range"])
    names = [r["attr"] for r in attrs_sorted]
    ranges = [r["mean_range"] for r in attrs_sorted]
    x = np.arange(len(names))
    ax1.bar(x, ranges, color="#0e7490")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("mean(max−min) of seg-label count along α-sweep", fontsize=10)
    ax1.set_title("Bedroom C2 — segmentation-label variability per attribute",
                  fontsize=11, weight="bold", pad=8)
    ax1.grid(alpha=0.25, axis="y")

    if rs and cs:
        ax2.scatter(rs, cs, s=80, c="#7c2d12", alpha=0.85, edgecolors="white",
                    linewidths=1.5)
        for r in results:
            if r["attr"] in ratios_known:
                ax2.annotate(r["attr"], (ratios_known[r["attr"]], r["mean_range"]),
                             fontsize=8, alpha=0.7, xytext=(4, 2),
                             textcoords="offset points")
        ax2.set_xlabel(r"non-linearity ratio $\bar\rho$ (saliency C2)", fontsize=10)
        ax2.set_ylabel("seg-label range along α-sweep", fontsize=10)
        title = "saliency curvature vs segmentation variability"
        if sp_r is not None:
            title += f"\n(Spearman r={sp_r:+.2f}, p={sp_p:.2g})"
        ax2.set_title(title, fontsize=11, weight="bold", pad=8)
        ax2.set_xscale("log")
        ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "bedroom_c2_seg.png")
    print(f"\nsaved {out / 'bedroom_c2_seg.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"per_attr": results,
                   "spearman_saliency_vs_seg": {"r": float(sp_r) if sp_r else None,
                                                 "p": float(sp_p) if sp_p else None}},
                  f, indent=2)


if __name__ == "__main__":
    main()
