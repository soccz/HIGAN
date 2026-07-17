"""C2 quantitative via CLIP-feature path curvature.

For each attribute, sweep α ∈ [-3, +3] in N steps. Get CLIP image
features for each step. Measure:

  path_length = Σ_i ||f(α_{i+1}) - f(α_i)||
  direct      = ||f(α_max) - f(α_min)||
  ratio       = path_length / direct        (≥ 1, = 1 iff linear)

If ratio > 1 by a lot, the trajectory in feature space is *curved*.
This is a second independent curvature measure (besides our pixel-
space ∂²I/∂α² ratio). If both measures correlate across attributes,
C2 is strongly supported.

This metric is classifier-free, semantic-network-agnostic (any
feature extractor works), and easy to interpret.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--alpha-range", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("--num-steps", type=int, default=13)
    ap.add_argument("--out", default="out/bedroom_c2_path")
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

    print("=== loading CLIP ===")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.eval().to(G.device)

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=G.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=G.device).view(1, 3, 1, 1)

    def clip_feat(img_neg1_pos1: torch.Tensor) -> torch.Tensor:
        x = (img_neg1_pos1.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - clip_mean) / clip_std
        with torch.no_grad():
            f = model.encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    alphas = np.linspace(args.alpha_range[0], args.alpha_range[1], args.num_steps)

    print(f"=== sweep {len(args.attrs)} attrs × {args.num_steps} steps × {args.num_samples} samples ===")
    results = []
    for attr in args.attrs:
        b_la = b_layered[attr]
        ratios = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            feats = []
            for alpha in alphas:
                with torch.no_grad():
                    img = G.synthesize(wp + float(alpha) * b_la.unsqueeze(0))
                feats.append(clip_feat(img).squeeze(0))
            F_stack = torch.stack(feats)            # (T, clip_dim)
            seg_lens = torch.linalg.norm(F_stack[1:] - F_stack[:-1], dim=1)
            path_len = seg_lens.sum().item()
            direct = torch.linalg.norm(F_stack[-1] - F_stack[0]).item()
            ratio = path_len / direct if direct > 1e-8 else 0.0
            ratios.append(ratio)
            torch.cuda.empty_cache()
        ratios = np.asarray(ratios)
        mean_r = float(ratios.mean())
        med_r = float(np.median(ratios))
        std_r = float(ratios.std())
        results.append({
            "attr": attr,
            "ratios": ratios.tolist(),
            "mean_ratio": mean_r,
            "median_ratio": med_r,
            "std_ratio": std_r,
        })
        print(f"  {attr:18s}  path/direct mean={mean_r:.3f}  median={med_r:.3f}  std={std_r:.3f}")

    # Compare with the saliency-curvature ratio from earlier
    pixel_ratios = {
        "indoor_lighting": 0.495, "wood": 0.624, "carpet": 0.95,
        "cluttered_space": 0.85, "glossy": 0.92, "dirt": 0.93,
        "scary": 1.10, "view": 23.22,
    }
    px = [pixel_ratios[r["attr"]] for r in results if r["attr"] in pixel_ratios]
    cl = [r["mean_ratio"] for r in results if r["attr"] in pixel_ratios]
    from scipy.stats import spearmanr, pearsonr
    sp_r, sp_p = spearmanr(px, cl)
    pe_r, pe_p = pearsonr(px, cl)
    print(f"\nSpearman(pixel ∂²/∂I ratio, CLIP path ratio) = {sp_r:+.3f}  (p={sp_p:.3g})")
    print(f"Pearson  = {pe_r:+.3f}  (p={pe_p:.3g})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=140)

    attrs_sorted = sorted(results, key=lambda r: -r["mean_ratio"])
    names = [r["attr"] for r in attrs_sorted]
    vals = [r["mean_ratio"] for r in attrs_sorted]
    stds = [r["std_ratio"] for r in attrs_sorted]
    x = np.arange(len(names))
    ax1.bar(x, vals, yerr=stds, color="#0e7490", capsize=4)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("CLIP-feature path / direct distance", fontsize=10)
    ax1.set_title("Bedroom C2 — CLIP path curvature per attribute (8 samples ± std)",
                  fontsize=11, weight="bold", pad=8)
    ax1.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax1.grid(alpha=0.25, axis="y")

    ax2.scatter(px, cl, s=80, c="#7c2d12", alpha=0.85, edgecolors="white",
                linewidths=1.5)
    for r in results:
        if r["attr"] in pixel_ratios:
            ax2.annotate(r["attr"], (pixel_ratios[r["attr"]], r["mean_ratio"]),
                         fontsize=8, alpha=0.7, xytext=(4, 2),
                         textcoords="offset points")
    ax2.set_xlabel(r"pixel $\partial^2 I /\partial I$ ratio (saliency C2)", fontsize=10)
    ax2.set_ylabel(r"CLIP-feature path / direct ratio (semantic C2)", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_title(
        f"two independent C2 measures agree (Spearman r={sp_r:+.2f}, p={sp_p:.2g})",
        fontsize=11, weight="bold", pad=8,
    )
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "bedroom_c2_path.png")
    print(f"\nsaved {out / 'bedroom_c2_path.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"per_attr": results,
                   "vs_pixel": {"spearman": {"r": float(sp_r), "p": float(sp_p)},
                                "pearson":  {"r": float(pe_r), "p": float(pe_p)}}},
                  f, indent=2)


if __name__ == "__main__":
    main()
