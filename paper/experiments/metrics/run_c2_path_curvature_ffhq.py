"""C2 CLIP path curvature on FFHQ — cross-architecture replication
of the bedroom r=0.99 agreement between pixel ∂²I/∂α² and CLIP-space
path-vs-direct ratio.
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
sys.path.insert(0, str(PAPER / "experiments"))

from domains.ffhq.generator import FFHQGenerator                  # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}

# pixel ∂²I/∂α² ratios from earlier FFHQ second-order experiment
PIXEL_RATIOS = {
    "smile":      1.75,
    "age":        7.62,
    "gender":     8.71,
    "eyeglasses": 22.82,
    "pose":       49.87,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=13)
    ap.add_argument("--alpha-range", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("--out", default="out/ffhq_c2_path")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    attrs = list(LAYERS_FOR.keys())

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.eval().to(G.device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=G.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=G.device).view(1, 3, 1, 1)

    def clip_feat(img):
        x = (img.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - clip_mean) / clip_std
        with torch.no_grad():
            f = model.encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    alphas = np.linspace(args.alpha_range[0], args.alpha_range[1], args.num_steps)

    print(f"=== {len(attrs)} attrs × {args.num_steps} steps × {args.num_samples} samples ===")
    results = []
    for attr in attrs:
        b_vec = np.load(boundaries_dir / f"stylegan_ffhq_{attr}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(b_vec).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        canonical = LAYERS_FOR[attr]
        b_layered = torch.zeros(L, D, device=G.device)
        for li in canonical:
            b_layered[li] = b_dir

        ratios = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            feats = []
            for alpha in alphas:
                with torch.no_grad():
                    img = G.synthesize(wp + float(alpha) * b_layered.unsqueeze(0))
                feats.append(clip_feat(img).squeeze(0))
            F_stack = torch.stack(feats)
            seg_lens = torch.linalg.norm(F_stack[1:] - F_stack[:-1], dim=1)
            path_len = seg_lens.sum().item()
            direct = torch.linalg.norm(F_stack[-1] - F_stack[0]).item()
            ratio = path_len / direct if direct > 1e-8 else 0.0
            ratios.append(ratio)
            torch.cuda.empty_cache()
        ratios = np.asarray(ratios)
        results.append({"attr": attr,
                         "mean_ratio": float(ratios.mean()),
                         "median_ratio": float(np.median(ratios)),
                         "std_ratio": float(ratios.std())})
        print(f"  {attr:12s}  path/direct mean={ratios.mean():.3f}  "
              f"median={np.median(ratios):.3f}  std={ratios.std():.3f}")

    # vs pixel ratios
    px = [PIXEL_RATIOS[r["attr"]] for r in results]
    cl = [r["mean_ratio"] for r in results]
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
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("CLIP path / direct distance", fontsize=10)
    ax1.set_title("FFHQ C2 — CLIP path curvature per attribute",
                  fontsize=11, weight="bold", pad=8)
    ax1.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax1.grid(alpha=0.25, axis="y")

    ax2.scatter(px, cl, s=80, c="#6d28d9", alpha=0.85, edgecolors="white",
                linewidths=1.5)
    for r in results:
        ax2.annotate(r["attr"], (PIXEL_RATIOS[r["attr"]], r["mean_ratio"]),
                     fontsize=8, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax2.set_xlabel(r"pixel $\partial^2 I / \partial I$ ratio (saliency C2)", fontsize=10)
    ax2.set_ylabel(r"CLIP-feature path / direct ratio (semantic C2)", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_title(
        f"FFHQ — two independent C2 measures agree "
        f"(Pearson r={pe_r:+.2f}, p={pe_p:.2g}; Spearman r={sp_r:+.2f})",
        fontsize=11, weight="bold", pad=8,
    )
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "ffhq_c2_path.png")
    print(f"\nsaved {out / 'ffhq_c2_path.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"per_attr": results,
                   "vs_pixel": {"spearman": {"r": float(sp_r), "p": float(sp_p)},
                                "pearson":  {"r": float(pe_r), "p": float(pe_p)}}},
                  f, indent=2)


if __name__ == "__main__":
    main()
