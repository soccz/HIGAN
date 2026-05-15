"""Spatial diversity of discovered directions — generator-grounded metric.

For each baseline method (GANSpace, SeFa, random+CLIP) at K=8, compute
the K saliency maps via our JVP framework, then for each method:

  spatial_diversity = 1 - mean_pairwise_IoU(top-20% masks)

Higher = more orthogonal spatial coverage of the image.

A complementary number to the semantic-label diversity already in
the sweep. Only run on bedroom (StyleGAN1 LSUN); FFHQ at 1024² would
add hours without changing the conceptual story.
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                       # noqa: E402
from higan_dev.manipulate import load_boundary                       # noqa: E402
from baselines.ganspace import ganspace_directions                  # noqa: E402
from baselines.sefa import sefa_directions                          # noqa: E402

# Layer set used for applying each direction (mid-layer texture range,
# matching the sweep)
APPLY_LAYERS = list(range(6, 12))


def saliency_for_direction(G, v, base_wp, n_samples):
    """Mean |dG·v| saliency at top-frac threshold-compatible scale."""
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    v = v / v.norm().clamp_min(1e-8)
    bl = torch.zeros(L, D, device=G.device)
    for li in APPLY_LAYERS:
        bl[li] = v
    acc = torch.zeros(H, W, device=G.device)
    for s in range(n_samples):
        wp = base_wp[s:s + 1].detach()
        def f(alpha):
            return G.synthesize(wp + alpha.view(1, 1, 1) * bl.unsqueeze(0))
        _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                      (torch.ones(1, device=G.device),))
        acc += dimg.abs().mean(dim=1).squeeze(0)
        torch.cuda.empty_cache()
    return (acc / n_samples).cpu().numpy()


def spatial_diversity(sal_maps, top_frac=0.2):
    """1 - mean pairwise IoU over top-frac binary masks."""
    masks = [s >= np.quantile(s, 1 - top_frac) for s in sal_maps]
    ious = []
    for i, j in combinations(range(len(masks)), 2):
        inter = float((masks[i] & masks[j]).sum())
        union = float((masks[i] | masks[j]).sum())
        ious.append(inter / union if union > 0 else 0.0)
    return 1.0 - float(np.mean(ious)), float(np.mean(ious)), ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--out", default="out/bedroom_spatial_diversity")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading HiGAN bedroom ===")
    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    methods: dict[str, np.ndarray] = {}

    print(f"\n=== GANSpace-W (K={args.K}) ===")
    gs = ganspace_directions(G, n_samples=5000, n_components=args.K, seed=0)
    methods["ganspace"] = gs.components

    print(f"\n=== SeFa (K={args.K}) ===")
    try:
        sefa = sefa_directions(G, n_components=args.K)
        methods["sefa"] = sefa.components
    except Exception as e:
        print(f"  SeFa failed: {e}")
        methods["sefa"] = None

    print(f"\n=== Random ({args.K} unit directions in W) ===")
    rng_r = torch.Generator(device=G.device).manual_seed(13)
    rand_dirs = torch.randn(args.K, G.w_dim, generator=rng_r,
                             device=G.device).cpu().numpy()
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True).clip(min=1e-8)
    methods["random"] = rand_dirs

    # Also: the ground-truth HiGAN boundaries themselves — what diversity
    # do the human-curated boundaries show?
    print(f"\n=== HiGAN ground-truth boundaries ({args.K} of 8) ===")
    bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    gt_attrs = ["indoor_lighting", "wood", "view", "carpet",
                "cluttered_space", "glossy", "dirt", "scary"][:args.K]
    gt_dirs = []
    for a in gt_attrs:
        b = load_boundary(str(bdir), a, num_layers=G.num_layers)
        gt_dirs.append(b.direction.cpu().numpy().astype(np.float32))
    methods["higan_gt"] = np.stack(gt_dirs)

    summary = {}
    for name, dirs in methods.items():
        if dirs is None:
            continue
        print(f"\n--- saliency for {name} ({dirs.shape[0]} directions) ---")
        sal_maps = []
        for k in range(dirs.shape[0]):
            v = torch.from_numpy(dirs[k]).to(G.device).float()
            sal = saliency_for_direction(G, v, base_wp, args.num_samples)
            sal_maps.append(sal)
            print(f"  dir {k+1}/{dirs.shape[0]}")
        div, mean_iou, ious = spatial_diversity(sal_maps)
        print(f"  spatial_diversity = {div:.3f}  (mean pairwise IoU = {mean_iou:.3f})")
        summary[name] = {"spatial_diversity": div,
                          "mean_pairwise_iou": mean_iou,
                          "all_ious": ious,
                          "n_directions": int(dirs.shape[0])}

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summary.keys())
    pretty = {"ganspace": "GANSpace-W", "sefa": "SeFa",
              "random": "Random unit", "higan_gt": "HiGAN GT (human)"}
    divs = [summary[n]["spatial_diversity"] for n in names]
    mious = [summary[n]["mean_pairwise_iou"] for n in names]
    colors = {"ganspace": "#0e7490", "sefa": "#6d28d9",
              "random": "#16a34a", "higan_gt": "#dc2626"}

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    x = np.arange(len(names))
    ax.bar(x - 0.18, divs, width=0.36,
           color=[colors[n] for n in names], alpha=0.92,
           label="spatial diversity (1 - mean IoU)")
    ax.bar(x + 0.18, mious, width=0.36,
           color=[colors[n] for n in names], alpha=0.45,
           label="mean pairwise IoU (overlap)")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[n] for n in names], fontsize=10)
    ax.set_ylabel("score", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Bedroom — spatial diversity of K={args.K} directions "
                 f"(top-20% saliency masks, n_samples={args.num_samples})",
                 fontsize=11, weight="bold", pad=8)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "spatial_diversity.png")
    print(f"\nsaved {out / 'spatial_diversity.png'}")

    print("\n=== summary ===")
    for n in names:
        print(f"  {pretty[n]:20s}  spatial_div={summary[n]['spatial_diversity']:.3f}  "
              f"(mean pairwise IoU {summary[n]['mean_pairwise_iou']:.3f})")

    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
