"""C3 quantitative: layer-localisation IoU.

For each attribute a and each layer ℓ, compute saliency S_{a,ℓ} when
the boundary direction is placed on layer ℓ alone. C3 predicts that
S_{a,ℓ} has high IoU with S_{a,ℓ'} for ℓ, ℓ' in the canonical
manipulate-layers of a, and low IoU for non-canonical pairs.

Procedure:
  Take the saliency at each layer's top 20% pixels as a binary mask.
  Pairwise IoU within canonical and within non-canonical layers.

Outputs:
  - For each attribute: 14×14 IoU matrix (visualised as heatmap).
  - Scalar "C3 score" = mean_IoU(canonical, canonical) -
                         mean_IoU(canonical, non-canonical).
    Positive = boundary is layer-localised; near zero = no localisation.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                  # noqa: E402
from higan_dev.manipulate import load_boundary                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--top-frac", type=float, default=0.2,
                    help="top fraction of pixels considered 'salient' for IoU")
    ap.add_argument("--out", default="out/bedroom_c3_iou")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    attrs = ["indoor_lighting", "wood", "view", "carpet",
             "cluttered_space", "glossy", "dirt", "scary"]
    bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    boundaries = {a: load_boundary(str(bdir), a, num_layers=L) for a in attrs}

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    # For each (attr, layer), compute averaged saliency map
    print(f"computing saliency for {len(attrs)} attrs × {L} layers ...")
    sal_per_layer: dict[str, list[np.ndarray]] = {a: [] for a in attrs}
    for attr in attrs:
        b_dir = boundaries[attr].direction.to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        for li in range(L):
            b_layered = torch.zeros(L, D, device=G.device)
            b_layered[li] = b_dir
            acc = torch.zeros(H, W, device=G.device)
            for s in range(args.num_samples):
                wp = base_wp[s:s + 1].detach()
                def f(alpha):
                    return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
                _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                              (torch.ones(1, device=G.device),))
                acc += dimg.abs().mean(dim=1).squeeze(0)
                torch.cuda.empty_cache()
            sal_per_layer[attr].append((acc / args.num_samples).cpu().numpy())
        print(f"  {attr:18s} done")

    # Compute pairwise IoU at top-frac threshold for each attribute
    def topk_mask(sal: np.ndarray, frac: float) -> np.ndarray:
        thresh = np.quantile(sal, 1 - frac)
        return sal >= thresh

    iou_matrices: dict[str, np.ndarray] = {}
    c3_scores: dict[str, dict] = {}
    for attr in attrs:
        masks = [topk_mask(s, args.top_frac) for s in sal_per_layer[attr]]
        M = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                inter = float((masks[i] & masks[j]).sum())
                union = float((masks[i] | masks[j]).sum())
                M[i, j] = inter / union if union > 0 else 0.0
        iou_matrices[attr] = M

        canonical = set(boundaries[attr].manipulate_layers)
        non_canonical = set(range(L)) - canonical
        cc_pairs = [(i, j) for i in canonical for j in canonical if i < j]
        cn_pairs = [(i, j) for i in canonical for j in non_canonical]
        nn_pairs = [(i, j) for i in non_canonical for j in non_canonical if i < j]
        iou_cc = np.mean([M[i, j] for i, j in cc_pairs]) if cc_pairs else 0.0
        iou_cn = np.mean([M[i, j] for i, j in cn_pairs]) if cn_pairs else 0.0
        iou_nn = np.mean([M[i, j] for i, j in nn_pairs]) if nn_pairs else 0.0
        c3_score = float(iou_cc - iou_cn)
        c3_scores[attr] = {
            "iou_canonical_canonical": float(iou_cc),
            "iou_canonical_noncanonical": float(iou_cn),
            "iou_noncanonical_noncanonical": float(iou_nn),
            "c3_score": c3_score,
            "canonical_layers": sorted(canonical),
        }
        print(f"  {attr:18s} IoU_cc={iou_cc:.3f}  IoU_cn={iou_cn:.3f}  "
              f"IoU_nn={iou_nn:.3f}  C3={c3_score:+.3f}")

    # render all 14x14 matrices side by side
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(attrs)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=120)
    for k, attr in enumerate(attrs):
        ax = axes[k // 4, k % 4]
        M = iou_matrices[attr]
        canonical = boundaries[attr].manipulate_layers
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        # highlight canonical layer rows/cols
        for li in canonical:
            ax.axvline(li, color="red", lw=0.4, alpha=0.6)
            ax.axhline(li, color="red", lw=0.4, alpha=0.6)
        ax.set_title(f"{attr}  (C3={c3_scores[attr]['c3_score']:+.2f})",
                     fontsize=10, weight="bold")
        ax.set_xticks(range(0, L, 2))
        ax.set_yticks(range(0, L, 2))
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("layer", fontsize=8)
    fig.suptitle(
        "Bedroom C3 — pairwise saliency IoU across layers\n"
        "(red lines mark canonical manipulate-layers)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "bedroom_c3_iou.png")
    print(f"\nsaved {out / 'bedroom_c3_iou.png'}")

    # aggregate statistic
    mean_c3 = float(np.mean([s["c3_score"] for s in c3_scores.values()]))
    print(f"\nMean C3 score across 8 attributes: {mean_c3:+.3f}")
    print(f"  positive ⇒ boundaries are layer-localised (C3 holds)")

    with open(out / "metrics.json", "w") as f:
        json.dump({"c3_scores": c3_scores,
                   "mean_c3": mean_c3,
                   "top_frac": args.top_frac,
                   "num_samples": args.num_samples}, f, indent=2)


if __name__ == "__main__":
    main()
