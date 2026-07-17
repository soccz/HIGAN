"""C3 quantitative on FFHQ — replicates the bedroom C3 IoU on
InterFaceGAN's 5 boundaries.
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
sys.path.insert(0, str(PAPER / "experiments"))

from domains.ffhq.generator import FFHQGenerator               # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--out", default="out/ffhq_c3_iou")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"

    attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
    boundaries = {}
    for a in attrs:
        b_vec = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
        boundaries[a] = torch.from_numpy(b_vec).to(G.device)

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    print(f"computing saliency for {len(attrs)} attrs × {L} layers ...")
    sal_per_layer: dict[str, list[np.ndarray]] = {a: [] for a in attrs}
    for attr in attrs:
        b_dir = boundaries[attr]
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
        print(f"  {attr:12s} done")

    def topk_mask(sal, frac):
        thresh = np.quantile(sal, 1 - frac)
        return sal >= thresh

    iou_matrices = {}
    c3_scores = {}
    for attr in attrs:
        masks = [topk_mask(s, args.top_frac) for s in sal_per_layer[attr]]
        M = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                inter = float((masks[i] & masks[j]).sum())
                union = float((masks[i] | masks[j]).sum())
                M[i, j] = inter / union if union > 0 else 0.0
        iou_matrices[attr] = M

        canonical = set(LAYERS_FOR[attr])
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
        print(f"  {attr:12s} IoU_cc={iou_cc:.3f}  IoU_cn={iou_cn:.3f}  "
              f"IoU_nn={iou_nn:.3f}  C3={c3_score:+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(attrs)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=120)
    for k, attr in enumerate(attrs):
        ax = axes[k]
        M = iou_matrices[attr]
        canonical = LAYERS_FOR[attr]
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for li in canonical:
            ax.axvline(li, color="red", lw=0.3, alpha=0.6)
            ax.axhline(li, color="red", lw=0.3, alpha=0.6)
        ax.set_title(f"{attr}  (C3={c3_scores[attr]['c3_score']:+.2f})",
                     fontsize=10, weight="bold")
        ax.set_xticks(range(0, L, 3))
        ax.set_yticks(range(0, L, 3))
    fig.suptitle(
        "FFHQ C3 — pairwise saliency IoU across W+ layers\n"
        "(red lines = InterFaceGAN canonical manipulate-layers)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "ffhq_c3_iou.png")
    print(f"\nsaved {out / 'ffhq_c3_iou.png'}")

    mean_c3 = float(np.mean([s["c3_score"] for s in c3_scores.values()]))
    print(f"\nMean C3 score across 5 FFHQ attributes: {mean_c3:+.3f}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"c3_scores": c3_scores, "mean_c3": mean_c3,
                   "top_frac": args.top_frac,
                   "num_samples": args.num_samples}, f, indent=2)


if __name__ == "__main__":
    main()
