"""C4 on bedroom — 8 boundaries × C(8,2)=28 pairs for higher statistical power.

Reuses higan_dev/ pipeline. For each pair (a, b), compute compositional
non-linearity (1 - corr(sal(a+b), sal(a)+sal(b))) and the mixed-Hessian
predictor ||d²G(b_a, b_b)|| / (||dG b_a|| · ||dG b_b||).

Spearman rank correlation across the 28 pairs is our main test for C4.
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import torch
from torch.func import jvp
from PIL import Image
from scipy.stats import spearmanr, pearsonr

# higan_dev already has HiGANGenerator + load_boundary
HIGAN_DEV = Path(__file__).resolve().parents[4] / "higan_dev"
sys.path.insert(0, str(HIGAN_DEV))

from higan_dev.generator import HiGANGenerator               # noqa: E402
from higan_dev.manipulate import load_boundary               # noqa: E402
from higan_dev.cam.grad_saliency import _layered_direction   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", default="out/bedroom_c4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=str(HIGAN_DEV / "data" / "higan_repo"))
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    attrs = ["indoor_lighting", "wood", "view", "carpet",
             "cluttered_space", "glossy", "dirt", "scary"]
    boundaries_dir = HIGAN_DEV / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    boundaries = {a: load_boundary(str(boundaries_dir), a, num_layers=L).to(G.device)
                  for a in attrs}
    b_layered = {a: _layered_direction(boundaries[a], L, D, G.device)
                 for a in attrs}

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    # first-order pass
    first_maps: dict[str, np.ndarray] = {}
    first_norms: dict[str, float] = {}
    print("first-order pass...")
    for a in attrs:
        b_a = b_layered[a]
        acc = torch.zeros(H, W, device=G.device)
        norm_acc = 0.0
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_a.unsqueeze(0))
            _, d = jvp(f, (torch.zeros(1, device=G.device),),
                      (torch.ones(1, device=G.device),))
            acc += d.abs().mean(dim=1).squeeze(0)
            norm_acc += float(d.norm().item())
            torch.cuda.empty_cache()
        first_maps[a] = (acc / args.num_samples).cpu().numpy()
        first_norms[a] = norm_acc / args.num_samples
        print(f"  {a:18s} |dG b_a|~{first_norms[a]:.2f}")

    # pairwise
    rows = []
    n_pairs = len(list(combinations(attrs, 2)))
    print(f"\npairwise composition + mixed Hessian for {n_pairs} pairs...")
    for k, (a, b) in enumerate(combinations(attrs, 2)):
        b_a = b_layered[a]
        b_b = b_layered[b]
        b_sum = b_a + b_b

        acc_sum = torch.zeros(H, W, device=G.device)
        mixed_acc = 0.0
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f_sum(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_sum.unsqueeze(0))
            _, dsum = jvp(f_sum, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            acc_sum += dsum.abs().mean(dim=1).squeeze(0)

            def f_two(alpha, beta):
                return G.synthesize(
                    wp + alpha.view(1, 1, 1) * b_a.unsqueeze(0)
                       + beta.view(1, 1, 1) * b_b.unsqueeze(0))
            def inner(beta):
                def f_alpha(alpha):
                    return f_two(alpha, beta)
                _, d_alpha = jvp(f_alpha, (torch.zeros(1, device=G.device),),
                                 (torch.ones(1, device=G.device),))
                return d_alpha
            _, mixed = jvp(inner, (torch.zeros(1, device=G.device),),
                           (torch.ones(1, device=G.device),))
            mixed_acc += float(mixed.norm().item())
            torch.cuda.empty_cache()

        sal_ab = (acc_sum / args.num_samples).cpu().numpy()
        sal_a = first_maps[a]
        sal_b = first_maps[b]
        expected = sal_a + sal_b
        flat_obs = sal_ab.flatten()
        flat_exp = expected.flatten()
        corr_ab = float(np.corrcoef(flat_obs, flat_exp)[0, 1])
        nonlinearity = 1.0 - corr_ab

        mixed_mean = mixed_acc / args.num_samples
        denom = first_norms[a] * first_norms[b]
        predictor = mixed_mean / (denom + 1e-12)

        rows.append({"a": a, "b": b, "corr": corr_ab,
                     "nonlinearity": nonlinearity,
                     "mixed_norm": mixed_mean,
                     "predictor": predictor})
        print(f"  [{k+1:2d}/{n_pairs}] {a:18s} + {b:18s}  corr={corr_ab:+.3f}  "
              f"P={predictor:.5f}")

    P = np.array([r["predictor"] for r in rows])
    Y = np.array([r["nonlinearity"] for r in rows])
    sp_r, sp_p = spearmanr(P, Y)
    pe_r, pe_p = pearsonr(P, Y)
    print(f"\n=== Result ===")
    print(f"Spearman (predictor, 1-corr) = {sp_r:+.3f}  (p={sp_p:.3g})  n={len(P)}")
    print(f"Pearson  (predictor, 1-corr) = {pe_r:+.3f}  (p={pe_p:.3g})  n={len(P)}")

    # scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=140)
    ax.scatter(P, Y, s=80, c="#44403c", alpha=0.85, edgecolors="white",
               linewidths=1.5)
    for r in rows:
        ax.annotate(f"{r['a'][:4]}+{r['b'][:4]}",
                    (r["predictor"], r["nonlinearity"]),
                    fontsize=7, alpha=0.7, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel(r"Predictor  $\|d^2G(b_a, b_b)\| / (\|dG\, b_a\| \cdot \|dG\, b_b\|)$",
                  fontsize=10)
    ax.set_ylabel(r"Compositional non-linearity  $1 - \mathrm{corr}(\mathrm{sal}(a+b),\, \mathrm{sal}(a) + \mathrm{sal}(b))$",
                  fontsize=10)
    ax.set_title(
        f"Bedroom — C4 predictor vs observed "
        f"(Spearman {sp_r:+.3f}, p={sp_p:.2g}, n={len(P)} pairs)",
        fontsize=11, weight="bold", pad=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "c4_scatter.png")

    with open(out / "metrics.json", "w") as f:
        json.dump({"pairs": rows,
                   "spearman": {"r": float(sp_r), "p": float(sp_p)},
                   "pearson":  {"r": float(pe_r), "p": float(pe_p)},
                   "first_norms": first_norms,
                   "n_pairs": len(rows),
                   "n_samples_per_pair": args.num_samples}, f, indent=2)
    print(f"\nsaved {out / 'c4_scatter.png'}")


if __name__ == "__main__":
    main()
