"""C4 on church — only 3 attributes (3 pairs). Third domain data point.

Reuses ChurchGenerator. With n=3 pairs the Spearman r is essentially
binary (-1, 0, +1) — useful as a sign-confirmation, not as a power
statistic.
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

EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))

from domains.church.generator import ChurchGenerator               # noqa: E402

LAYERS_FOR = {
    "clouds":     list(range(0, 8)),
    "sunny":      list(range(0, 8)),
    "vegetation": list(range(6, 12)),
}
CHURCH_BOUNDARY_DIR = (
    Path(__file__).resolve().parents[4]
    / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan2_church"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", default="out/church_c4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = ChurchGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    attrs = ["clouds", "sunny", "vegetation"]
    b_layered = {}
    for a in attrs:
        v = np.load(CHURCH_BOUNDARY_DIR / f"{a}_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        d = torch.from_numpy(v).to(G.device)
        d = d / d.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=G.device)
        for li in LAYERS_FOR[a]:
            bl[li] = d
        b_layered[a] = bl

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    first_maps = {}
    first_norms = {}
    print("first-order pass...")
    for a in attrs:
        ba = b_layered[a]
        acc = torch.zeros(H, W, device=G.device)
        norm_acc = 0.0
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * ba.unsqueeze(0))
            _, d = jvp(f, (torch.zeros(1, device=G.device),),
                       (torch.ones(1, device=G.device),))
            acc += d.abs().mean(dim=1).squeeze(0)
            norm_acc += float(d.norm().item())
            torch.cuda.empty_cache()
        first_maps[a] = (acc / args.num_samples).cpu().numpy()
        first_norms[a] = norm_acc / args.num_samples
        print(f"  {a:12s}  |dG b|~{first_norms[a]:.2f}")

    rows = []
    print("\npairwise composition + mixed Hessian...")
    for k, (a, b) in enumerate(combinations(attrs, 2)):
        ba = b_layered[a]
        bb = b_layered[b]
        bsum = ba + bb

        acc_sum = torch.zeros(H, W, device=G.device)
        mixed_acc = 0.0
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f_sum(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * bsum.unsqueeze(0))
            _, dsum = jvp(f_sum, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            acc_sum += dsum.abs().mean(dim=1).squeeze(0)

            def f_two(alpha, beta):
                return G.synthesize(
                    wp + alpha.view(1, 1, 1) * ba.unsqueeze(0)
                       + beta.view(1, 1, 1) * bb.unsqueeze(0))
            def inner(beta):
                def f_a(alpha):
                    return f_two(alpha, beta)
                _, d_a = jvp(f_a, (torch.zeros(1, device=G.device),),
                             (torch.ones(1, device=G.device),))
                return d_a
            _, mixed = jvp(inner, (torch.zeros(1, device=G.device),),
                           (torch.ones(1, device=G.device),))
            mixed_acc += float(mixed.norm().item())
            torch.cuda.empty_cache()

        sal_ab = (acc_sum / args.num_samples).cpu().numpy()
        sa = first_maps[a]
        sb = first_maps[b]
        exp_ = sa + sb
        corr_ab = float(np.corrcoef(sal_ab.flatten(), exp_.flatten())[0, 1])
        nonlinearity = 1.0 - corr_ab
        mixed_mean = mixed_acc / args.num_samples
        predictor = mixed_mean / (first_norms[a] * first_norms[b] + 1e-12)

        rows.append({"a": a, "b": b, "corr": corr_ab,
                     "nonlinearity": nonlinearity,
                     "mixed_norm": mixed_mean,
                     "predictor": predictor})
        print(f"  [{k+1}/3] {a:12s} + {b:12s}  corr={corr_ab:+.3f}  P={predictor:.5f}  "
              f"nl={nonlinearity:+.4f}")

    P = np.array([r["predictor"] for r in rows])
    Y = np.array([r["nonlinearity"] for r in rows])
    sp_r, sp_p = spearmanr(P, Y) if len(P) > 1 else (float("nan"), float("nan"))
    pe_r, pe_p = pearsonr(P, Y) if len(P) > 1 else (float("nan"), float("nan"))
    print(f"\n=== Church (n=3 pairs) ===")
    print(f"Spearman = {sp_r:+.3f}  (p={sp_p:.3g})")
    print(f"Pearson  = {pe_r:+.3f}  (p={pe_p:.3g})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    ax.scatter(P, Y, s=110, c="#1f2937", alpha=0.85, edgecolors="white",
               linewidths=1.6)
    for r in rows:
        ax.annotate(f"{r['a'][:3]}+{r['b'][:3]}",
                    (r["predictor"], r["nonlinearity"]),
                    fontsize=10, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel(r"Predictor  $\|d^2G(b_a,b_b)\|/(\|dG b_a\|\,\|dG b_b\|)$",
                  fontsize=10)
    ax.set_ylabel(r"Compositional non-linearity  $1-\mathrm{corr}$", fontsize=10)
    ax.set_title(
        f"Church — C4 predictor vs observed  (Spearman {sp_r:+.2f}, n=3)",
        fontsize=11, weight="bold", pad=8,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "c4_scatter.png")
    print(f"saved {out / 'c4_scatter.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"pairs": rows,
                   "spearman": {"r": float(sp_r), "p": float(sp_p)},
                   "pearson":  {"r": float(pe_r), "p": float(pe_p)},
                   "first_norms": first_norms,
                   "n_pairs": len(rows),
                   "n_samples_per_pair": args.num_samples}, f, indent=2)


if __name__ == "__main__":
    main()
