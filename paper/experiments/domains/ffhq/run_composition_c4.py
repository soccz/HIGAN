"""C4 (FFHQ): mixed-Hessian predictor of compositional editing failure.

For every attribute pair (a, b) we measure:
  (i)  compositional non-linearity = 1 - corr( sal(a+b), sal(a) + sal(b) )
       — the empirical "how much does superposition fail"
  (ii) predictor P(a, b) = ||d²G(b_a, b_b)|| / (||dG b_a|| * ||dG b_b||)
       — the mixed-Hessian-based predictor from theory ch. 04

Hypothesis: rank-correlation Spearman(P, 1 - corr) > 0.6.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from itertools import combinations

import numpy as np
import torch
from torch.func import jvp
from PIL import Image
from scipy.stats import spearmanr, pearsonr

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from domains.ffhq.generator import FFHQGenerator     # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}


def load_dir(name: str, G: FFHQGenerator, layered: bool = True) -> torch.Tensor:
    bpath = (EXPERIMENTS_DIR / "data" / "interfacegan" / "boundaries"
             / f"stylegan_ffhq_{name}_w_boundary.npy")
    b_vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
    b_dir = torch.from_numpy(b_vec).to(G.device)
    b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
    if not layered:
        return b_dir
    canonical = LAYERS_FOR.get(name, list(range(G.num_layers)))
    out = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for li in canonical:
        out[li] = b_dir
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out", default="out/ffhq_c4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    boundaries = {a: load_dir(a, G) for a in args.attrs}

    # First-order saliency norms per attribute (denominator of predictor)
    first_norms: dict[str, float] = {}
    first_maps: dict[str, np.ndarray] = {}
    print("first-order pass...")
    for a in args.attrs:
        b_a = boundaries[a]
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
        print(f"  {a}: |dG b_a| ~ {first_norms[a]:.2f}")

    # Pair-wise: compositional saliency corr + mixed Hessian
    rows = []
    print("\npairwise (composition + mixed Hessian)...")
    for (a, b) in combinations(args.attrs, 2):
        b_a = boundaries[a]
        b_b = boundaries[b]
        b_sum = b_a + b_b

        # sal(a+b) (combined direction sum, JVP at 0)
        acc_sum = torch.zeros(H, W, device=G.device)
        # mixed Hessian
        mixed_acc = 0.0
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()

            # sal(a+b): apply combined direction
            def f_sum(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_sum.unsqueeze(0))
            _, dsum = jvp(f_sum, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            acc_sum += dsum.abs().mean(dim=1).squeeze(0)

            # mixed Hessian: d²G(b_a, b_b)
            # f(α, β) = G(wp + α b_a + β b_b)
            # inner(β) = d/dα f(α, β) at α=0  -> use jvp on f w.r.t. α
            # outer = d/dβ inner(β) at β=0  -> jvp on inner w.r.t. β
            def f_two_params(alpha, beta):
                return G.synthesize(
                    wp + alpha.view(1, 1, 1) * b_a.unsqueeze(0)
                       + beta.view(1, 1, 1) * b_b.unsqueeze(0))
            def inner(beta):
                def f_alpha(alpha):
                    return f_two_params(alpha, beta)
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
        # pixel-wise Pearson between observed (sal_ab) and expected
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
        print(f"  {a:12s} + {b:12s}  corr={corr_ab:+.3f}  "
              f"P={predictor:.3f}  mixed|d²|={mixed_mean:.2f}")

    # Rank correlation
    P = np.array([r["predictor"] for r in rows])
    Y = np.array([r["nonlinearity"] for r in rows])
    sp_r, sp_p = spearmanr(P, Y)
    pe_r, pe_p = pearsonr(P, Y)
    print(f"\nSpearman corr(predictor, 1-corr_ab) = {sp_r:+.3f}  (p={sp_p:.3g})")
    print(f"Pearson  corr(predictor, 1-corr_ab) = {pe_r:+.3f}  (p={pe_p:.3g})")

    # scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.scatter(P, Y, s=80, c="#6d28d9", alpha=0.85, edgecolors="white",
               linewidths=1.5)
    for r in rows:
        ax.annotate(f"{r['a'][:3]}+{r['b'][:3]}",
                    (r["predictor"], r["nonlinearity"]),
                    fontsize=8, alpha=0.7, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel(r"Predictor  $\|d^2G(b_a, b_b)\| / (\|dG\, b_a\| \cdot \|dG\, b_b\|)$",
                  fontsize=10)
    ax.set_ylabel(r"Compositional non-linearity  $1 - \mathrm{corr}(\mathrm{sal}(a+b),\, \mathrm{sal}(a) + \mathrm{sal}(b))$",
                  fontsize=10)
    ax.set_title(
        f"FFHQ — C4 predictor vs observed (Spearman {sp_r:+.3f}, p={sp_p:.2g})",
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
                   "first_norms": first_norms}, f, indent=2)
    print(f"\nsaved {out / 'c4_scatter.png'}")


if __name__ == "__main__":
    main()
