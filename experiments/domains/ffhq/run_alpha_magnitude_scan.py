"""Track 20 — FFHQ boundary-magnitude scan: ρ(α) for α ∈ {0.25..3}.

Measures how the curvature ratio scales with the perturbation
magnitude. Hypothesis: structural attributes show ρ ∝ α^β with
β ≥ 0.5; textural attributes have β < 0.3.

See designs/20_boundary_magnitude_scan.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

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


def per_sample_ratio_at_alpha(G, wp_idx, base_wp, b_layered, alpha):
    """ρ measured around wp + α*b (not at the origin)."""
    wp_shift = base_wp[wp_idx:wp_idx + 1].detach() + \
               alpha * b_layered.unsqueeze(0)

    def f(beta):
        return G.synthesize(wp_shift + beta.view(1, 1, 1) * b_layered.unsqueeze(0))

    def df(beta):
        _, d = jvp(f, (beta,), (torch.ones_like(beta),))
        return d

    b0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)
    _, first = jvp(f, (b0,), (one,))
    _, second = jvp(df, (b0,), (one,))
    return second.abs().mean().item() / max(first.abs().mean().item(), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.25, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default="experiments/out/ffhq_alpha_scan")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
    boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                      / "boundaries")
    bl_dict = {}
    for a in attrs:
        v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        d = torch.from_numpy(v).to(G.device)
        d = d / d.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=G.device)
        for li in LAYERS_FOR[a]:
            bl[li] = d
        bl_dict[a] = bl

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_samples, generator=rng)

    print(f"[{time.strftime('%H:%M:%S')}] FFHQ α magnitude scan: "
          f"{len(attrs)} attrs × {len(args.alphas)} α × "
          f"{args.n_samples} seeds")

    results = {}
    for attr in attrs:
        per_alpha = {}
        for alpha in args.alphas:
            ratios = []
            for i in range(args.n_samples):
                r = per_sample_ratio_at_alpha(G, i, base_wp, bl_dict[attr],
                                              alpha)
                ratios.append(r)
                torch.cuda.empty_cache()
            per_alpha[str(alpha)] = {"mean": float(np.mean(ratios)),
                                      "std": float(np.std(ratios))}
            print(f"  {attr:12s} α={alpha}  mean={np.mean(ratios):.3f}")

        # log-log slope
        log_a = np.log(args.alphas)
        log_r = np.log([per_alpha[str(a)]["mean"] for a in args.alphas])
        slope, intercept = np.polyfit(log_a, log_r, 1)
        per_alpha["log_slope"] = float(slope)
        per_alpha["log_intercept"] = float(intercept)
        print(f"  {attr:12s} log-log slope = {slope:+.3f}")
        results[attr] = per_alpha

    print("\n=== summary ===")
    for attr in attrs:
        print(f"  {attr:12s} slope = {results[attr]['log_slope']:+.3f}")

    (out / "metrics.json").write_text(json.dumps({
        "results": results,
        "config": vars(args),
    }, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
