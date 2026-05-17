"""Track 18 — finite-difference validation of composed-JVP second-derivative.

Three independent methods for d^2 G(b, b):
  M1. Composed JVP (default)
  M2. FD of JVP: (jvp at +ε - jvp at -ε) / (2ε)
  M3. Second-order FD: (G(wp+ε) - 2G(wp) + G(wp-ε)) / ε²

At ε ∈ {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}, report relative error of M2
and M3 vs M1 across N=8 (latent, attribute) pairs.

See designs/18_finite_difference_validation.md.
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
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402


def composed_jvp(G, wp, bl):
    """Reference: |d²G(b, b)| pixel map via composed JVP."""
    def f(alpha):
        return G.synthesize(wp + alpha.view(1, 1, 1) * bl.unsqueeze(0))
    def df(alpha):
        _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
        return d
    a0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)
    _, sec = jvp(df, (a0,), (one,))
    return sec.detach()


def fd_of_jvp(G, wp, bl, eps):
    def f(alpha):
        return G.synthesize(wp + alpha.view(1, 1, 1) * bl.unsqueeze(0))
    def at(a_val):
        ap = torch.tensor(a_val, device=G.device)
        at = torch.ones_like(ap)
        _, d = jvp(f, (ap,), (at,))
        return d
    dp = at(+eps); dm = at(-eps)
    return ((dp - dm) / (2 * eps)).detach()


def second_fd(G, wp, bl, eps):
    with torch.no_grad():
        xp = G.synthesize(wp + eps * bl.unsqueeze(0))
        x0 = G.synthesize(wp)
        xn = G.synthesize(wp - eps * bl.unsqueeze(0))
        return ((xp - 2 * x0 + xn) / (eps ** 2)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom"], default="bedroom")
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--epsilons", nargs="+", type=float,
                    default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ap.add_argument("--out", default="experiments/out/fd_validation")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary
    from higan_dev.cam.grad_saliency import _layered_direction
    G = HiGANGenerator(higan_repo=str(
        PAPER.parent / "higan_dev" / "data" / "higan_repo"
    ))
    L, D = G.num_layers, G.w_dim
    bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
            / "boundaries" / "stylegan_bedroom")
    attrs = ["view", "indoor_lighting", "wood"]
    bl_dict = {
        a: _layered_direction(
            load_boundary(str(bdir), a, num_layers=L).to(G.device),
            L, D, G.device
        ) for a in attrs
    }

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_pairs, generator=rng)

    print(f"[{time.strftime('%H:%M:%S')}] FD validation: "
          f"{args.n_pairs} latents × {len(attrs)} attrs × "
          f"{len(args.epsilons)} ε values")

    results = {}
    for attr in attrs:
        bl = bl_dict[attr]
        per_eps = {str(eps): {"m2_rel": [], "m3_rel": []}
                    for eps in args.epsilons}
        for i in range(args.n_pairs):
            wp = base_wp[i:i + 1].detach()
            ref = composed_jvp(G, wp, bl)
            ref_mag = ref.abs().mean().clamp_min(1e-12)
            for eps in args.epsilons:
                m2 = fd_of_jvp(G, wp, bl, eps)
                m3 = second_fd(G, wp, bl, eps)
                rel2 = ((m2 - ref).abs().mean() / ref_mag).item()
                rel3 = ((m3 - ref).abs().mean() / ref_mag).item()
                per_eps[str(eps)]["m2_rel"].append(rel2)
                per_eps[str(eps)]["m3_rel"].append(rel3)
                torch.cuda.empty_cache()
        # aggregate per ε
        agg = {}
        for eps in args.epsilons:
            agg[str(eps)] = {
                "m2_rel_mean": float(np.mean(per_eps[str(eps)]["m2_rel"])),
                "m3_rel_mean": float(np.mean(per_eps[str(eps)]["m3_rel"])),
            }
            print(f"  attr={attr} ε={eps:g}  "
                  f"M2 rel-err={agg[str(eps)]['m2_rel_mean']:.3e}  "
                  f"M3 rel-err={agg[str(eps)]['m3_rel_mean']:.3e}")
        results[attr] = agg

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
