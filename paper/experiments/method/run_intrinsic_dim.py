"""Track 17 — intrinsic image-manifold dimension via Jacobian rank.

For each of N latents, probe K random directions in W+, compute
their pushforward via JVP, stack into a Jacobian-like matrix
J ∈ R^{3HW × K}, take SVD, and report the entropy-weighted
effective rank (Roy & Vetterli 2007).

Establishes a third geometric characterisation of the image manifold,
complementing C1 (curvature ratio) and Stanczuk 2024 (score-Jacobian
rank).

See designs/17_intrinsic_dimension.md.
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


def effective_rank(s: np.ndarray) -> float:
    """Roy-Vetterli 2007 effective rank from singular values."""
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "ffhq"], required=True)
    ap.add_argument("--n-latents", type=int, default=16)
    ap.add_argument("--K-probes", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out or
               f"experiments/out/intrinsic_dim_{args.domain}")
    out.mkdir(parents=True, exist_ok=True)

    if args.domain == "bedroom":
        from higan_dev.generator import HiGANGenerator
        G = HiGANGenerator(higan_repo=str(
            PAPER.parent / "higan_dev" / "data" / "higan_repo"
        ))
    else:
        from domains.ffhq.generator import FFHQGenerator
        G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_latents, generator=rng)

    print(f"[{time.strftime('%H:%M:%S')}] intrinsic dim on {args.domain}: "
          f"{args.n_latents} latents × {args.K_probes} probes")

    per_latent_rank = []
    per_latent_spectrum = []
    rng_probe = torch.Generator(device=G.device).manual_seed(13)

    for i in range(args.n_latents):
        wp = base_wp[i:i + 1].detach()

        # build (3HW, K_probes) matrix of JVP outputs
        J_cols = []
        for k in range(args.K_probes):
            v = torch.randn(L, D, device=G.device, generator=rng_probe)
            v_flat = v.flatten()
            v = (v / v_flat.norm().clamp_min(1e-8)).view(L, D)

            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * v.unsqueeze(0))

            a0 = torch.zeros(1, device=G.device)
            one = torch.ones(1, device=G.device)
            _, d = jvp(f, (a0,), (one,))
            J_cols.append(d.flatten().cpu().float().numpy())
            torch.cuda.empty_cache()
        J = np.stack(J_cols, axis=1)         # (3HW, K)
        # SVD (thin)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        eff = effective_rank(S)
        per_latent_rank.append(eff)
        per_latent_spectrum.append(S.tolist())
        print(f"  latent {i+1}/{args.n_latents}  "
              f"effective rank = {eff:.1f}  "
              f"top-σ = {S[0]:.3g}, σ_K = {S[-1]:.3g}")

    median_rank = float(np.median(per_latent_rank))
    iqr = float(np.percentile(per_latent_rank, 75)
                - np.percentile(per_latent_rank, 25))
    print(f"\n=== {args.domain} ===")
    print(f"effective rank: median={median_rank:.1f}  IQR={iqr:.1f}")
    print(f"(probe ceiling K={args.K_probes}, ambient = 3HW = "
          f"{3 * G.resolution * G.resolution:,})")

    payload = {
        "domain": args.domain,
        "n_latents": args.n_latents,
        "K_probes": args.K_probes,
        "per_latent_effective_rank": per_latent_rank,
        "per_latent_spectrum_top16": [s[:16] for s in per_latent_spectrum],
        "median_effective_rank": median_rank,
        "iqr_effective_rank": iqr,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
