"""Track 8 — truncation-ψ ablation on FFHQ C1 (per-attribute ρ).

For each ψ ∈ {0.5, 0.7, 1.0}, re-run the FFHQ second-order ratio
measurement on the 5 InterFaceGAN attributes and check whether the
attribute ordering is preserved.

See designs/08_truncation_ablation.md.
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
from scipy.stats import spearmanr

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


def per_sample_ratio(G, base_wp, b_layered, idx):
    wp = base_wp[idx:idx + 1].detach()

    def f(alpha):
        return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
    def df(alpha):
        _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
        return d
    a0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)
    _, first = jvp(f, (a0,), (one,))
    _, second = jvp(df, (a0,), (one,))
    return second.abs().mean().item() / max(first.abs().mean().item(), 1e-8)


def measure_for_psi(psi: float, num_samples: int):
    G = FFHQGenerator()
    # apply ψ to mapping output during sampling
    # FFHQGenerator wraps InterFaceGAN's StyleGANGenerator which exposes
    # truncation_psi as an instance attribute on the inner module.
    original_psi = G.truncation_psi
    G.truncation_psi = psi
    if hasattr(G._net, "truncation"):
        # InterFaceGAN's TruncationModule stores `psi` as a buffer
        if hasattr(G._net.truncation, "psi"):
            G._net.truncation.psi.fill_(psi)
    L, D = G.num_layers, G.w_dim
    boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                      / "boundaries")
    attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
    b_layered = {}
    for a in attrs:
        v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        d = torch.from_numpy(v).to(G.device)
        d = d / d.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=G.device)
        for li in LAYERS_FOR[a]:
            bl[li] = d
        b_layered[a] = bl

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(num_samples, generator=rng)

    out = {}
    for attr in attrs:
        ratios = []
        for s in range(num_samples):
            r = per_sample_ratio(G, base_wp, b_layered[attr], s)
            ratios.append(r)
            torch.cuda.empty_cache()
        out[attr] = {"mean": float(np.mean(ratios)),
                     "median": float(np.median(ratios)),
                     "std": float(np.std(ratios)),
                     "values": ratios}
        print(f"  ψ={psi}  {attr:12s} mean={np.mean(ratios):.3f}  "
              f"median={np.median(ratios):.3f}")
    # restore
    G.truncation_psi = original_psi
    del G
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psis", nargs="+", type=float,
                    default=[0.5, 0.7, 1.0])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", default="experiments/out/ffhq_truncation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] FFHQ ψ-ablation, "
          f"{args.num_samples} seeds per (ψ, attr)")
    results = {}
    for psi in args.psis:
        print(f"\n=== ψ = {psi} ===")
        results[str(psi)] = measure_for_psi(psi, args.num_samples)

    # cross-ψ rank stability
    attrs = list(results[str(args.psis[0])].keys())
    print("\n=== rank stability across ψ ===")
    rank_table = {}
    for i, p1 in enumerate(args.psis):
        for j, p2 in enumerate(args.psis):
            if i >= j:
                continue
            ord1 = [results[str(p1)][a]["mean"] for a in attrs]
            ord2 = [results[str(p2)][a]["mean"] for a in attrs]
            r, p = spearmanr(ord1, ord2)
            rank_table[f"{p1}_vs_{p2}"] = {"r": float(r), "p": float(p)}
            print(f"  ψ={p1} vs ψ={p2}: Spearman r={r:+.3f} (p={p:.3g})")

    (out / "metrics.json").write_text(json.dumps(
        {"per_psi": results, "rank_stability": rank_table,
         "config": vars(args)}, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
