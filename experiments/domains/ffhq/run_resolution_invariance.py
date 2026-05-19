"""Track 13 — FFHQ resolution-invariance ablation via lod override.

For each lod ∈ {0, 1, 2}: 1024² / 512² / 256² rendering of FFHQ,
re-measure C1 ratios on 5 InterFaceGAN attributes, test rank
preservation.

See designs/13_resolution_invariance.md.
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

PAPER = Path(__file__).resolve().parents[3]   # paper/
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402

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


def measure_for_lod(lod: float, num_samples: int):
    G = FFHQGenerator(lod_override=lod)
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
    res = G.resolution // (2 ** int(lod))
    print(f"  effective resolution at lod={lod}: ~{res}²")

    out = {}
    for attr in attrs:
        ratios = []
        for s in range(num_samples):
            r = per_sample_ratio(G, base_wp, b_layered[attr], s)
            ratios.append(r)
            torch.cuda.empty_cache()
        out[attr] = {"mean": float(np.mean(ratios)),
                     "median": float(np.median(ratios)),
                     "std": float(np.std(ratios))}
        print(f"  lod={lod} {attr:12s} mean={np.mean(ratios):.3f}")
    del G; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lods", nargs="+", type=float, default=[0, 1, 2])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", default="experiments/out/ffhq_resolution")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for lod in args.lods:
        print(f"\n=== lod = {lod} ===")
        results[str(lod)] = measure_for_lod(lod, args.num_samples)

    attrs = list(results[str(args.lods[0])].keys())
    rank_table = {}
    for i, l1 in enumerate(args.lods):
        for j, l2 in enumerate(args.lods):
            if i >= j: continue
            o1 = [results[str(l1)][a]["mean"] for a in attrs]
            o2 = [results[str(l2)][a]["mean"] for a in attrs]
            r, p = spearmanr(o1, o2)
            rank_table[f"lod{l1}_vs_lod{l2}"] = {"r": float(r), "p": float(p)}
            print(f"  lod={l1} vs lod={l2}: Spearman r={r:+.3f} p={p:.3g}")

    (out / "metrics.json").write_text(json.dumps(
        {"per_lod": results, "rank_stability": rank_table,
         "config": vars(args)}, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
