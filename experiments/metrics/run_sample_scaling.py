"""Track 3 — sample-size scaling + bootstrap CI on C1 ratios.

For each (domain, attribute), compute per-sample second-order ratio
ρ_i = |∂²I/∂α²|_mean / |∂I/∂α|_mean for i = 1..N_max, then report:
  - mean ratio at N ∈ {8, 16, 32, 64, 128}
  - 95% bootstrap percentile CI (10 000 resamples)
  - Spearman rank correlation between the attribute ordering at N and
    at N_max (stability of the ordering).

Supports --domain bedroom | ffhq | church.
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
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))


def per_sample_ratio(G, base_wp, b_layered, idx) -> float:
    """ratio = mean |∂²I/∂α²| / mean |∂I/∂α|, single sample."""
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
    first_mean = first.abs().mean().item()
    second_mean = second.abs().mean().item()
    return second_mean / max(first_mean, 1e-8)


def bootstrap_mean_ci(values: list[float], n_boot: int = 10_000,
                       seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 2.5)), \
           float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "ffhq", "church"],
                    required=True)
    ap.add_argument("--n-max", type=int, default=128)
    ap.add_argument("--Ns", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or
               f"experiments/out/sample_scaling_{args.domain}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] sample scaling for {args.domain} "
          f"with N_max={args.n_max}")

    if args.domain == "bedroom":
        from higan_dev.generator import HiGANGenerator
        from higan_dev.manipulate import load_boundary
        from higan_dev.cam.grad_saliency import _layered_direction
        G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
        attrs = ["indoor_lighting", "wood", "view", "carpet",
                 "cluttered_space", "glossy", "dirt", "scary"]
        bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
        b_layered = {a: _layered_direction(
            load_boundary(str(bdir), a, num_layers=G.num_layers).to(G.device),
            G.num_layers, G.w_dim, G.device) for a in attrs}
    elif args.domain == "ffhq":
        from domains.ffhq.generator import FFHQGenerator
        G = FFHQGenerator()
        attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
        boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
        LAYERS_FOR = {
            "pose":        list(range(0, 4)),
            "gender":      list(range(0, 8)),
            "age":         list(range(0, 8)),
            "eyeglasses":  list(range(0, 8)),
            "smile":       list(range(4, 8)),
        }
        b_layered = {}
        for a in attrs:
            v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            d = d / d.norm().clamp_min(1e-8)
            bl = torch.zeros(G.num_layers, G.w_dim, device=G.device)
            for li in LAYERS_FOR[a]:
                bl[li] = d
            b_layered[a] = bl
    else:  # church
        from domains.church.generator import ChurchGenerator
        G = ChurchGenerator()
        attrs = ["clouds", "sunny", "vegetation"]
        LAYERS_FOR = {
            "clouds":     list(range(0, 8)),
            "sunny":      list(range(0, 8)),
            "vegetation": list(range(6, 12)),
        }
        bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan2_church"
        b_layered = {}
        for a in attrs:
            v = np.load(bdir / f"{a}_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            d = d / d.norm().clamp_min(1e-8)
            bl = torch.zeros(G.num_layers, G.w_dim, device=G.device)
            for li in LAYERS_FOR[a]:
                bl[li] = d
            b_layered[a] = bl

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_max, generator=rng)
    print(f"sampled {args.n_max} base wps")

    per_attr_ratios = {}
    for attr in attrs:
        t_attr = time.time()
        ratios = []
        for i in range(args.n_max):
            r = per_sample_ratio(G, base_wp, b_layered[attr], i)
            ratios.append(r)
            torch.cuda.empty_cache()
            if (i + 1) % 16 == 0:
                print(f"  {attr:18s} {i+1}/{args.n_max}  "
                      f"running mean={np.mean(ratios):.3f}  "
                      f"({time.time()-t_attr:.1f}s)")
        per_attr_ratios[attr] = ratios
        print(f"  {attr:18s} done: mean={np.mean(ratios):.3f}  "
              f"median={np.median(ratios):.3f}  "
              f"({time.time()-t_attr:.1f}s)")

    # Bootstrap CI at each N
    print("\n=== bootstrap CI table ===")
    table = {}
    for N in args.Ns:
        if N > args.n_max:
            continue
        row = {}
        for attr in attrs:
            vals = per_attr_ratios[attr][:N]
            mean_, lo, hi = bootstrap_mean_ci(vals)
            row[attr] = {"mean": mean_, "ci_lo": lo, "ci_hi": hi,
                         "ci_width_rel": (hi - lo) / max(mean_, 1e-8)}
        table[str(N)] = row
        print(f"N={N:3d}:  " + "  ".join(
            f"{a}={row[a]['mean']:.3f}±{(row[a]['ci_hi']-row[a]['ci_lo'])/2:.3f}"
            for a in attrs
        ))

    # Rank-stability vs N_max
    n_max_means = np.array([np.mean(per_attr_ratios[a]) for a in attrs])
    stability = {}
    for N in args.Ns:
        if N > args.n_max:
            continue
        n_means = np.array([np.mean(per_attr_ratios[a][:N]) for a in attrs])
        if len(attrs) > 2:
            r, p = spearmanr(n_means, n_max_means)
            stability[str(N)] = {"spearman_vs_nmax": float(r),
                                 "p": float(p)}
            print(f"N={N:3d} rank-vs-Nmax  Spearman={r:+.3f}  p={p:.3g}")

    payload = {
        "domain": args.domain,
        "n_max": args.n_max,
        "Ns": args.Ns,
        "attrs": attrs,
        "per_attr_ratios": {a: per_attr_ratios[a] for a in attrs},
        "bootstrap_ci": table,
        "rank_stability": stability,
    }
    with open(out / "metrics.json", "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
