"""Track 14 — noise-seed robustness for C1 ratios.

Re-runs the second-order ratio measurement on a domain across 5
independent seeds and reports per-attribute mean ± std + cross-seed
Spearman rank correlation.

See designs/14_noise_robustness.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from torch.func import jvp
from scipy.stats import spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "ffhq", "church"],
                    required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[2027, 2028, 2029, 2030, 2031])
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out or
               f"experiments/out/noise_robustness_{args.domain}")
    out.mkdir(parents=True, exist_ok=True)

    if args.domain == "bedroom":
        from higan_dev.generator import HiGANGenerator
        from higan_dev.manipulate import load_boundary
        from higan_dev.cam.grad_saliency import _layered_direction
        G = HiGANGenerator(higan_repo=str(
            PAPER.parent / "higan_dev" / "data" / "higan_repo"
        ))
        attrs = ["indoor_lighting", "wood", "view", "carpet",
                 "cluttered_space", "glossy", "dirt", "scary"]
        bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
                / "boundaries" / "stylegan_bedroom")
        bl_dict = {
            a: _layered_direction(
                load_boundary(str(bdir), a, num_layers=G.num_layers).to(G.device),
                G.num_layers, G.w_dim, G.device
            ) for a in attrs
        }
    elif args.domain == "ffhq":
        from domains.ffhq.generator import FFHQGenerator
        G = FFHQGenerator()
        attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
        boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                          / "boundaries")
        LAYERS_FOR = {
            "pose":        list(range(0, 4)),
            "gender":      list(range(0, 8)),
            "age":         list(range(0, 8)),
            "eyeglasses":  list(range(0, 8)),
            "smile":       list(range(4, 8)),
        }
        bl_dict = {}
        for a in attrs:
            v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            d = d / d.norm().clamp_min(1e-8)
            bl = torch.zeros(G.num_layers, G.w_dim, device=G.device)
            for li in LAYERS_FOR[a]:
                bl[li] = d
            bl_dict[a] = bl
    else:
        from domains.church.generator import ChurchGenerator
        G = ChurchGenerator()
        attrs = ["clouds", "sunny", "vegetation"]
        LAYERS_FOR = {
            "clouds":     list(range(0, 8)),
            "sunny":      list(range(0, 8)),
            "vegetation": list(range(6, 12)),
        }
        bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
                / "boundaries" / "stylegan2_church")
        bl_dict = {}
        for a in attrs:
            v = np.load(bdir / f"{a}_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            d = d / d.norm().clamp_min(1e-8)
            bl = torch.zeros(G.num_layers, G.w_dim, device=G.device)
            for li in LAYERS_FOR[a]:
                bl[li] = d
            bl_dict[a] = bl

    print(f"[{time.strftime('%H:%M:%S')}] noise robustness on "
          f"{args.domain}: {len(args.seeds)} seeds × "
          f"{len(attrs)} attrs × {args.num_samples} samples")

    per_seed = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        rng = torch.Generator(device=G.device).manual_seed(seed)
        base_wp = G.sample_wp(args.num_samples, generator=rng)
        seed_results = {}
        for attr in attrs:
            ratios = []
            for i in range(args.num_samples):
                r = per_sample_ratio(G, base_wp, bl_dict[attr], i)
                ratios.append(r)
                torch.cuda.empty_cache()
            seed_results[attr] = {"mean": float(np.mean(ratios)),
                                   "std": float(np.std(ratios))}
            print(f"  {attr:18s} mean={np.mean(ratios):.3f}")
        per_seed[str(seed)] = seed_results

    # aggregate: mean of means, std of means across seeds
    agg = {}
    for attr in attrs:
        means = [per_seed[str(s)][attr]["mean"] for s in args.seeds]
        agg[attr] = {
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "rel_cv": float(np.std(means) / max(abs(np.mean(means)), 1e-8)),
            "seed_means": means,
        }
    print("\n=== aggregate (mean ± std of per-seed means) ===")
    for attr in attrs:
        r = agg[attr]
        print(f"  {attr:18s} {r['mean_of_means']:.3f} ± {r['std_of_means']:.3f}  "
              f"CV={r['rel_cv']:.2%}")

    # cross-seed Spearman rank correlation
    pair_spearman = {}
    for s1, s2 in combinations(args.seeds, 2):
        o1 = [per_seed[str(s1)][a]["mean"] for a in attrs]
        o2 = [per_seed[str(s2)][a]["mean"] for a in attrs]
        r, p = spearmanr(o1, o2)
        pair_spearman[f"{s1}_vs_{s2}"] = {"r": float(r), "p": float(p)}
        print(f"  seed {s1} vs {s2}: Spearman r={r:+.3f} (p={p:.3g})")

    mean_pair_r = float(np.mean([v["r"] for v in pair_spearman.values()]))
    print(f"\nmean pairwise Spearman r across seeds: {mean_pair_r:+.3f}")

    payload = {"per_seed": per_seed, "aggregate": agg,
               "pairwise_spearman": pair_spearman,
               "mean_pairwise_r": mean_pair_r,
               "config": vars(args)}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
