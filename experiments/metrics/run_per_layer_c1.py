"""Track 10 — per-layer C1 decomposition.

For each (domain, attribute), measure the second-order ratio
ρ_ℓ(a) = E[|d²G(b_a^(ℓ), b_a^(ℓ))|] / E[|dG b_a^(ℓ)|]
applying the boundary direction to a single W+ layer ℓ at a time,
then plot ρ_ℓ across ℓ ∈ {0..L-1}.

Tests whether structural attributes concentrate their second-order
energy in coarse layers (matching canonical InterFaceGAN ranges).

See designs/10_per_layer_c1_decomposition.md.
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


def per_layer_ratio(G, b_dir, base_wp, layer_idx, n_samples):
    """ρ for boundary applied only to a single layer."""
    L, D = G.num_layers, G.w_dim
    bl = torch.zeros(L, D, device=G.device)
    bl[layer_idx] = b_dir
    ratios = []
    for s in range(n_samples):
        wp = base_wp[s:s + 1].detach()
        def f(alpha):
            return G.synthesize(wp + alpha.view(1, 1, 1) * bl.unsqueeze(0))
        def df(alpha):
            _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
            return d
        a0 = torch.zeros(1, device=G.device)
        one = torch.ones(1, device=G.device)
        _, first = jvp(f, (a0,), (one,))
        _, second = jvp(df, (a0,), (one,))
        first_m = first.abs().mean().item()
        second_m = second.abs().mean().item()
        ratios.append(second_m / max(first_m, 1e-8))
        torch.cuda.empty_cache()
    return float(np.mean(ratios)), float(np.std(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "ffhq"], required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or
               f"experiments/out/per_layer_c1_{args.domain}")
    out.mkdir(parents=True, exist_ok=True)

    if args.domain == "bedroom":
        from higan_dev.generator import HiGANGenerator
        from higan_dev.manipulate import load_boundary
        G = HiGANGenerator(higan_repo=str(
            PAPER.parent / "higan_dev" / "data" / "higan_repo"
        ))
        attrs = ["indoor_lighting", "wood", "view", "carpet",
                 "cluttered_space", "glossy", "dirt", "scary"]
        bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
                / "boundaries" / "stylegan_bedroom")
        b_dirs = {}
        canonical = {}
        for a in attrs:
            b = load_boundary(str(bdir), a, num_layers=G.num_layers)
            d = b.direction.to(G.device)
            b_dirs[a] = d / d.norm().clamp_min(1e-8)
            canonical[a] = sorted(b.manipulate_layers)
    else:
        from domains.ffhq.generator import FFHQGenerator
        G = FFHQGenerator()
        attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
        boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                          / "boundaries")
        canonical = {
            "pose":        list(range(0, 4)),
            "gender":      list(range(0, 8)),
            "age":         list(range(0, 8)),
            "eyeglasses":  list(range(0, 8)),
            "smile":       list(range(4, 8)),
        }
        b_dirs = {}
        for a in attrs:
            v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            b_dirs[a] = d / d.norm().clamp_min(1e-8)

    L = G.num_layers
    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_samples, generator=rng)

    print(f"[{time.strftime('%H:%M:%S')}] per-layer C1 on {args.domain}: "
          f"{len(attrs)} attrs × {L} layers × {args.n_samples} seeds")

    results = {"per_attr": {}, "canonical": canonical,
               "domain": args.domain, "L": L, "attrs": attrs}
    for attr in attrs:
        t0 = time.time()
        ρ_per_layer = []
        std_per_layer = []
        for ℓ in range(L):
            ρ, sd = per_layer_ratio(G, b_dirs[attr], base_wp, ℓ,
                                     args.n_samples)
            ρ_per_layer.append(ρ)
            std_per_layer.append(sd)
            print(f"  {attr:14s} layer {ℓ:2d}  ρ={ρ:.3f}")
        argmax_layer = int(np.argmax(ρ_per_layer))
        in_canonical = argmax_layer in canonical[attr]
        results["per_attr"][attr] = {
            "rho_per_layer": ρ_per_layer,
            "std_per_layer": std_per_layer,
            "argmax_layer": argmax_layer,
            "argmax_in_canonical": in_canonical,
            "canonical_layers": canonical[attr],
        }
        print(f"  {attr:14s} argmax layer={argmax_layer}  "
              f"canonical={canonical[attr]}  "
              f"hit={'✓' if in_canonical else '✗'}  ({time.time()-t0:.0f}s)")

    n_hit = sum(r["argmax_in_canonical"]
                for r in results["per_attr"].values())
    results["argmax_canonical_hit_rate"] = n_hit / len(attrs)
    print(f"\nargmax-in-canonical rate: {n_hit}/{len(attrs)}")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out / 'metrics.json'}")

    # plot per-attribute line plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, (len(attrs) + 1) // 2,
                              figsize=(3 * ((len(attrs) + 1) // 2), 6),
                              dpi=140, squeeze=False)
    for k, attr in enumerate(attrs):
        ax = axes[k // ((len(attrs) + 1) // 2)][k % ((len(attrs) + 1) // 2)]
        ρ = results["per_attr"][attr]["rho_per_layer"]
        ax.bar(range(L), ρ, color=["#dc2626" if ℓ in canonical[attr]
                                    else "#0e7490" for ℓ in range(L)])
        ax.set_title(f"{attr}", fontsize=10, weight="bold")
        ax.set_xlabel("W+ layer"); ax.set_ylabel("ρ")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{args.domain} — per-layer ρ. Red = canonical layer",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "per_layer_c1.png")
    print(f"saved {out / 'per_layer_c1.png'}")


if __name__ == "__main__":
    main()
