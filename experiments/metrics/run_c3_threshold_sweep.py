"""C3 threshold sweep — does the layer-localisation finding survive
across top-k thresholds k ∈ {0.10, 0.20, 0.30, 0.50}?

Approach: compute saliency once per (attr, layer) — the expensive
part — then re-derive C3 score across multiple thresholds without
re-running JVPs.

Supports both bedroom (HiGAN, 8 attrs × 14 layers) and FFHQ
(InterFaceGAN, 5 attrs × 18 layers) via --domain flag.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

THRESHOLDS = [0.10, 0.20, 0.30, 0.50]


def setup_bedroom():
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary
    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
    attrs = ["indoor_lighting", "wood", "view", "carpet",
             "cluttered_space", "glossy", "dirt", "scary"]
    bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    boundaries = {a: load_boundary(str(bdir), a, num_layers=G.num_layers)
                  for a in attrs}
    b_dirs = {a: boundaries[a].direction.to(G.device) for a in attrs}
    canonical = {a: sorted(boundaries[a].manipulate_layers) for a in attrs}
    return G, attrs, b_dirs, canonical


def setup_ffhq():
    from domains.ffhq.generator import FFHQGenerator
    G = FFHQGenerator()
    boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    LAYERS_FOR = {
        "pose":        list(range(0, 4)),
        "gender":      list(range(0, 8)),
        "age":         list(range(0, 8)),
        "eyeglasses":  list(range(0, 8)),
        "smile":       list(range(4, 8)),
    }
    attrs = list(LAYERS_FOR.keys())
    b_dirs = {}
    for a in attrs:
        v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        b_dirs[a] = torch.from_numpy(v).to(G.device)
    return G, attrs, b_dirs, LAYERS_FOR


def compute_saliency_per_layer(G, attrs, b_dirs, n_samples):
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(n_samples, generator=rng)

    sal: dict[str, list[np.ndarray]] = {a: [] for a in attrs}
    for attr in attrs:
        b_dir = b_dirs[attr] / b_dirs[attr].norm().clamp_min(1e-8)
        for li in range(L):
            b_layered = torch.zeros(L, D, device=G.device)
            b_layered[li] = b_dir
            acc = torch.zeros(H, W, device=G.device)
            for s in range(n_samples):
                wp = base_wp[s:s + 1].detach()
                def f(alpha):
                    return G.synthesize(wp + alpha.view(1, 1, 1)
                                        * b_layered.unsqueeze(0))
                _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                              (torch.ones(1, device=G.device),))
                acc += dimg.abs().mean(dim=1).squeeze(0)
                torch.cuda.empty_cache()
            sal[attr].append((acc / n_samples).cpu().numpy())
        print(f"  {attr:14s} done")
    return sal


def c3_at_threshold(sal_per_layer, canonical, L, frac):
    def topk(s):
        return s >= np.quantile(s, 1 - frac)
    scores = {}
    for attr, sals in sal_per_layer.items():
        masks = [topk(s) for s in sals]
        canon = set(canonical[attr])
        noncanon = set(range(L)) - canon
        cc = [(i, j) for i in canon for j in canon if i < j]
        cn = [(i, j) for i in canon for j in noncanon]
        if not cc or not cn:
            scores[attr] = float("nan")
            continue
        iou_cc = np.mean([_iou(masks[i], masks[j]) for i, j in cc])
        iou_cn = np.mean([_iou(masks[i], masks[j]) for i, j in cn])
        scores[attr] = float(iou_cc - iou_cn)
    return scores


def _iou(m1, m2):
    inter = float((m1 & m2).sum())
    union = float((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "ffhq"], required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"out/{args.domain}_c3_threshold")
    out.mkdir(parents=True, exist_ok=True)

    if args.domain == "bedroom":
        G, attrs, b_dirs, canonical = setup_bedroom()
    else:
        G, attrs, b_dirs, canonical = setup_ffhq()
    L = G.num_layers
    print(f"=== {args.domain} ===  L={L}  attrs={attrs}  n_samples={args.num_samples}")

    sal = compute_saliency_per_layer(G, attrs, b_dirs, args.num_samples)

    results = {}
    for frac in THRESHOLDS:
        scores = c3_at_threshold(sal, canonical, L, frac)
        mean = float(np.nanmean(list(scores.values())))
        n_pos = sum(1 for v in scores.values() if v > 0)
        results[str(frac)] = {"scores": scores,
                               "mean": mean,
                               "fraction_positive": n_pos / len(attrs)}
        print(f"  top-{int(frac*100):2d}%   mean={mean:+.3f}  "
              f"positive={n_pos}/{len(attrs)}  per-attr=" +
              " ".join(f"{a}:{v:+.2f}" for a, v in scores.items()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=140)

    fr_arr = np.array(THRESHOLDS)
    for attr in attrs:
        ys = [results[str(f)]["scores"][attr] for f in THRESHOLDS]
        ax1.plot(fr_arr, ys, "o-", label=attr, lw=1.5, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax1.set_xlabel("top-k saliency fraction", fontsize=10)
    ax1.set_ylabel("C3 score (IoU_cc − IoU_cn)", fontsize=10)
    ax1.set_title(f"{args.domain.capitalize()} C3 vs threshold — per attribute",
                  fontsize=11, weight="bold", pad=8)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="best", ncol=2)

    means = [results[str(f)]["mean"] for f in THRESHOLDS]
    fracs = [results[str(f)]["fraction_positive"] for f in THRESHOLDS]
    ax2.bar(np.arange(len(THRESHOLDS)) - 0.18, means, width=0.36,
            color="#0e7490", label="mean C3 across attrs")
    ax2t = ax2.twinx()
    ax2t.bar(np.arange(len(THRESHOLDS)) + 0.18, fracs, width=0.36,
             color="#c2410c", label="fraction positive")
    ax2.set_xticks(range(len(THRESHOLDS)))
    ax2.set_xticklabels([f"{int(f*100)}%" for f in THRESHOLDS])
    ax2.set_xlabel("top-k threshold", fontsize=10)
    ax2.set_ylabel("mean C3", color="#0e7490", fontsize=10)
    ax2t.set_ylabel("fraction positive", color="#c2410c", fontsize=10)
    ax2.set_title("C3 aggregate stability across thresholds",
                  fontsize=11, weight="bold", pad=8)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / f"{args.domain}_c3_threshold.png")
    print(f"\nsaved {out / f'{args.domain}_c3_threshold.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"per_threshold": results,
                   "thresholds": THRESHOLDS,
                   "num_samples": args.num_samples}, f, indent=2)


if __name__ == "__main__":
    main()
