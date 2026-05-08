"""Discover new interpretable directions by random sampling in W+.

For each randomly sampled direction in W+ (or just one layer of W+), compute
the JVP saliency. Rank directions by spatial localisation (negative entropy),
keeping only directions whose effect concentrates in a recognisable region.
This demonstrates that our analysis tool generalises beyond the 8 hand-curated
HiGAN boundaries — *no extra training, no labels.*
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.cam.diff_map import colorize_heat, overlay


def _top_k_mass(p: np.ndarray, frac: float = 0.05) -> float:
    """Fraction of total mass concentrated in the top-frac fraction of pixels.

    Uniform: returns frac (e.g., 0.05).
    Single hot spot: approaches 1.0.
    Higher = more spatially concentrated.
    """
    flat = p.flatten()
    s = flat.sum()
    if s < 1e-8:
        return 0.0
    k = max(1, int(flat.size * frac))
    top = np.partition(flat, -k)[-k:]
    return float(top.sum() / s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-directions", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--keep-top", type=int, default=8)
    ap.add_argument("--layer-mode", choices=["all", "single", "fine", "coarse"],
                    default="single",
                    help="all = perturb all 14 layers; single = pick one random layer;"
                         " fine = layers 10-13; coarse = layers 0-3")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="out/random_directions")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    rng_t = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng_t)
    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1)

    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    saliencies: list[np.ndarray] = []
    metas: list[dict] = []

    for d in range(args.num_directions):
        # build a random layered direction
        b_layered = torch.zeros(L, D, device=G.device)
        if args.layer_mode == "all":
            v = torch.randn(L, D, generator=rng_dir, device=G.device)
            v = v / v.flatten().norm().clamp_min(1e-8)
            b_layered = v
            chosen = list(range(L))
        elif args.layer_mode == "single":
            # exclude layer 0 (StyleGAN's constant-input layer; effect is too weak)
            li = int(torch.randint(1, L, (1,), generator=rng_dir, device=G.device).item())
            v = torch.randn(D, generator=rng_dir, device=G.device)
            v = v / v.norm().clamp_min(1e-8)
            b_layered[li] = v
            chosen = [li]
        elif args.layer_mode == "fine":
            li = int(torch.randint(10, 14, (1,), generator=rng_dir, device=G.device).item())
            v = torch.randn(D, generator=rng_dir, device=G.device)
            v = v / v.norm().clamp_min(1e-8)
            b_layered[li] = v
            chosen = [li]
        else:  # coarse: layers 1-4 only (skip 0, which is the constant layer)
            li = int(torch.randint(1, 5, (1,), generator=rng_dir, device=G.device).item())
            v = torch.randn(D, generator=rng_dir, device=G.device)
            v = v / v.norm().clamp_min(1e-8)
            b_layered[li] = v
            chosen = [li]

        # accumulate JVP across samples
        acc = torch.zeros(H, W, device=G.device)
        chunks = 4
        for s in range(0, args.num_samples, chunks):
            wp_chunk = base_wp[s:s + chunks].detach()
            B = wp_chunk.shape[0]

            def f(a: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp_chunk + a.view(B, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(f, (torch.zeros(B, device=G.device),),
                          (torch.ones(B, device=G.device),))
            acc += dimg.abs().mean(dim=1).sum(dim=0)
        sal = (acc / args.num_samples).cpu().numpy()
        raw_strength = float(sal.mean())                # absolute magnitude of effect
        m = sal.max()
        if m > 1e-8:
            sal = sal / m
        # mass concentration in top-5% pixels (1.0 = single point, 0.05 = uniform)
        top5 = _top_k_mass(sal, frac=0.05)
        saliencies.append(sal.astype(np.float32))
        metas.append({
            "id": d, "layer": chosen,
            "top5_mass": top5,
            "strength": raw_strength,
        })

    # filter out directions with no real effect (noise-driven), then rank by
    # spatial concentration descending.
    strengths = np.asarray([m["strength"] for m in metas])
    median_strength = float(np.median(strengths))
    valid = [i for i in range(len(metas)) if metas[i]["strength"] >= median_strength]
    valid.sort(key=lambda i: -metas[i]["top5_mass"])
    keep = valid[: args.keep_top]
    print("\nMost localised random directions (highest top-5% mass):")
    for i in keep:
        print(f"  dir{metas[i]['id']:03d}  layer={metas[i]['layer']}  "
              f"top5={metas[i]['top5_mass']:.3f}  strength={metas[i]['strength']:.4f}")

    # render mean image once for overlay
    with torch.no_grad():
        # average rendering of base latents (for context)
        chunks = []
        for s in range(0, base_wp.shape[0], 4):
            chunks.append(G.synthesize(base_wp[s:s + 4]))
        imgs = torch.cat(chunks, dim=0)
        mean_img = (imgs.clamp(-1, 1) + 1) / 2
        mean_img = (mean_img.permute(0, 2, 3, 1).cpu().numpy() * 255
                    ).astype(np.uint8)
        # use first sample's image as the static reference
        ref_img = mean_img[0]

    # build a gallery: rows of [heat | overlay] for each kept dir
    rows = []
    for i in keep:
        sal = saliencies[i]
        heat = colorize_heat(sal)
        ov = overlay(ref_img, sal, alpha=0.55)
        meta = metas[i]
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        label = Image.new("RGB", (heat.shape[1], 24), (245, 245, 244))
        d = ImageDraw.Draw(label)
        d.text((6, 4),
               f"dir #{meta['id']:03d} · layer {meta['layer']} · top5 {meta['top5_mass']:.2f}",
               fill=(40, 40, 40), font=font)
        labelN = np.asarray(label)
        row = np.concatenate([heat, ov], axis=1)
        labels_full = np.concatenate(
            [labelN, np.full((24, heat.shape[1], 3), 245, dtype=np.uint8)], axis=1)
        rows.append(np.concatenate([labels_full, row], axis=0))
    final = np.concatenate(rows, axis=0)
    out_path = out / f"discovered_top{args.keep_top}_{args.layer_mode}.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    np.savez(out / f"raw_{args.layer_mode}.npz",
             saliencies=np.stack(saliencies),
             top5_mass=np.asarray([m["top5_mass"] for m in metas]),
             strengths=np.asarray([m["strength"] for m in metas]),
             layers=np.asarray([m["layer"] for m in metas], dtype=object))


if __name__ == "__main__":
    main()
