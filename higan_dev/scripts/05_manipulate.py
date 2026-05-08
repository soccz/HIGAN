"""Generate boundary-manipulation grids (one row per attribute).

Picks a few sample latents (random or from --wp-file), then for each requested
attribute renders a -delta..+delta sweep next to the original.

Example:
    python scripts/05_manipulate.py --attrs indoor_lighting wood view \
        --num-samples 4 --steps 5 --delta 3 --out out/manipulate
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, manipulate_wp, list_available_boundaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--wp-file", help="optional .npy of (N,L,D) anchor codes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)

    if args.wp_file:
        wp = torch.from_numpy(np.load(args.wp_file)).float().to(G.device)
        if wp.shape[0] > args.num_samples:
            wp = wp[: args.num_samples]
    else:
        gen = torch.Generator(device=G.device).manual_seed(args.seed)
        wp = G.sample_wp(args.num_samples, generator=gen)

    distances = np.linspace(-args.delta, args.delta, args.steps).tolist()

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers).to(G.device)
        except FileNotFoundError:
            print(f"[skip] {attr}: no boundary")
            continue
        manip = manipulate_wp(wp, b, distances=distances)  # (B, K, L, D)
        B, K = manip.shape[:2]
        flat = manip.reshape(B * K, manip.shape[2], manip.shape[3])
        with torch.no_grad():
            imgs = G.synthesize(flat)                          # (B*K, 3, H, W)
        u8 = G.to_uint8(imgs)                                   # (B*K, H, W, 3)
        H, W, _ = u8.shape[1:]
        u8 = u8.reshape(B, K, H, W, 3)
        # build a grid: rows = B samples, cols = K distances
        grid = np.concatenate(
            [np.concatenate(list(u8[i]), axis=1) for i in range(B)], axis=0
        )
        Image.fromarray(grid).save(out_dir / f"{attr}.png")
        print(f"{attr}: saved {out_dir / f'{attr}.png'}  (rows=samples, cols={distances})")


if __name__ == "__main__":
    main()
