"""Decompose grad-saliency by individual W+ layer.

For each attribute boundary that operates on multiple layers (e.g.,
indoor_lighting on layers 6..11), compute the JVP-based saliency that would
result from perturbing one layer at a time. Saves a per-attribute grid:
rows = single layer, cols = mean | abs heatmap.

Example:
    python scripts/12_per_layer_saliency.py --attrs indoor_lighting wood view \
        --num-samples 32 --out out/per_layer
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
from higan_dev.cam.grad_saliency import compute_per_layer_saliency
from higan_dev.cam.diff_map import colorize_heat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--out", default="out/per_layer")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers)
        except FileNotFoundError:
            continue

        per = compute_per_layer_saliency(
            G, b, num_samples=args.num_samples, micro_batch=args.micro_batch,
        )
        sub = out_dir / attr
        sub.mkdir(parents=True, exist_ok=True)

        # row per layer: layer label | abs heatmap
        layer_ids = sorted(per.keys())
        row_imgs = []
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        for li in layer_ids:
            heat = colorize_heat(per[li])             # (H, W, 3) uint8
            label = Image.new("RGB", (heat.shape[0], 28), (245, 245, 244))
            draw = ImageDraw.Draw(label)
            draw.text((6, 4), f"layer {li}", fill=(40, 40, 40), font=font)
            label_np = np.asarray(label)
            cell = np.concatenate([label_np, heat], axis=0)
            row_imgs.append(cell)
            Image.fromarray(heat).save(sub / f"layer_{li}.png")

        montage = np.concatenate(row_imgs, axis=1)
        Image.fromarray(montage).save(sub / "all_layers.png")

        np.savez(sub / "raw.npz",
                 **{f"layer_{k}": v for k, v in per.items()},
                 layer_ids=np.asarray(layer_ids))
        print(f"{attr}: layers={layer_ids}  -> {sub/'all_layers.png'}")

    print(f"\nresults in {out_dir}")


if __name__ == "__main__":
    main()
