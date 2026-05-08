"""Saliency-guided local edits — attribute changes only where saliency is high.

Loads encoder, encodes test images, then for each attribute renders:
    [original | saliency | mask | global edit | local edit]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.utils import label_bar as _label
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.manipulate import load_boundary
from higan_dev.cam.local_edit import saliency_guided_edit
from higan_dev.cam.diff_map import colorize_heat
from higan_dev.utils import load_image_tensor



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--image-idx", type=int, default=2)
    ap.add_argument("--out", default="out/local_edit")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))
    target = load_image_tensor(img_paths[args.image_idx],
                               size=cfg.generator.resolution,
                               normalize="-1_1").to(G.device)
    res = encode_image(target, enc, G, w_avg=w_avg)
    wp_enc = res.wp                                              # (1, L, D)

    rows = []
    cell_w = cfg.generator.resolution
    labels = ["original", "saliency", "mask", "global edit", "local edit"]
    label_strip = np.concatenate([_label(l, cell_w) for l in labels], axis=1)

    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr,
                          num_layers=G.num_layers).to(G.device)
        r = saliency_guided_edit(
            G, wp_enc, b,
            alpha=args.alpha, sigma=args.sigma, threshold=args.threshold,
        )
        sal_rgb = colorize_heat(r.saliency)
        mask_rgb = colorize_heat(r.mask, cmap="viridis")
        row = np.concatenate([
            r.original, sal_rgb, mask_rgb, r.global_edit, r.local_edit,
        ], axis=1)
        eyebrow = _label(f"━━ {attr.upper()} ━━", row.shape[1], h=28, fs=16)
        rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

    final = np.concatenate(rows, axis=0)
    out_path = out_dir / f"img_{args.image_idx}_local_edit.png"
    Image.fromarray(final).save(out_path)
    print(f"saved {out_path}  ({final.shape[1]} x {final.shape[0]})")


if __name__ == "__main__":
    main()
