"""Encoder attention: where in the input image does the encoder *look*
to determine each attribute direction in W+?

For each test image and each attribute, render a row:
    [target | encoder attention | overlay]
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
from higan_dev.inversion.encode import load_encoder
from higan_dev.manipulate import load_boundary
from higan_dev.cam.encoder_attention import encoder_attention
from higan_dev.cam.diff_map import colorize_heat, overlay
from higan_dev.utils import load_image_tensor



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--out", default="out/encoder_attention")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))[:4]
    cell_w = cfg.generator.resolution

    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr,
                          num_layers=G.num_layers).to(G.device)

        rows = []
        for p in img_paths:
            target = load_image_tensor(p, size=cell_w, normalize="-1_1").to(G.device)
            r = encoder_attention(enc, target, b, w_avg=w_avg)
            target_u8 = ((target.clamp(-1, 1) + 1) / 2)
            target_u8 = (target_u8.permute(0, 2, 3, 1).cpu().numpy() * 255
                         ).astype(np.uint8)[0]
            sal_rgb = colorize_heat(r.saliency)
            ov = overlay(target_u8, r.saliency, alpha=0.55)
            row = np.concatenate([target_u8, sal_rgb, ov], axis=1)
            rows.append(row)
        labels = ["input image", "encoder attention", "overlay"]
        label_strip = np.concatenate([_label(l, cell_w) for l in labels], axis=1)
        eyebrow = _label(f"━━ {attr.upper()} ━━", label_strip.shape[1], h=28, fs=16)
        grid = np.concatenate([eyebrow, label_strip] + rows, axis=0)
        Image.fromarray(grid).save(out_dir / f"{attr}.png")
        print(f"{attr}: {len(rows)} rows -> {out_dir / f'{attr}.png'}")


if __name__ == "__main__":
    main()
