"""End-to-end demo figure: one bedroom, four pipeline stages.

For a chosen test image, render a single horizontal strip:
    target  |  encoder recon  |  grad-saliency overlay  |  edit -d  |  edit +d

This is the "show me my bedroom" moment: from a real photo to "where the
lamp is" to "now make it brighter".
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
from higan_dev.manipulate import load_boundary, manipulate_wp
from higan_dev.cam.grad_saliency import compute_grad_saliency
from higan_dev.cam.diff_map import colorize_heat, overlay
from higan_dev.utils import load_image_tensor



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="+", default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--image-idx", type=int, default=0)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--out", default="out/full_cycle")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))
    img_path = img_paths[args.image_idx]

    target = load_image_tensor(img_path, size=cfg.generator.resolution,
                               normalize="-1_1").to(G.device)
    res = encode_image(target, enc, G, w_avg=w_avg)
    wp_enc = res.wp                                          # (1, L, D)
    recon_u8 = G.to_uint8(res.image)[0]
    target_u8 = G.to_uint8(target)[0]

    cells_per_attr: list[np.ndarray] = []
    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr,
                          num_layers=G.num_layers).to(G.device)

        # JVP saliency on this single image's encoded latent
        sal_res = compute_grad_saliency(
            G, b, num_samples=1, micro_batch=1, base_wp=wp_enc, keep_per_sample=1,
        )
        # the per_sample_image and per_sample_abs hold the one sample
        sal_img = sal_res.per_sample_image[0]                  # (H, W, 3) uint8
        sal_map = sal_res.per_sample_abs[0]                    # (H, W) in [0,1]
        sal_overlay = overlay(sal_img, sal_map, alpha=0.6)
        sal_heat = colorize_heat(sal_map)

        # edits: -d and +d
        manip = manipulate_wp(wp_enc, b, distances=[-args.delta, args.delta])
        N, K, L, D = manip.shape
        with torch.no_grad():
            imgs = G.synthesize(manip.reshape(N * K, L, D))
        edit_u8 = G.to_uint8(imgs).reshape(N, K, *imgs.shape[2:][:2], 3)
        edit_neg = edit_u8[0, 0]
        edit_pos = edit_u8[0, 1]

        # build a row for this attribute:
        #   [target | encoder recon | saliency heat | overlay | edit -d | edit +d]
        row = np.concatenate(
            [target_u8, recon_u8, sal_heat, sal_overlay, edit_neg, edit_pos],
            axis=1,
        )
        # column labels for this row
        cell_w = target_u8.shape[1]
        labels = ["target", "encoder recon",
                  f"grad-saliency · {attr}", "overlay",
                  f"edit -{args.delta}", f"edit +{args.delta}"]
        label_strip = np.concatenate(
            [_label_strip(l, cell_w) for l in labels], axis=1
        )
        # attribute eyebrow
        eyebrow = _label_strip(
            f"━━ {attr.upper()} ━━",
            cell_w * 6, h=32, font_size=18,
        )
        cells_per_attr.append(np.concatenate([eyebrow, label_strip, row], axis=0))

    final = np.concatenate(cells_per_attr, axis=0)
    out_path = out_dir / f"img_{args.image_idx}_full_cycle.png"
    Image.fromarray(final).save(out_path)
    print(f"saved {out_path}  ({final.shape[1]} x {final.shape[0]})")


if __name__ == "__main__":
    main()
