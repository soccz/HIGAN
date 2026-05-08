"""End-to-end real-image attribute editing via the trained encoder.

For each test image:  image -> encoder -> wp_pred -> wp_pred + t*boundary -> render
Saves one grid per attribute with rows = images, cols = distance steps.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.manipulate import load_boundary, manipulate_wp, list_available_boundaries
from higan_dev.utils import load_image_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--out", default="out/edit_real")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)
    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)

    # encode all test images
    img_paths = sorted(Path(args.testset).glob("img_*.png"))
    if not img_paths:
        raise SystemExit("no test images")
    targets = torch.cat([
        load_image_tensor(p, size=cfg.generator.resolution, normalize="-1_1")
        for p in img_paths
    ], dim=0).to(G.device)

    res = encode_image(targets, enc, G, w_avg=w_avg)
    wp_enc = res.wp                                          # (N, L, D)
    distances = np.linspace(-args.delta, args.delta, args.steps).tolist()

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers).to(G.device)
        except FileNotFoundError:
            continue
        manip = manipulate_wp(wp_enc, b, distances=distances)   # (N, K, L, D)
        N, K = manip.shape[:2]
        flat = manip.reshape(N * K, manip.shape[2], manip.shape[3])
        with torch.no_grad():
            imgs = G.synthesize(flat)
        u8 = G.to_uint8(imgs)
        H, W, _ = u8.shape[1:]
        u8 = u8.reshape(N, K, H, W, 3)

        # build grid: prepend the original real image as col 0 for context
        rows = []
        targets_u8 = G.to_uint8(targets)
        for i in range(N):
            row = np.concatenate(
                [targets_u8[i]] + list(u8[i]), axis=1
            )
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        Image.fromarray(grid).save(out / f"{attr}.png")
        print(f"{attr}: rows=N={N}  cols=[real|{distances}]  ->  {out/f'{attr}.png'}")


if __name__ == "__main__":
    main()
