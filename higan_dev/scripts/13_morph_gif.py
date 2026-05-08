"""Render smooth boundary-edit morphs as GIF/MP4.

Per attribute: take a few test images, encode them with the trained encoder,
then linearly walk wp + α·b for α in [-d ... +d] in N frames. Save as both
animated GIF and a vertical strip PNG.
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
from higan_dev.manipulate import load_boundary, list_available_boundaries, manipulate_wp
from higan_dev.utils import load_image_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--n-images", type=int, default=2,
                    help="number of test images to morph in parallel (rows)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--bounce", action="store_true",
                    help="ping-pong: -d -> +d -> -d (cleaner loop)")
    ap.add_argument("--out", default="out/morph")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))[: args.n_images]
    if not img_paths:
        raise SystemExit("no test images")
    targets = torch.cat([
        load_image_tensor(p, size=cfg.generator.resolution, normalize="-1_1")
        for p in img_paths
    ], dim=0).to(G.device)
    res = encode_image(targets, enc, G, w_avg=w_avg)
    wp_enc = res.wp                                          # (N, L, D)

    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)

    distances = np.linspace(-args.delta, args.delta, args.frames).tolist()
    if args.bounce:
        distances = distances + distances[-2:0:-1]           # ping-pong

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers).to(G.device)
        except FileNotFoundError:
            continue

        manip = manipulate_wp(wp_enc, b, distances=distances)  # (N, K, L, D)
        N, K, L, D = manip.shape
        flat = manip.reshape(N * K, L, D)
        # render in chunks to fit in 8 GB
        chunks = []
        chunk = 8
        with torch.no_grad():
            for s in range(0, flat.shape[0], chunk):
                chunks.append(G.to_uint8(G.synthesize(flat[s:s + chunk])))
        u8 = np.concatenate(chunks, axis=0)
        H, W, _ = u8.shape[1:]
        u8 = u8.reshape(N, K, H, W, 3)

        # build per-frame image: rows of N images
        frames = []
        for k in range(K):
            row = np.concatenate(list(u8[:, k]), axis=1)        # (H, N*W, 3)
            frames.append(Image.fromarray(row))
        gif_path = out / f"{attr}.gif"
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / args.fps), loop=0, optimize=True,
        )

        # also save first/middle/last frames + strip for static fallback
        strip = np.concatenate([np.asarray(frames[0]),
                                np.asarray(frames[len(frames) // 2]),
                                np.asarray(frames[-1])], axis=0)
        Image.fromarray(strip).save(out / f"{attr}_strip.png")
        print(f"{attr}: {len(frames)} frames -> {gif_path}")


if __name__ == "__main__":
    main()
