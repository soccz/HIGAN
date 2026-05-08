"""Build a fixed test set of synthetic bedroom images + ground-truth wp.

Used to compare optimisation-based vs encoder-based inversion on the same
images.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.utils import save_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="out/testset")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    gen = torch.Generator(device=G.device).manual_seed(args.seed)
    with torch.no_grad():
        wp_gt = G.sample_wp(args.n, generator=gen)
        images = G.synthesize(wp_gt)

    np.save(out / "wp_gt.npy", wp_gt.cpu().numpy())
    u8 = G.to_uint8(images)
    for i, im in enumerate(u8):
        save_image(im, out / f"img_{i:02d}.png")
    print(f"saved {args.n} images + wp_gt.npy in {out}")


if __name__ == "__main__":
    main()
