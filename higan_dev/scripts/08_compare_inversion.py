"""Compare optimisation-based and encoder-based inversion on the same test set.

Reads test images from `out/testset/`, the optim-inversion outputs from
`out/optim_inv/img_<i>/`, runs the trained encoder on the same images, and
builds a side-by-side comparison grid plus a per-image metric table.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.losses import LPIPSLoss
from higan_dev.utils import load_image_tensor, save_image


def lpips_value(lpips_fn, a, b):
    return float(lpips_fn(a, b).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--optim-dir", default="out/optim_inv")
    ap.add_argument("--out", default="out/compare")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)
    lpips_fn = LPIPSLoss().to(G.device)

    testset = Path(args.testset)
    optim_dir = Path(args.optim_dir)
    img_paths = sorted(testset.glob("img_*.png"))
    if not img_paths:
        raise SystemExit(f"no test images in {testset}")

    rows = []
    metrics = []
    for i, img_path in enumerate(img_paths):
        target = load_image_tensor(img_path, size=cfg.generator.resolution,
                                   normalize="-1_1").to(G.device)

        # encoder inversion
        res = encode_image(target, enc, G, w_avg=w_avg)
        recon_enc = res.image

        # optim inversion result was saved as recon.png
        optim_recon_path = optim_dir / f"img_{i}" / "recon.png"
        if optim_recon_path.exists():
            recon_optim = load_image_tensor(optim_recon_path,
                                            size=cfg.generator.resolution,
                                            normalize="-1_1").to(G.device)
        else:
            recon_optim = torch.zeros_like(target)

        with torch.no_grad():
            l_pix_enc = float(F.mse_loss(recon_enc, target).item())
            l_pix_opt = float(F.mse_loss(recon_optim, target).item())
            l_lpips_enc = lpips_value(lpips_fn, recon_enc, target)
            l_lpips_opt = lpips_value(lpips_fn, recon_optim, target)

        # row image: [target | optim | encoder]
        row = np.concatenate([
            G.to_uint8(target)[0],
            G.to_uint8(recon_optim)[0],
            G.to_uint8(recon_enc)[0],
        ], axis=1)
        rows.append(row)
        metrics.append({
            "image": img_path.name,
            "pixel_mse": {"optim": l_pix_opt, "encoder": l_pix_enc},
            "lpips":     {"optim": l_lpips_opt, "encoder": l_lpips_enc},
        })

    grid = np.concatenate(rows, axis=0)
    save_image(grid, out / "grid_target_optim_encoder.png")

    # caption banner: tile column titles
    H, W, _ = rows[0].shape
    titles = ["target", "optim", "encoder"]
    banner = np.full((22, W, 3), 240, dtype=np.uint8)
    Image.fromarray(banner).save(out / "banner_hint.png")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nper-image metrics (lower is better):\n")
    print(f"{'image':14s} {'pixel_mse_optim':>16s} {'pixel_mse_enc':>16s}"
          f" {'lpips_optim':>14s} {'lpips_enc':>14s}")
    for m in metrics:
        print(f"{m['image']:14s} "
              f"{m['pixel_mse']['optim']:16.4f} {m['pixel_mse']['encoder']:16.4f}"
              f" {m['lpips']['optim']:14.4f} {m['lpips']['encoder']:14.4f}")

    avg_pix = lambda key: np.mean([m["pixel_mse"][key] for m in metrics])
    avg_lp = lambda key: np.mean([m["lpips"][key] for m in metrics])
    print(f"\n{'mean':14s} "
          f"{avg_pix('optim'):16.4f} {avg_pix('encoder'):16.4f}"
          f" {avg_lp('optim'):14.4f} {avg_lp('encoder'):14.4f}")
    print(f"\ngrid -> {out / 'grid_target_optim_encoder.png'}")


if __name__ == "__main__":
    main()
