"""End-to-end CAM-style spatial analysis of HiGAN attribute boundaries.

Runs the perturbation sweep across many random latents and saves heatmaps
showing where each attribute lives in pixel space. Optionally overlays the
heatmap on the mean rendering.

Example:
    python scripts/06_cam_analysis.py --attrs indoor_lighting wood view \
        --num-samples 32 --delta 1.5 --out out/cam
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
from higan_dev.cam.diff_map import (
    compute_diff_map, colorize_heat, colorize_signed, overlay,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="*", default=None,
                    help="attribute names; default = all available")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--out", default="out/cam")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)
    delta = args.delta if args.delta is not None else cfg.cam.delta
    num_samples = args.num_samples or cfg.cam.num_samples

    summary_path = out_dir / "summary.txt"
    summary_lines = [
        f"# CAM-style attribute saliency",
        f"num_samples={num_samples}  delta={delta}",
        f"attrs={attrs}",
        "",
    ]

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers)
        except FileNotFoundError:
            print(f"[skip] {attr}: boundary not found")
            continue
        result = compute_diff_map(
            G, b,
            delta=delta,
            num_samples=num_samples,
            micro_batch=args.micro_batch,
        )
        # save heatmaps
        sub = out_dir / attr
        sub.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.mean_image).save(sub / "mean.png")
        Image.fromarray(colorize_heat(result.abs_diff)).save(sub / "abs_diff.png")
        Image.fromarray(colorize_heat(result.variance, cmap="viridis")).save(
            sub / "variance.png"
        )
        Image.fromarray(colorize_signed(result.signed_diff)).save(sub / "signed.png")
        Image.fromarray(overlay(result.mean_image, result.abs_diff, alpha=0.55)).save(
            sub / "overlay.png"
        )
        # save raw arrays too, for downstream quantitative work
        np.savez(sub / "raw.npz",
                 abs_diff=result.abs_diff,
                 variance=result.variance,
                 signed_diff=result.signed_diff,
                 layers=np.asarray(b.manipulate_layers))
        line = (f"{attr:18s}  layers={b.manipulate_layers}  "
                f"abs_max={result.abs_diff.max():.3f}  "
                f"var_max={result.variance.max():.3f}")
        print(line)
        summary_lines.append(line)

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nresults in {out_dir}")


if __name__ == "__main__":
    main()
