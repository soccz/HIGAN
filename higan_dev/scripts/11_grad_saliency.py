"""Gradient-based pixel saliency analysis (forward-mode JVP through G).

Distinct from `06_cam_analysis.py` which uses forward perturbation finite
differences. This one runs the generator's autograd backward / forward-tangent
to get the *exact* first-order pixel sensitivity per attribute boundary.

Example:
    python scripts/11_grad_saliency.py --num-samples 64 --out out/grad_saliency
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
from higan_dev.cam.grad_saliency import compute_grad_saliency
from higan_dev.cam.diff_map import colorize_heat, colorize_signed, overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--out", default="out/grad_saliency")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    bdir = cfg.paths.boundaries_dir
    attrs = args.attrs or list_available_boundaries(bdir)

    summary = [
        f"# Gradient-based pixel saliency",
        f"num_samples={args.num_samples}",
        f"attrs={attrs}",
        "",
    ]

    for attr in attrs:
        try:
            b = load_boundary(bdir, attr, num_layers=G.num_layers)
        except FileNotFoundError:
            print(f"[skip] {attr}: boundary not found")
            continue
        result = compute_grad_saliency(
            G, b, num_samples=args.num_samples, micro_batch=args.micro_batch,
        )
        sub = out_dir / attr
        sub.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.mean_image).save(sub / "mean.png")
        Image.fromarray(colorize_heat(result.abs_saliency)).save(sub / "abs.png")
        Image.fromarray(colorize_signed(result.signed_saliency)).save(sub / "signed.png")
        Image.fromarray(overlay(result.mean_image, result.abs_saliency, alpha=0.55)).save(
            sub / "overlay.png"
        )
        # per-sample saliency overlays (compact grid: img | sal | overlay)
        per_dir = sub / "per_sample"
        per_dir.mkdir(exist_ok=True)
        rows = []
        for k in range(len(result.per_sample_abs)):
            img = result.per_sample_image[k]
            sal = result.per_sample_abs[k]
            sal_rgb = colorize_heat(sal)
            ov = overlay(img, sal, alpha=0.55)
            row = np.concatenate([img, sal_rgb, ov], axis=1)
            rows.append(row)
            Image.fromarray(row).save(per_dir / f"sample_{k:02d}.png")
        if rows:
            Image.fromarray(np.concatenate(rows, axis=0)).save(sub / "per_sample_grid.png")
        np.savez(sub / "raw.npz",
                 abs_saliency=result.abs_saliency,
                 signed_saliency=result.signed_saliency,
                 layers=np.asarray(b.manipulate_layers))
        line = (f"{attr:18s}  layers={b.manipulate_layers}  "
                f"abs_max={result.abs_saliency.max():.3f}")
        print(line)
        summary.append(line)

    (out_dir / "summary.txt").write_text("\n".join(summary))
    print(f"\nresults in {out_dir}")


if __name__ == "__main__":
    main()
