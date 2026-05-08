"""Render the 8×8 attribute disentanglement matrix.

Saves two heatmaps: |abs| saliency correlation and signed saliency correlation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
from higan_dev.cam.disentangle import compute_disentanglement, render_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--out", default="out/disentangle")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir
    attrs = list_available_boundaries(bdir)
    boundaries = [load_boundary(bdir, a, num_layers=G.num_layers) for a in attrs]

    res = compute_disentanglement(
        G, boundaries, num_samples=args.num_samples,
    )

    abs_img = render_matrix(
        res["abs_corr"], res["names"],
        title="|abs| saliency  ·  pixel-wise Pearson",
        vmin=-1, vmax=1,
    )
    sgn_img = render_matrix(
        res["signed_corr"], res["names"],
        title="signed saliency  ·  pixel-wise Pearson",
        vmin=-1, vmax=1,
    )
    Image.fromarray(abs_img).save(out / "abs_corr.png")
    Image.fromarray(sgn_img).save(out / "signed_corr.png")
    np.savez(out / "raw.npz",
             abs_corr=res["abs_corr"],
             signed_corr=res["signed_corr"],
             names=np.asarray(res["names"]))
    print(f"saved heatmaps to {out}")
    print("\n|abs| corr (top entanglements):")
    n = len(res["names"])
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((res["abs_corr"][i, j], res["names"][i], res["names"][j]))
    pairs.sort(key=lambda x: -x[0])
    for c, a, b in pairs[:5]:
        print(f"  {a:18s} <-> {b:18s}  corr={c:+.3f}")
    print("\n  most disentangled (lowest):")
    for c, a, b in pairs[-3:]:
        print(f"  {a:18s} <-> {b:18s}  corr={c:+.3f}")


if __name__ == "__main__":
    main()
