"""Saliency robustness: how stable is each attribute's saliency map?

For each attribute, compute the saliency at K different random latents.
Measure inter-sample consistency (mean pairwise correlation) and intensity
spread (mean per-pixel std relative to mean).

High consistency = "this attribute's spatial behaviour is stable across
bedrooms" (robust attribute).  Low consistency = "saliency depends on
which bedroom we look at" (scene-specific attribute).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.utils import label_bar as _label
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
from higan_dev.cam.grad_saliency import _layered_direction



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--out", default="out/robustness")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir
    attrs = list_available_boundaries(bdir)

    # shared base latents — same set for every attribute so the
    # comparison reflects attribute-level variation, not seed luck.
    rng = torch.Generator(device=G.device).manual_seed(99)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    rows: list[dict] = []
    L, D = G.num_layers, G.w_dim

    for attr in attrs:
        b = load_boundary(bdir, attr, num_layers=G.num_layers).to(G.device)
        b_layered = _layered_direction(b, L, D, G.device)

        per_sample: list[np.ndarray] = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(
                f,
                (torch.zeros(1, device=G.device),),
                (torch.ones(1, device=G.device),),
            )
            sal = dimg.abs().mean(dim=1).squeeze(0).cpu().numpy()
            m = sal.max()
            per_sample.append((sal / m if m > 1e-8 else sal).astype(np.float32))

        sal_stack = np.stack(per_sample, axis=0)            # (K, H, W)
        K = sal_stack.shape[0]

        # mean pairwise pixel-wise correlation across the K samples
        flat = sal_stack.reshape(K, -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(flat, axis=1) + 1e-8
        corr_matrix = (flat @ flat.T) / (norms[:, None] * norms[None, :])
        # exclude diagonal
        triu = corr_matrix[np.triu_indices(K, k=1)]
        consistency = float(triu.mean())

        # intensity coefficient of variation
        mean_map = sal_stack.mean(0)
        std_map = sal_stack.std(0)
        cv = float((std_map.mean() / (mean_map.mean() + 1e-8)))

        rows.append({"attr": attr, "consistency": consistency, "cv": cv,
                     "mean_map": mean_map, "std_map": std_map})

    rows.sort(key=lambda r: -r["consistency"])
    print("\nRobustness ranking (high consistency = stable across bedrooms):")
    print(f"  {'attr':18s} {'consistency':>14s} {'CV (std/mean)':>16s}")
    for r in rows:
        print(f"  {r['attr']:18s} {r['consistency']:+14.3f} {r['cv']:16.3f}")

    # build a 2×N visualisation: row 0 = mean saliency, row 1 = per-pixel std
    from higan_dev.cam.diff_map import colorize_heat
    H = G.resolution
    cell_w = H
    name_strip = np.concatenate(
        [_label(r["attr"], cell_w) for r in rows], axis=1
    )
    mean_row = np.concatenate(
        [colorize_heat(r["mean_map"] / max(r["mean_map"].max(), 1e-8))
         for r in rows], axis=1
    )
    std_row = np.concatenate(
        [colorize_heat(r["std_map"] / max(r["std_map"].max(), 1e-8), cmap="viridis")
         for r in rows], axis=1
    )
    label_left = _label("mean saliency", cell_w * len(rows), h=22)
    label_left2 = _label("per-pixel std (across samples)", cell_w * len(rows), h=22)
    grid = np.concatenate(
        [name_strip, label_left, mean_row, label_left2, std_row], axis=0
    )
    Image.fromarray(grid).save(out / "grid.png")
    np.savez(out / "raw.npz",
             names=np.asarray([r["attr"] for r in rows]),
             consistency=np.asarray([r["consistency"] for r in rows]),
             cv=np.asarray([r["cv"] for r in rows]))
    print(f"\nsaved {out / 'grid.png'}")


if __name__ == "__main__":
    main()
