"""Saliency at intermediate generator activations — closer to true Grad-CAM.

Standard Grad-CAM operates on conv feature maps, not raw pixels. Here we
mirror that by computing  ∂(activation_at_block_k) / ∂α  for each block
k ∈ [0..6] of the StyleGAN bedroom synthesis tower (spatial sizes
4, 8, 16, 32, 64, 128, 256). The saliency at smaller blocks shows where
the *abstraction* of the attribute lives in feature space, before the
output is upsampled.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary
from higan_dev.cam.grad_saliency import _layered_direction
from higan_dev.cam.diff_map import colorize_heat


def _label(text: str, w: int, h: int = 22, fs: int = 13) -> np.ndarray:
    img = Image.new("RGB", (w, h), (245, 245, 244))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 3), text, fill=(40, 40, 40), font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--max-block", type=int, default=6,
                    help="up to block index 6 (spatial 256 for bedroom256)")
    ap.add_argument("--target-size", type=int, default=128,
                    help="display size each cell is upsampled to (px)")
    ap.add_argument("--out", default="out/layer_saliency")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir

    # shared base latents
    rng = torch.Generator(device=G.device).manual_seed(31)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    L, D = G.num_layers, G.w_dim
    block_resolutions = [4, 8, 16, 32, 64, 128, 256]

    # results: dict[(attr, block_idx)] = (H, W) saliency
    grids: dict[str, list[np.ndarray]] = {}
    for attr in args.attrs:
        b = load_boundary(bdir, attr, num_layers=L).to(G.device)
        b_layered = _layered_direction(b, L, D, G.device)

        per_block: list[np.ndarray] = []
        for blk in range(args.max_block + 1):
            acc = None
            for s in range(0, args.num_samples, 4):
                wp = base_wp[s:s + 4].detach()
                B = wp.shape[0]
                def f(alpha):
                    return G.synthesize_at_block(
                        wp + alpha.view(B, 1, 1) * b_layered.unsqueeze(0), blk)
                a0 = torch.zeros(B, device=G.device)
                ones = torch.ones(B, device=G.device)
                _, dact = jvp(f, (a0,), (ones,))
                # dact: (B, C, H_blk, W_blk)
                sal = dact.abs().mean(dim=1)        # (B, H_blk, W_blk)
                if acc is None:
                    acc = sal.sum(dim=0)
                else:
                    acc = acc + sal.sum(dim=0)
            sal_np = (acc / args.num_samples).cpu().numpy()
            m = sal_np.max()
            sal_np = (sal_np / m).astype(np.float32) if m > 1e-8 else sal_np.astype(np.float32)
            per_block.append(sal_np)
            print(f"  {attr:18s} block {blk}  spatial={sal_np.shape}  max={m:.4g}")
        grids[attr] = per_block

    # build a montage: rows = attribute, cols = block
    ts = args.target_size
    rows = []
    for attr in args.attrs:
        cells = []
        for blk, sal in enumerate(grids[attr]):
            heat = colorize_heat(sal)
            heat_pil = Image.fromarray(heat).resize((ts, ts), Image.BILINEAR)
            arr = np.asarray(heat_pil)
            label = _label(f"block {blk} · {block_resolutions[blk]}px", ts)
            cells.append(np.concatenate([label, arr], axis=0))
        row = np.concatenate(cells, axis=1)
        eyebrow = _label(f"━━ {attr.upper()} ━━", row.shape[1], h=28, fs=16)
        rows.append(np.concatenate([eyebrow, row], axis=0))
    final = np.concatenate(rows, axis=0)
    Image.fromarray(final).save(out / "intermediate_layers.png")
    print(f"\nsaved {out / 'intermediate_layers.png'}")


if __name__ == "__main__":
    main()
