"""Exhaustive 8 × 14 attribute × W+ layer saliency matrix.

Until now per-layer decomposition was on a single attribute's
manipulate_layers (e.g., indoor_lighting on layers 6–11). Here we evaluate
*every attribute* at *every layer*, including non-canonical ones, to see
whether HiGAN's hand-curated layer ranges are tight or whether spillover
exists.

Output: a single 8-row × 14-column heatmap montage.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary, list_available_boundaries
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
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--cell-size", type=int, default=96,
                    help="display size of each (attr, layer) cell")
    ap.add_argument("--out", default="out/full_matrix")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir
    attrs = list_available_boundaries(bdir)

    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    rng = torch.Generator(device=G.device).manual_seed(101)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    matrix: dict[str, list[np.ndarray]] = {}     # attr -> [14 saliency maps]
    canonical: dict[str, list[int]] = {}         # attr -> manipulate_layers
    intensities: dict[str, np.ndarray] = {}      # attr -> (14,) raw mean per layer

    for attr in attrs:
        b = load_boundary(bdir, attr, num_layers=L).to(G.device)
        canonical[attr] = list(b.manipulate_layers)

        per_layer_maps: list[np.ndarray] = []
        per_layer_intensity = np.zeros(L, dtype=np.float32)

        for layer_idx in range(L):
            b_layered = torch.zeros(L, D, device=G.device)
            b_layered[layer_idx] = b.direction.to(G.device)

            acc = torch.zeros(H, W, device=G.device)
            for s in range(0, args.num_samples, 4):
                wp = base_wp[s:s + 4].detach()
                B = wp.shape[0]
                def f(alpha):
                    return G.synthesize(
                        wp + alpha.view(B, 1, 1) * b_layered.unsqueeze(0))
                _, dimg = jvp(
                    f,
                    (torch.zeros(B, device=G.device),),
                    (torch.ones(B, device=G.device),),
                )
                acc += dimg.abs().mean(dim=1).sum(dim=0)
            sal = (acc / args.num_samples).cpu().numpy()
            per_layer_intensity[layer_idx] = float(sal.mean())
            m = sal.max()
            sal = (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)
            per_layer_maps.append(sal)
        matrix[attr] = per_layer_maps
        intensities[attr] = per_layer_intensity
        print(f"  {attr:18s}  canonical layers={canonical[attr]}  "
              f"max-intensity layer={int(np.argmax(per_layer_intensity))}")

    # build the (rows = attribute, cols = layer 0-13) montage
    cs = args.cell_size
    header = [_label(f"layer {li}", cs, h=22, fs=11) for li in range(L)]
    header_strip = np.concatenate(header, axis=1)
    pad_left = _label("attribute", cs, h=22, fs=11)
    header_strip = np.concatenate([pad_left, header_strip], axis=1)

    rows = [header_strip]
    for attr in attrs:
        attr_label = _label(attr, cs, h=cs, fs=12)
        cells = []
        canon = set(canonical[attr])
        peak_layer = int(np.argmax(intensities[attr]))
        for li in range(L):
            heat = colorize_heat(matrix[attr][li])
            heat_pil = Image.fromarray(heat).resize((cs, cs), Image.BILINEAR)
            arr = np.asarray(heat_pil).copy()
            # frame canonical layers in green, peak layer in red (dominant)
            border = None
            if li == peak_layer:
                border = (185, 28, 28)         # red — peak intensity
            elif li in canon:
                border = (21, 128, 61)         # green — canonical
            if border is not None:
                arr[:3, :, :] = border
                arr[-3:, :, :] = border
                arr[:, :3, :] = border
                arr[:, -3:, :] = border
            cells.append(arr)
        row = np.concatenate([attr_label] + cells, axis=1)
        rows.append(row)
    final = np.concatenate(rows, axis=0)
    out_path = out / "matrix_8x14.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    # also save intensity table
    int_table = np.stack([intensities[a] for a in attrs])
    np.savez(out / "intensities.npz",
             attrs=np.asarray(attrs),
             intensities=int_table,
             canonical_layers=np.asarray(
                 [canonical[a] for a in attrs], dtype=object),
             peak_layer=np.asarray([int(np.argmax(intensities[a])) for a in attrs]))
    # print intensity table
    print("\nlayer intensity (mean ∂I/∂α magnitude per layer):")
    print("  attr".ljust(20) + "".join(f"L{l:>2d} " for l in range(L)))
    for a in attrs:
        line = "  " + a.ljust(18)
        for li in range(L):
            v = intensities[a][li]
            mark = "*" if li == int(np.argmax(intensities[a])) else " "
            line += f"{v:6.3f}{mark}"
        print(line)


if __name__ == "__main__":
    main()
