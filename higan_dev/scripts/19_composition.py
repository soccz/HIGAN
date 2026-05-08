"""Compositional editing: saliency of (a+b) vs sum of individual saliencies.

For a few attribute pairs, render the 4 saliency maps side-by-side and
print the pixel-wise correlation between the actual joint saliency and
the linear sum.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary
from higan_dev.cam.composition import compositional_saliency
from higan_dev.cam.diff_map import colorize_heat


def _label(text: str, w: int, h: int = 26, fs: int = 16) -> np.ndarray:
    img = Image.new("RGB", (w, h), (245, 245, 244))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(40, 40, 40), font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", nargs="+",
                    default=["indoor_lighting,wood",
                             "indoor_lighting,view",
                             "wood,view",
                             "carpet,wood",
                             "cluttered_space,glossy"])
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--out", default="out/composition")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    bdir = cfg.paths.boundaries_dir

    rows = []
    table = []
    for pair in args.pairs:
        a, b = pair.split(",")
        ba = load_boundary(bdir, a, num_layers=G.num_layers)
        bb = load_boundary(bdir, b, num_layers=G.num_layers)
        r = compositional_saliency(G, ba, bb, num_samples=args.num_samples)
        cell_w = cfg.generator.resolution
        sal_a = colorize_heat(r.sal_a)
        sal_b = colorize_heat(r.sal_b)
        sal_sum = colorize_heat(r.sal_sum)
        expected = colorize_heat(r.expected_sum)
        # diff map between actual sum and expected
        diff = np.abs(r.sal_sum - r.expected_sum)
        diff_rgb = colorize_heat(diff / max(diff.max(), 1e-8), cmap="magma")

        labels = [f"sal({a})", f"sal({b})", f"sal({a}+{b})", f"sal({a})+sal({b})", "|actual − expected|"]
        label_strip = np.concatenate([_label(l, cell_w) for l in labels], axis=1)
        eyebrow = _label(f"━━ {a.upper()} + {b.upper()}  ·  corr = {r.corr:.3f}",
                         label_strip.shape[1], h=30, fs=18)
        row = np.concatenate([sal_a, sal_b, sal_sum, expected, diff_rgb], axis=1)
        rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))
        table.append((a, b, r.corr))

    final = np.concatenate(rows, axis=0)
    out_path = out / "composition_grid.png"
    Image.fromarray(final).save(out_path)
    print(f"saved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    print("\npair correlations (actual vs expected sum):")
    for a, b, c in sorted(table, key=lambda x: -x[2]):
        marker = "linear ✓" if c > 0.95 else ("near-linear" if c > 0.85 else "non-linear ⚠")
        print(f"  {a:18s} + {b:18s}  corr={c:+.3f}   [{marker}]")


if __name__ == "__main__":
    main()
