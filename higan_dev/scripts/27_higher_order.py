"""Second-order saliency  ∂²I/∂α²  — measures pixel-wise non-linearity.

For boundary direction b and scalar perturbation α, first-order saliency
is  ∂I/∂α  (already covered). Second-order is  ∂²I/∂α².

Interpretation:
  pixels where ∂²I/∂α² is small ≈ first-order linear regime — moving along
    the boundary changes the pixel uniformly.
  pixels where ∂²I/∂α² is large ≈ non-linear regime — the same boundary
    direction has rapidly accelerating / decelerating pixel changes,
    suggesting saturation or other non-linearities.

We compute it via jvp(jvp(...)). Forward-mode is composable, so two
nested JVP calls give us per-pixel curvature in one combined pass.
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
from higan_dev.manipulate import load_boundary
from higan_dev.cam.grad_saliency import _layered_direction
from higan_dev.cam.diff_map import colorize_heat


def _label(text: str, w: int, h: int = 24, fs: int = 14) -> np.ndarray:
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
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", default="out/higher_order")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    rng = torch.Generator(device=G.device).manual_seed(31)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    rows = []
    cell_w = H

    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr,
                          num_layers=L).to(G.device)
        b_layered = _layered_direction(b, L, D, G.device)

        first_acc = torch.zeros(H, W, device=G.device)
        second_acc = torch.zeros(H, W, device=G.device)
        n_done = 0

        for s in range(0, args.num_samples, 4):
            wp_chunk = base_wp[s:s + 4].detach()
            B = wp_chunk.shape[0]

            # First derivative function: f(α) -> image
            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(
                    wp_chunk + alpha.view(B, 1, 1) * b_layered.unsqueeze(0))

            # Second derivative via composed JVP:
            #   df/dα at α=0 in direction tangent
            # We need ∂²f/∂α² which is jvp of (jvp of f). Concretely:
            #   jvp(lambda a: jvp(f, (a,), (ones,))[1], (a0,), (ones,))
            # gives us the directional derivative of df/dα along α.
            def df_dalpha(alpha: torch.Tensor) -> torch.Tensor:
                _, dimg = jvp(
                    f, (alpha,),
                    (torch.ones_like(alpha),),
                )
                return dimg                                       # (B, 3, H, W)

            alpha0 = torch.zeros(B, device=G.device)
            ones = torch.ones(B, device=G.device)
            # First-order saliency
            _, first = jvp(f, (alpha0,), (ones,))                 # (B, 3, H, W)
            # Second-order saliency
            _, second = jvp(df_dalpha, (alpha0,), (ones,))        # (B, 3, H, W)

            first_acc += first.abs().mean(dim=1).sum(dim=0)
            second_acc += second.abs().mean(dim=1).sum(dim=0)
            n_done += B

        first_map = (first_acc / n_done).cpu().numpy()
        second_map = (second_acc / n_done).cpu().numpy()
        # ratio = second / first  (per-pixel non-linearity)
        ratio = second_map / (first_map + 1e-6)
        # normalise each map for display
        first_n = first_map / max(first_map.max(), 1e-8)
        second_n = second_map / max(second_map.max(), 1e-8)
        ratio_n = ratio / max(ratio.max(), 1e-8)

        first_rgb = colorize_heat(first_n.astype(np.float32))
        second_rgb = colorize_heat(second_n.astype(np.float32))
        ratio_rgb = colorize_heat(ratio_n.astype(np.float32), cmap="viridis")

        labels = ["first order |∂I/∂α|", "second order |∂²I/∂α²|", "non-linearity ratio"]
        label_strip = np.concatenate(
            [_label(l, cell_w) for l in labels], axis=1)
        eyebrow = _label(f"━━ {attr.upper()} ━━",
                          cell_w * 3, h=28, fs=16)
        row = np.concatenate([first_rgb, second_rgb, ratio_rgb], axis=1)
        rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

        # also report scalar non-linearity statistics
        ratio_mean = float(ratio.mean())
        ratio_p95 = float(np.percentile(ratio, 95))
        print(f"  {attr:18s}  mean(∂²/∂)/(∂I/∂α)={ratio_mean:.3f}  "
              f"p95={ratio_p95:.3f}  "
              f"first_max={first_map.max():.3f}  second_max={second_map.max():.3f}")

    final = np.concatenate(rows, axis=0)
    out_path = out_dir / "higher_order.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")


if __name__ == "__main__":
    main()
