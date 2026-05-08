"""Animated WebP: as we sweep along a boundary, how does the *saliency*
itself evolve frame-by-frame?

Most morph videos show only the rendered image changing. Here each frame
also visualises the local first-order saliency at that perturbation point —
i.e., we re-evaluate ∂I/∂α at every wp along the trajectory. The result
shows whether saliency is stable along the trajectory (mostly linear edits)
or rapidly shifting (non-linear, curving).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.manipulate import load_boundary, manipulate_wp
from higan_dev.cam.grad_saliency import _layered_direction
from higan_dev.cam.diff_map import colorize_heat, overlay
from higan_dev.utils import load_image_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--image-idx", type=int, default=2)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--bounce", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="out/saliency_morph")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))
    target = load_image_tensor(img_paths[args.image_idx],
                               size=cfg.generator.resolution,
                               normalize="-1_1").to(G.device)
    res = encode_image(target, enc, G, w_avg=w_avg)
    wp_enc = res.wp                                        # (1, L, D)
    L, D = G.num_layers, G.w_dim

    distances = np.linspace(-args.delta, args.delta, args.frames).tolist()
    if args.bounce:
        distances = distances + distances[-2:0:-1]

    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr,
                          num_layers=L).to(G.device)
        b_layered = _layered_direction(b, L, D, G.device)

        frames: list[Image.Image] = []
        for dist in distances:
            wp_t = wp_enc + dist * b_layered.unsqueeze(0)
            # render image
            with torch.no_grad():
                img = G.synthesize(wp_t)
            img_u8 = G.to_uint8(img)[0]

            # compute local first-order saliency at this point
            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp_t + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(
                f,
                (torch.zeros(1, device=G.device),),
                (torch.ones(1, device=G.device),),
            )
            sal = dimg.abs().mean(dim=1).squeeze(0).cpu().numpy()
            sal = sal / max(sal.max(), 1e-8)

            sal_rgb = colorize_heat(sal.astype(np.float32))
            ov = overlay(img_u8, sal, alpha=0.55)
            frame = np.concatenate([img_u8, sal_rgb, ov], axis=1)
            frames.append(Image.fromarray(frame))

        out_webp = out / f"{attr}.webp"
        frames[0].save(
            out_webp, save_all=True, append_images=frames[1:],
            duration=int(1000 / args.fps), loop=0, quality=80, method=6,
        )
        # also save a 3-frame static strip (begin / mid / end)
        mid = len(frames) // 2
        strip = np.concatenate(
            [np.asarray(frames[0]),
             np.asarray(frames[mid]),
             np.asarray(frames[-1])], axis=0,
        )
        Image.fromarray(strip).save(out / f"{attr}_strip.png")
        print(f"{attr}: {len(frames)} frames -> {out_webp.name}")


if __name__ == "__main__":
    main()
