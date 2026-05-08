"""Optimization-based inversion CLI.

Examples:
    # Invert a real image
    python scripts/02_invert_optim.py --image data/my_room.png --steps 800 \
        --out out/inv_my_room

    # Sanity test: invert an image we generated ourselves (should converge to ~0 loss)
    python scripts/02_invert_optim.py --self-test --steps 300 --out out/inv_selftest
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.optim import invert_image
from higan_dev.utils import load_image_tensor, save_image, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", type=str, help="path to target image (RGB)")
    ap.add_argument("--self-test", action="store_true",
                    help="generate a target with the GAN itself and try to invert it")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-inits", type=int, default=1)
    ap.add_argument("--init-mode", choices=["w_avg", "random"], default="w_avg")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    if args.self_test:
        gen = torch.Generator(device=G.device).manual_seed(args.seed)
        with torch.no_grad():
            wp_true = G.sample_wp(1, generator=gen)
            target = G.synthesize(wp_true)
        save_image(G.to_uint8(target)[0], out_dir / "target.png")
    else:
        if not args.image:
            ap.error("--image is required when --self-test is not set")
        target = load_image_tensor(args.image, size=cfg.generator.resolution, normalize="-1_1") \
            .to(G.device)
        save_image(G.to_uint8(target)[0], out_dir / "target.png")

    steps = args.steps or cfg.inversion_optim.num_steps
    lr = args.lr or cfg.inversion_optim.lr

    result = invert_image(
        target=target,
        generator=G,
        num_steps=steps,
        lr=lr,
        num_inits=args.num_inits,
        init_mode=args.init_mode,
        loss_weights={
            "pixel_l2": 1.0,
            "lpips": 0.8,
            "tv": 1e-4,
        },
        log_every=max(1, steps // 10),
    )

    save_image(G.to_uint8(result.image)[0], out_dir / "recon.png")
    np.save(out_dir / "wp.npy", result.wp.cpu().numpy())
    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {"final_loss": result.loss, "history": result.history[::max(1, steps // 50)]},
            f, indent=2,
        )

    if args.self_test:
        wp_err = (result.wp - wp_true).norm().item()
        print(f"\nself-test: |wp_recon - wp_true| = {wp_err:.4f}")

    print(f"\ndone. final loss = {result.loss:.4f}")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
