"""Track 4 — FFHQ W+ encoder training (synthetic supervision).

Re-uses the bedroom WPlusEncoder + ImageReconLoss with num_layers=18 for
the StyleGAN1-FFHQ generator. Differences vs bedroom training:
  - 1024² output → recon LPIPS at 256² downscaled (memory)
  - batch 1 (FFHQ-1024 + ResNet-50 + LPIPS all in 8 GB)
  - 40 k iterations total, checkpoints at 1k/5k/10k/20k/40k
    (matches bedroom's evaluation schedule for the C5 comparison)

See designs/04_c5_ffhq_encoder.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from domains.ffhq.generator import FFHQGenerator           # noqa: E402
from higan_dev.encoder.model import WPlusEncoder, WPlusEncoderCfg  # noqa: E402
from higan_dev.losses import ImageReconLoss                # noqa: E402


def _sample_targets(G: FFHQGenerator, batch: int):
    with torch.no_grad():
        wp_gt = G.sample_wp(batch)
        image = G.synthesize(wp_gt)
    return wp_gt, image


def downscale(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear",
                         align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/out/ffhq_c5")
    ap.add_argument("--num-iters", type=int, default=40000)
    ap.add_argument("--ckpt-iters", nargs="+", type=int,
                    default=[1000, 5000, 10000, 20000, 40000])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-mse-weight", type=float, default=0.1)
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=0.8)
    ap.add_argument("--lpips-size", type=int, default=256,
                    help="resolution for LPIPS computation (downscaled)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "vis").mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    if args.resume is None:
        log_path.unlink(missing_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator...")
    G = FFHQGenerator()
    device = G.device

    enc_cfg = WPlusEncoderCfg(
        backbone=args.backbone, num_layers=G.num_layers,
        latent_dim=G.w_dim, pretrained=True,
    )
    enc = WPlusEncoder(enc_cfg).to(device)

    start_iter = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        enc.load_state_dict(state["model"])
        start_iter = int(state.get("iter", 0))
        print(f"[resume] from {args.resume}, start_iter={start_iter}")

    enc.train()

    img_loss = ImageReconLoss(
        weights={
            "pixel_l2": args.pixel_weight,
            "lpips": args.lpips_weight,
            "perceptual": 0.0,
            "tv": 1e-4,
        },
        device=device,
    )

    w_avg = G._net.truncation.w_avg.detach().clone().to(device) \
        if hasattr(G._net, "truncation") else None

    opt = torch.optim.Adam(enc.parameters(), lr=args.lr,
                            betas=(0.9, 0.999))

    print(f"[{time.strftime('%H:%M:%S')}] start training, "
          f"iters {start_iter}..{args.num_iters}")
    t0 = time.time()
    for it in range(start_iter + 1, args.num_iters + 1):
        wp_gt, target = _sample_targets(G, args.batch)

        opt.zero_grad(set_to_none=True)
        # encoder works at 256² downscaled images for ResNet input
        target_small = downscale(target, 256)
        wp_pred = enc(target_small, w_avg=w_avg)
        recon = G.synthesize(wp_pred)

        # losses
        w_mse = F.mse_loss(wp_pred, wp_gt) * args.w_mse_weight
        # downscale recon + target for LPIPS to save memory
        recon_s = downscale(recon, args.lpips_size)
        target_s = downscale(target, args.lpips_size)
        img_l, img_log = img_loss(recon_s, target_s)
        loss = w_mse + img_l
        loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
        opt.step()

        if it % args.log_every == 0:
            row = {"it": it, "total": float(loss.item()),
                   "w_mse": float(w_mse.item()),
                   "sec_per_it": (time.time() - t0) / max(1, it - start_iter)}
            row.update({k: v for k, v in img_log.items() if isinstance(v, float)})
            with open(log_path, "a") as fp:
                fp.write(json.dumps(row) + "\n")
            print(f"  it {it:6d}  loss {loss.item():.4f}  "
                  f"w_mse {w_mse.item():.4f}  "
                  f"lpips {img_log.get('lpips', 0):.4f}  "
                  f"({row['sec_per_it']:.2f}s/it)")

        if it in args.ckpt_iters or it % args.ckpt_every == 0:
            ckpt = {"iter": it,
                    "encoder_cfg": asdict(enc_cfg),
                    "model": enc.state_dict()}
            torch.save(ckpt, out / "ckpt" / f"enc_{it:06d}.pt")
            print(f"  saved ckpt enc_{it:06d}.pt")

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. "
          f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
