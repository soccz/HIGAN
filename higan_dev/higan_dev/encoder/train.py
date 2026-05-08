"""Synthetic-supervision training loop for the W+ encoder.

Each step:
    1. Sample latent code wp_gt ~ G's prior (via z -> mapping -> truncation).
    2. Generate target image x = G.synthesize(wp_gt). (Frozen, no grad through G.)
    3. Encoder predicts wp_pred = E(x).
    4. Reconstruct x_hat = G.synthesize(wp_pred).  (Now with grad through G.)
    5. Loss = lambda_w * MSE(wp_pred, wp_gt)
           + image-recon losses on (x_hat, x) (LPIPS / pixel L2 / VGG / TV)
    6. Update encoder params only.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ..config import Config
from ..generator import HiGANGenerator
from ..losses import ImageReconLoss
from ..utils import AverageMeter, ensure_dir, save_image, set_seed
from .model import WPlusEncoder, WPlusEncoderCfg


def _sample_targets(G: HiGANGenerator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (wp_gt, image) without tracking gradients through G's prior."""
    with torch.no_grad():
        wp_gt = G.sample_wp(batch)
        image = G.synthesize(wp_gt)
    return wp_gt, image


def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)
    out_dir = ensure_dir(Path(cfg.paths.out_dir) / "encoder_train")
    ckpt_dir = ensure_dir(out_dir / "ckpt")
    vis_dir = ensure_dir(out_dir / "vis")
    log_path = out_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)

    # ----- generator (frozen) -----
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    w_avg = G._net.truncation.w_avg.detach().clone().to(device)

    # ----- encoder -----
    enc = WPlusEncoder(WPlusEncoderCfg(
        backbone=cfg.encoder.backbone,
        num_layers=cfg.encoder.num_layers,
        latent_dim=cfg.encoder.latent_dim,
        pretrained=True,
    )).to(device)
    enc.train()

    # ----- losses -----
    img_loss = ImageReconLoss(
        weights={
            "pixel_l2": cfg.train.loss_weights.get("pixel_l2", 1.0),
            "lpips": cfg.train.loss_weights.get("lpips", 0.8),
            "perceptual": cfg.train.loss_weights.get("perceptual", 0.0),
            "tv": cfg.train.loss_weights.get("tv", 1e-4),
        },
        device=device,
    )
    w_mse_weight = cfg.train.loss_weights.get("w_mse", 1.0)

    # ----- optimizer -----
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)

    avg = {k: AverageMeter() for k in ("total", "w_mse", "pixel_l2", "lpips", "perceptual", "tv")}
    t0 = time.time()
    pbar = tqdm(range(1, cfg.train.num_iters + 1), ncols=100, dynamic_ncols=True)

    for it in pbar:
        # warmup lr
        if it <= cfg.train.warmup_iters:
            cur_lr = cfg.train.lr * it / cfg.train.warmup_iters
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

        wp_gt, target = _sample_targets(G, cfg.train.batch_size)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.train.amp):
            wp_pred = enc(target, w_avg=w_avg)
            recon = G.synthesize(wp_pred)
            w_mse = F.mse_loss(wp_pred, wp_gt) * w_mse_weight
            img_l, img_log = img_loss(recon, target)
            loss = w_mse + img_l

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
        scaler.step(opt)
        scaler.update()

        # ----- logging -----
        avg["total"].update(float(loss.item()))
        avg["w_mse"].update(float(w_mse.item()))
        for k in ("pixel_l2", "lpips", "perceptual", "tv"):
            if k in img_log:
                avg[k].update(img_log[k])

        if it % cfg.train.log_every == 0:
            row = {"it": it, **{k: m.avg for k, m in avg.items() if m.n > 0}}
            row["sec_per_it"] = (time.time() - t0) / it
            pbar.set_postfix({"total": f"{row['total']:.3f}",
                              "w_mse": f"{row['w_mse']:.3f}",
                              "lpips": f"{row.get('lpips', 0):.3f}"})
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            for m in avg.values():
                m.reset()

        # ----- visualisation -----
        if it % cfg.train.vis_every == 0:
            with torch.no_grad():
                vis_pairs = []
                for i in range(min(4, cfg.train.batch_size)):
                    pair = torch.cat([target[i:i+1], recon[i:i+1]], dim=3)  # side-by-side
                    vis_pairs.append(pair)
                grid = torch.cat(vis_pairs, dim=2)  # stack vertically
            save_image(G.to_uint8(grid)[0], vis_dir / f"it{it:06d}.png")

        # ----- checkpoint -----
        if it % cfg.train.ckpt_every == 0 or it == cfg.train.num_iters:
            torch.save(
                {
                    "iter": it,
                    "encoder_cfg": asdict(WPlusEncoderCfg(
                        backbone=cfg.encoder.backbone,
                        num_layers=cfg.encoder.num_layers,
                        latent_dim=cfg.encoder.latent_dim,
                        pretrained=False,
                    )),
                    "model": enc.state_dict(),
                    "w_avg": w_avg.cpu(),
                },
                ckpt_dir / f"enc_{it:06d}.pt",
            )

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min")
