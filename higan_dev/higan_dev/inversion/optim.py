"""Optimization-based GAN inversion in W+ space.

Unlike the original notebook, gradients now flow correctly from the loss
through the generator into `wp`, so the optimizer actually moves the latent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math

import torch
from tqdm import tqdm

from ..generator import HiGANGenerator
from ..losses import ImageReconLoss


def _lr_schedule(t: float, *, ramp_up: float = 0.05, ramp_down: float = 0.25) -> float:
    """Standard StyleGAN inversion lr multiplier in [0,1].
    t in [0, 1]. Ramps up over `ramp_up` fraction, holds, then ramps down.
    """
    lr_ramp = min(1.0, (1.0 - t) / ramp_down)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1.0, t / ramp_up)
    return lr_ramp


@dataclass
class InvertResult:
    wp: torch.Tensor          # (1, num_layers, w_dim) best latent
    image: torch.Tensor       # (1, 3, H, W) in [-1, 1]
    loss: float
    history: list[dict]       # per-step loss components for the best run


def _init_wp(generator: HiGANGenerator, mode: str, n: int, std: float = 0.1) -> torch.Tensor:
    if mode == "random":
        return torch.randn(n, generator.num_layers, generator.w_dim,
                           device=generator.device) * std
    if mode == "w_avg":
        # use the StyleGAN running average of w as init, broadcast across layers
        w_avg = generator._net.truncation.w_avg.detach().clone()
        if w_avg.dim() == 1:
            w_avg = w_avg.unsqueeze(0)
        return w_avg.unsqueeze(1).repeat(n, generator.num_layers, 1)
    raise ValueError(f"unknown init mode: {mode}")


def invert_image(
    target: torch.Tensor,                 # (1,3,H,W) in [-1,1]
    generator: HiGANGenerator,
    *,
    num_steps: int = 1500,
    lr: float = 0.01,
    num_inits: int = 1,
    init_mode: str = "w_avg",             # 'w_avg' or 'random'
    loss_weights: Optional[dict] = None,
    log_every: int = 100,
    progress: bool = True,
) -> InvertResult:
    """Invert a single image into W+ space."""
    assert target.dim() == 4 and target.shape[0] == 1, "target must be (1,3,H,W)"
    target = target.to(generator.device)

    weights = dict(loss_weights or {
        "pixel_l2": 1.0, "lpips": 0.8, "tv": 1e-4,
    })
    criterion = ImageReconLoss(weights, device=generator.device)

    best: Optional[InvertResult] = None

    for init_idx in range(num_inits):
        wp = _init_wp(generator, init_mode, n=1).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([wp], lr=lr)

        history: list[dict] = []
        rng = range(num_steps)
        if progress:
            rng = tqdm(rng, desc=f"init {init_idx + 1}/{num_inits}", ncols=88)

        for step in rng:
            t = step / max(1, num_steps - 1)
            cur_lr = lr * _lr_schedule(t)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            img = generator.synthesize(wp)
            loss, log = criterion(img, target)
            loss.backward()
            opt.step()
            log["step"] = step
            log["lr"] = cur_lr
            history.append(log)
            if progress and step % log_every == 0:
                tqdm.write(
                    f"  step {step:4d}  lr={cur_lr:.4f}  total={log['total']:.4f}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in log.items()
                                 if k not in ("total", "step", "lr"))
                )

        with torch.no_grad():
            final_img = generator.synthesize(wp)
            final_loss, _ = criterion(final_img, target)
        result = InvertResult(
            wp=wp.detach().clone(),
            image=final_img.detach().clone(),
            loss=float(final_loss.item()),
            history=history,
        )
        if best is None or result.loss < best.loss:
            best = result

    assert best is not None
    return best
