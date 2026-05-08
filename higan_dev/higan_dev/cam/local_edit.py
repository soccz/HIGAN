"""Saliency-guided local editing.

Workflow:
    1. compute grad-saliency map M(x, b) for a single (image, boundary).
    2. smooth + threshold M to a soft mask m ∈ [0, 1].
    3. global edit  : I_pos = G(wp + α b)
       local  edit  : I_local = I_pos · m + I_orig · (1 - m)

So the attribute change applies only to spatially salient pixels, leaving
everything else (e.g., bedspread when toggling lamp) untouched.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

from ..generator import HiGANGenerator
from ..manipulate import Boundary, manipulate_wp
from .grad_saliency import _layered_direction


def _gaussian_blur_2d(x: torch.Tensor, sigma: float = 4.0,
                      radius: int = 12) -> torch.Tensor:
    """x: (B, C, H, W) -> blurred."""
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    kernel = (g1[:, None] * g1[None, :]).expand(C, 1, -1, -1)
    return F.conv2d(x, kernel, padding=radius, groups=C)


@dataclass
class LocalEditResult:
    original: np.ndarray         # (H, W, 3) uint8
    saliency: np.ndarray         # (H, W) in [0, 1]
    mask: np.ndarray             # (H, W) in [0, 1]
    global_edit: np.ndarray      # (H, W, 3) uint8
    local_edit: np.ndarray       # (H, W, 3) uint8


@torch.no_grad()
def _to_uint8(img: torch.Tensor) -> np.ndarray:
    x = (img.clamp(-1, 1) + 1) / 2
    return (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def saliency_guided_edit(
    generator: HiGANGenerator,
    wp: torch.Tensor,             # (1, L, D)
    boundary: Boundary,
    *,
    alpha: float = 3.0,
    sigma: float = 4.0,
    threshold: float = 0.25,
) -> LocalEditResult:
    device = generator.device
    boundary = boundary.to(device)

    # 1) saliency for THIS specific wp
    b_layered = _layered_direction(boundary, generator.num_layers,
                                   generator.w_dim, device)

    def f(a: torch.Tensor) -> torch.Tensor:
        return generator.synthesize(wp + a.view(1, 1, 1) * b_layered.unsqueeze(0))
    img0, dimg = jvp(f, (torch.zeros(1, device=device),),
                     (torch.ones(1, device=device),))
    sal = dimg.abs().mean(dim=1)                      # (1, H, W)
    sal = sal / sal.amax().clamp_min(1e-8)            # normalise to [0, 1]

    # 2) build soft mask from saliency
    mask = sal.unsqueeze(1)                           # (1, 1, H, W)
    if sigma > 0:
        mask = _gaussian_blur_2d(mask, sigma=sigma, radius=int(sigma * 3))
        mask = mask / mask.amax().clamp_min(1e-8)
    mask = ((mask - threshold) / (1.0 - threshold)).clamp(0, 1)

    # 3) global edit
    edit_pos = manipulate_wp(wp, boundary, distances=[alpha])           # (1, 1, L, D)
    edit_pos = edit_pos.reshape(1, generator.num_layers, generator.w_dim)
    img_global = generator.synthesize(edit_pos)

    # 4) local edit
    img_local = img_global * mask + img0 * (1 - mask)

    return LocalEditResult(
        original=_to_uint8(img0)[0],
        saliency=sal.squeeze(0).detach().cpu().numpy(),
        mask=mask.squeeze().detach().cpu().numpy(),
        global_edit=_to_uint8(img_global)[0],
        local_edit=_to_uint8(img_local)[0],
    )
