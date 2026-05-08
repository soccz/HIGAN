"""Latent-perturbation pixel-attribution heatmaps for HiGAN attributes.

Idea: for a given semantic boundary direction `b`, perturb the latent code
along ±delta and measure how the rendered image changes spatially. Averaging
over many samples gives a heatmap of "where this attribute lives in pixel
space". This is the simplest classifier-free CAM-like analysis we can run on
the HiGAN bedroom model.

We compute three signals per attribute:
    1. abs_diff   = mean_n |I(z_n + delta b) - I(z_n - delta b)| pooled over RGB.
    2. signed_diff = mean_n  (I(z_n + delta b) - I(z_n - delta b)).mean(rgb)
    3. variance   = var across the perturbation sweep.
The signed_diff is colour-aware: red regions get brighter when the attribute
turns on, etc. The abs_diff is the basic spatial saliency.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..generator import HiGANGenerator
from ..manipulate import Boundary, manipulate_wp


@dataclass
class DiffMapResult:
    abs_diff: np.ndarray      # (H, W) in [0, 1]
    signed_diff: np.ndarray   # (H, W) in [-1, 1]
    variance: np.ndarray      # (H, W) in [0, 1]
    mean_image: np.ndarray    # (H, W, 3) uint8 — average rendering for context
    delta: float
    num_samples: int
    boundary_name: str


@torch.no_grad()
def compute_diff_map(
    generator: HiGANGenerator,
    boundary: Boundary,
    *,
    delta: float = 1.5,
    num_samples: int = 32,
    micro_batch: int = 4,
    base_wp: torch.Tensor | None = None,
) -> DiffMapResult:
    """Run the perturbation sweep and accumulate spatial signals.

    Args:
        base_wp: optional (num_samples, num_layers, latent_dim) anchor codes.
            If None, samples random latents from the prior.
    """
    device = generator.device
    boundary = boundary.to(device)

    if base_wp is None:
        gen = torch.Generator(device=device).manual_seed(0)
        base_wp = generator.sample_wp(num_samples, generator=gen)
    else:
        base_wp = base_wp.to(device)
        num_samples = base_wp.shape[0]

    H = W = generator.resolution
    abs_diff = torch.zeros(H, W, device=device)
    signed_diff = torch.zeros(H, W, device=device)
    var_acc = torch.zeros(H, W, device=device)
    mean_img = torch.zeros(3, H, W, device=device)

    dists = [-delta, 0.0, delta]
    n_done = 0
    for start in tqdm(range(0, num_samples, micro_batch), desc=f"diff[{boundary.name}]",
                      ncols=84):
        chunk = base_wp[start:start + micro_batch]
        manip = manipulate_wp(chunk, boundary, distances=dists)
        # manip: (b, 3, L, D)  -> reshape to (b*3, L, D)
        b = manip.shape[0]
        flat = manip.reshape(b * 3, manip.shape[2], manip.shape[3])
        imgs = generator.synthesize(flat)            # (b*3, 3, H, W) in ~[-1,1]
        imgs = imgs.clamp(-1, 1)
        imgs = (imgs + 1) / 2                         # [0,1]
        imgs = imgs.view(b, 3, 3, H, W)               # (b, K=3 dist, C=3, H, W)

        diff = (imgs[:, 2] - imgs[:, 0])              # (b, 3, H, W) signed
        abs_diff += diff.abs().mean(1).sum(0)
        signed_diff += diff.mean(1).sum(0)            # mean over rgb
        var_acc += imgs.var(dim=1).mean(1).sum(0)     # var across distances, mean rgb
        mean_img += imgs[:, 1].sum(0)                  # 0-distance image avg
        n_done += b

    abs_diff = (abs_diff / n_done).clamp_min(0).cpu().numpy()
    signed_diff = (signed_diff / n_done).cpu().numpy()
    var_map = (var_acc / n_done).clamp_min(0).cpu().numpy()
    mean_img = (mean_img / n_done).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    mean_img_u8 = (mean_img * 255).astype(np.uint8)

    # normalise heatmaps to [0,1] by their own max for visualisation convenience
    if abs_diff.max() > 1e-8:
        abs_diff_n = abs_diff / abs_diff.max()
    else:
        abs_diff_n = abs_diff
    if var_map.max() > 1e-8:
        var_n = var_map / var_map.max()
    else:
        var_n = var_map
    sgn_max = max(abs(signed_diff.max()), abs(signed_diff.min()), 1e-8)
    signed_n = signed_diff / sgn_max

    return DiffMapResult(
        abs_diff=abs_diff_n.astype(np.float32),
        signed_diff=signed_n.astype(np.float32),
        variance=var_n.astype(np.float32),
        mean_image=mean_img_u8,
        delta=float(delta),
        num_samples=int(n_done),
        boundary_name=boundary.name,
    )


def colorize_heat(h: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """h: (H,W) in [0,1] -> (H,W,3) uint8 RGB using matplotlib colormap."""
    import matplotlib.cm as cm
    import matplotlib.pyplot as _plt  # noqa: F401  (ensures registry loaded)
    rgba = cm.get_cmap(cmap)(np.clip(h, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(image_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.5,
            cmap: str = "magma") -> np.ndarray:
    heat_rgb = colorize_heat(heat, cmap=cmap)
    return ((1 - alpha) * image_u8 + alpha * heat_rgb).astype(np.uint8)


def colorize_signed(h: np.ndarray) -> np.ndarray:
    """h: (H,W) in [-1,1] -> (H,W,3) uint8 with diverging RdBu colormap."""
    import matplotlib.cm as cm
    rgba = cm.get_cmap("RdBu_r")((h + 1) / 2)
    return (rgba[..., :3] * 255).astype(np.uint8)
