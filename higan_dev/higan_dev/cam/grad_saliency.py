"""Gradient-based pixel saliency for HiGAN attribute boundaries.

Where `diff_map.py` measures `I(wp + δb) - I(wp - δb)` (forward-only finite
difference), this module computes the *exact first-order* sensitivity by
running a Jacobian-vector product through the generator:

    α : R          (scalar perturbation along boundary direction)
    wp(α) = wp + α · b̃            ( b̃ = b placed only on manipulate_layers )
    I(α)  = G(wp(α))               ( our differentiable HiGAN wrapper )
    saliency[h,w]  =  | ∂I[c,h,w] / ∂α |  averaged over RGB

Because α is a scalar per sample (1-input) but the output has 3·H·W channels
(many-output), forward-mode autodiff is the efficient direction here:
`torch.func.jvp` returns the full pixel sensitivity tensor in a single
forward-with-tangent pass — equivalent to a backward call for any single
linear functional of the image, but tracking every pixel simultaneously.

Mechanistically this is what makes the analysis "Grad-CAM-spirit": gradients
flow through the entire generator, just as Grad-CAM flows gradients through
a classifier — except here the score function is a direction in latent space
rather than a class logit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.func import jvp
from tqdm import tqdm

from ..generator import HiGANGenerator
from ..manipulate import Boundary


@dataclass
class GradSaliencyResult:
    abs_saliency: np.ndarray         # (H, W) in [0, 1] — averaged
    signed_saliency: np.ndarray      # (H, W) in [-1, 1] — averaged
    mean_image: np.ndarray           # (H, W, 3) uint8 — average rendering
    per_sample_abs: np.ndarray       # (K, H, W) per-sample saliency, normalised individually
    per_sample_image: np.ndarray     # (K, H, W, 3) uint8 — corresponding base image
    num_samples: int
    boundary_name: str


def _layered_direction(boundary: Boundary, num_layers: int, latent_dim: int,
                       device: torch.device,
                       only_layer: Optional[int] = None) -> torch.Tensor:
    """Return (L, D) tensor with boundary direction on manipulate_layers, zero elsewhere.

    If `only_layer` is given, place the direction *only* on that single layer
    (useful for per-layer decomposition).
    """
    b = torch.zeros(num_layers, latent_dim, device=device)
    if only_layer is not None:
        if 0 <= only_layer < num_layers:
            b[only_layer] = boundary.direction.to(device)
        return b
    for li in boundary.manipulate_layers:
        if 0 <= li < num_layers:
            b[li] = boundary.direction.to(device)
    return b


def compute_per_layer_saliency(
    generator: HiGANGenerator,
    boundary: Boundary,
    *,
    num_samples: int = 32,
    micro_batch: int = 4,
    base_wp: Optional[torch.Tensor] = None,
) -> dict[int, np.ndarray]:
    """Decompose the saliency by manipulate-layer.

    For each layer in `boundary.manipulate_layers`, compute the JVP-based
    saliency that would result from perturbing *only that layer* of wp along
    the boundary direction. Returns {layer_idx: (H, W) abs-saliency in [0, 1]}.
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
    out: dict[int, np.ndarray] = {}

    for li in boundary.manipulate_layers:
        b_layered = _layered_direction(boundary, generator.num_layers,
                                       generator.w_dim, device, only_layer=li)
        acc = torch.zeros(H, W, device=device)
        n_done = 0
        for start in range(0, num_samples, micro_batch):
            wp_chunk = base_wp[start:start + micro_batch].detach()
            B = wp_chunk.shape[0]

            def f(alpha: torch.Tensor) -> torch.Tensor:
                wp_p = wp_chunk + alpha.view(B, 1, 1) * b_layered.unsqueeze(0)
                return generator.synthesize(wp_p)

            alpha0 = torch.zeros(B, device=device)
            tangent = torch.ones(B, device=device)
            _, dimg = jvp(f, (alpha0,), (tangent,))
            acc += dimg.abs().mean(dim=1).sum(dim=0)
            n_done += B
        sal = (acc / n_done).cpu().numpy()
        if sal.max() > 1e-8:
            sal = sal / sal.max()
        out[li] = sal.astype(np.float32)
    return out


def compute_grad_saliency(
    generator: HiGANGenerator,
    boundary: Boundary,
    *,
    num_samples: int = 64,
    micro_batch: int = 4,
    base_wp: Optional[torch.Tensor] = None,
    keep_per_sample: int = 4,
) -> GradSaliencyResult:
    """Run JVP through the generator and accumulate per-pixel sensitivity."""
    device = generator.device
    boundary = boundary.to(device)

    if base_wp is None:
        gen = torch.Generator(device=device).manual_seed(0)
        base_wp = generator.sample_wp(num_samples, generator=gen)
    else:
        base_wp = base_wp.to(device)
        num_samples = base_wp.shape[0]

    b_layered = _layered_direction(boundary, generator.num_layers,
                                   generator.w_dim, device)

    H = W = generator.resolution
    abs_acc = torch.zeros(H, W, device=device)
    signed_acc = torch.zeros(H, W, device=device)
    mean_img = torch.zeros(3, H, W, device=device)
    n_done = 0
    per_sample_abs: list[np.ndarray] = []
    per_sample_img: list[np.ndarray] = []

    for start in tqdm(range(0, num_samples, micro_batch),
                      desc=f"grad[{boundary.name}]", ncols=84):
        wp_chunk = base_wp[start:start + micro_batch].detach()
        B = wp_chunk.shape[0]

        def f(alpha: torch.Tensor) -> torch.Tensor:
            # alpha: (B,)  -> wp_p: (B, L, D)
            wp_p = wp_chunk + alpha.view(B, 1, 1) * b_layered.unsqueeze(0)
            return generator.synthesize(wp_p)

        alpha0 = torch.zeros(B, device=device)
        tangent = torch.ones(B, device=device)
        img0, dimg_dalpha = jvp(f, (alpha0,), (tangent,))
        # img0: (B, 3, H, W)  in roughly [-1, 1]
        # dimg_dalpha: (B, 3, H, W) — first-order pixel sensitivity to α

        sal_b = dimg_dalpha.abs().mean(dim=1)        # (B, H, W)
        abs_acc += sal_b.sum(dim=0)
        signed_acc += dimg_dalpha.mean(dim=1).sum(dim=0)
        mean_img += ((img0.clamp(-1, 1) + 1) / 2).sum(dim=0)

        # keep per-sample saliency for first K samples for visualisation
        if len(per_sample_abs) < keep_per_sample:
            need = keep_per_sample - len(per_sample_abs)
            take = min(need, B)
            sal_np = sal_b[:take].detach().cpu().numpy()
            for k in range(take):
                m = sal_np[k].max()
                per_sample_abs.append(sal_np[k] / (m if m > 1e-8 else 1.0))
            img_np = ((img0[:take].clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
            per_sample_img.extend(img_np.detach().cpu().numpy().astype(np.uint8))

        n_done += B

    abs_map = (abs_acc / n_done).cpu().numpy()
    signed_map = (signed_acc / n_done).cpu().numpy()
    mean_u8 = ((mean_img / n_done).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
               ).astype(np.uint8)

    if abs_map.max() > 1e-8:
        abs_map_n = abs_map / abs_map.max()
    else:
        abs_map_n = abs_map
    sgn_max = max(abs(signed_map.max()), abs(signed_map.min()), 1e-8)
    signed_n = signed_map / sgn_max

    return GradSaliencyResult(
        abs_saliency=abs_map_n.astype(np.float32),
        signed_saliency=signed_n.astype(np.float32),
        mean_image=mean_u8,
        per_sample_abs=np.stack(per_sample_abs).astype(np.float32) if per_sample_abs
            else np.zeros((0, H, W), dtype=np.float32),
        per_sample_image=np.stack(per_sample_img).astype(np.uint8) if per_sample_img
            else np.zeros((0, H, W, 3), dtype=np.uint8),
        num_samples=int(n_done),
        boundary_name=boundary.name,
    )
