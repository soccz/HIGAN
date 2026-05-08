"""Encoder attention: for which input pixels does E(x)·b respond?

The generator-side `grad_saliency.py` answered "where does this latent
direction *act* in the output". The mirror question on the encoder side
is "where does the encoder *look* in the input to estimate the projection
of wp onto a given direction".

We compute  s(x) = ⟨E(x) restricted to manipulate_layers,  b̂⟩   (a scalar)
and then  saliency(x) = | ∂s / ∂x |  averaged over RGB.

Reverse-mode autograd is the right tool here: scalar output, many-input.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from ..encoder.model import WPlusEncoder
from ..manipulate import Boundary


@dataclass
class EncoderAttentionResult:
    saliency: np.ndarray          # (H, W) in [0, 1]
    score: float                  # current value of E(x) · b


def encoder_attention(
    encoder: WPlusEncoder,
    image: torch.Tensor,          # (1, 3, H, W) in [-1, 1]
    boundary: Boundary,
    *,
    w_avg: torch.Tensor | None = None,
) -> EncoderAttentionResult:
    """Reverse-mode saliency of the encoder's projection onto boundary."""
    device = image.device
    boundary = boundary.to(device)
    image = image.detach().clone().requires_grad_(True)

    wp = encoder(image, w_avg=w_avg)               # (1, L, D)
    # restrict to manipulate_layers and project onto boundary direction
    layers = boundary.manipulate_layers
    selected = wp[:, layers, :]                    # (1, |layers|, D)
    score = (selected * boundary.direction.view(1, 1, -1)).sum()
    grads = torch.autograd.grad(score, image)[0]   # (1, 3, H, W)

    sal = grads.abs().mean(dim=1).squeeze(0)       # (H, W)
    # percentile-based normalisation: clip top 1% to be max (= 1.0).
    # Robust against single hot pixels dominating the heatmap.
    flat = sal.flatten()
    p99 = torch.quantile(flat, 0.99)
    p05 = torch.quantile(flat, 0.05)
    sal_n = ((sal - p05) / (p99 - p05).clamp_min(1e-8)).clamp(0, 1)
    return EncoderAttentionResult(
        saliency=sal_n.detach().cpu().numpy().astype(np.float32),
        score=float(score.item()),
    )
