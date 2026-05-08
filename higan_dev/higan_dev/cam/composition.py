"""Compositional editing analysis.

Question: when we edit along  α·b₁ + β·b₂  simultaneously, is the resulting
saliency the *sum* of the individual saliencies, or do interference effects
appear?

Method:
    sal_a       = | ∂I/∂α at α=0,β=0 |          (direction a alone)
    sal_b       = | ∂I/∂β at α=0,β=0 |          (direction b alone)
    sal_sum     = | ∂I/∂γ at γ=0     | where γ = α=β = γ shared scalar
                                                (combined direction a+b)

If linear: sal_sum ≈ sal_a + sal_b (up to sign of cross terms).
If non-linear interference: sal_sum diverges from the sum.

We measure pixel-wise correlation between sal_sum and (sal_a + sal_b).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import jvp

from ..generator import HiGANGenerator
from ..manipulate import Boundary
from .grad_saliency import _layered_direction


@dataclass
class CompositionResult:
    sal_a: np.ndarray         # (H, W) in [0, 1]
    sal_b: np.ndarray
    sal_sum: np.ndarray
    expected_sum: np.ndarray  # sal_a + sal_b (re-normalised)
    corr: float               # pixel-wise correlation between sal_sum and expected
    name_a: str
    name_b: str


def compositional_saliency(
    generator: HiGANGenerator,
    boundary_a: Boundary,
    boundary_b: Boundary,
    *,
    num_samples: int = 32,
    micro_batch: int = 4,
) -> CompositionResult:
    device = generator.device
    boundary_a = boundary_a.to(device)
    boundary_b = boundary_b.to(device)

    L, D = generator.num_layers, generator.w_dim
    b_a = _layered_direction(boundary_a, L, D, device)
    b_b = _layered_direction(boundary_b, L, D, device)
    b_sum = b_a + b_b

    gen = torch.Generator(device=device).manual_seed(13)
    base_wp = generator.sample_wp(num_samples, generator=gen)

    H = W = generator.resolution
    acc_a = torch.zeros(H, W, device=device)
    acc_b = torch.zeros(H, W, device=device)
    acc_sum = torch.zeros(H, W, device=device)
    n_done = 0

    for s in range(0, num_samples, micro_batch):
        wp = base_wp[s:s + micro_batch].detach()
        B = wp.shape[0]
        a0 = torch.zeros(B, device=device)
        ones = torch.ones(B, device=device)

        for tag, b_used, acc in [("a", b_a, acc_a),
                                  ("b", b_b, acc_b),
                                  ("sum", b_sum, acc_sum)]:
            def f(alpha):
                return generator.synthesize(wp + alpha.view(B, 1, 1) * b_used.unsqueeze(0))
            _, dimg = jvp(f, (a0,), (ones,))
            acc.add_(dimg.abs().mean(dim=1).sum(dim=0))
        n_done += B

    def _norm(t):
        a = (t / n_done).cpu().numpy()
        m = a.max()
        return (a / m).astype(np.float32) if m > 1e-8 else a.astype(np.float32)

    sal_a = _norm(acc_a)
    sal_b = _norm(acc_b)
    sal_sum = _norm(acc_sum)
    expected = (sal_a + sal_b)
    expected = expected / (expected.max() + 1e-8)

    corr = float(np.corrcoef(sal_sum.flatten(), expected.flatten())[0, 1])

    return CompositionResult(
        sal_a=sal_a, sal_b=sal_b, sal_sum=sal_sum, expected_sum=expected,
        corr=corr, name_a=boundary_a.name, name_b=boundary_b.name,
    )
