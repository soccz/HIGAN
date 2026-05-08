"""Pairwise disentanglement / interference between attribute boundaries.

For each pair of HiGAN attributes (a, b), we ask:
    "Does perturbing along boundary a also move pixels that boundary b would
     move?"

Two metrics:
    1. saliency_corr[a,b] = pixel-wise Pearson correlation between the
       averaged grad-saliency maps of a and b. High = entangled.
    2. signed_corr[a,b]   = same but on signed saliency (∂I/∂α with sign).
       Captures whether they push pixels in the *same* or opposite directions.

Returns a `(num_attrs, num_attrs)` matrix that we render as a heatmap.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from ..generator import HiGANGenerator
from ..manipulate import Boundary
from .grad_saliency import compute_grad_saliency


def compute_disentanglement(
    generator: HiGANGenerator,
    boundaries: list[Boundary],
    *,
    num_samples: int = 32,
    micro_batch: int = 4,
) -> dict[str, np.ndarray]:
    """Compute attribute × attribute saliency correlation."""
    # Use the SAME base latents for every attribute so correlations are fair.
    gen = torch.Generator(device=generator.device).manual_seed(7)
    base_wp = generator.sample_wp(num_samples, generator=gen)

    abs_maps: list[np.ndarray] = []
    sgn_maps: list[np.ndarray] = []
    names: list[str] = []
    for b in boundaries:
        r = compute_grad_saliency(
            generator, b, num_samples=num_samples, micro_batch=micro_batch,
            base_wp=base_wp, keep_per_sample=0,
        )
        abs_maps.append(r.abs_saliency.flatten())
        sgn_maps.append(r.signed_saliency.flatten())
        names.append(b.name)

    n = len(boundaries)
    abs_corr = np.zeros((n, n), dtype=np.float32)
    sgn_corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            abs_corr[i, j] = float(np.corrcoef(abs_maps[i], abs_maps[j])[0, 1])
            sgn_corr[i, j] = float(np.corrcoef(sgn_maps[i], sgn_maps[j])[0, 1])
    return {"abs_corr": abs_corr, "signed_corr": sgn_corr, "names": names}


def render_matrix(matrix: np.ndarray, names: list[str], *, title: str = "",
                  vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """Render a labelled (n, n) matrix as RGB uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            v = matrix[i, j]
            color = "white" if abs(v) > 0.55 else "#1c1917"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    if title:
        ax.set_title(title, fontsize=11, pad=10, color="#1c1917", weight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr
