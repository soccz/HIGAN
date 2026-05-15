"""SeFa baseline (Shen & Zhou, CVPR 2021).

Closed-form decomposition of the first affine modulation layer of
StyleGAN: the weights W̃ ∈ R^{d_out × d_in} map a W-space vector to
per-layer modulation coefficients. Top eigenvectors of W̃^T W̃ are
"semantic directions" — they maximally affect the modulation.

Reference: https://github.com/genforce/sefa

For HiGAN / InterFaceGAN style generators, the per-layer style is
applied as `style * x + bias`, where `style` is produced by a dense
layer from the layer's wp slice. We concatenate the style weights
across all layers (or pick a subset) and do SVD on the combined matrix.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SeFaResult:
    components: np.ndarray   # (k, w_dim) top-k semantic directions
    eigenvalues: np.ndarray  # (k,) sorted descending


def find_style_weights(generator) -> list[np.ndarray]:
    """Heuristically find the StyleGAN style-projection weights.

    These are dense layers that map w (or wp[:, l]) to per-feature
    modulation coefficients. In genforce StyleGAN1, each
    `layer<i>.epilogue.style_mod.dense.fc.weight` is one such matrix.
    """
    net = generator._net
    weights = []
    for name, module in net.named_modules():
        # genforce StyleGAN1 path: synthesis.layerN.epilogue.style_mod.dense.fc
        if isinstance(module, nn.Linear) and (
            "style_mod" in name or "style" in name.lower()
        ):
            w = module.weight.detach().cpu().numpy().astype(np.float32)
            weights.append(w)
    return weights


def sefa_directions(
    generator,
    n_components: int = 8,
    use_layers: list[int] | None = None,
) -> SeFaResult:
    """Stack style-projection weights, SVD, return top eigenvectors."""
    weights = find_style_weights(generator)
    if not weights:
        raise RuntimeError("No style projection weights found in generator. "
                           "Try adapting `find_style_weights` for this model.")
    print(f"  found {len(weights)} style projection matrices")
    if use_layers is not None:
        weights = [weights[i] for i in use_layers if i < len(weights)]
    # Stack along rows (concatenate output dims) - SeFa's "merged" decomposition
    W = np.concatenate(weights, axis=0)        # (sum_l d_out_l, w_dim)
    print(f"  merged weight matrix: {W.shape}, computing W^T W eigendecomp...")
    A = W.T @ W                                  # (w_dim, w_dim)
    vals, vecs = np.linalg.eigh(A)
    # eigenvalues come ascending; reverse
    idx = np.argsort(-vals)[:n_components]
    components = vecs[:, idx].T                  # (k, w_dim)
    eigenvalues = vals[idx]
    return SeFaResult(
        components=components.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
    )
