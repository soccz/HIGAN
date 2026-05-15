"""GANSpace baseline (Härkönen et al., NeurIPS 2020).

Idea: sample many latents, capture intermediate activations at a
chosen synthesis layer, do PCA on those activation vectors, and use
the top principal components as "discovered" semantic directions.

We then push the same PCA directions through our JVP-saliency
pipeline (running our analysis tool on a competing method's discovered
directions) — this is the apples-to-apples comparison the paper needs.

Reference: https://github.com/harskish/ganspace
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GANSpaceResult:
    components: np.ndarray   # (k, w_dim) top-k principal directions in W-space
    explained: np.ndarray    # (k,) variance ratio per component
    mean_w: np.ndarray       # (w_dim,) mean of sampled W


def collect_w_samples(
    generator,
    n_samples: int = 1024,
    batch_size: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Sample n W vectors from the generator's mapping network."""
    gen = torch.Generator(device=generator.device).manual_seed(seed)
    chunks = []
    for s in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - s)
        z = torch.randn(b, generator.z_dim, device=generator.device, generator=gen)
        with torch.no_grad():
            w = generator.z_to_w(z)
        chunks.append(w.detach().cpu())
    return torch.cat(chunks, dim=0)


def ganspace_directions(
    generator,
    n_samples: int = 5000,
    n_components: int = 8,
    seed: int = 0,
) -> GANSpaceResult:
    """GANSpace-W: PCA on W vectors directly.

    The original GANSpace paper does PCA on early-layer activations
    (StyleGAN1 mapping output, StyleGAN2 layer-1 features). For W
    that's equivalent to PCA on the (mean-centred) W samples themselves;
    we use that simpler version here. For early-conv activations
    we provide `ganspace_activations()` below.
    """
    print(f"  sampling {n_samples} W vectors...")
    W = collect_w_samples(generator, n_samples=n_samples, seed=seed).numpy()
    W_mean = W.mean(0)
    W_c = W - W_mean
    # PCA via SVD
    print("  PCA via SVD...")
    U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    components = Vt[:n_components]
    return GANSpaceResult(
        components=components.astype(np.float32),
        explained=explained[:n_components].astype(np.float32),
        mean_w=W_mean.astype(np.float32),
    )


def ganspace_activations(
    generator,
    layer_name_or_idx: int | str,
    n_samples: int = 1024,
    batch_size: int = 8,
    n_components: int = 8,
    seed: int = 0,
) -> GANSpaceResult:
    """GANSpace-A: PCA on intermediate synthesis-layer activations.

    Slower but closer to the original paper's prescription. Uses a
    forward hook to capture activations at the named layer.
    """
    handles = []
    activations: list[torch.Tensor] = []

    def hook(_module, _input, output):
        activations.append(output.detach().mean(dim=(2, 3)).cpu())

    # locate the target layer
    if isinstance(layer_name_or_idx, int):
        target = getattr(generator._net.synthesis, f"layer{layer_name_or_idx}")
    else:
        target = generator._net.synthesis.__getattr__(layer_name_or_idx)
    handles.append(target.register_forward_hook(hook))

    gen = torch.Generator(device=generator.device).manual_seed(seed)
    try:
        for s in range(0, n_samples, batch_size):
            b = min(batch_size, n_samples - s)
            z = torch.randn(b, generator.z_dim, device=generator.device, generator=gen)
            with torch.no_grad():
                wp = generator.z_to_wp(z)
                _ = generator.synthesize(wp)
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    A = torch.cat(activations, dim=0).numpy()
    A_mean = A.mean(0)
    A_c = A - A_mean
    print(f"  activation matrix: {A.shape}, PCA via SVD...")
    U, S, Vt = np.linalg.svd(A_c, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    components = Vt[:n_components]
    return GANSpaceResult(
        components=components.astype(np.float32),
        explained=explained[:n_components].astype(np.float32),
        mean_w=np.zeros(generator.w_dim, dtype=np.float32),  # not applicable
    )
