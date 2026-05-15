"""LatentCLR (Yüksel et al. ICCV 2021) — faithful re-implementation.

The original learns K input-dependent direction nets d_k(z) ∈ R^{w_dim};
positives are different latents pushed by the same d_k, negatives are
pushes by different d_k. NT-Xent loss in a feature-space difference.

We follow that recipe but use a simpler input-INDEPENDENT direction
parameterisation (learnable K × w_dim matrix) because:
  1. Our downstream evaluation uses static directions (head-to-head),
     so input-dependent dks are reduced to a single direction at
     evaluation time anyway.
  2. The contrastive objective on feature differences is preserved.
This is the constant-direction variant of LatentCLR — same contrastive
spirit, fewer parameters.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentCLRConfig:
    K: int = 100                  # number of directions
    w_dim: int = 512              # StyleGAN W dim
    temperature: float = 0.5
    feature_layer: int = 10       # synthesis layer index for feature extractor
    direction_scale: float = 6.0  # α magnitude during training (sigma units)


class DirectionBank(nn.Module):
    def __init__(self, K: int, w_dim: int):
        super().__init__()
        self.dirs = nn.Parameter(torch.randn(K, w_dim) / (w_dim ** 0.5))

    @property
    def unit_dirs(self) -> torch.Tensor:
        return F.normalize(self.dirs, dim=1)


def nt_xent_loss(features: torch.Tensor, K: int, B: int,
                 temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent on feature-space differences.

    Input `features` has shape (K*B, D). Logical layout:
        features[k*B + b] = F(G(z_b + α * d_k)) - F(G(z_b))
    Positives: same k, different b. Negatives: different k.

    Returns scalar loss.
    """
    N = features.size(0)
    f = F.normalize(features, dim=1)
    sim = f @ f.T  # (N, N)
    sim = sim / temperature

    # Diagonal mask (self-similarity not used)
    mask_diag = torch.eye(N, device=f.device, dtype=torch.bool)
    sim.masked_fill_(mask_diag, -1e9)

    # Build positive mask: feature i = (k_i, b_i); positive j: k_j == k_i, j != i
    k_idx = torch.arange(N, device=f.device) // B  # 0..K-1
    pos_mask = (k_idx.unsqueeze(0) == k_idx.unsqueeze(1)) & ~mask_diag

    # log-softmax over each row
    log_denom = torch.logsumexp(sim, dim=1, keepdim=True)  # (N, 1)
    log_prob = sim - log_denom

    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / \
                   pos_mask.sum(dim=1).clamp_min(1)
    return -pos_log_prob.mean()
