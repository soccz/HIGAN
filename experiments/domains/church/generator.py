"""Differentiable wrapper around genforce/higan stylegan2_church256.

StyleGAN2 synthesis is cleaner than StyleGAN1 — no `.lod.cpu().tolist()`
buffer to monkey-patch, just bypass `easy_synthesize` to keep autograd
alive.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


_HIGAN_REPO = (
    Path(__file__).resolve().parents[4] / "higan_dev" / "data" / "higan_repo"
)


def _ensure_on_path() -> None:
    p = str(_HIGAN_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class ChurchGenerator(nn.Module):
    """Frozen, differentiable StyleGAN2 LSUN church 256x256 generator."""

    def __init__(self, model_name: str = "stylegan2_church", device: str = "cuda"):
        super().__init__()
        _ensure_on_path()
        from models.helper import build_generator                # noqa: E402

        G = build_generator(model_name)
        # genforce stylegan2 uses `.net` like HiGAN1, not `.model`
        self._net: nn.Module = getattr(G, "net", None) or G.model
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)

        self.num_layers: int = G.num_layers
        self.w_dim: int = G.w_space_dim
        self.z_dim: int = G.z_space_dim
        self.resolution: int = G.resolution
        self.truncation_psi: float = getattr(G, "truncation_psi", 0.7)
        self.truncation_layers: int = getattr(G, "truncation_layers", 0)
        self.device = torch.device(device)
        self.to(self.device)

    # ----- latent conversions -----
    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        return self._net.mapping(z)

    def w_to_wp(self, w: torch.Tensor) -> torch.Tensor:
        return self._net.truncation(w)

    def z_to_wp(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_to_wp(self.z_to_w(z))

    # ----- synthesis (differentiable, no patch needed) -----
    def synthesize(self, wp: torch.Tensor) -> torch.Tensor:
        if wp.dim() == 2:
            wp = wp.unsqueeze(1).repeat(1, self.num_layers, 1)
        return self._net.synthesis(wp)

    def forward(self, latent: torch.Tensor,
                space: Literal["z", "w", "wp"] = "wp") -> torch.Tensor:
        if space == "z":
            wp = self.z_to_wp(latent)
        elif space == "w":
            wp = self.w_to_wp(latent)
        elif space == "wp":
            wp = latent
        else:
            raise ValueError(f"unknown space: {space}")
        return self.synthesize(wp)

    @torch.no_grad()
    def sample_w(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=self.device, generator=generator)
        return self.z_to_w(z)

    @torch.no_grad()
    def sample_wp(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=self.device, generator=generator)
        return self.z_to_wp(z)

    @staticmethod
    def to_uint8(images: torch.Tensor) -> np.ndarray:
        x = images.detach().cpu().float()
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        x = (x.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        return x
