"""Differentiable wrapper around genforce/interfacegan StyleGAN FFHQ generator.

Analogous to higan_dev.generator.HiGANGenerator but for the StyleGAN1 FFHQ
checkpoint (1024^2, 18 layers).

Same playbook:
  - direct call to G.net.synthesis (bypass detach/numpy in easy_synthesize)
  - monkey-patch synthesis.forward to remove .cpu().tolist() on lod buffer
    so torch.func.jvp can compose through it.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


_INTERFACEGAN_REPO = Path(__file__).resolve().parents[2] / "data" / "interfacegan"


def _ensure_on_path() -> None:
    p = str(_INTERFACEGAN_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class FFHQGenerator(nn.Module):
    """Frozen, differentiable StyleGAN FFHQ generator at 1024x1024.

    Outputs images in [-1, 1], shape (B, 3, 1024, 1024).

    Note on memory: a single 1024^2 forward pass uses ~2.5 GB on the
    8 GB RTX 3070; composed JVP roughly doubles this. Batch size 1 is
    recommended; for second-order analyses we provide an optional
    `lod=2` mode that renders at 256^2 with ~10x less memory.
    """

    def __init__(self, model_name: str = "stylegan_ffhq", device: str = "cuda",
                 lod_override: float | None = None):
        super().__init__()
        _ensure_on_path()
        from models.stylegan_generator import StyleGANGenerator           # noqa: E402

        G = StyleGANGenerator(model_name)
        # InterFaceGAN names the inner module `model`; HiGAN uses `net`.
        self._net: nn.Module = getattr(G, "net", None) or G.model
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)

        self.num_layers: int = G.num_layers
        self.w_dim: int = G.w_space_dim
        self.z_dim: int = G.latent_space_dim
        self.resolution: int = G.resolution
        self.truncation_psi: float = G.truncation_psi
        self.truncation_layers: int = G.truncation_layers
        self.device = torch.device(device)
        self.to(self.device)

        # cache lod as a python float (jvp-safe), allow override for memory mode
        cached_lod = float(self._net.synthesis.lod.detach().cpu().item())
        if lod_override is not None:
            cached_lod = float(lod_override)
        self._cached_lod = cached_lod
        self._patch_synthesis_for_jvp()

    def _patch_synthesis_for_jvp(self) -> None:
        """Patch InterFaceGAN's SynthesisModule.forward to be jvp-safe.

        Difference from HiGAN's patch: InterFaceGAN's forward iterates
        over `block_idx in range(1, len(channels))` with layer0 as a
        special-case starting block, and uses layer indices
        2*b-2 and 2*b-1 for each block. The original line
        `lod = self.lod.cpu().tolist()` breaks the dual tensor.
        """
        synth = self._net.synthesis
        cached_lod = self._cached_lod
        n_blocks = len(synth.channels)

        def jvp_safe_forward(self_synth, w):
            lod = cached_lod
            x = self_synth.layer0(w[:, 0])
            image = None
            for block_idx in range(1, n_blocks):
                if block_idx + lod < n_blocks:
                    layer_idx = 2 * block_idx - 2
                    if layer_idx == 0:
                        x = self_synth.__getattr__(f"layer{layer_idx}")(w[:, layer_idx])
                    else:
                        x = self_synth.__getattr__(f"layer{layer_idx}")(x, w[:, layer_idx])
                    layer_idx = 2 * block_idx - 1
                    x = self_synth.__getattr__(f"layer{layer_idx}")(x, w[:, layer_idx])
                    image = self_synth.__getattr__(f"output{block_idx - 1}")(x)
                else:
                    image = self_synth.upsample(image)
            return image

        synth.forward = types.MethodType(jvp_safe_forward, synth)

    # ----- latent conversions -----
    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        return self._net.mapping(z)

    def w_to_wp(self, w: torch.Tensor) -> torch.Tensor:
        return self._net.truncation(w)

    def z_to_wp(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_to_wp(self.z_to_w(z))

    # ----- synthesis (differentiable) -----
    def synthesize(self, wp: torch.Tensor) -> torch.Tensor:
        """wp: (B, num_layers, w_dim) -> image (B, 3, H, W) in [-1, 1]."""
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
