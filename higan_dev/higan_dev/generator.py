"""Differentiable wrapper around genforce/higan StyleGAN bedroom generator.

The genforce `easy_synthesize` API detaches and converts to numpy uint8, which
breaks autograd. We bypass it and call `G.net.synthesis` directly so that
gradients can flow back to latent codes (needed for optimization-based
inversion and for training an encoder with image-space losses).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


def _ensure_higan_on_path(higan_repo: str | Path) -> None:
    p = str(Path(higan_repo).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


class HiGANGenerator(nn.Module):
    """Frozen, differentiable StyleGAN bedroom256 generator.

    Outputs images in [-1, 1] range, shape (B, 3, 256, 256).
    """

    def __init__(self, higan_repo: str | Path, model_name: str = "stylegan_bedroom",
                 device: str = "cuda"):
        super().__init__()
        _ensure_higan_on_path(higan_repo)
        from models.helper import build_generator  # noqa: E402

        G = build_generator(model_name)
        # G.net is the actual nn.Module with .mapping / .truncation / .synthesis
        self._net: nn.Module = G.net
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)
        # Monkey-patch synthesis.forward to avoid `self.lod.cpu().tolist()`,
        # which breaks torch.func.jvp (can't dereference dual tensors via
        # .cpu()). For a frozen, fixed-resolution model, `lod` is constant
        # (0.0 for stylegan_bedroom256), so we read it once and substitute
        # a python float into the closure.
        self._patch_synthesis_for_jvp()

        self.num_layers: int = G.num_layers
        self.w_dim: int = G.w_space_dim
        self.z_dim: int = G.z_space_dim
        self.resolution: int = G.resolution
        self.truncation_psi: float = G.truncation_psi
        self.truncation_layers: int = G.truncation_layers
        self.device = torch.device(device)
        self.to(self.device)

    # ----- monkey-patch -----
    def _patch_synthesis_for_jvp(self) -> None:
        synth = self._net.synthesis
        cached_lod = float(synth.lod.detach().cpu().item())
        init_log2 = synth.init_res_log2
        final_log2 = synth.final_res_log2

        def jvp_safe_forward(self_synth, w):
            # Replace `lod = self.lod.cpu().tolist()` with cached float so the
            # forward-mode autodiff doesn't try to dereference a dual tensor.
            lod = cached_lod
            for res_log2 in range(init_log2, final_log2 + 1):
                if res_log2 + lod <= final_log2:
                    block_idx = res_log2 - init_log2
                    if block_idx == 0:
                        x = self_synth.__getattr__(f"layer{2 * block_idx}")(w[:, 2 * block_idx])
                    else:
                        x = self_synth.__getattr__(f"layer{2 * block_idx}")(x, w[:, 2 * block_idx])
                    x = self_synth.__getattr__(f"layer{2 * block_idx + 1}")(
                        x, w[:, 2 * block_idx + 1]
                    )
                    image = self_synth.__getattr__(f"output{block_idx}")(x)
                else:
                    image = self_synth.upsample(image)
            return image

        # bind the patched method to this specific synth instance
        import types
        synth.forward = types.MethodType(jvp_safe_forward, synth)

    # ----- latent space conversions -----
    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        return self._net.mapping(z)

    def w_to_wp(self, w: torch.Tensor) -> torch.Tensor:
        # truncation expects (B, w_dim); produces (B, num_layers, w_dim)
        return self._net.truncation(w)

    def z_to_wp(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_to_wp(self.z_to_w(z))

    # ----- image synthesis (differentiable) -----
    def synthesize(self, wp: torch.Tensor) -> torch.Tensor:
        """wp: (B, num_layers, w_dim) -> image (B, 3, H, W) in [-1, 1]."""
        if wp.dim() == 2:
            # (B, w_dim) -> broadcast to (B, num_layers, w_dim)
            wp = wp.unsqueeze(1).repeat(1, self.num_layers, 1)
        return self._net.synthesis(wp)

    def synthesize_at_block(self, wp: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Return the activation right after block `block_idx` (before its
        per-block output projection). For bedroom256, block_idx in [0..6]
        corresponds to spatial 4, 8, 16, 32, 64, 128, 256.
        """
        if wp.dim() == 2:
            wp = wp.unsqueeze(1).repeat(1, self.num_layers, 1)
        synth = self._net.synthesis
        init_log2 = synth.init_res_log2
        x = None
        for res_log2 in range(init_log2, init_log2 + block_idx + 1):
            bi = res_log2 - init_log2
            if bi == 0:
                x = synth.__getattr__(f"layer{2 * bi}")(wp[:, 2 * bi])
            else:
                x = synth.__getattr__(f"layer{2 * bi}")(x, wp[:, 2 * bi])
            x = synth.__getattr__(f"layer{2 * bi + 1}")(x, wp[:, 2 * bi + 1])
        return x

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

    # ----- helpers -----
    @torch.no_grad()
    def sample_w(self, n: int, *, generator: torch.Generator | None = None,
                 truncation: bool = False) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=self.device, generator=generator)
        w = self.z_to_w(z)
        if truncation:
            # apply average-w truncation manually if requested
            w = self._net.truncation.w_avg + self.truncation_psi * (w - self._net.truncation.w_avg)
        return w

    @torch.no_grad()
    def sample_wp(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=self.device, generator=generator)
        return self.z_to_wp(z)

    @staticmethod
    def to_uint8(images: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) in [-1,1] -> (B,H,W,3) uint8."""
        x = images.detach().cpu().float()
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        x = (x.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        return x

    @staticmethod
    def from_uint8(images: np.ndarray, device: str | torch.device = "cuda") -> torch.Tensor:
        """(B,H,W,3) uint8 -> (B,3,H,W) float in [-1,1]."""
        x = torch.from_numpy(images).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * 2.0 - 1.0
        return x.to(device)
