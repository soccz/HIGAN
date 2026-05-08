"""Common image / latent losses for inversion and encoder training."""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


# -----------------------------------------------------------------------------
# Perceptual loss (VGG16 features) — input expected in [-1, 1].
# -----------------------------------------------------------------------------
class VGGPerceptual(nn.Module):
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, layer_idx: int = 16):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # input in [-1, 1] -> [0, 1] -> ImageNet-normalised
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.vgg(self._normalize(a)), self.vgg(self._normalize(b)))


# -----------------------------------------------------------------------------
# Total variation (smoothness) loss
# -----------------------------------------------------------------------------
def total_variation(image: torch.Tensor) -> torch.Tensor:
    dh = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    dw = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    return dh + dw


# -----------------------------------------------------------------------------
# LPIPS lazy loader (avoids hard dependency at import time)
# -----------------------------------------------------------------------------
class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips  # noqa: F401, lazy import
        self._lpips = lpips.LPIPS(net=net, verbose=False).eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # lpips expects inputs in [-1, 1]
        return self._lpips(a, b).mean()


# -----------------------------------------------------------------------------
# Combined loss helper
# -----------------------------------------------------------------------------
class ImageReconLoss(nn.Module):
    """L2 + VGG perceptual + LPIPS + TV. All weights configurable."""

    def __init__(self, weights: dict[str, float], device: str | torch.device = "cuda"):
        super().__init__()
        self.weights = dict(weights)
        self.vgg: Optional[VGGPerceptual] = None
        self.lpips_fn: Optional[LPIPSLoss] = None
        if self.weights.get("perceptual", 0.0) > 0:
            self.vgg = VGGPerceptual().to(device)
        if self.weights.get("lpips", 0.0) > 0:
            self.lpips_fn = LPIPSLoss().to(device)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        components: dict[str, torch.Tensor] = {}
        if self.weights.get("pixel_l2", 0.0) > 0 or self.weights.get("mse", 0.0) > 0:
            w = self.weights.get("pixel_l2", self.weights.get("mse", 0.0))
            components["pixel_l2"] = F.mse_loss(recon, target) * w
        if self.weights.get("perceptual", 0.0) > 0:
            assert self.vgg is not None
            components["perceptual"] = self.vgg(recon, target) * self.weights["perceptual"]
        if self.weights.get("lpips", 0.0) > 0:
            assert self.lpips_fn is not None
            components["lpips"] = self.lpips_fn(recon, target) * self.weights["lpips"]
        if self.weights.get("tv", 0.0) > 0:
            components["tv"] = total_variation(recon) * self.weights["tv"]
        total = sum(components.values()) if components else recon.new_zeros(())
        log = {k: v.item() for k, v in components.items()}
        log["total"] = float(total.item()) if isinstance(total, torch.Tensor) else float(total)
        return total, log
