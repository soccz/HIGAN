"""Custom W+ encoder for HiGAN bedroom256.

Architecture: ResNet backbone + per-scale shared "neck" (small conv stack to a
512-d feature vector) + per-layer linear style head. We split the 14 W+ layers
across 3 backbone scales (coarse / mid / fine), matching how StyleGAN layer
indices correspond to spatial scales:
    0..3   (coarse: structure / layout)  ← deepest 7x7 features
    4..9   (mid:    shape / object)      ← 14x14 features
    10..13 (fine:   colour / texture)    ← 28x28 features
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
)


_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, [64, 128, 256, 512]),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, [64, 128, 256, 512]),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, [256, 512, 1024, 2048]),
}


_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _Neck(nn.Module):
    """Shared per-scale neck: (B, C_in, H, W) -> (B, 512)."""

    def __init__(self, in_ch: int, hid: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hid, hid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hid),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


@dataclass
class WPlusEncoderCfg:
    backbone: str = "resnet50"
    num_layers: int = 14
    latent_dim: int = 512
    pretrained: bool = True


class WPlusEncoder(nn.Module):
    """Image -> (B, num_layers, latent_dim) prediction in W+ space."""

    def __init__(self, cfg: WPlusEncoderCfg):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {cfg.backbone}")
        builder, weights, ch = _BACKBONES[cfg.backbone]
        net = builder(weights=weights if cfg.pretrained else None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 64x64  -> 56x56 after stem; ch[0]
        self.layer2 = net.layer2   # 28x28  ch[1]
        self.layer3 = net.layer3   # 14x14  ch[2]
        self.layer4 = net.layer4   #  7x7   ch[3]

        n = cfg.num_layers
        self.split = (
            list(range(0, max(1, n * 4 // 14))),
            list(range(max(1, n * 4 // 14), max(2, n * 10 // 14))),
            list(range(max(2, n * 10 // 14), n)),
        )

        self.neck_coarse = _Neck(ch[3], hid=cfg.latent_dim)
        self.neck_mid = _Neck(ch[2], hid=cfg.latent_dim)
        self.neck_fine = _Neck(ch[1], hid=cfg.latent_dim)

        # Per-layer linear heads
        self.head_coarse = nn.ModuleList(
            [nn.Linear(cfg.latent_dim, cfg.latent_dim) for _ in self.split[0]]
        )
        self.head_mid = nn.ModuleList(
            [nn.Linear(cfg.latent_dim, cfg.latent_dim) for _ in self.split[1]]
        )
        self.head_fine = nn.ModuleList(
            [nn.Linear(cfg.latent_dim, cfg.latent_dim) for _ in self.split[2]]
        )

        self.register_buffer("inet_mean", _INET_MEAN.clone())
        self.register_buffer("inet_std", _INET_STD.clone())

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        return (x - self.inet_mean) / self.inet_std

    def forward(self, x: torch.Tensor, *, w_avg: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B,3,H,W) in [-1,1]. Returns (B, num_layers, latent_dim).

        If `w_avg` provided, predicts deltas around it (residual mode).
        """
        x = self._normalize(x)
        x = self.stem(x)
        f1 = self.layer1(x)     # 56x56
        f2 = self.layer2(f1)    # 28x28
        f3 = self.layer3(f2)    # 14x14
        f4 = self.layer4(f3)    # 7x7

        v_coarse = self.neck_coarse(f4)   # (B, 512)
        v_mid = self.neck_mid(f3)
        v_fine = self.neck_fine(f2)

        styles = []
        for h in self.head_coarse:
            styles.append(h(v_coarse))
        for h in self.head_mid:
            styles.append(h(v_mid))
        for h in self.head_fine:
            styles.append(h(v_fine))

        wp = torch.stack(styles, dim=1)
        if w_avg is not None:
            if w_avg.dim() == 1:
                w_avg = w_avg.unsqueeze(0)
            wp = wp + w_avg.unsqueeze(0)
        return wp
