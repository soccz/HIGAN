"""Encoder-based GAN inversion: one forward pass."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch

from ..encoder.model import WPlusEncoder, WPlusEncoderCfg
from ..generator import HiGANGenerator


@dataclass
class EncodeResult:
    wp: torch.Tensor      # (B, num_layers, latent_dim)
    image: torch.Tensor   # (B, 3, H, W) in [-1, 1]


def load_encoder(ckpt_path: str | Path, device: str | torch.device = "cuda"
                 ) -> tuple[WPlusEncoder, torch.Tensor]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["encoder_cfg"]
    cfg_dict["pretrained"] = False
    enc = WPlusEncoder(WPlusEncoderCfg(**cfg_dict))
    enc.load_state_dict(state["model"])
    enc.eval().to(device)
    w_avg = state["w_avg"].to(device) if "w_avg" in state else None
    return enc, w_avg


@torch.no_grad()
def encode_image(image: torch.Tensor, encoder: WPlusEncoder,
                 generator: HiGANGenerator, w_avg: torch.Tensor | None = None
                 ) -> EncodeResult:
    """image: (B,3,H,W) in [-1,1]."""
    image = image.to(generator.device)
    wp = encoder(image, w_avg=w_avg)
    recon = generator.synthesize(wp)
    return EncodeResult(wp=wp, image=recon)
