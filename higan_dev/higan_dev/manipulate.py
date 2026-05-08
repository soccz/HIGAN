"""Boundary manipulation in W+ space — torch-native, differentiable.

Replaces the numpy-based `manipulate` from the genforce HiGAN repo so it
integrates with our generator wrapper and supports gradient flow (useful for
the CAM/diff-map analysis below).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# Default manipulate-layers per attribute, copied from the HiGAN paper /
# notebook. `view` is global (low-frequency layout); the rest target mid layers.
DEFAULT_MANIPULATE_LAYERS = {
    "view": list(range(0, 5)),                  # 0..4
    "indoor_lighting": list(range(6, 12)),
    "wood": list(range(6, 12)),
    "cluttered_space": list(range(6, 12)),
    "carpet": list(range(6, 12)),
    "dirt": list(range(6, 12)),
    "glossy": list(range(6, 12)),
    "scary": list(range(6, 12)),
}


@dataclass
class Boundary:
    """Loaded HiGAN attribute boundary."""
    name: str
    direction: torch.Tensor               # (latent_dim,) unit vector
    manipulate_layers: list[int]
    raw_meta: dict | None = None

    def to(self, device) -> "Boundary":
        return Boundary(self.name, self.direction.to(device), self.manipulate_layers,
                        self.raw_meta)


def load_boundary(boundaries_dir: str | Path, name: str,
                  num_layers: int = 14) -> Boundary:
    """Load a HiGAN boundary .npy file. Falls back to default manipulate_layers."""
    path = Path(boundaries_dir) / f"{name}_boundary.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=True)
    meta = None
    if raw.dtype == object:
        d = raw.item()
        boundary_np = d["boundary"]
        if "meta_data" in d and "manipulate_layers" in d["meta_data"]:
            spec = d["meta_data"]["manipulate_layers"]
            manipulate_layers = _parse_layer_spec(spec, num_layers)
        else:
            manipulate_layers = list(DEFAULT_MANIPULATE_LAYERS.get(
                name, list(range(num_layers))))
        meta = d.get("meta_data")
    else:
        boundary_np = raw
        manipulate_layers = list(DEFAULT_MANIPULATE_LAYERS.get(
            name, list(range(num_layers))))
    direction = torch.from_numpy(np.asarray(boundary_np)).float().squeeze()
    if direction.dim() != 1:
        # boundaries sometimes come as (1, 512). Take the first row.
        direction = direction.flatten()[: direction.numel()].view(-1)
        direction = direction.flatten()
    # normalise just in case
    direction = direction / direction.norm().clamp_min(1e-8)
    return Boundary(name=name, direction=direction,
                    manipulate_layers=manipulate_layers, raw_meta=meta)


def _parse_layer_spec(spec, num_layers: int) -> list[int]:
    if isinstance(spec, str):
        out: list[int] = []
        for token in spec.replace(" ", "").split(","):
            if "-" in token:
                a, b = token.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(token))
        return [i for i in out if 0 <= i < num_layers]
    if isinstance(spec, (list, tuple, np.ndarray)):
        return [int(i) for i in spec if 0 <= int(i) < num_layers]
    return list(range(num_layers))


def manipulate_wp(
    wp: torch.Tensor,                          # (B, num_layers, latent_dim)
    boundary: Boundary,
    distances: Iterable[float],
    *,
    layer_strength: torch.Tensor | None = None,  # (num_layers,) optional scaling
) -> torch.Tensor:
    """Return wp shifted along the boundary by each given distance.

    Returns: (B, num_distances, num_layers, latent_dim)
    """
    B, L, D = wp.shape
    direction = boundary.direction.to(wp.device).view(1, 1, D)  # broadcastable
    # mask: (L,) with 1 on layers to manipulate else 0
    mask = wp.new_zeros(L)
    for li in boundary.manipulate_layers:
        if 0 <= li < L:
            mask[li] = 1.0
    if layer_strength is not None:
        mask = mask * layer_strength.to(wp.device).clamp_min(0)

    distances = torch.as_tensor(list(distances), dtype=wp.dtype, device=wp.device)
    # (B, K, L, D)
    delta = distances.view(1, -1, 1, 1) * mask.view(1, 1, L, 1) * direction.view(1, 1, 1, D)
    return wp.unsqueeze(1) + delta


def list_available_boundaries(boundaries_dir: str | Path) -> list[str]:
    out = []
    for p in sorted(Path(boundaries_dir).glob("*_boundary.npy")):
        out.append(p.stem.replace("_boundary", ""))
    return out
