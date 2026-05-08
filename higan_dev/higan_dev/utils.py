from __future__ import annotations
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) or (B,C,H,W) in [-1,1] or [0,1] -> uint8 numpy (H,W,C) or (B,H,W,C)."""
    if t.dim() == 4:
        return np.stack([to_uint8_image(x) for x in t])
    x = t.detach().cpu().float()
    if x.min() < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def save_image(t: torch.Tensor | np.ndarray, path: str | Path) -> None:
    if isinstance(t, torch.Tensor):
        arr = to_uint8_image(t)
    else:
        arr = t
    Image.fromarray(arr).save(path)


def load_image_tensor(path: str | Path, size: int = 256, normalize: str = "0_1") -> torch.Tensor:
    """Load image as tensor (1,3,H,W). normalize: '0_1' or '-1_1'."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,C in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if normalize == "-1_1":
        t = t * 2.0 - 1.0
    return t


def grid(images: np.ndarray, cols: int) -> np.ndarray:
    """Tile (N,H,W,C) uint8 into (rows*H, cols*W, C)."""
    n, h, w, c = images.shape
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, im in enumerate(images):
        r, cc = divmod(i, cols)
        canvas[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = im
    return canvas


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, n: int = 1) -> None:
        self.sum += v * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)
