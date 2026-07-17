"""Smoke test for SDH wrapper. Verifies:
 1. SD v1.5 loads on the available GPU.
 2. h-space shape probe works.
 3. A single sample() round trip produces a valid image.
 4. sample_with_h_perturb() with alpha=0 reproduces the no-perturb result.
 5. torch.func.jvp through a single UNet eval gives nonzero tangent.
Run with `python3 -u smoke_test.py` from `paper/`.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.generator import SDH, SDConfig


def main():
    print(f"[t={0:5.1f}s] start")
    t0 = time.time()

    # 1. load
    sdh = SDH(SDConfig(resolution=512))
    print(f"[t={time.time()-t0:5.1f}s] loaded. cuda mem free: "
          f"{torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")

    # 2. h-space shape
    hs = sdh.h_space_shape()
    print(f"[t={time.time()-t0:5.1f}s] mid_block output shape: {hs}")
    # SD v1.5 at 512²: expect (1, 1280, 8, 8)

    # 3. sample
    print(f"[t={time.time()-t0:5.1f}s] sampling 'a photo of a cat' seed=2027...")
    out = sdh.sample("a photo of a cat", seed=2027)
    img = out["image"]
    print(f"[t={time.time()-t0:5.1f}s] image shape {tuple(img.shape)} "
          f"range [{img.min().item():.3f}, {img.max().item():.3f}]")

    # 4. perturb-zero equivalence
    print(f"[t={time.time()-t0:5.1f}s] verifying alpha=0 is identity...")
    v = torch.zeros(hs, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    a = torch.zeros((), device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    _, img_p = sdh.sample_with_h_perturb("a photo of a cat", 2027,
                                         t_edit_idx=25, v=v, alpha=a)
    diff = (img - img_p).abs().mean().item()
    print(f"[t={time.time()-t0:5.1f}s] mean |Δimage| at alpha=0: {diff:.6f}")
    assert diff < 1e-3, f"alpha=0 should be identity, got diff={diff}"

    # 5. JVP through a single UNet eval
    print(f"[t={time.time()-t0:5.1f}s] running torch.func.jvp through 1 UNet eval...")
    cond, uncond = sdh.encode_prompt("a smiling face", "")
    H = W = sdh.cfg.resolution // 8
    x = torch.randn(1, 4, H, W, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    # JVP w.r.t. a scalar alpha that scales an arbitrary h-direction
    v = torch.randn(hs, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    v = v / v.norm() * 1e-2  # small magnitude so JVP is well-conditioned

    def f(alpha):
        sdh._h_v = v
        sdh._h_alpha = alpha
        sdh._h_active = True
        try:
            out = sdh.epsilon(x, 25, cond, uncond)
        finally:
            sdh._h_active = False
            sdh._h_v = None
            sdh._h_alpha = None
        return out

    from torch.func import jvp
    a0 = torch.zeros((), device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    a1 = torch.ones((), device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    try:
        eps0, deps = jvp(f, (a0,), (a1,))
        print(f"[t={time.time()-t0:5.1f}s] jvp eps0 shape={tuple(eps0.shape)}, "
              f"|deps|={deps.norm().item():.4g}")
        assert deps.norm().item() > 0, "JVP tangent is zero — hook not in path?"
    except Exception as e:
        print(f"[t={time.time()-t0:5.1f}s] JVP failed: {type(e).__name__}: {e}")
        raise

    print(f"[t={time.time()-t0:5.1f}s] SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
