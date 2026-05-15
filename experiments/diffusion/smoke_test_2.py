"""Smoke test 2: JVP through the full DDIM chain (25 steps from t_edit=25 to t=0).

This is what C1/C2 actually require. If this fits in memory and produces
a finite tangent, the experiment is feasible. If OOM here, we need to
fall back to 256² or single-step-prediction.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.generator import SDH, SDConfig


def main():
    t0 = time.time()
    sdh = SDH(SDConfig(resolution=512))
    print(f"[t={time.time()-t0:5.1f}s] SDH loaded")

    cond, uncond = sdh.encode_prompt("a smiling face", "")
    H = W = sdh.cfg.resolution // 8
    gen = torch.Generator(device=sdh.cfg.device).manual_seed(2027)
    x_init = torch.randn(1, 4, H, W, generator=gen,
                          device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    hs = sdh.h_space_shape()
    v = torch.randn(hs, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    v = v / v.norm()
    print(f"[t={time.time()-t0:5.1f}s] v shape {tuple(v.shape)} norm {v.norm().item():.4f}")

    # Run unperturbed sampling up to t_edit=25 (no JVP needed, no grad)
    t_edit_idx = 25
    with torch.no_grad():
        x = x_init.clone()
        for i in range(t_edit_idx):
            eps = sdh.epsilon(x, i, cond, uncond)
            x = sdh.ddim_step(x, eps, i)
    print(f"[t={time.time()-t0:5.1f}s] reached t_edit={t_edit_idx}, "
          f"x_t stats: mean={x.mean().item():.3f} std={x.std().item():.3f}, "
          f"free GPU mem {torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")

    x_at_edit = x.detach().clone()

    # f(alpha) = decoded x_0 image when h is perturbed by alpha*v at the
    # edit step, then DDIM continues to t=0.
    def f(alpha):
        x_local = x_at_edit
        sdh._h_v = v
        sdh._h_alpha = alpha
        for i in range(t_edit_idx, sdh.cfg.num_inference_steps):
            sdh._h_active = (i == t_edit_idx)  # only inject at the edit step
            eps = sdh.epsilon(x_local, i, cond, uncond)
            x_local = sdh.ddim_step(x_local, eps, i)
        sdh._h_active = False
        sdh._h_v = None
        sdh._h_alpha = None
        return x_local

    a0 = torch.zeros((), device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    a1 = torch.ones((), device=sdh.cfg.device, dtype=sdh.cfg.dtype)

    print(f"[t={time.time()-t0:5.1f}s] JVP through {sdh.cfg.num_inference_steps - t_edit_idx} DDIM steps...")
    try:
        x_final, dx_final = jvp(f, (a0,), (a1,))
        print(f"[t={time.time()-t0:5.1f}s] x_final shape={tuple(x_final.shape)} "
              f"|dx_final|={dx_final.norm().item():.4g}, "
              f"free GPU mem {torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[t={time.time()-t0:5.1f}s] OOM in full-chain JVP: {e}")
        return

    # Now also try the COMPOSED JVP for the second derivative
    print(f"[t={time.time()-t0:5.1f}s] composed JVP for 2nd derivative...")

    def df_dalpha(alpha):
        _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
        return d

    try:
        d1, d2 = jvp(df_dalpha, (a0,), (a1,))
        print(f"[t={time.time()-t0:5.1f}s] 1st-deriv |d|={d1.norm().item():.4g}  "
              f"2nd-deriv |d2|={d2.norm().item():.4g}")
        print(f"free GPU mem {torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[t={time.time()-t0:5.1f}s] OOM in composed JVP: {e}")
        return

    print(f"[t={time.time()-t0:5.1f}s] SMOKE TEST 2 PASSED — full chain + composed JVP works")


if __name__ == "__main__":
    main()
