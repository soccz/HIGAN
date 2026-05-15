"""Track 11 — wall-clock benchmark for first-order pushforward.

Three methods at matched accuracy:
  1. forward JVP (torch.func.jvp) — reference
  2. finite-difference: (G(wp+εb) - G(wp-εb)) / 2ε
  3. per-pixel reverse mode via vmap+vjp

Reports wall-clock, peak GPU memory, numerical agreement with JVP
reference. On bedroom (cheap) only — FFHQ-1024 likely OOMs vmap+vjp.

See designs/11_walltime_benchmark.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp, vmap, vjp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))


def time_fn(fn, n_runs: int = 5, warmup: int = 1):
    """Returns (mean_s, std_s, peak_mb)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = fn()
        torch.cuda.synchronize()
        durations.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    return float(np.mean(durations)), float(np.std(durations)), float(peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom"], default="bedroom")
    ap.add_argument("--epsilon-fd", type=float, default=1e-3)
    ap.add_argument("--vjp-pixel-cap", type=int, default=256,
                    help="how many output pixels to vmap+vjp through "
                    "(capped for the head-to-head sanity number; full "
                    "3HW is impractical and extrapolated theoretically)")
    ap.add_argument("--out", default="experiments/out/walltime")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary
    from higan_dev.cam.grad_saliency import _layered_direction

    G = HiGANGenerator(higan_repo=str(
        PAPER.parent / "higan_dev" / "data" / "higan_repo"
    ))
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
            / "boundaries" / "stylegan_bedroom")
    b = load_boundary(str(bdir), "view", num_layers=L)
    b_l = _layered_direction(b, L, D, G.device).to(G.device)
    rng = torch.Generator(device=G.device).manual_seed(0)
    wp = G.sample_wp(1, generator=rng).detach()

    def synth_with_dir(scalar):
        return G.synthesize(wp + scalar.view(1, 1, 1) * b_l.unsqueeze(0))

    a0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)

    # ---- 1. JVP reference ----
    def call_jvp():
        return jvp(synth_with_dir, (a0,), (one,))
    mean_jvp, std_jvp, peak_jvp = time_fn(call_jvp)
    jvp_ref = call_jvp()[1].detach()
    print(f"JVP    : {mean_jvp*1000:7.1f}±{std_jvp*1000:.1f} ms, "
          f"peak {peak_jvp:.0f} MB")

    # ---- 2. finite difference ----
    eps = args.epsilon_fd
    def call_fd():
        with torch.no_grad():
            xp = G.synthesize(wp + eps * b_l.unsqueeze(0))
            xn = G.synthesize(wp - eps * b_l.unsqueeze(0))
            return (xp - xn) / (2 * eps)
    mean_fd, std_fd, peak_fd = time_fn(call_fd)
    fd_ref = call_fd().detach()
    fd_err = (fd_ref - jvp_ref).abs().mean().item()
    jvp_mag = jvp_ref.abs().mean().item()
    print(f"FD ε={eps:g}: {mean_fd*1000:7.1f}±{std_fd*1000:.1f} ms, "
          f"peak {peak_fd:.0f} MB, "
          f"rel err vs JVP = {fd_err/jvp_mag:.2e}")

    # ---- 3. vmap+vjp at first K pixels ----
    K = args.vjp_pixel_cap
    print(f"vmap+vjp first {K} pixels (extrapolating to {3*H*W} for "
          f"the full Jacobian)")

    def synth_wp(wp_arg):
        return G.synthesize(wp_arg.unsqueeze(0)).flatten()  # (3*H*W,)

    # Use a single VJP per pixel index then vmap
    # vjp gives ∂y_i/∂wp; we want sum over i of vjp_i applied to b_l.
    # Strategy: produce K standard basis vectors and use vmap-vjp.
    def vjp_one(idx):
        _, f_vjp = vjp(synth_wp, wp.squeeze(0))
        e_i = torch.zeros(3 * H * W, device=G.device)
        e_i[idx] = 1.0
        grad_wp, = f_vjp(e_i)
        # project gradient onto direction b_l
        return (grad_wp * b_l).sum()

    indices = torch.arange(K, device=G.device)

    def call_vmap_vjp():
        return vmap(vjp_one)(indices)

    try:
        mean_vmap, std_vmap, peak_vmap = time_fn(call_vmap_vjp,
                                                  n_runs=2, warmup=1)
        print(f"vmap+vjp ({K} pixels): "
              f"{mean_vmap*1000:7.1f}±{std_vmap*1000:.1f} ms, "
              f"peak {peak_vmap:.0f} MB")
        full_extrap_s = mean_vmap * (3 * H * W) / K
        print(f"  → extrapolated to full {3*H*W} pixels: "
              f"~{full_extrap_s:.1f} s")
    except Exception as e:
        print(f"vmap+vjp failed: {type(e).__name__}: {e}")
        mean_vmap, std_vmap, peak_vmap, full_extrap_s = (
            None, None, None, None
        )

    results = {
        "domain": args.domain,
        "resolution": H,
        "jvp":    {"mean_ms": mean_jvp*1000, "std_ms": std_jvp*1000,
                   "peak_mb": peak_jvp},
        "fd":     {"mean_ms": mean_fd*1000, "std_ms": std_fd*1000,
                   "peak_mb": peak_fd, "epsilon": eps,
                   "rel_err_vs_jvp": fd_err / jvp_mag},
        "vmap_vjp_cap": {"K": K,
                          "mean_ms": mean_vmap*1000 if mean_vmap else None,
                          "peak_mb": peak_vmap,
                          "extrap_full_s": full_extrap_s},
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
