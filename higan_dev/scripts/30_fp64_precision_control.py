"""Precision control (TMLR reviewer L10's #1 ask): is the StyleGAN FD curvature
floor a *precision* artifact? Run the exact-vs-FD second-difference comparison in
fp32 AND fp64 and compare the best-step relative-error floor.

Theory (note non-recoverability): the central second-difference relative-error
floor scales as sqrt(eps_mach) * (function constant). So fp32 (eps_mach~1.2e-7)
should floor far higher than fp64 (eps_mach~2.2e-16). If the fp32 45% floor drops
toward ~0 in fp64, the honest claim sharpens to: the floor is a *single-precision*
non-recoverability (the precision generators actually run in), not a fundamental
one. Either outcome is reported honestly.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.manipulate import load_boundary
from higan_dev.cam.grad_saliency import _layered_direction


def curve_fn(G, wp, v, B, dt):
    def f(alpha):
        return G.synthesize(wp + alpha.view(B, 1, 1) * v.unsqueeze(0))
    return f


def exact_second(G, f, B, dt):
    ones = torch.ones(B, device=G.device, dtype=dt)
    a0 = torch.zeros(B, device=G.device, dtype=dt)
    def df(a): return jvp(f, (a,), (torch.ones_like(a),))[1]
    _, second = jvp(df, (a0,), (ones,))
    _, first = jvp(f, (a0,), (ones,))
    return second.abs().mean().item(), first.abs().mean().item()


def fd_second(G, f, B, d, dt):
    ap = torch.full((B,), +d, device=G.device, dtype=dt)
    am = torch.full((B,), -d, device=G.device, dtype=dt)
    a0 = torch.zeros(B, device=G.device, dtype=dt)
    with torch.no_grad():
        gp = f(ap); g0 = f(a0); gm = f(am)
        second = (gp - 2.0 * g0 + gm) / (d * d)
    return second.abs().mean().item()


def run_one_dtype(G, dt, dirs, base_z, d_grid, num_samples):
    """Return mean-relerr per step over (dir,sample), and the floor."""
    relerr = {d: [] for d in d_grid}
    for dname, v in dirs:
        v = v.to(dt)
        for s in range(num_samples):
            with torch.no_grad():
                wp = G.z_to_wp(base_z[s:s+1].to(dt))
            f = curve_fn(G, wp, v, 1, dt)
            c_ex, _ = exact_second(G, f, 1, dt)
            for d in d_grid:
                c_fd = fd_second(G, f, 1, d, dt)
                relerr[d].append(abs(c_fd - c_ex) / max(c_ex, 1e-30))
    agg = {d: float(np.mean(relerr[d])) for d in d_grid}
    best_d = min(d_grid, key=lambda d: agg[d])
    return agg, best_d, agg[best_d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--d-grid", type=float, nargs="+",
                    default=[3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001,
                             3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="out/fp64_precision_control")
    args = ap.parse_args()

    cfg = Config.load(resolve("configs/default.yaml"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name, device=cfg.train.device)
    G.double()                       # cast once; we feed dtype-correct inputs per run
    L, D = G.num_layers, G.w_dim
    # fixed base latents (as fp64 z; cast per-dtype inside)
    gz = torch.Generator(device=G.device).manual_seed(args.seed)
    base_z = torch.randn(args.num_samples, G.z_dim, device=G.device,
                         dtype=torch.float64, generator=gz)
    dirs = []
    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr, num_layers=L).to(G.device)
        dirs.append((attr, _layered_direction(b, L, D, G.device).double()))
    print(f"[{time.strftime('%H:%M:%S')}] G loaded; {len(dirs)} dirs x {args.num_samples} samples")

    results = {}
    for name, dt in [("float32", torch.float32), ("float64", torch.float64)]:
        # G is double; for fp32 run we cast G to float and inputs to float
        if dt == torch.float32:
            G.float()
        else:
            G.double()
        t0 = time.time()
        agg, best_d, floor = run_one_dtype(G, dt, dirs, base_z, args.d_grid, args.num_samples)
        results[name] = {"agg_by_step": {str(k): v for k, v in agg.items()},
                         "best_step": best_d, "floor_relerr": floor}
        print(f"  {name}: floor = {floor*100:.2f}%  at d={best_d}  ({time.time()-t0:.0f}s)")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 64)
    print("Precision control — StyleGAN FD 2nd-difference curvature floor")
    print("=" * 64)
    f32 = results["float32"]["floor_relerr"] * 100
    f64 = results["float64"]["floor_relerr"] * 100
    print(f"  fp32 floor: {f32:.2f}%  (best step {results['float32']['best_step']})")
    print(f"  fp64 floor: {f64:.4f}%  (best step {results['float64']['best_step']})")
    print(f"  ratio fp32/fp64: {f32/max(f64,1e-9):.0f}x")
    print(f"\n  Interpretation: if fp64 floor << fp32 floor, the 45%-class floor is a")
    print(f"  SINGLE-PRECISION non-recoverability -- exactly the precision generators")
    print(f"  (and diffusion U-Nets) actually run in. The exact JVP removes it at any precision.")
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
