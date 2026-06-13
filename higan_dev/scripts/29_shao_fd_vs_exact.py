"""Pillar C (TMLR §5b) — Shao-style FD 2nd-difference curvature vs EXACT composed-JVP
on a real StyleGAN/HiGAN generator.

Shao, Kumar & Fletcher (CVPR-W 2018, arXiv 1711.08014) deliberately discard the
Christoffel symbols and estimate the second-order term of the generator along a latent
curve by a FINITE-DIFFERENCE second difference on a COARSE discretization (their T=10):

    d^2 g / dt^2  ~=  ( g(z_{i+1}) - 2 g(z_i) + g(z_{i-1}) ) / dt^2 .

Their headline conclusion ("manifolds have surprisingly little curvature => straight lines
are close to geodesics") is read off from this FD-estimated second-order signal.

We isolate that exact numerical primitive and measure its accuracy against the EXACT
second-order pushforward computed by a composed JVP (jvp of jvp), which needs no step size.

At a base latent z with unit tangent v (a boundary direction or a random w-direction):
  exact:  c* = mean| d^2/dalpha^2 g(z + alpha v) |_{alpha=0}        (no step size)
  FD(d):  c_d = mean| ( g(z+d v) - 2 g(z) + g(z-d v) ) / d^2 |      (Shao's primitive)

Theory (note/main.tex): the FD second difference has error O(d^2) (truncation) +
O(eps_machine * |g| / d^2) (cancellation). In single precision (StyleGAN runs fp32) these
cannot be made simultaneously small; the optimal d gives a relative-error FLOOR, and a
COARSE d (Shao's T=10 regime) sits in the truncation branch -> curvature UNDER-estimated
-> a spurious "near-flat" reading. We sweep d to expose the U-curve and locate the floor.
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


def curve_fn(G, wp_base, v_layered, B):
    """f(alpha) -> image batch for the straight latent curve z(alpha) = wp_base + alpha v."""
    def f(alpha):
        return G.synthesize(wp_base + alpha.view(B, 1, 1) * v_layered.unsqueeze(0))
    return f


def exact_second(G, f, B):
    """mean| d^2/dalpha^2 g | at alpha=0 via composed JVP (no step size)."""
    ones = torch.ones(B, device=G.device)
    a0 = torch.zeros(B, device=G.device)

    def df(alpha):
        return jvp(f, (alpha,), (torch.ones_like(alpha),))[1]

    _, second = jvp(df, (a0,), (ones,))           # (B,3,H,W)
    _, first = jvp(f, (a0,), (ones,))
    return second.abs().mean().item(), first.abs().mean().item()


def fd_both(G, f, B, d):
    """Return (2nd-order FD, 1st-order FD) at step d.
    2nd (Shao): mean| (g+  - 2 g0 + g-) / d^2 | .
    1st (sanity, should be robust): mean| (g+ - g-) / (2 d) | .
    The 1st-order central difference is well-conditioned; if it matches the exact
    JVP first-order magnitude, the pipeline is sound and any 2nd-order blow-up is
    genuine non-recoverability, not a bug."""
    ap = torch.full((B,), +d, device=G.device)
    am = torch.full((B,), -d, device=G.device)
    a0 = torch.zeros(B, device=G.device)
    with torch.no_grad():
        gp = f(ap); g0 = f(a0); gm = f(am)
        second = (gp - 2.0 * g0 + gm) / (d * d)
        first = (gp - gm) / (2.0 * d)
    return second.abs().mean().item(), first.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--n-random-dirs", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--d-grid", type=float, nargs="+",
                    default=[3.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05,
                             0.02, 0.01, 0.005, 0.002, 0.001, 0.0005])
    ap.add_argument("--shao-T", type=int, default=10,
                    help="Shao's discretization; d_shao = path_tmax / T")
    ap.add_argument("--path-tmax", type=float, default=3.0,
                    help="latent-curve half-length used to map T -> step")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="out/shao_fd_vs_exact")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    print(f"[{time.strftime('%H:%M:%S')}] HiGAN loaded; L={L} D={D} "
          f"res={G.resolution} dtype={base_wp.dtype}")

    # build direction list: boundary attrs (layered) + random w-directions
    dirs = []
    for attr in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, attr, num_layers=L).to(G.device)
        dirs.append((attr, _layered_direction(b, L, D, G.device)))
    for r in range(args.n_random_dirs):
        g = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=g)
        rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv))

    d_shao = args.path_tmax / args.shao_T
    records = []
    for dname, v_layered in dirs:
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            f = curve_fn(G, wp, v_layered, 1)
            c_exact, c_first = exact_second(G, f, 1)
            fds = {d: fd_both(G, f, 1, d) for d in args.d_grid}
            records.append({"dir": dname, "sample": s,
                            "c_exact": c_exact, "c_first_exact": c_first,
                            "c_fd2": {str(d): fds[d][0] for d in args.d_grid},
                            "c_fd1": {str(d): fds[d][1] for d in args.d_grid}})
        print(f"  {dname:16s} done ({time.strftime('%H:%M:%S')})")

    # aggregate: relative error of FD vs exact per step d
    def relerr(cfd, cex):
        return abs(cfd - cex) / max(cex, 1e-12)
    agg = {}
    for d in args.d_grid:
        errs2 = [relerr(r["c_fd2"][str(d)], r["c_exact"]) for r in records]
        errs1 = [relerr(r["c_fd1"][str(d)], r["c_first_exact"]) for r in records]
        agg[str(d)] = {"mean_relerr": float(np.mean(errs2)),
                       "median_relerr": float(np.median(errs2)),
                       "mean_c_fd": float(np.mean([r["c_fd2"][str(d)] for r in records])),
                       "mean_relerr_FIRST": float(np.mean(errs1)),
                       "mean_c_fd1": float(np.mean([r["c_fd1"][str(d)] for r in records]))}
    mean_exact = float(np.mean([r["c_exact"] for r in records]))
    mean_first_exact = float(np.mean([r["c_first_exact"] for r in records]))
    # best 1st-order FD agreement = sanity floor (should be tiny at some d)
    best_d1 = min(args.d_grid, key=lambda d: agg[str(d)]["mean_relerr_FIRST"])
    # best FD step and its floor
    best_d = min(args.d_grid, key=lambda d: agg[str(d)]["mean_relerr"])
    # Shao's-step relerr (nearest grid d to d_shao, plus exact compute)
    near_shao = min(args.d_grid, key=lambda d: abs(d - d_shao))

    result = {"records": records, "agg_by_step": agg,
              "mean_c_exact": mean_exact,
              "best_step": best_d,
              "best_step_relerr": agg[str(best_d)]["mean_relerr"],
              "d_shao": d_shao, "near_shao_grid_d": near_shao,
              "near_shao_relerr": agg[str(near_shao)]["mean_relerr"],
              "config": vars(args)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 72)
    print("Pillar C — Shao FD 2nd-difference vs EXACT composed-JVP curvature (StyleGAN)")
    print("=" * 72)
    print(f"  SANITY — 1st-order: exact JVP |dg/da|={mean_first_exact:.4e}; "
          f"best 1st-order central-FD agreement = {agg[str(best_d1)]['mean_relerr_FIRST']:.2%} "
          f"at d={best_d1}")
    print(f"  (1st-order FD should match exact to ~<1%; if so pipeline is sound "
          f"and the 2nd-order blow-up below is genuine non-recoverability.)")
    print(f"\n  mean exact curvature |d^2 g/da^2| (ground truth, step-free): {mean_exact:.4e}")
    print(f"  {'step d':>10}  {'|c_fd2|':>11}  {'2nd rel.err':>12}  {'1st rel.err':>12}")
    for d in args.d_grid:
        flag = ""
        if d == best_d: flag += "  <- best 2nd step (FD floor)"
        if d == near_shao: flag += f"  <- Shao T={args.shao_T} (d~{d_shao:.3g})"
        print(f"  {d:>10.4g}  {agg[str(d)]['mean_c_fd']:>11.3e}  "
              f"{agg[str(d)]['mean_relerr']:>11.1%}  "
              f"{agg[str(d)]['mean_relerr_FIRST']:>11.1%}{flag}")
    print(f"\n  FD 2nd-order error floor (best step d={best_d}): {agg[str(best_d)]['mean_relerr']:.1%}")
    print(f"  Shao T={args.shao_T} (d~{d_shao:.3g}): "
          f"{agg[str(near_shao)]['mean_relerr']:.1%} rel.err  "
          f"-> FD reads curvature {agg[str(near_shao)]['mean_c_fd']/mean_exact:.2f}x the truth")
    print(f"\n  HONEST SCOPE: Shao's core numerical PRIMITIVE (FD second difference of g)")
    print(f"  is non-recoverable on a real generator -- >={agg[str(best_d)]['mean_relerr']:.0%} error at any step,")
    print(f"  validated against the exact instrument (1st-order FD agrees to "
          f"{agg[str(best_d1)]['mean_relerr_FIRST']:.1%}).")
    print(f"  We do NOT claim Shao's near-flat VERDICT is false here: that verdict is read")
    print(f"  off geodesic distances/MDS and needs the full geodesic pipeline to re-test.")
    print(f"  Claim = the primitive any such conclusion rests on is numerically unreliable.")
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
