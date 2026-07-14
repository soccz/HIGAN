"""ⓐ-ii (TMLR consequence test) — does the naive /δ² second-difference curvature
estimator INVERT a curvature-based ORDERING that the exact composed-JVP reveals?

Two passes (design: note/AII_curvature_ordering_design.md):

  PASS B (PRIMARY, per-direction ordering flip): order a set of latent directions
    (the 8 annotated attribute boundaries in their natural manipulate-layers + K random
    w-directions) by mean curvature intensity, exact composed-JVP vs naive /δ². The flip
    lives here: directions whose true curvatures sit near the fp32 floor get a
    floor-driven (≈noise) FD reading, so "which direction is most curved" — a natural
    latent-geometry question — is reordered. Metric: Spearman(exact, FD) over directions.

  PASS A (SECONDARY, per-layer magnitude floor): restrict the tangent to each synthesis
    layer; exact orders layers by monotone coarse→fine decay (the §5a claim). The /δ²
    floor crushes the magnitudes (off by ~100x) but, because the trend spans orders of
    magnitude, the RANK survives — an honest non-flip that motivates why the flip needs
    similar-magnitude directions (Pass B), not the gross layer trend.

  CONTROL: 1st-order central difference at a small step vs the exact first-order JVP must
    agree to ~few % — proves the pipeline is sound and the 2nd-order collapse is genuine
    non-recoverability, not a bug.

We do NOT cherry-pick a step; the per-direction flip claim requires low rank agreement at
every δ in the grid. ρ here is curvature intensity (a standard quantity), not a self-
defined ratio. We do not claim any published paper is wrong (see design doc firewall).
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
    def f(alpha):
        return G.synthesize(wp_base + alpha.view(B, 1, 1) * v_layered.unsqueeze(0))
    return f


def exact_second(G, f, B):
    ones = torch.ones(B, device=G.device)
    a0 = torch.zeros(B, device=G.device)

    def df(alpha):
        return jvp(f, (alpha,), (torch.ones_like(alpha),))[1]

    _, second = jvp(df, (a0,), (ones,))
    _, first = jvp(f, (a0,), (ones,))
    return second.abs().mean().item(), first.abs().mean().item()


def fd_both(G, f, B, d):
    ap = torch.full((B,), +d, device=G.device)
    am = torch.full((B,), -d, device=G.device)
    a0 = torch.zeros(B, device=G.device)
    with torch.no_grad():
        gp = f(ap); g0 = f(a0); gm = f(am)
        second = (gp - 2.0 * g0 + gm) / (d * d)
        first = (gp - gm) / (2.0 * d)
    return second.abs().mean().item(), first.abs().mean().item()


def _avg_rank(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x, y):
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _pair_reversals(ex, fd):
    """fraction of direction pairs whose order disagrees between exact and FD."""
    n = len(ex); bad = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ex[i] == ex[j]:
                continue
            tot += 1
            if (ex[i] - ex[j]) * (fd[i] - fd[j]) < 0:
                bad += 1
    return bad / max(tot, 1)


def measure_direction(G, base_wp, v, N, d_grid, dstrs, ctrl_d):
    ce, cf1e = [], []
    cfd2 = {ds: [] for ds in dstrs}
    ctrl_fd1 = []
    for s in range(N):
        wp = base_wp[s:s + 1].detach()
        f = curve_fn(G, wp, v, 1)
        c2, c1 = exact_second(G, f, 1)
        ce.append(c2); cf1e.append(c1)
        for d, ds in zip(d_grid, dstrs):
            cfd2[ds].append(fd_both(G, f, 1, d)[0])
        ctrl_fd1.append(fd_both(G, f, 1, ctrl_d)[1])
    return {"c_exact": float(np.mean(ce)),
            "c_first_exact": float(np.mean(cf1e)),
            "c_fd2": {ds: float(np.mean(cfd2[ds])) for ds in dstrs},
            "ctrl_first_relerr": float(np.mean(
                [abs(a - b) / max(b, 1e-12) for a, b in zip(ctrl_fd1, cf1e)]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view", "carpet",
                             "cluttered_space", "dirt", "glossy", "scary"])
    ap.add_argument("--n-random-dirs", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--d-grid", type=float, nargs="+", default=[3.0, 1.0, 0.3, 0.1])
    ap.add_argument("--ctrl-step", type=float, default=0.02,
                    help="small step for the 1st-order soundness control")
    ap.add_argument("--per-layer-attrs", nargs="+",
                    default=["indoor_lighting", "view", "wood"])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="out/curvature_ordering_flip")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    d_grid, dstrs = args.d_grid, [str(d) for d in args.d_grid]
    print(f"[{time.strftime('%H:%M:%S')}] G loaded L={L} D={D} res={G.resolution} "
          f"attrs={len(args.attrs)} rand={args.n_random_dirs} N={args.num_samples} "
          f"d_grid={d_grid} ctrl={args.ctrl_step}")
    t0 = time.time()

    # ── PASS B: per-direction ordering ───────────────────────────────────────
    dirs = []
    for a in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, a, num_layers=L).to(G.device)
        dirs.append((a, _layered_direction(b, L, D, G.device)))  # natural manipulate-layers
    for r in range(args.n_random_dirs):
        g = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=g); rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv))

    dir_rows = []
    for name, v in dirs:
        m = measure_direction(G, base_wp, v, args.num_samples, d_grid, dstrs, args.ctrl_step)
        m["name"] = name; dir_rows.append(m)
        print(f"  [B] {name:16s} c*={m['c_exact']:.3e} "
              f"cFD@{dstrs[0]}={m['c_fd2'][dstrs[0]]:.3e} ({time.time()-t0:.0f}s)")

    names = [r["name"] for r in dir_rows]
    ex = [r["c_exact"] for r in dir_rows]
    rho_dir = {ds: spearman(ex, [r["c_fd2"][ds] for r in dir_rows]) for ds in dstrs}
    rev_dir = {ds: _pair_reversals(ex, [r["c_fd2"][ds] for r in dir_rows]) for ds in dstrs}
    order_exact = [names[i] for i in np.argsort(ex)[::-1]]
    order_fd = {ds: [names[i] for i in np.argsort([r["c_fd2"][ds] for r in dir_rows])[::-1]]
                for ds in dstrs}
    ctrl_relerr = float(np.mean([r["ctrl_first_relerr"] for r in dir_rows]))

    # ── PASS A: per-layer (magnitude floor; expected non-flip) ───────────────
    layers = list(range(L))
    pl_records = []
    for a in args.per_layer_attrs:
        b = load_boundary(cfg.paths.boundaries_dir, a, num_layers=L).to(G.device)
        for ell in layers:
            v = _layered_direction(b, L, D, G.device, only_layer=ell)
            m = measure_direction(G, base_wp, v, args.num_samples, d_grid, dstrs, args.ctrl_step)
            m["attr"] = a; m["layer"] = ell; pl_records.append(m)
        print(f"  [A] per-layer {a:16s} done ({time.time()-t0:.0f}s)")
    c_exact_layer = [float(np.mean([r["c_exact"] for r in pl_records if r["layer"] == e]))
                     for e in layers]
    c_fd_layer = {ds: [float(np.mean([r["c_fd2"][ds] for r in pl_records if r["layer"] == e]))
                       for e in layers] for ds in dstrs}
    rho_layer_exact = spearman(layers, c_exact_layer)
    rho_layer_fd = {ds: spearman(layers, c_fd_layer[ds]) for ds in dstrs}

    # ── verdict ──────────────────────────────────────────────────────────────
    control_ok = ctrl_relerr < 0.10
    dir_flip = control_ok and all(rho_dir[ds] < 0.5 for ds in dstrs)

    result = {
        "pass_B_directions": dir_rows, "dir_names": names,
        "rho_dir_exact_vs_fd": rho_dir, "pair_reversal_frac": rev_dir,
        "order_exact": order_exact, "order_fd": order_fd,
        "control_first_relerr": ctrl_relerr,
        "pass_A_per_layer": {"c_exact_layer": c_exact_layer, "c_fd_layer": c_fd_layer,
                             "rho_layer_exact": rho_layer_exact, "rho_layer_fd": rho_layer_fd},
        "verdict_direction_flip": bool(dir_flip), "verdict_control_ok": bool(control_ok),
        "config": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 78)
    print("ⓐ-ii — curvature-ordering flip: exact composed-JVP vs naive /δ² second diff")
    print("=" * 78)
    print(f"  CONTROL 1st-order central-FD @δ={args.ctrl_step} vs exact: "
          f"{ctrl_relerr:.2%}  -> {'SOUND' if control_ok else 'CHECK PIPELINE'}")
    print(f"\n  PASS B — per-direction ordering ({len(dirs)} dirs = "
          f"{len(args.attrs)} attrs + {args.n_random_dirs} random):")
    print(f"    Spearman(exact, FD) over directions  |  pairwise-reversal fraction:")
    for ds in dstrs:
        print(f"      δ={ds:>5}:  ρ = {rho_dir[ds]:+.3f}   reversals = {rev_dir[ds]:.1%}")
    print(f"    most-curved (exact):  {' > '.join(order_exact[:5])} ...")
    print(f"    most-curved (FD δ={dstrs[0]}): {' > '.join(order_fd[dstrs[0]][:5])} ...")
    print(f"\n  PASS A — per-layer (magnitude floor; expect rank-robust non-flip):")
    print(f"    exact ρ(layer,curv) = {rho_layer_exact:+.3f}; "
          f"FD ρ = {{{', '.join(f'{ds}:{rho_layer_fd[ds]:+.2f}' for ds in dstrs)}}}")
    print(f"    layer0 curvature: exact={c_exact_layer[0]:.3e} vs FD@{dstrs[0]}="
          f"{c_fd_layer[dstrs[0]][0]:.3e}  (floor crushes magnitude)")
    print(f"\n  VERDICT: per-direction ordering flip = {dir_flip}  (control sound = {control_ok})")
    print("  " + ("-> H1 CONFIRMED: naive /δ² reorders which directions are most curved "
                   "(TMLR consequence)" if dir_flip
                   else "-> NO ROBUST FLIP across the grid — read ρ_dir; lean ⓑ if ρ stays high"))
    print(f"\nsaved {out/'metrics.json'}")

    if not args.no_fig:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ax = axes[0]
            fd0 = [r["c_fd2"][dstrs[0]] for r in dir_rows]
            ax.scatter(ex, fd0, s=18)
            for i, nm in enumerate(names):
                if nm in args.attrs:
                    ax.annotate(nm, (ex[i], fd0[i]), fontsize=6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("exact curvature (truth)"); ax.set_ylabel(f"naive /δ² δ={dstrs[0]}")
            ax.set_title(f"per-direction: ρ={rho_dir[dstrs[0]]:+.2f}, "
                         f"rev={rev_dir[dstrs[0]]:.0%}")
            ax2 = axes[1]
            ax2.plot(layers, c_exact_layer, "o-", lw=2, label="exact (truth)")
            for ds in dstrs:
                ax2.plot(layers, c_fd_layer[ds], ".--", alpha=.7, label=f"/δ² δ={ds}")
            ax2.set_yscale("log"); ax2.set_xlabel("synthesis layer"); ax2.legend(fontsize=7)
            ax2.set_ylabel(r"mean $|\partial^2_\alpha G|$"); ax2.set_title("per-layer floor")
            fig.tight_layout(); fig.savefig(out / "ordering_flip.pdf")
            print(f"saved {out/'ordering_flip.pdf'}")
        except Exception as e:
            print(f"[fig skipped] {e}")


if __name__ == "__main__":
    main()
