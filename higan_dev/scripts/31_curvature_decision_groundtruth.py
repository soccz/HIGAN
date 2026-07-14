"""ⓐ-ii Experiment C — does the FD curvature proxy drive a real interpolation decision
WRONG, judged against an estimator-free ground truth?

The decision (real latent-editing practice): "which directions are curved enough to need
GEODESIC (careful) interpolation, vs linear is fine?" An analyst ranks directions by a
local curvature proxy. We break circularity by judging against a proxy-FREE ground truth:

  GROUND TRUTH (no FD, no AD): densely render the path G(z+αv), α∈[−T,T] at K points, and
    measure NONLINEARITY = mean image-space deviation of the rendered path from the straight
    chord, normalized by chord length. This is exactly "how wrong is linear interpolation" —
    the quantity the decision is about — computed by rendering only.

  PROXIES: c_AD = exact composed-JVP |∂²_α G| (step-free); c_FD = naive /δ² second diff.

  TEST: over directions, does c_AD predict the ground-truth nonlinearity ranking better than
    c_FD? If AD matches NL but FD does not, the cheap /δ² proxy mis-selects which directions
    need geodesic interpolation -> a real decision changes with the estimator (demonstrated
    consequence, TMLR criterion 2). If both match -> FD is safe even here (use-license).

Metrics: Spearman(c_AD, NL) vs Spearman(c_FD[δ], NL) over directions; and top-k(NL) recovery
by each proxy. Decided by result, no cherry-picking.
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
    ones = torch.ones(B, device=G.device); a0 = torch.zeros(B, device=G.device)
    def df(alpha):
        return jvp(f, (alpha,), (torch.ones_like(alpha),))[1]
    _, second = jvp(df, (a0,), (ones,))
    return second.abs().mean().item()


def fd_second(G, f, B, d):
    ap = torch.full((B,), +d, device=G.device); am = torch.full((B,), -d, device=G.device)
    a0 = torch.zeros(B, device=G.device)
    with torch.no_grad():
        gp = f(ap); g0 = f(a0); gm = f(am)
        return ((gp - 2.0 * g0 + gm) / (d * d)).abs().mean().item()


def path_nonlinearity(G, wp, v, T, K, chunk=8):
    """estimator-free ground truth: mean normalized deviation of the densely rendered path
    from the straight chord in image space."""
    alphas = torch.linspace(-T, T, K, device=G.device)
    imgs = []
    with torch.no_grad():
        for i in range(0, K, chunk):
            ac = alphas[i:i + chunk]
            wpb = wp + ac.view(-1, 1, 1) * v.unsqueeze(0)   # (c, L, D)
            imgs.append(G.synthesize(wpb))                  # (c,3,H,W)
    imgs = torch.cat(imgs, 0)                               # (K,3,H,W)
    g0, gT = imgs[0], imgs[-1]
    chord = gT - g0
    cn = chord.norm().clamp_min(1e-12)
    ts = (alphas + T) / (2 * T)
    devs = []
    for k in range(1, K - 1):
        lin = g0 + ts[k] * chord
        devs.append(((imgs[k] - lin).norm() / cn).item())
    return float(np.mean(devs))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def topk_recovery(score, truth, k):
    ts = set(np.argsort(truth)[::-1][:k]); ss = set(np.argsort(score)[::-1][:k])
    return len(ts & ss) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view", "carpet",
                             "cluttered_space", "dirt", "glossy", "scary"])
    ap.add_argument("--n-random-dirs", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--d-grid", type=float, nargs="+", default=[3.0, 1.0, 0.3, 0.1])
    ap.add_argument("--path-tmax", type=float, default=3.0)
    ap.add_argument("--n-path-pts", type=int, default=33)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="out/curvature_decision_gt")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name, device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    d_grid, dstrs = args.d_grid, [str(d) for d in args.d_grid]
    print(f"[{time.strftime('%H:%M:%S')}] G L={L} D={D} res={G.resolution} "
          f"dirs={len(args.attrs)}+{args.n_random_dirs} N={args.num_samples} "
          f"T={args.path_tmax} K={args.n_path_pts}")
    t0 = time.time()

    dirs = []
    for a in args.attrs:
        b = load_boundary(cfg.paths.boundaries_dir, a, num_layers=L).to(G.device)
        dirs.append((a, _layered_direction(b, L, D, G.device)))
    for r in range(args.n_random_dirs):
        g = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=g); rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv))

    rows = []
    for name, v in dirs:
        nl, cad = [], []
        cfd = {ds: [] for ds in dstrs}
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            f = curve_fn(G, wp, v, 1)
            nl.append(path_nonlinearity(G, wp, v, args.path_tmax, args.n_path_pts))
            cad.append(exact_second(G, f, 1))
            for d, ds in zip(d_grid, dstrs):
                cfd[ds].append(fd_second(G, f, 1, d))
        rows.append({"name": name, "NL": float(np.mean(nl)),
                     "c_AD": float(np.mean(cad)),
                     "c_FD": {ds: float(np.mean(cfd[ds])) for ds in dstrs}})
        print(f"  {name:16s} NL={rows[-1]['NL']:.4f} c_AD={rows[-1]['c_AD']:.3e} "
              f"c_FD@{dstrs[0]}={rows[-1]['c_FD'][dstrs[0]]:.3e} ({time.time()-t0:.0f}s)")

    names = [r["name"] for r in rows]
    NL = [r["NL"] for r in rows]; cAD = [r["c_AD"] for r in rows]
    rho_AD = spearman(cAD, NL)
    rho_FD = {ds: spearman([r["c_FD"][ds] for r in rows], NL) for ds in dstrs}
    rec_AD = {k: topk_recovery(cAD, NL, k) for k in (3, 5)}
    rec_FD = {ds: {k: topk_recovery([r["c_FD"][ds] for r in rows], NL, k) for k in (3, 5)}
              for ds in dstrs}

    # AD must predict the GT well (else the instrument is not the right proxy either);
    # consequence demonstrated iff AD predicts NL substantially better than FD at every step.
    ad_predicts = rho_AD >= 0.7
    fd_worse = all(rho_FD[ds] < rho_AD - 0.15 for ds in dstrs)
    consequence = ad_predicts and fd_worse

    result = {"rows": rows, "names": names,
              "rho_AD_vs_NL": rho_AD, "rho_FD_vs_NL": rho_FD,
              "topk_recovery_AD": rec_AD, "topk_recovery_FD": rec_FD,
              "verdict_ad_predicts_GT": bool(ad_predicts),
              "verdict_fd_worse_than_ad": bool(fd_worse),
              "verdict_consequence_demonstrated": bool(consequence),
              "config": vars(args)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 78)
    print("Experiment C — does FD curvature mis-drive the interpolation decision vs GT?")
    print("=" * 78)
    print(f"  ground-truth NONLINEARITY (estimator-free, dense render):")
    print(f"    most-nonlinear (true): {' > '.join([names[i] for i in np.argsort(NL)[::-1][:5]])} ...")
    print(f"\n  How well does each curvature proxy predict the TRUE nonlinearity ranking?")
    print(f"    exact AD :  Spearman(c_AD, NL) = {rho_AD:+.3f}   "
          f"top3-recovery={rec_AD[3]:.2f} top5={rec_AD[5]:.2f}")
    for ds in dstrs:
        print(f"    FD /δ² δ={ds:>5}: Spearman(c_FD, NL) = {rho_FD[ds]:+.3f}   "
              f"top3={rec_FD[ds][3]:.2f} top5={rec_FD[ds][5]:.2f}")
    print(f"\n  VERDICT: AD predicts GT (ρ≥0.7): {ad_predicts} | "
          f"FD worse than AD everywhere: {fd_worse}")
    print("  " + ("-> CONSEQUENCE DEMONSTRATED: the cheap /δ² proxy mis-selects which "
                  "directions need geodesic interpolation; exact AD is required (TMLR)."
                  if consequence else
                  "-> NO demonstrated decision flip: FD predicts the GT about as well as AD "
                  "(use-license; lean numerical venue / borderline TMLR)."))
    print(f"\nsaved {out/'metrics.json'}")

    if not args.no_fig:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            ax[0].scatter(cAD, NL, s=20, label=f"AD ρ={rho_AD:+.2f}")
            ax[0].scatter([r["c_FD"][dstrs[0]] for r in rows], NL, s=20, marker="x",
                          label=f"FD δ={dstrs[0]} ρ={rho_FD[dstrs[0]]:+.2f}")
            ax[0].set_xscale("log"); ax[0].set_xlabel("curvature proxy")
            ax[0].set_ylabel("true nonlinearity NL"); ax[0].legend(fontsize=8)
            ax[0].set_title("proxy vs ground-truth")
            xs = ["AD"] + [f"FD δ={ds}" for ds in dstrs]
            ax[1].bar(range(len(xs)), [rho_AD] + [rho_FD[ds] for ds in dstrs],
                      color=["C0"] + ["C3"] * len(dstrs))
            ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs, rotation=30, fontsize=8)
            ax[1].set_ylabel("Spearman(proxy, NL)"); ax[1].set_title("which proxy predicts the decision")
            fig.tight_layout(); fig.savefig(out / "decision_gt.pdf")
            print(f"saved {out/'decision_gt.pdf'}")
        except Exception as e:
            print(f"[fig skipped] {e}")


if __name__ == "__main__":
    main()
