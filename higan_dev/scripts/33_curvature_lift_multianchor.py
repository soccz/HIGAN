"""GO/NO-GO (multi-anchor, larger-N, bootstrap) — is the curvature LIFT over first-order
displacement (scripts/32: partial-Spearman(c_AD, NL | jvp)=+0.45, p=.016, N=24, 1 anchor)
ROBUST, or an N=24/single-anchor artifact?

Strengthens scripts/32 along the three stated weaknesses:
  - larger N : 8 attrs + N_RANDOM random directions  (correlation points, vs 24)
  - multiple base-latent ANCHORS : recompute the whole per-direction correlation for several
    independent base-latent seeds -> a DISTRIBUTION of the partial, not a point estimate
  - multiple control simultaneously : partial(c_AD, NL | jvp, chord) via rank least-squares
  - bootstrap CI over directions, per anchor

Pre-registered (synthesis go/no-go):
  GO  (positive TMLR): >=4/5 anchors partial(c_AD,NL|jvp) > 0 AND median >~ 0.25 AND the
       multiple-control partial bootstrap CI excludes 0 in most anchors.
  NEG (clean-negative TMLR): medians positive but anchors straddle 0 / inconsistent
       -> "curvature lift is anchor-fragile; 1st-order displacement is the sufficient statistic".
  STOP: median partial <= 0 -> already-dead magnitude dominance.
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


def curve_fn(G, wp_base, v, B):
    def f(alpha):
        return G.synthesize(wp_base + alpha.view(B, 1, 1) * v.unsqueeze(0))
    return f


def exact_second(G, f, B):
    a0 = torch.zeros(B, device=G.device)
    def df(alpha):
        return jvp(f, (alpha,), (torch.ones_like(alpha),))[1]
    _, second = jvp(df, (a0,), (torch.ones(B, device=G.device),))
    return second.abs().mean().item()


def exact_first(G, f, B):
    a0 = torch.zeros(B, device=G.device)
    _, first = jvp(f, (a0,), (torch.ones_like(a0),))
    return first.norm().item()


def path_nl_chord(G, wp, v, T, K, chunk=8):
    alphas = torch.linspace(-T, T, K, device=G.device)
    imgs = []
    with torch.no_grad():
        for i in range(0, K, chunk):
            ac = alphas[i:i + chunk]
            imgs.append(G.synthesize(wp + ac.view(-1, 1, 1) * v.unsqueeze(0)))
    imgs = torch.cat(imgs, 0)
    g0, gT = imgs[0], imgs[-1]; chord = gT - g0; cn = chord.norm().clamp_min(1e-12)
    ts = (alphas + T) / (2 * T)
    devs = [((imgs[k] - (g0 + ts[k] * chord)).norm() / cn).item() for k in range(1, K - 1)]
    return float(np.mean(devs)), float(cn.item())


def rankvec(x):
    r = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    return r - r.mean()


def spearman(x, y):
    rx, ry = rankvec(x), rankvec(y)
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float((rx @ ry) / den) if den > 0 else float("nan")


def partial(x, y, controls):
    """partial Spearman of x,y controlling a list of control vectors (rank least-squares)."""
    rx, ry = rankvec(x), rankvec(y)
    Z = np.column_stack([rankvec(c) for c in controls])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None); ex = rx - Z @ bx
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None); ey = ry - Z @ by
    den = np.sqrt((ex @ ex) * (ey @ ey))
    return float((ex @ ey) / den) if den > 0 else float("nan")


def boot_ci(fn, n_items, n=2000, seed=0):
    rg = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rg.integers(0, n_items, n_items)
        v = fn(idx)
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-name", default=None,
                    help="override generator (e.g. stylegan2_church); default uses config")
    ap.add_argument("--random-only", action="store_true",
                    help="use only random w-directions (no attribute boundaries)")
    ap.add_argument("--n-random", type=int, default=42)
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--anchors", type=int, default=5)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--K", type=int, default=33)
    ap.add_argument("--base-seed0", type=int, default=31)
    ap.add_argument("--out", default="out/curvature_lift_multianchor")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    attrs = ["indoor_lighting", "wood", "view", "carpet",
             "cluttered_space", "dirt", "glossy", "scary"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=args.model_name or cfg.generator.model_name,
                       device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    print(f"[{time.strftime('%H:%M:%S')}] L={L} D={D} dirs={8+args.n_random} "
          f"n_base={args.n_base} anchors={args.anchors} K={args.K}")
    t0 = time.time()

    # fixed direction set across anchors
    dirs = []
    if not args.random_only:
        for a in attrs:
            b = load_boundary(cfg.paths.boundaries_dir, a, num_layers=L).to(G.device)
            dirs.append((a, _layered_direction(b, L, D, G.device)))
    for r in range(args.n_random):
        g = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=g); rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv))
    names = [n for n, _ in dirs]

    anchor_results = []
    for a in range(args.anchors):
        rng = torch.Generator(device=G.device).manual_seed(args.base_seed0 + a)
        base_wp = G.sample_wp(args.n_base, generator=rng)
        NL, cAD, jvpn, chord = [], [], [], []
        for name, v in dirs:
            nl, ad, jv, ch = [], [], [], []
            for s in range(args.n_base):
                wp = base_wp[s:s + 1].detach()
                f = curve_fn(G, wp, v, 1)
                n_, c_ = path_nl_chord(G, wp, v, args.T, args.K)
                nl.append(n_); ch.append(c_)
                ad.append(exact_second(G, f, 1)); jv.append(exact_first(G, f, 1))
            NL.append(np.mean(nl)); cAD.append(np.mean(ad))
            jvpn.append(np.mean(jv)); chord.append(np.mean(ch))
        NL, cAD, jvpn, chord = map(np.array, (NL, cAD, jvpn, chord))

        p_jvp = partial(cAD, NL, [jvpn])
        p_both = partial(cAD, NL, [jvpn, chord])
        rg = np.random.default_rng(7)
        null = np.array([partial(rg.permutation(cAD), NL, [jvpn]) for _ in range(5000)])
        perm_p = float((np.sum(null >= p_jvp) + 1) / 5001)
        lo_j, hi_j, med_j = boot_ci(lambda idx: partial(cAD[idx], NL[idx], [jvpn[idx]]), len(names))
        lo_b, hi_b, med_b = boot_ci(lambda idx: partial(cAD[idx], NL[idx], [jvpn[idx], chord[idx]]), len(names))
        ar = {"anchor_seed": args.base_seed0 + a,
              "rho_cAD_NL": spearman(cAD, NL), "rho_jvp_NL": spearman(jvpn, NL),
              "rho_cAD_jvp": spearman(cAD, jvpn),
              "partial_given_jvp": p_jvp, "perm_p_given_jvp": perm_p,
              "partial_given_jvp_chord": p_both,
              "boot95_given_jvp": [lo_j, hi_j], "boot95_given_jvp_chord": [lo_b, hi_b]}
        anchor_results.append(ar)
        print(f"  anchor {args.base_seed0+a}: partial(c|jvp)={p_jvp:+.3f} "
              f"(perm p={perm_p:.3f}, 95%CI[{lo_j:+.2f},{hi_j:+.2f}])  "
              f"partial(c|jvp,chord)={p_both:+.3f} (95%CI[{lo_b:+.2f},{hi_b:+.2f}])  "
              f"({time.time()-t0:.0f}s)")

    pj = np.array([r["partial_given_jvp"] for r in anchor_results])
    pb = np.array([r["partial_given_jvp_chord"] for r in anchor_results])
    n_pos = int(np.sum(pj > 0))
    ci_excl_0 = int(np.sum([r["boot95_given_jvp"][0] > 0 for r in anchor_results]))
    go = (n_pos >= max(1, int(np.ceil(0.8 * args.anchors)))) and (np.median(pj) >= 0.25) \
        and (ci_excl_0 >= int(np.ceil(0.5 * args.anchors)))
    straddle = (np.median(pj) > 0) and not go
    verdict = "GO_positive_TMLR" if go else ("NEGATIVE_TMLR_anchor_fragile" if straddle else "STOP_magnitude_dominates")

    result = {"anchors": anchor_results,
              "partial_given_jvp_distribution": pj.tolist(),
              "partial_given_jvp_chord_distribution": pb.tolist(),
              "n_anchors": args.anchors, "n_positive": n_pos,
              "median_partial_given_jvp": float(np.median(pj)),
              "median_partial_given_jvp_chord": float(np.median(pb)),
              "n_anchors_CI_excludes_0": ci_excl_0,
              "verdict": verdict, "config": vars(args)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 78)
    print("Curvature LIFT over displacement — multi-anchor robustness")
    print("=" * 78)
    print(f"  partial(c_AD, NL | jvp) across {args.anchors} anchors: "
          f"{[f'{x:+.2f}' for x in pj]}")
    print(f"    median = {np.median(pj):+.3f}, positive in {n_pos}/{args.anchors}, "
          f"CI-excludes-0 in {ci_excl_0}/{args.anchors}")
    print(f"  partial(c_AD, NL | jvp, chord) across anchors: {[f'{x:+.2f}' for x in pb]}")
    print(f"    median = {np.median(pb):+.3f}")
    print(f"\n  VERDICT: {verdict}")
    print({"GO_positive_TMLR": "  -> curvature carries robust displacement-orthogonal lift; pursue positive TMLR + FFHQ generality.",
           "NEGATIVE_TMLR_anchor_fragile": "  -> lift is anchor-fragile; clean-negative TMLR ('1st-order displacement suffices').",
           "STOP_magnitude_dominates": "  -> magnitude dominates; instrument-only numerical venue."}[verdict])
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
