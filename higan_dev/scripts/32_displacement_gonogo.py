"""GO/NO-GO: does exact local curvature c_AD predict path nonlinearity NL
INCREMENTALLY over first-order displacement magnitude?

Adds to the existing decision-GT harness the trivial baselines the negative claim
must beat:
  jvp_norm  = |d_alpha G| at alpha=0  (exact first-order JVP, one call)
  chord_len = ||G(z+Tv) - G(z)||      (raw global displacement, the NL normalizer)
  vnorm     = ||v||                    (degenerate control, ~const)

Pre-registered NEGATIVE-CONFIRMED iff:
  partial-Spearman(c_AD, NL | jvp_norm) NOT significantly > 0  AND
  max(rho(jvp_norm,NL), rho(chord_len,NL)) >= rho(c_AD, NL).
"""
from __future__ import annotations
import json, time
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


def exact_first(G, f, B):
    a0 = torch.zeros(B, device=G.device)
    _, first = jvp(f, (a0,), (torch.ones_like(a0),))
    return first.norm().item()


def path_nl_and_chord(G, wp, v, T, K, chunk=8):
    alphas = torch.linspace(-T, T, K, device=G.device)
    imgs = []
    with torch.no_grad():
        for i in range(0, K, chunk):
            ac = alphas[i:i + chunk]
            wpb = wp + ac.view(-1, 1, 1) * v.unsqueeze(0)
            imgs.append(G.synthesize(wpb))
    imgs = torch.cat(imgs, 0)
    g0, gT = imgs[0], imgs[-1]
    chord = gT - g0
    cn = chord.norm().clamp_min(1e-12)
    ts = (alphas + T) / (2 * T)
    devs = [((imgs[k] - (g0 + ts[k] * chord)).norm() / cn).item() for k in range(1, K - 1)]
    return float(np.mean(devs)), float(cn.item())


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def rankvec(x):
    r = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    return r - r.mean()


def partial_spearman(x, y, z):
    """partial Spearman of x,y controlling z: residualize ranks, then correlate."""
    rx, ry, rz = rankvec(x), rankvec(y), rankvec(z)
    bx = (rx @ rz) / (rz @ rz); ex = rx - bx * rz
    by = (ry @ rz) / (rz @ rz); ey = ry - by * rz
    den = np.sqrt((ex @ ex) * (ey @ ey))
    return float((ex @ ey) / den) if den > 0 else float("nan")


def main():
    cfg = Config.load(resolve("configs/default.yaml"))
    attrs = ["indoor_lighting", "wood", "view", "carpet",
             "cluttered_space", "dirt", "glossy", "scary"]
    N, T, K, seed = 24, 3.0, 33, 31
    out = Path("out/displacement_gonogo"); out.mkdir(parents=True, exist_ok=True)
    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name, device=cfg.train.device)
    L, D = G.num_layers, G.w_dim
    rng = torch.Generator(device=G.device).manual_seed(seed)
    base_wp = G.sample_wp(N, generator=rng)
    print(f"[{time.strftime('%H:%M:%S')}] L={L} D={D} N={N}")
    t0 = time.time()

    dirs = []
    for a in attrs:
        b = load_boundary(cfg.paths.boundaries_dir, a, num_layers=L).to(G.device)
        dirs.append((a, _layered_direction(b, L, D, G.device)))
    for r in range(16):
        g = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=g); rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv))

    rows = []
    for name, v in dirs:
        nl, cad, jvpn, chord, vn = [], [], [], [], []
        for s in range(N):
            wp = base_wp[s:s + 1].detach()
            f = curve_fn(G, wp, v, 1)
            n, c = path_nl_and_chord(G, wp, v, T, K)
            nl.append(n); chord.append(c)
            cad.append(exact_second(G, f, 1))
            jvpn.append(exact_first(G, f, 1))
            vn.append(float(v.norm().item()))
        rows.append({"name": name, "NL": float(np.mean(nl)),
                     "c_AD": float(np.mean(cad)),
                     "jvp_norm": float(np.mean(jvpn)),
                     "chord_len": float(np.mean(chord)),
                     "v_norm": float(np.mean(vn))})
        print(f"  {name:16s} NL={rows[-1]['NL']:.3f} c_AD={rows[-1]['c_AD']:.3e} "
              f"jvp={rows[-1]['jvp_norm']:.3e} chord={rows[-1]['chord_len']:.3e} "
              f"({time.time()-t0:.0f}s)")

    names = [r["name"] for r in rows]
    NL = np.array([r["NL"] for r in rows])
    cAD = np.array([r["c_AD"] for r in rows])
    jvpn = np.array([r["jvp_norm"] for r in rows])
    chord = np.array([r["chord_len"] for r in rows])

    def perm_p(stat_fn, obs, *arrs, n=10000):
        rg = np.random.default_rng(0)
        null = np.array([stat_fn(rg.permutation(arrs[0]), *arrs[1:]) for _ in range(n)])
        return float((np.sum(null >= obs) + 1) / (n + 1))

    res = {}
    for tag, mask in [("full", np.ones(len(names), bool)),
                      ("no_view", np.array([n != "view" for n in names]))]:
        nl_, ad_, jv_, ch_ = NL[mask], cAD[mask], jvpn[mask], chord[mask]
        r_ad = spearman(ad_, nl_); r_jv = spearman(jv_, nl_); r_ch = spearman(ch_, nl_)
        p_ad = perm_p(spearman, r_ad, ad_, nl_)
        p_jv = perm_p(spearman, r_jv, jv_, nl_)
        p_ch = perm_p(spearman, r_ch, ch_, nl_)
        pr_ad_jv = partial_spearman(ad_, nl_, jv_)
        pr_ad_ch = partial_spearman(ad_, nl_, ch_)
        # permutation null for partial: permute c_AD ranks, recompute partial
        rg = np.random.default_rng(1)
        null_pr = np.array([partial_spearman(rg.permutation(ad_), nl_, jv_) for _ in range(10000)])
        p_partial_jv = float((np.sum(null_pr >= pr_ad_jv) + 1) / 10001)
        null_pr2 = np.array([partial_spearman(rg.permutation(ad_), nl_, ch_) for _ in range(10000)])
        p_partial_ch = float((np.sum(null_pr2 >= pr_ad_ch) + 1) / 10001)

        baseline_ties_or_beats = max(r_jv, r_ch) >= r_ad
        partial_not_sig = (p_partial_jv > 0.05)
        negative_confirmed = baseline_ties_or_beats and partial_not_sig
        res[tag] = {
            "rho_cAD_NL": r_ad, "p_cAD": p_ad,
            "rho_jvp_NL": r_jv, "p_jvp": p_jv,
            "rho_chord_NL": r_ch, "p_chord": p_ch,
            "rho_jvp_chord": spearman(jv_, ch_),
            "rho_cAD_jvp": spearman(ad_, jv_),
            "partial_cAD_NL_given_jvp": pr_ad_jv, "p_partial_given_jvp": p_partial_jv,
            "partial_cAD_NL_given_chord": pr_ad_ch, "p_partial_given_chord": p_partial_ch,
            "baseline_ties_or_beats_cAD": bool(baseline_ties_or_beats),
            "partial_not_significant": bool(partial_not_sig),
            "NEGATIVE_CONFIRMED": bool(negative_confirmed),
        }

    out_json = {"rows": rows, "results": res}
    (out / "metrics.json").write_text(json.dumps(out_json, indent=2))
    print("\n" + "=" * 78)
    for tag in res:
        r = res[tag]
        print(f"[{tag}]")
        print(f"  rho(c_AD,NL)={r['rho_cAD_NL']:+.3f} (p={r['p_cAD']:.3f})  "
              f"rho(jvp,NL)={r['rho_jvp_NL']:+.3f} (p={r['p_jvp']:.3f})  "
              f"rho(chord,NL)={r['rho_chord_NL']:+.3f} (p={r['p_chord']:.3f})")
        print(f"  partial(c_AD,NL|jvp)={r['partial_cAD_NL_given_jvp']:+.3f} "
              f"(perm p={r['p_partial_given_jvp']:.3f})  "
              f"partial(c_AD,NL|chord)={r['partial_cAD_NL_given_chord']:+.3f} "
              f"(perm p={r['p_partial_given_chord']:.3f})")
        print(f"  rho(jvp,chord)={r['rho_jvp_chord']:+.3f}  rho(c_AD,jvp)={r['rho_cAD_jvp']:+.3f}")
        print(f"  >> NEGATIVE_CONFIRMED = {r['NEGATIVE_CONFIRMED']}")
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
