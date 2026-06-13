"""Challenge A — does FD vs exact composed-JVP change the Park-NeurIPS23
reproduction's conclusion?

Park's 2nd-order ratio rho_i = |d^2 x / d alpha^2| / |dx/d alpha| along the top
right-singular directions. The reproduction (run_park_repro.py) estimates the
SECOND derivative by FINITE-DIFFERENCING two JVPs:  (jvp(+eps) - jvp(-eps))/(2 eps).
That is FD on a second-order quantity through a deep DDIM chain — exactly the
regime where FD truncation/cancellation is unreliable (see note/: toy shows FD
cannot recover 2nd order at any eps).

Here we recompute rho TWO ways and ask whether the conclusion
(Spearman(sigma, rho)) changes:
  (a) FD-of-JVP at several eps  (the reproduction's method)
  (b) EXACT composed JVP        (jvp of jvp; no eps)
If the sign / magnitude of Spearman(sigma, rho) differs between (a) and (b),
the published-style conclusion depends on the numerical method, not the geometry.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp
from scipy.stats import spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
from lib.reproducibility import set_deterministic, run_metadata   # noqa: E402
from diffusion.generator import SDH, SDConfig                     # noqa: E402


def chain_fn(sdh, x_edit, v, t_idx, cond, uncond):
    """Return f(alpha) = x_T given an h-space perturbation alpha*v injected at t_idx."""
    def f(alpha):
        x = x_edit
        sdh._h_v = v; sdh._h_alpha = alpha
        for i in range(t_idx, sdh.cfg.num_inference_steps):
            sdh._h_active = (i == t_idx)
            eps = sdh.epsilon(x, i, cond, uncond)
            x = sdh.ddim_step(x, eps, i)
        sdh._h_active = False; sdh._h_v = None; sdh._h_alpha = None
        return x
    return f


def jvp1(sdh, x_edit, v, alpha_val, t_idx, cond, uncond):
    f = chain_fn(sdh, x_edit, v, t_idx, cond, uncond)
    ap = torch.tensor(alpha_val, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    at = torch.tensor(1.0, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    return jvp(f, (ap,), (at,))[1]


def second_fd(sdh, x_edit, v, eps, t_idx, cond, uncond):
    """FD-of-JVP 2nd derivative: (jvp(+eps) - jvp(-eps)) / (2 eps)."""
    dxp = jvp1(sdh, x_edit, v, +eps, t_idx, cond, uncond)
    dxm = jvp1(sdh, x_edit, v, -eps, t_idx, cond, uncond)
    first = 0.5 * (dxp.abs().mean() + dxm.abs().mean()).item()
    second = ((dxp - dxm) / (2 * eps)).abs().mean().item()
    return second, first


def second_exact(sdh, x_edit, v, t_idx, cond, uncond):
    """EXACT 2nd derivative via composed JVP at alpha=0 (no eps)."""
    f = chain_fn(sdh, x_edit, v, t_idx, cond, uncond)
    one = torch.tensor(1.0, device=sdh.cfg.device, dtype=sdh.cfg.dtype)

    def df(alpha):
        return jvp(f, (alpha,), (one,))[1]
    a0 = torch.tensor(0.0, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    d1 = jvp(f, (a0,), (one,))[1]
    _, d2 = jvp(df, (a0,), (one,))
    first = d1.abs().mean().item()
    second = d2.abs().mean().item()
    return second, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--K-probes", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--t-edit", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[0.1, 0.05, 0.01, 0.001])
    ap.add_argument("--paper-eps", type=float, default=0.05)
    ap.add_argument("--prompt", default="a photograph of a face")
    ap.add_argument("--out", default="experiments/out/park_fd_vs_exact")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sdh = SDH(SDConfig(resolution=args.resolution, num_inference_steps=args.steps))
    cond, uncond = sdh.encode_prompt(args.prompt, "")
    hs = sdh.h_space_shape()
    flat_dim = int(np.prod(hs[1:]))
    print(f"[{time.strftime('%H:%M:%S')}] SD loaded; h={hs}")

    per_seed = []
    for seed in range(args.n_seeds):
        t0 = time.time()
        gen = torch.Generator(device=sdh.cfg.device).manual_seed(seed)
        probes = torch.randn(args.K_probes, flat_dim, device=sdh.cfg.device,
                             dtype=sdh.cfg.dtype, generator=gen)
        probes = probes / probes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        probes_h = probes.reshape(args.K_probes, *hs[1:])

        H = W = sdh.cfg.resolution // 8
        gx = torch.Generator(device=sdh.cfg.device).manual_seed(seed * 17 + 3)
        with torch.no_grad():
            x = torch.randn(1, 4, H, W, generator=gx, device=sdh.cfg.device,
                            dtype=sdh.cfg.dtype)
            for i in range(args.t_edit):
                x = sdh.ddim_step(x, sdh.epsilon(x, i, cond, uncond), i)
        x_edit = x.detach()

        # build first-order J (FD-of-JVP averaged, same as repro) -> SVD basis
        J_cols = []
        for k in range(args.K_probes):
            dxp = jvp1(sdh, x_edit, probes_h[k:k+1], +args.paper_eps, args.t_edit, cond, uncond)
            dxm = jvp1(sdh, x_edit, probes_h[k:k+1], -args.paper_eps, args.t_edit, cond, uncond)
            J_cols.append(((dxp + dxm).flatten() / 2.0).detach().cpu().float().numpy())
            torch.cuda.empty_cache()
        J = np.stack(J_cols, axis=1)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        top_V = Vt[:args.top_k]

        # for each top direction: rho via FD(eps grid) and exact
        rho_fd = {e: [] for e in args.eps_grid}
        rho_exact = []
        for i in range(args.top_k):
            v_i = torch.zeros(*hs, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
            for k in range(args.K_probes):
                v_i = v_i + float(top_V[i, k]) * probes_h[k:k+1]
            v_i = v_i / v_i.norm().clamp_min(1e-8)
            for e in args.eps_grid:
                s2, s1 = second_fd(sdh, x_edit, v_i, e, args.t_edit, cond, uncond)
                rho_fd[e].append(s2 / max(s1, 1e-8))
                torch.cuda.empty_cache()
            s2e, s1e = second_exact(sdh, x_edit, v_i, args.t_edit, cond, uncond)
            rho_exact.append(s2e / max(s1e, 1e-8))
            torch.cuda.empty_cache()

        sig = S[:args.top_k]
        sp_fd = {str(e): float(spearmanr(sig, rho_fd[e])[0]) for e in args.eps_grid}
        sp_exact = float(spearmanr(sig, rho_exact)[0])
        per_seed.append({"seed": seed, "sigma": sig.tolist(),
                         "rho_fd": {str(e): rho_fd[e] for e in args.eps_grid},
                         "rho_exact": rho_exact,
                         "spearman_fd": sp_fd, "spearman_exact": sp_exact})
        print(f"seed {seed}: Spearman(sig,rho)  "
              f"FD@{args.paper_eps}={sp_fd[str(args.paper_eps)]:+.3f}  "
              f"exact={sp_exact:+.3f}  ({time.time()-t0:.0f}s)")

    # aggregate
    def mean_sp(getter):
        vals = [getter(s) for s in per_seed]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    agg = {
        "mean_spearman_FD_at_paper_eps": mean_sp(lambda s: s["spearman_fd"][str(args.paper_eps)]),
        "mean_spearman_exact": mean_sp(lambda s: s["spearman_exact"]),
        "mean_spearman_FD_by_eps": {str(e): mean_sp(lambda s: s["spearman_fd"][str(e)]) for e in args.eps_grid},
    }
    res = {"per_seed": per_seed, "aggregate": agg, "config": vars(args),
           "_meta": run_metadata(seed=args.seed)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 70)
    print("CHALLENGE A — Park rho: FD vs EXACT composed-JVP")
    print("=" * 70)
    print(f"  Spearman(sigma, rho), mean over {args.n_seeds} seeds:")
    for e in args.eps_grid:
        print(f"    FD  eps={e:<7}: {agg['mean_spearman_FD_by_eps'][str(e)]:+.3f}")
    print(f"    EXACT (composed JVP): {agg['mean_spearman_exact']:+.3f}")
    print(f"\n  Park's claim: rho NEGATIVELY correlates with sigma (complementary).")
    print(f"  If FD@{args.paper_eps} and exact give different sign/magnitude,")
    print(f"  the reproduced conclusion depends on the numerical method.")
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
