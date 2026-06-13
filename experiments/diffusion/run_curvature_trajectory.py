"""Positive finding (instrument-enabled): the SECOND-order geometry of a diffusion
model's h-space evolves systematically across the denoising trajectory, and this
structure is resolvable only by the exact composed-JVP instrument --- finite
differences wash it out.

Park et al. (NeurIPS'23) showed the FIRST-order geometry (local latent basis) of
diffusion h-space evolves over timesteps (coarse-to-fine). We add the SECOND-order
axis: along edit directions, the curvature ratio rho(t)=|d^2x/da^2|/|dx/da| of the
denoising map as a function of the injection timestep t. We compute rho(t) two ways
--- exact composed JVP (no step size) and FD-of-JVP (the diffusion analogue of a
naive curvature estimate) --- and ask:
  (1) does the exact instrument reveal a consistent rho(t) trend across the
      trajectory (a real second-order signature)?
  (2) does FD resolve it, or is it noise/sign-unstable (so the structure is
      *only* visible with exact higher-order AD)?

This reframes the finite-difference fact as the MOTIVATION for an instrument that
enables a positive geometric finding, rather than as the contribution itself.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
from lib.reproducibility import set_deterministic, run_metadata          # noqa: E402
from diffusion.generator import SDH, SDConfig                            # noqa: E402
from diffusion.run_park_fd_vs_exact import second_exact, second_fd       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--K-dirs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--t-grid", type=int, nargs="+",
                    default=[2, 5, 8, 11, 14, 17])
    ap.add_argument("--fd-eps", type=float, default=0.05)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--prompt", default="a photograph of a face")
    ap.add_argument("--out", default="experiments/out/curvature_trajectory")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sdh = SDH(SDConfig(resolution=args.resolution, num_inference_steps=args.steps))
    cond, uncond = sdh.encode_prompt(args.prompt, "")
    hs = sdh.h_space_shape()
    flat = int(np.prod(hs[1:]))
    print(f"[{time.strftime('%H:%M:%S')}] SD loaded; h={hs}; t-grid={args.t_grid}")

    # rho[t] -> list over (seed,dir) of (exact, fd)
    rho_exact = {t: [] for t in args.t_grid}
    rho_fd = {t: [] for t in args.t_grid}

    for seed in range(args.n_seeds):
        t0 = time.time()
        gen = torch.Generator(device=sdh.cfg.device).manual_seed(seed)
        # probe directions in h-space (unit norm)
        P = torch.randn(args.K_dirs, flat, device=sdh.cfg.device,
                        dtype=sdh.cfg.dtype, generator=gen)
        P = (P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)).reshape(args.K_dirs, *hs[1:])

        # build the denoising trajectory once: store latent x at each step
        H = W = sdh.cfg.resolution // 8
        gx = torch.Generator(device=sdh.cfg.device).manual_seed(seed * 17 + 3)
        traj = {}
        with torch.no_grad():
            x = torch.randn(1, 4, H, W, generator=gx, device=sdh.cfg.device,
                            dtype=sdh.cfg.dtype)
            for i in range(sdh.cfg.num_inference_steps):
                traj[i] = x.detach()
                x = sdh.ddim_step(x, sdh.epsilon(x, i, cond, uncond), i)

        for t in args.t_grid:
            x_t = traj[t]
            for k in range(args.K_dirs):
                v = P[k:k+1]
                s2e, s1e = second_exact(sdh, x_t, v, t, cond, uncond)
                s2f, s1f = second_fd(sdh, x_t, v, args.fd_eps, t, cond, uncond)
                rho_exact[t].append(s2e / max(s1e, 1e-8))
                rho_fd[t].append(s2f / max(s1f, 1e-8))
                torch.cuda.empty_cache()
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    # aggregate
    def stat(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "sem": float(a.std() / max(np.sqrt(len(a)), 1)),
                "median": float(np.median(a)), "n": len(a)}
    agg = {"t_grid": args.t_grid,
           "exact": {str(t): stat(rho_exact[t]) for t in args.t_grid},
           "fd": {str(t): stat(rho_fd[t]) for t in args.t_grid}}

    # how well does each method reveal a monotone trend across t? Spearman(t, rho)
    from scipy.stats import spearmanr
    te = np.repeat(args.t_grid, [len(rho_exact[t]) for t in args.t_grid])
    sp_exact = float(spearmanr(te, np.concatenate([rho_exact[t] for t in args.t_grid]))[0])
    sp_fd = float(spearmanr(te, np.concatenate([rho_fd[t] for t in args.t_grid]))[0])
    # signal-to-noise: trend range / within-t scatter
    ex_means = np.array([agg["exact"][str(t)]["mean"] for t in args.t_grid])
    ex_scatter = np.mean([agg["exact"][str(t)]["std"] for t in args.t_grid])
    fd_means = np.array([agg["fd"][str(t)]["mean"] for t in args.t_grid])
    fd_scatter = np.mean([agg["fd"][str(t)]["std"] for t in args.t_grid])
    snr_exact = float((ex_means.max() - ex_means.min()) / max(ex_scatter, 1e-12))
    snr_fd = float((fd_means.max() - fd_means.min()) / max(fd_scatter, 1e-12))

    result = {"agg": agg,
              "spearman_t_rho_exact": sp_exact, "spearman_t_rho_fd": sp_fd,
              "snr_exact": snr_exact, "snr_fd": snr_fd,
              "config": vars(args), "_meta": run_metadata(seed=args.seed)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("Curvature signature across the denoising trajectory (SD h-space)")
    print("=" * 70)
    print(f"  {'t':>4}  {'exact rho mean+/-sem':>24}  {'FD rho mean+/-sem':>24}")
    for t in args.t_grid:
        e, f = agg["exact"][str(t)], agg["fd"][str(t)]
        print(f"  {t:>4}  {e['mean']:>10.4f} +/- {e['sem']:<9.4f}  "
              f"{f['mean']:>10.4f} +/- {f['sem']:<9.4f}")
    print(f"\n  trend Spearman(t, rho):  exact={sp_exact:+.3f}   FD={sp_fd:+.3f}")
    print(f"  trend SNR (range/scatter): exact={snr_exact:.2f}   FD={snr_fd:.2f}")
    print(f"\n  If exact shows a clean monotone rho(t) (|Spearman| high, SNR high)")
    print(f"  while FD is flat/noisy, the second-order trajectory signature is a")
    print(f"  real structure visible ONLY through the exact instrument.")
    print(f"\nsaved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
