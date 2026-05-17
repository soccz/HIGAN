"""Track 7 — Park-NeurIPS23 Riemannian-metric reproduction + 2nd-order extension.

For each noise seed:
  1. Build a probe basis {e_1, ..., e_K} of K=32 random h-space directions.
  2. Compute J e_i via FD-of-JVPs (Park's construction).
  3. SVD J = U Σ V^T. Top right-singular vectors = editing basis.
  4. For top-k=4 right-singular vectors, compute the second-order ratio
     ρ_i = |d²G(v_i, v_i)| / σ_i, where σ_i is the singular value.
  5. Report Spearman correlation between σ and ρ.

If ρ is negatively correlated with σ, first- and second-order
geometric structures carry complementary information — our paper's
delta over Park.

See designs/07_park_riemannian_reproduction.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from scipy.stats import spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402

from diffusion.generator import SDH, SDConfig                      # noqa: E402


def jvp_through_chain(sdh, x_edit, v, alpha_val, t_idx, cond, uncond):
    def f(alpha):
        x = x_edit
        sdh._h_v = v; sdh._h_alpha = alpha
        for i in range(t_idx, sdh.cfg.num_inference_steps):
            sdh._h_active = (i == t_idx)
            eps = sdh.epsilon(x, i, cond, uncond)
            x = sdh.ddim_step(x, eps, i)
        sdh._h_active = False; sdh._h_v = None; sdh._h_alpha = None
        return x
    ap = torch.tensor(alpha_val, device=sdh.cfg.device,
                       dtype=sdh.cfg.dtype)
    at = torch.tensor(1.0, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    return jvp(f, (ap,), (at,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--K-probes", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--t-edit", type=int, default=25)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--prompt", default="a photograph of a face")
    ap.add_argument("--out", default="experiments/out/sd_park_repro")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sdh = SDH(SDConfig(resolution=512))
    print(f"[{time.strftime('%H:%M:%S')}] SD loaded for Park reproduction")

    cond, uncond = sdh.encode_prompt(args.prompt, "")
    hs = sdh.h_space_shape()             # (1, 1280, 8, 8)
    flat_dim = int(np.prod(hs[1:]))      # 1280*8*8 = 81920

    summary = []
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed+1}/{args.n_seeds} ===")
        t_seed = time.time()
        # 1. random probe basis
        gen = torch.Generator(device=sdh.cfg.device).manual_seed(seed)
        probes = torch.randn(args.K_probes, flat_dim, device=sdh.cfg.device,
                              dtype=sdh.cfg.dtype, generator=gen)
        probes = probes / probes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        # reshape each to hs spatial
        probes_h = probes.reshape(args.K_probes, *hs[1:])

        # 2. reach t_edit unperturbed
        H = W = sdh.cfg.resolution // 8
        gen_x = torch.Generator(device=sdh.cfg.device).manual_seed(
            seed * 17 + 3
        )
        with torch.no_grad():
            x = torch.randn(1, 4, H, W, generator=gen_x,
                            device=sdh.cfg.device, dtype=sdh.cfg.dtype)
            for i in range(args.t_edit):
                eps = sdh.epsilon(x, i, cond, uncond)
                x = sdh.ddim_step(x, eps, i)
        x_edit = x.detach()

        # 3. Build J via FD-of-JVPs: J e_k = (x_+ε - x_-ε)/(2ε)
        J_cols = []
        for k in range(args.K_probes):
            v_k = probes_h[k:k+1]
            _, dx_p = jvp_through_chain(sdh, x_edit, v_k,
                                         +args.epsilon, args.t_edit,
                                         cond, uncond)
            _, dx_m = jvp_through_chain(sdh, x_edit, v_k,
                                         -args.epsilon, args.t_edit,
                                         cond, uncond)
            # FD second-derivative dimension: but Park's J is first-order,
            # so really J e_k = (x_+ - x_-)/(2ε * 1) which equals d x_0
            # because the chain rule contracts. Use dx_p (or average +/-)
            # as the column.
            J_col = (dx_p + dx_m).flatten() / 2.0
            J_cols.append(J_col.detach().cpu().float().numpy())
            torch.cuda.empty_cache()
            if (k + 1) % 8 == 0:
                print(f"  probe {k+1}/{args.K_probes}")
        J = np.stack(J_cols, axis=1)                # (3HW, K)
        print(f"  J shape={J.shape}")

        # 4. SVD
        # J = U Σ V^T,  U:(3HW, K), Σ:(K,), V:(K, K)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        print(f"  singular values: {S[:8]}")
        # top right-singular vectors → linear combinations of probes
        # v_i = Σ_k Vt[i, k] * probes_h[k]   (in h-space)
        top_V = Vt[:args.top_k]                      # (top_k, K)

        # 5. second-order ratio along top SVD directions
        ρ_per_top = []
        for i in range(args.top_k):
            # build the h-space direction
            v_i = torch.zeros(*hs, device=sdh.cfg.device,
                              dtype=sdh.cfg.dtype)
            for k in range(args.K_probes):
                v_i = v_i + float(top_V[i, k]) * probes_h[k:k+1]
            v_i = v_i / v_i.norm().clamp_min(1e-8)
            # ρ via FD of two JVPs (second derivative)
            _, dxp = jvp_through_chain(sdh, x_edit, v_i,
                                        +args.epsilon, args.t_edit,
                                        cond, uncond)
            _, dxm = jvp_through_chain(sdh, x_edit, v_i,
                                        -args.epsilon, args.t_edit,
                                        cond, uncond)
            first_mag = 0.5 * (dxp.abs().mean() + dxm.abs().mean()).item()
            second_mag = ((dxp - dxm) / (2 * args.epsilon)).abs().mean().item()
            ρ_i = second_mag / max(first_mag, 1e-8)
            ρ_per_top.append(ρ_i)
            print(f"  top-SVD #{i+1}  σ={S[i]:.3g}  ρ={ρ_i:.3f}")
            torch.cuda.empty_cache()

        # rank correlation of σ vs ρ
        if args.top_k > 2:
            r, p = spearmanr(S[:args.top_k], ρ_per_top)
        else:
            r, p = float("nan"), float("nan")
        summary.append({
            "seed": seed,
            "singular_values": S.tolist(),
            "rho_top": ρ_per_top,
            "spearman_sigma_rho": {"r": float(r), "p": float(p)},
            "wall_s": time.time() - t_seed,
        })
        print(f"  σ↔ρ Spearman r={r:+.3f} (p={p:.3g})")

    out_path = out / "metrics.json"
    out_path.write_text(json.dumps(
        {"per_seed": summary,
         "aggregate": {
             "mean_top1_sigma": float(np.mean(
                 [s["singular_values"][0] for s in summary]
             )),
             "mean_top1_rho": float(np.mean(
                 [s["rho_top"][0] for s in summary]
             )),
             "mean_spearman_sigma_rho": float(np.mean(
                 [s["spearman_sigma_rho"]["r"]
                  for s in summary
                  if not np.isnan(s["spearman_sigma_rho"]["r"])]
             )),
         },
         "config": vars(args)},
        indent=2,
    ))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
