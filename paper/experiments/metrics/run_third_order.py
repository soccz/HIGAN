"""3rd-order curvature analysis via composed JVP (∂³I/∂α³).

Extension of Track 27 (ffhq_higher_order) to the next derivative.
For each FFHQ attribute, compute:
  - |∂I/∂α|     (1st-order pushforward, mean image-pixel magnitude)
  - |∂²I/∂α²|   (2nd-order pushforward via composed JVP)
  - |∂³I/∂α³|   (3rd-order pushforward via triple-composed JVP)

Report ratio mean for r2 = 2nd/1st and r3 = 3rd/1st per attribute.
Hypothesis: structural attrs (pose) have higher non-linearity at all
orders, but the gap grows fastest at 3rd order — quantifies how much
of the curvature signal is captured at 2nd order vs needs higher orders.
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

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from lib.reproducibility import set_deterministic, run_metadata  # noqa: E402
from domains.ffhq.generator import FFHQGenerator  # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}


def load_boundary(attr: str, w_dim: int, num_layers: int):
    """Load FFHQ InterFaceGAN boundary, apply to canonical layers."""
    bdir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    for cand in [bdir / f"stylegan_ffhq_{attr}_w_boundary.npy",
                 bdir / f"stylegan_ffhq_{attr}_boundary.npy"]:
        if cand.exists():
            b = np.load(cand).astype(np.float32).squeeze()
            v = torch.tensor(b / (np.linalg.norm(b) + 1e-8))
            d = torch.zeros(num_layers, w_dim)
            for li in LAYERS_FOR.get(attr, list(range(num_layers))):
                d[li] = v
            return d
    raise FileNotFoundError(f"no boundary for {attr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--attrs", nargs="+",
                    default=list(LAYERS_FOR.keys()))
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="experiments/out/ffhq_third_order")
    ap.add_argument("--lod", type=float, default=2.0,
                    help="StyleGAN1 lod: 2=256², 1=512², 0=1024². "
                    "3rd-order JVP at 1024² OOMs on 8GB; use lod=2.")
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator(lod_override=args.lod)
    L, D = G.num_layers, G.w_dim

    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    results = []
    for attr in args.attrs:
        b_layered = load_boundary(attr, D, L).to(G.device)
        print(f"\n=== {attr} ===")

        mag_first = []
        mag_second = []
        mag_third = []

        for s in range(args.num_samples):
            wp = base_wp[s:s+1].detach()

            # f(α) = synthesize(wp + α * b_layered)
            def f_at(a_val):
                a = torch.tensor([a_val], device=G.device)
                return G.synthesize(
                    wp + a.view(-1, 1, 1) * b_layered.unsqueeze(0))

            def f(a):
                return G.synthesize(
                    wp + a.view(-1, 1, 1) * b_layered.unsqueeze(0))

            # 1st-order at α=0: ∂I/∂α
            alpha = torch.tensor([0.0], device=G.device)
            _, dimg1 = jvp(f, (alpha,), (torch.ones_like(alpha),))
            mag_first.append(float(dimg1.abs().mean().item()))

            # 2nd-order at α=0: ∂²I/∂α² via composed JVP
            def df(a):
                _, d = jvp(f, (a,), (torch.ones_like(a),))
                return d

            _, dimg2_at0 = jvp(df, (alpha,), (torch.ones_like(alpha),))
            mag_second.append(float(dimg2_at0.abs().mean().item()))
            del dimg2_at0
            torch.cuda.empty_cache()

            # 3rd-order: FD on 2nd-order (triple-nested JVP OOMs at 8GB).
            # ∂³I/∂α³ ≈ [d²f(ε) − d²f(0)] / ε with ε small.
            eps = 0.05
            alpha_plus = torch.tensor([eps], device=G.device)
            _, dimg2_plus = jvp(df, (alpha_plus,), (torch.ones_like(alpha_plus),))
            _, dimg2_zero = jvp(df, (alpha,), (torch.ones_like(alpha),))
            dimg3 = (dimg2_plus - dimg2_zero) / eps
            mag_third.append(float(dimg3.abs().mean().item()))
            del dimg2_plus, dimg2_zero, dimg3
            torch.cuda.empty_cache()

        m1 = np.array(mag_first)
        m2 = np.array(mag_second)
        m3 = np.array(mag_third)
        r2 = m2 / (m1 + 1e-8)
        r3 = m3 / (m1 + 1e-8)
        r32 = m3 / (m2 + 1e-8)

        row = {
            "attr": attr,
            "first_mean": float(m1.mean()),
            "second_mean": float(m2.mean()),
            "third_mean": float(m3.mean()),
            "ratio_2nd_over_1st_mean": float(r2.mean()),
            "ratio_2nd_over_1st_p95": float(np.percentile(r2, 95)),
            "ratio_3rd_over_1st_mean": float(r3.mean()),
            "ratio_3rd_over_1st_p95": float(np.percentile(r3, 95)),
            "ratio_3rd_over_2nd_mean": float(r32.mean()),
        }
        results.append(row)
        print(f"  1st mean: {m1.mean():.4f}")
        print(f"  2nd mean: {m2.mean():.4f}  ratio 2/1: {r2.mean():.4f}")
        print(f"  3rd mean: {m3.mean():.4f}  ratio 3/1: {r3.mean():.4f}  "
              f"ratio 3/2: {r32.mean():.4f}")

    output = {
        "results": results,
        "num_samples": args.num_samples,
        "attrs": args.attrs,
        "_meta": run_metadata(seed=args.seed),
    }
    (out / "metrics.json").write_text(json.dumps(output, indent=2))
    print(f"\nsaved {out}/metrics.json")


if __name__ == "__main__":
    main()
