"""Hard reproducibility test: composed-JVP ρ measurement bit-identical?

Runs the actual code path used by run_higher_order.py / run_sample_scaling.py
on bedroom HiGAN (256², small footprint — fits alongside running SD job).
Computes one ρ ratio for the `view` attribute, twice with same seed,
compares bit-by-bit.

If this fails, set_deterministic() is not sufficient for our use case.
"""
from __future__ import annotations
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic


def one_measurement(seed: int) -> dict:
    """Returns (first_mean, second_mean, ρ) for one bedroom-view JVP."""
    set_deterministic(seed)
    import torch
    from torch.func import jvp
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary
    from higan_dev.cam.grad_saliency import _layered_direction

    G = HiGANGenerator(higan_repo=str(
        PAPER.parent / "higan_dev" / "data" / "higan_repo"
    ))
    bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
            / "boundaries" / "stylegan_bedroom")
    b = load_boundary(str(bdir), "view", num_layers=G.num_layers)
    b_la = _layered_direction(b, G.num_layers, G.w_dim, G.device).to(G.device)

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(4, generator=rng)

    wp = base_wp[0:1].detach()

    def f(alpha):
        return G.synthesize(wp + alpha.view(1, 1, 1) * b_la.unsqueeze(0))
    def df(alpha):
        _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
        return d
    a0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)
    _, first = jvp(f, (a0,), (one,))
    _, second = jvp(df, (a0,), (one,))
    first_mean = first.abs().mean().item()
    second_mean = second.abs().mean().item()
    rho = second_mean / max(first_mean, 1e-8)
    del G
    import torch
    torch.cuda.empty_cache()
    return {"first_mean": first_mean,
            "second_mean": second_mean,
            "rho": rho}


def main():
    print("=== JVP-based ρ measurement bit-identical test ===")
    print("  bedroom HiGAN view attribute, composed-JVP, single base wp")
    print()

    r1 = one_measurement(seed=2027)
    print(f"  run 1 (seed=2027): first={r1['first_mean']:.18g}")
    print(f"                     second={r1['second_mean']:.18g}")
    print(f"                     ρ={r1['rho']:.18g}")
    print()

    r2 = one_measurement(seed=2027)
    print(f"  run 2 (seed=2027): first={r2['first_mean']:.18g}")
    print(f"                     second={r2['second_mean']:.18g}")
    print(f"                     ρ={r2['rho']:.18g}")
    print()

    same = (r1['first_mean'] == r2['first_mean'] and
            r1['second_mean'] == r2['second_mean'] and
            r1['rho'] == r2['rho'])
    if same:
        print("✓ BIT-IDENTICAL — composed-JVP through StyleGAN is deterministic")
    else:
        df1 = abs(r2['first_mean'] - r1['first_mean'])
        df2 = abs(r2['second_mean'] - r1['second_mean'])
        dfr = abs(r2['rho'] - r1['rho'])
        print(f"✗ DIFFERS — Δfirst={df1:.3e}, Δsecond={df2:.3e}, Δρ={dfr:.3e}")

    # Bonus: different seed should differ
    r3 = one_measurement(seed=2028)
    print()
    print(f"  run 3 (seed=2028): ρ={r3['rho']:.18g}")
    different = r1['rho'] != r3['rho']
    if different:
        print("  ✓ different seed gives different ρ (as expected)")
    else:
        print("  ⚠ different seeds gave SAME ρ — seeding ineffective?")


if __name__ == "__main__":
    main()
