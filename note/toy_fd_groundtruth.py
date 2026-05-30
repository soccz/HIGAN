"""Airtight demonstration that finite differences cannot recover the second-order
(curvature) term of a vector-valued map at ANY step size, while forward-mode
composed JVP recovers it to machine precision — on a TOY ANALYTIC map whose true
first and second derivatives are known in closed form (so JVP is NOT its own
referee; this dodges the tautology charge against the generator-only evidence).

Map (deliberately curved, mixed scales like a generator layer):
    f(a) = [ exp(0.7 a),  sin(3 a),  (a + 0.4)**3,  tanh(2 a),  a**2 * cos(a) ]
True derivatives are analytic (see f1_true / f2_true).

We compare, across epsilons:
  - central FD 1st  : (f(a+e) - f(a-e)) / (2e)          vs  f1_true
  - central FD 2nd  : (f(a+e) - 2 f(a) + f(a-e)) / e^2  vs  f2_true
  - forward JVP 1st : torch.func.jvp                     vs  f1_true
  - composed JVP 2nd: jvp of jvp                          vs  f2_true

Expected: FD-2nd relative error is U-shaped (truncation at large e, catastrophic
cancellation at small e) and never reaches machine precision; JVP-2nd is exact.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)   # give FD its BEST possible chance (fp64)


def f(a: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.exp(0.7 * a),
        torch.sin(3.0 * a),
        (a + 0.4) ** 3,
        torch.tanh(2.0 * a),
        a ** 2 * torch.cos(a),
    ])


def f1_true(a: float) -> np.ndarray:
    return np.array([
        0.7 * np.exp(0.7 * a),
        3.0 * np.cos(3.0 * a),
        3.0 * (a + 0.4) ** 2,
        2.0 * (1.0 - np.tanh(2.0 * a) ** 2),
        2.0 * a * np.cos(a) - a ** 2 * np.sin(a),
    ])


def f2_true(a: float) -> np.ndarray:
    return np.array([
        0.49 * np.exp(0.7 * a),
        -9.0 * np.sin(3.0 * a),
        6.0 * (a + 0.4),
        -8.0 * np.tanh(2.0 * a) * (1.0 - np.tanh(2.0 * a) ** 2),
        2.0 * np.cos(a) - 4.0 * a * np.sin(a) - a ** 2 * np.cos(a),
    ])


def relerr(est: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(est - true) / (np.linalg.norm(true) + 1e-300))


def main():
    a0 = 0.3
    a0_t = torch.tensor(a0)
    true1, true2 = f1_true(a0), f2_true(a0)

    # --- forward-mode JVP (exact) ---
    _, jvp1 = torch.func.jvp(f, (a0_t,), (torch.ones(()),))
    def df(a):
        _, d = torch.func.jvp(f, (a,), (torch.ones(()),))
        return d
    _, jvp2 = torch.func.jvp(df, (a0_t,), (torch.ones(()),))
    jvp1_err = relerr(jvp1.numpy(), true1)
    jvp2_err = relerr(jvp2.numpy(), true2)

    # --- central FD across epsilons (fp64, best case) ---
    epsilons = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    rows = []
    f0 = f(a0_t).numpy()
    for e in epsilons:
        fp = f(torch.tensor(a0 + e)).numpy()
        fm = f(torch.tensor(a0 - e)).numpy()
        fd1 = (fp - fm) / (2 * e)
        fd2 = (fp - 2 * f0 + fm) / (e ** 2)
        rows.append({"eps": e,
                     "fd1_relerr": relerr(fd1, true1),
                     "fd2_relerr": relerr(fd2, true2)})

    fd2_best = min(r["fd2_relerr"] for r in rows)
    fd2_best_eps = min(rows, key=lambda r: r["fd2_relerr"])["eps"]

    out = {
        "map": "exp(0.7a), sin(3a), (a+0.4)^3, tanh(2a), a^2 cos(a)",
        "eval_point": a0,
        "dtype": "float64 (FD best case)",
        "jvp1_relerr": jvp1_err,
        "jvp2_relerr": jvp2_err,
        "fd_sweep": rows,
        "fd2_best_relerr": fd2_best,
        "fd2_best_eps": fd2_best_eps,
        "ratio_fd2best_over_jvp2": fd2_best / (jvp2_err + 1e-300),
    }
    Path("/mnt/20t/study/HIGAN/note/toy_fd_groundtruth.json").write_text(
        json.dumps(out, indent=2))

    print("=" * 70)
    print("TOY ANALYTIC GROUND TRUTH (true derivatives known in closed form)")
    print("=" * 70)
    print(f"  eval point a0 = {a0},  dtype = float64 (FD's best case)")
    print(f"\n  forward JVP   1st-order rel.err = {jvp1_err:.2e}")
    print(f"  composed JVP  2nd-order rel.err = {jvp2_err:.2e}  (exact)")
    print(f"\n  central FD 2nd-order rel.err vs eps:")
    print(f"  {'eps':>10} {'FD-1st relerr':>16} {'FD-2nd relerr':>16}")
    for r in rows:
        print(f"  {r['eps']:>10.0e} {r['fd1_relerr']:>16.2e} {r['fd2_relerr']:>16.2e}")
    print(f"\n  BEST FD-2nd rel.err = {fd2_best:.2e} (at eps={fd2_best_eps:.0e})")
    print(f"  JVP-2nd rel.err     = {jvp2_err:.2e}")
    print(f"  FD's best is {out['ratio_fd2best_over_jvp2']:.1e}x worse than JVP.")
    print("\n  => Even in fp64 with the optimal eps, central FD cannot match the")
    print("     exact composed-JVP curvature; at small eps it diverges")
    print("     (catastrophic cancellation). JVP is not its own referee here —")
    print("     the ground truth is analytic.")


if __name__ == "__main__":
    main()
