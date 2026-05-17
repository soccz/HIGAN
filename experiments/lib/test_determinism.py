"""Quick smoke test: confirm set_deterministic() + same seed → bit-identical results.

Runs a small CUDA computation twice with the helper, expects exact match.
If this fails, the helper isn't actually pinning everything.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.reproducibility import set_deterministic, run_metadata


def one_run(seed: int) -> tuple[float, float]:
    set_deterministic(seed)
    import torch
    # Simulate a small computation that goes through cudnn + cublas
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 16, 32, 32, device=device)
    conv = torch.nn.Conv2d(16, 32, 3, padding=1).to(device)
    y = conv(x).sum().item()
    # Some pure torch matmul
    A = torch.randn(64, 64, device=device)
    B = torch.randn(64, 64, device=device)
    z = (A @ B).sum().item()
    return y, z


def main():
    print("=== determinism smoke test ===")
    print(f"  meta: {run_metadata(seed=2027)}")
    r1 = one_run(2027)
    r2 = one_run(2027)
    r3 = one_run(2028)
    print(f"  run1 (seed 2027): y={r1[0]:.10f}, z={r1[1]:.10f}")
    print(f"  run2 (seed 2027): y={r2[0]:.10f}, z={r2[1]:.10f}")
    print(f"  run3 (seed 2028): y={r3[0]:.10f}, z={r3[1]:.10f}")
    same = (r1[0] == r2[0]) and (r1[1] == r2[1])
    diff_from_other = (r1[0] != r3[0]) or (r1[1] != r3[1])
    if same and diff_from_other:
        print("✓ DETERMINISTIC — same seed gives identical, different seed differs")
    else:
        print("✗ NOT DETERMINISTIC")
        if not same:
            print(f"  run1 vs run2 differ: Δy={r2[0]-r1[0]:.4e} Δz={r2[1]-r1[1]:.4e}")
        if not diff_from_other:
            print(f"  different seeds gave same result — seeding not effective")
        sys.exit(1)


if __name__ == "__main__":
    main()
