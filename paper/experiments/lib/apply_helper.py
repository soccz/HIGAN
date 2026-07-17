"""Idempotent patcher: add set_deterministic + run_metadata to every run_*.py.

For each target file:
  - Inserts `from lib.reproducibility import set_deterministic, run_metadata`
    after the sys.path.insert lines if not already present.
  - Adds `--seed` argparse arg if missing.
  - Calls `set_deterministic(seed=args.seed)` after args = ap.parse_args().

This is a textual transformation; verify changes with `git diff` after running.
"""
from __future__ import annotations
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
EXP = PAPER / "experiments"


def patch_file(p: Path) -> bool:
    src = p.read_text()
    orig = src
    changed = False

    if "from lib.reproducibility" not in src:
        # find the LAST sys.path.insert line; add import after it
        matches = list(re.finditer(r"^sys\.path\.insert\(0, str.*\)$",
                                    src, flags=re.MULTILINE))
        if matches:
            insert_at = matches[-1].end()
            new_import = ("\n\nfrom lib.reproducibility import "
                          "set_deterministic, run_metadata    # noqa: E402")
            src = src[:insert_at] + new_import + src[insert_at:]
            changed = True

    if "set_deterministic(" not in src:
        # find `args = ap.parse_args()` and add set_deterministic after
        m = re.search(r"(^    args = ap\.parse_args\(\)$)",
                      src, flags=re.MULTILINE)
        if m:
            insert_at = m.end()
            line = "\n\n    set_deterministic(seed=getattr(args, 'seed', 2027))"
            src = src[:insert_at] + line + src[insert_at:]
            changed = True

    # Add --seed if missing
    if "--seed" not in src:
        m = re.search(r"(^    args = ap\.parse_args\(\)$)",
                      src, flags=re.MULTILINE)
        if m:
            insert_at = m.start()
            line = '    ap.add_argument("--seed", type=int, default=2027)\n'
            src = src[:insert_at] + line + src[insert_at:]
            changed = True

    if changed:
        p.write_text(src)
    return changed


TARGETS = [
    # Wave 2-4 scripts
    "diffusion/run_daam_comparison.py",
    "diffusion/run_park_repro.py",
    "domains/ffhq/run_truncation_ablation.py",
    "domains/ffhq/run_resolution_invariance.py",
    "domains/ffhq/run_alpha_magnitude_scan.py",
    "metrics/run_multi_clip_c2.py",
    "metrics/run_per_layer_c1.py",
    "metrics/run_c6_scaling.py",
    "metrics/run_dino_path_curvature.py",
    "metrics/run_noise_robustness.py",
    "metrics/run_sample_scaling.py",
    "method/run_walltime_benchmark.py",
    "method/run_intrinsic_dim.py",
    "method/run_fd_validation.py",
    "baselines/run_editing_head_to_head.py",
    "baselines/run_spatial_diversity.py",
]


def main():
    for rel in TARGETS:
        p = EXP / rel
        if not p.exists():
            print(f"  skip (missing): {rel}")
            continue
        ch = patch_file(p)
        print(f"  {'patched' if ch else 'already-ok'}: {rel}")


if __name__ == "__main__":
    main()
