"""Main-claim post-process: summarize curvature-guided edit selection.

The original FFHQ head-to-head track evaluates low-curvature, high-curvature,
and random direction selections at a fixed edit strength. This script turns that
raw table into paired attr-level effect sizes suitable for a paper table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import run_metadata  # noqa: E402

OUT = PAPER / "experiments" / "out"
GROUPS = ["curvature_low", "random", "curvature_high"]
METRICS = ["rho_this_dir", "mean_id_cos", "mean_lpips_proxy", "abs_delta_attr"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_ci(values: list[float], seed: int, n_boot: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    values = {
        "rho_this_dir": [float(r["rho_this_dir"]) for r in rows],
        "mean_id_cos": [float(r["mean_id_cos"]) for r in rows],
        "mean_lpips_proxy": [float(r["mean_lpips_proxy"]) for r in rows],
        "abs_delta_attr": [abs(float(r["mean_delta_attr"])) for r in rows],
    }
    for name, vals in values.items():
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_std"] = float(np.std(vals, ddof=0))
    return out


def build_attr_table(raw: dict[str, Any]) -> dict[str, Any]:
    table = {}
    for attr, groups in raw["per_attr"].items():
        table[attr] = {}
        for group in GROUPS:
            table[attr][group] = summarize_group(groups[group])
    return table


def paired_effects(attr_table: dict[str, Any], seed: int, n_boot: int) -> dict[str, Any]:
    attrs = sorted(attr_table)
    comparisons = {
        "curvature_low_minus_random": ("curvature_low", "random"),
        "curvature_low_minus_high": ("curvature_low", "curvature_high"),
    }
    out = {}
    for cname, (a, b) in comparisons.items():
        comp = {}
        for metric in METRICS:
            key = f"{metric}_mean"
            diffs = [attr_table[attr][a][key] - attr_table[attr][b][key]
                     for attr in attrs]
            comp[metric] = {
                **mean_ci(diffs, seed=seed, n_boot=n_boot),
                "wins_for_low": int(sum(d > 0 for d in diffs)),
                "n_attrs": len(attrs),
                "direction": (
                    "higher_is_better" if metric in {"mean_id_cos", "abs_delta_attr"}
                    else "lower_is_better"
                ),
            }
            if metric == "mean_lpips_proxy":
                comp[metric]["wins_for_low"] = int(sum(d < 0 for d in diffs))
            if metric == "rho_this_dir":
                comp[metric]["wins_for_low"] = int(sum(d < 0 for d in diffs))
        out[cname] = comp
    return out


def write_plot(attr_table: dict[str, Any], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attrs = sorted(attr_table)
    x = np.arange(len(attrs))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
    colors = {
        "curvature_low": "#2563eb",
        "random": "#6b7280",
        "curvature_high": "#c2410c",
    }
    for i, group in enumerate(GROUPS):
        offset = (i - 1) * width
        axes[0].bar(x + offset,
                    [attr_table[a][group]["mean_id_cos_mean"] for a in attrs],
                    width=width, color=colors[group], label=group)
        axes[1].bar(x + offset,
                    [attr_table[a][group]["mean_lpips_proxy_mean"] for a in attrs],
                    width=width, color=colors[group], label=group)
    axes[0].set_title("Identity preservation")
    axes[0].set_ylabel("CLIP image cosine")
    axes[1].set_title("Perceptual drift proxy")
    axes[1].set_ylabel("downsampled L2")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curvature_guided_selection_summary.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/out/editing_head_to_head/metrics.json")
    ap.add_argument("--out", default="experiments/out/main_curvature_guided_selection")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PAPER / input_path
    out = Path(args.out)
    if not out.is_absolute():
        out = PAPER / out
    out.mkdir(parents=True, exist_ok=True)

    raw = load_json(input_path)
    attr_table = build_attr_table(raw)
    effects = paired_effects(attr_table, seed=args.seed, n_boot=args.n_boot)
    payload = {
        "claim": (
            "Low-curvature pre-selection finds edit directions with smaller "
            "identity/perceptual drift than random or high-curvature selections "
            "under the fixed-alpha protocol."
        ),
        "protocol_note": (
            "This is a fixed-strength summary of the existing head-to-head track; "
            "attribute-change-matched evaluation should be used as the stronger "
            "main-paper version."
        ),
        "per_attr": attr_table,
        "paired_attr_effects": effects,
        "source_config": raw.get("config", {}),
        "_meta": run_metadata(extra={
            "script": "experiments/metrics/run_curvature_guided_selection_summary.py",
            "input": str(input_path),
            "seed": args.seed,
            "n_boot": args.n_boot,
        }),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_plot(attr_table, out)

    print("=== curvature-guided selection summary ===")
    for cname, comp in effects.items():
        did = comp["mean_id_cos"]["mean"]
        dlp = comp["mean_lpips_proxy"]["mean"]
        dda = comp["abs_delta_attr"]["mean"]
        print(f"{cname:28s} ΔID={did:+.4f} ΔLPIPS={dlp:+.4f} "
              f"Δ|attr|={dda:+.4f}")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
