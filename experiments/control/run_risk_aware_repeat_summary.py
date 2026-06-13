"""Summarize repeated risk-aware controller runs.

This script is intentionally a post-hoc reducer over locked per-seed runs.  It
does not rescore or reselect directions; it only reports paired deltas between
the predeclared risk-aware controller and each baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


COMPARISONS = {
    "risk_minus_gain_only": ("risk_aware", "gain_only"),
    "risk_minus_random": ("risk_aware", "random"),
    "risk_minus_high_risk": ("risk_aware", "high_risk"),
    "risk_minus_low_risk": ("risk_aware", "low_risk"),
}
METRICS = [
    "mean_probe_gain",
    "mean_rho",
    "target_hit_rate_calib",
    "mean_id_cos",
    "mean_lpips_proxy",
    "mean_abs_delta_attr",
    "mean_abs_alpha",
    "mean_lpips_true",
]
ORIENTATION = {
    "mean_probe_gain": "higher",
    "mean_rho": "lower",
    "target_hit_rate_calib": "higher",
    "mean_id_cos": "higher",
    "mean_lpips_proxy": "lower",
    "mean_abs_delta_attr": "matched",
    "mean_abs_alpha": "lower",
    "mean_lpips_true": "lower",
}


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def output_metrics_path(path: str | Path) -> Path:
    p = resolve_paper_path(path)
    return p if p.suffix == ".json" else p / "metrics.json"


def load_json(path: str | Path) -> dict[str, Any]:
    p = resolve_paper_path(path)
    return json.loads(p.read_text())


def summarize_values(values: list[float],
                     metric: str,
                     *,
                     conservative: bool = False) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": std,
        "sem": sem,
        "ci95_normal": [float(arr.mean() - 1.96 * sem),
                         float(arr.mean() + 1.96 * sem)],
        "mean_abs": float(np.abs(arr).mean()),
        "orientation": ORIENTATION[metric],
    }
    if ORIENTATION[metric] == "higher":
        wins = int((arr > 0).sum())
        out["wins"] = wins
        out["win_rate"] = float(wins / arr.size)
    elif ORIENTATION[metric] == "lower":
        wins = int((arr < 0).sum())
        out["wins"] = wins
        out["win_rate"] = float(wins / arr.size)
    else:
        out["note"] = "Raw risk-aware minus baseline semantic magnitude; closer to zero is preferred."
    if conservative:
        out["unit"] = "seed_mean"
    else:
        out["unit"] = "seed_attribute_cell"
    return out


def collect_rows(payload: dict[str, Any], source_path: str) -> list[dict[str, Any]]:
    seed = int(payload.get("config", {}).get("seed", -1))
    rows: list[dict[str, Any]] = []
    for attr in sorted(payload.get("per_attr", {})):
        summary = payload["per_attr"][attr]["summary"]
        for comp, (lhs, rhs) in COMPARISONS.items():
            for metric in METRICS:
                if metric not in summary[lhs] or metric not in summary[rhs]:
                    continue
                lhs_value = float(summary[lhs][metric])
                rhs_value = float(summary[rhs][metric])
                rows.append({
                    "seed": seed,
                    "source": source_path,
                    "attr": attr,
                    "comparison": comp,
                    "metric": metric,
                    "risk_value": lhs_value,
                    "baseline_value": rhs_value,
                    "diff": lhs_value - rhs_value,
                })
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_present = [
        metric for metric in METRICS
        if any(r["metric"] == metric for r in rows)
    ]
    cell_level: dict[str, Any] = {}
    for comp in COMPARISONS:
        cell_level[comp] = {}
        for metric in metrics_present:
            vals = [
                r["diff"] for r in rows
                if r["comparison"] == comp and r["metric"] == metric
            ]
            cell_level[comp][metric] = summarize_values(vals, metric)

    per_seed: dict[str, Any] = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        per_seed[str(seed)] = {}
        for comp in COMPARISONS:
            per_seed[str(seed)][comp] = {}
            for metric in metrics_present:
                vals = [
                    r["diff"] for r in seed_rows
                    if r["comparison"] == comp and r["metric"] == metric
                ]
                if vals:
                    per_seed[str(seed)][comp][metric] = float(np.mean(vals))

    seed_level: dict[str, Any] = {}
    for comp in COMPARISONS:
        seed_level[comp] = {}
        for metric in metrics_present:
            vals = [
                per_seed[str(seed)][comp][metric]
                for seed in sorted(int(s) for s in per_seed)
                if metric in per_seed[str(seed)][comp]
            ]
            seed_level[comp][metric] = summarize_values(
                vals, metric, conservative=True)

    return {
        "cell_level": cell_level,
        "seed_level": seed_level,
        "per_seed": per_seed,
        "metrics_present": metrics_present,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/out/control_risk_aware_robustness")
    ap.add_argument("--seed", type=int, default=3107)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="risk_aware_robustness_summary")
    args = ap.parse_args()

    set_deterministic(args.seed)
    rows: list[dict[str, Any]] = []
    inputs = []
    for input_path in args.inputs:
        p = resolve_paper_path(input_path)
        payload = load_json(p)
        inputs.append(str(p))
        rows.extend(collect_rows(payload, str(p)))

    seeds = sorted({int(r["seed"]) for r in rows})
    attrs = sorted({r["attr"] for r in rows})
    aggregate = aggregate_rows(rows)
    metrics_present = aggregate.pop("metrics_present")
    result = {
        "inputs": inputs,
        "n_runs": len(inputs),
        "seeds": seeds,
        "attrs": attrs,
        "comparisons": COMPARISONS,
        "metrics": metrics_present,
        **aggregate,
        "rows": rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_risk_aware_repeat_summary.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out_path = output_metrics_path(args.out)
    write_json_atomic(out_path, result)

    print(f"saved {out_path}")
    print(f"runs={len(inputs)} seeds={seeds} attrs={attrs}")
    for comp in COMPARISONS:
        sid = result["seed_level"][comp]["mean_id_cos"]
        slp = result["seed_level"][comp]["mean_lpips_proxy"]
        print(
            f"{comp}: seed-mean dID={sid['mean']:+.4f} "
            f"wins={sid.get('wins', '-')}/{sid['n']}  "
            f"dLPIPS={slp['mean']:+.4f} "
            f"wins={slp.get('wins', '-')}/{slp['n']}"
        )


if __name__ == "__main__":
    main()
