"""Compare actual risk-aware selection against shuffled/inverted-risk controls."""
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


def load_payloads(paths: list[str]) -> dict[int, dict[str, Any]]:
    loaded = {}
    for raw in paths:
        p = resolve_paper_path(raw)
        payload = json.loads(p.read_text())
        seed = int(payload.get("config", {}).get("seed", -1))
        loaded[seed] = payload
    return loaded


def summarize_values(values: list[float], metric: str, unit: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    out: dict[str, Any] = {
        "n": int(arr.size),
        "unit": unit,
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
        out["wins"] = int((arr > 0).sum())
        out["win_rate"] = float(out["wins"] / arr.size)
    elif ORIENTATION[metric] == "lower":
        out["wins"] = int((arr < 0).sum())
        out["win_rate"] = float(out["wins"] / arr.size)
    else:
        out["note"] = "Raw first mode minus second mode semantic magnitude; closer to zero is preferred."
    return out


def controller_rows(mode_payloads: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, by_seed in mode_payloads.items():
        for seed, payload in by_seed.items():
            for attr in sorted(payload.get("per_attr", {})):
                summary = payload["per_attr"][attr]["summary"]
                for metric in METRICS:
                    if metric not in summary["risk_aware"]:
                        continue
                    rows.append({
                        "mode": mode,
                        "seed": seed,
                        "attr": attr,
                        "metric": metric,
                        "risk_aware": float(summary["risk_aware"][metric]),
                        "gain_only": float(summary["gain_only"][metric]),
                        "random": float(summary["random"][metric]),
                        "high_risk": float(summary["high_risk"][metric]),
                        "low_risk": float(summary["low_risk"][metric]),
                    })
    return rows


def paired_mode_diffs(mode_payloads: dict[str, dict[int, dict[str, Any]]],
                      lhs: str,
                      rhs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_seeds = sorted(set(mode_payloads[lhs]) & set(mode_payloads[rhs]))
    for seed in common_seeds:
        lhs_payload = mode_payloads[lhs][seed]
        rhs_payload = mode_payloads[rhs][seed]
        common_attrs = sorted(set(lhs_payload.get("per_attr", {})) &
                              set(rhs_payload.get("per_attr", {})))
        for attr in common_attrs:
            lhs_summary = lhs_payload["per_attr"][attr]["summary"]["risk_aware"]
            rhs_summary = rhs_payload["per_attr"][attr]["summary"]["risk_aware"]
            for metric in METRICS:
                if metric not in lhs_summary or metric not in rhs_summary:
                    continue
                rows.append({
                    "comparison": f"{lhs}_minus_{rhs}",
                    "seed": seed,
                    "attr": attr,
                    "metric": metric,
                    "lhs_value": float(lhs_summary[metric]),
                    "rhs_value": float(rhs_summary[metric]),
                    "diff": float(lhs_summary[metric]) - float(rhs_summary[metric]),
                })
    return rows


def aggregate_diff_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell_level: dict[str, Any] = {}
    seed_level: dict[str, Any] = {}
    for comp in sorted({r["comparison"] for r in rows}):
        cell_level[comp] = {}
        seed_level[comp] = {}
        for metric in METRICS:
            vals = [
                r["diff"] for r in rows
                if r["comparison"] == comp and r["metric"] == metric
            ]
            cell_level[comp][metric] = summarize_values(
                vals, metric, "seed_attribute_cell")
            seed_means = []
            for seed in sorted({int(r["seed"]) for r in rows}):
                cur = [
                    r["diff"] for r in rows
                    if r["comparison"] == comp
                    and r["metric"] == metric
                    and int(r["seed"]) == seed
                ]
                if cur:
                    seed_means.append(float(np.mean(cur)))
            seed_level[comp][metric] = summarize_values(
                seed_means, metric, "seed_mean")
    return {"cell_level": cell_level, "seed_level": seed_level}


def aggregate_mode_vs_baseline(rows: list[dict[str, Any]],
                               baseline: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in rows}):
        out[mode] = {}
        for metric in METRICS:
            cell_vals = [
                r["risk_aware"] - r[baseline] for r in rows
                if r["mode"] == mode and r["metric"] == metric
            ]
            out[mode][metric] = summarize_values(
                cell_vals, metric, "seed_attribute_cell")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actual-inputs", nargs="+", required=True)
    ap.add_argument("--shuffled-inputs", nargs="+", required=True)
    ap.add_argument("--inverted-inputs", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/out/control_risk_signal_negative_controls")
    ap.add_argument("--seed", type=int, default=3307)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="risk_signal_negative_summary")
    args = ap.parse_args()

    set_deterministic(args.seed)
    mode_payloads = {
        "actual": load_payloads(args.actual_inputs),
        "shuffled": load_payloads(args.shuffled_inputs),
        "inverted": load_payloads(args.inverted_inputs),
    }
    diff_rows = []
    diff_rows.extend(paired_mode_diffs(mode_payloads, "actual", "shuffled"))
    diff_rows.extend(paired_mode_diffs(mode_payloads, "actual", "inverted"))
    rows = controller_rows(mode_payloads)
    result = {
        "modes": sorted(mode_payloads),
        "seeds": {
            mode: sorted(by_seed)
            for mode, by_seed in mode_payloads.items()
        },
        "metrics": METRICS,
        "paired_controller_diffs": aggregate_diff_rows(diff_rows),
        "controller_vs_gain_only": aggregate_mode_vs_baseline(rows, "gain_only"),
        "controller_vs_random": aggregate_mode_vs_baseline(rows, "random"),
        "controller_rows": rows,
        "diff_rows": diff_rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_risk_signal_negative_summary.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out_path = output_metrics_path(args.out)
    write_json_atomic(out_path, result)

    print(f"saved {out_path}")
    for comp, metrics in result["paired_controller_diffs"]["seed_level"].items():
        sid = metrics["mean_id_cos"]
        slp = metrics["mean_lpips_proxy"]
        print(
            f"{comp}: seed-mean dID={sid['mean']:+.4f} "
            f"wins={sid.get('wins', '-')}/{sid['n']}  "
            f"dLPIPS={slp['mean']:+.4f} "
            f"wins={slp.get('wins', '-')}/{slp['n']}"
        )


if __name__ == "__main__":
    main()
