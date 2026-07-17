"""Aggregate cross-domain risk predictive-validity runs."""
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


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def output_metrics_path(path: str | Path) -> Path:
    p = resolve(path)
    return p if p.suffix == ".json" else p / "metrics.json"


def summarize(values: list[float], orientation: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
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
        "ci95_normal": [
            float(arr.mean() - 1.96 * sem),
            float(arr.mean() + 1.96 * sem),
        ],
        "orientation": orientation,
    }
    if orientation == "higher":
        out["wins"] = int((arr > 0).sum())
    elif orientation == "lower":
        out["wins"] = int((arr < 0).sum())
    return out


def lpips_pair_metric(aggregate: dict[str, Any]) -> str:
    for key in aggregate["matched_pairs"]:
        if "lpips" in key:
            return key
    raise KeyError("no LPIPS matched-pair metric found")


def add_dino_row_metrics(row: dict[str, Any],
                         aggregate: dict[str, Any]) -> None:
    if "rho_vs_dino" not in aggregate.get("spearman", {}):
        return
    row["rho_vs_dino_spearman"] = aggregate["spearman"]["rho_vs_dino"]["r"]
    row["dino_beta_rho"] = aggregate["regression"]["dino"]["beta_rho"]
    row["matched_pair_dino_low_minus_high"] = (
        aggregate["matched_pairs"]["mean_dino_cos_diff_low_minus_high"]
        .get("mean", float("nan"))
    )


def add_seed_level_if_present(seed_level: dict[str, Any],
                              seed_rows: list[dict[str, Any]],
                              key: str,
                              orientation: str) -> None:
    if all(key in row for row in seed_rows):
        seed_level[key] = summarize([row[key] for row in seed_rows], orientation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4081)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="risk_predictive_summary")
    args = ap.parse_args()

    set_deterministic(args.seed)
    rows = []
    seed_rows = []
    for raw in args.inputs:
        p = resolve(raw)
        payload = json.loads(p.read_text())
        aggregate = payload["aggregate"]
        pair_key = lpips_pair_metric(aggregate)
        seed = int(payload.get("config", {}).get("seed", -1))
        row = {
            "seed": seed,
            "source": str(p),
            "domain": payload.get("config", {}).get("domain"),
            "risk_estimator": payload.get("config", {}).get("risk_estimator"),
            "prompt_style": payload.get("config", {}).get("prompt_style"),
            "n_rows": aggregate["n_rows"],
            "n_matched_pairs": aggregate["n_matched_pairs"],
            "rho_vs_id_spearman": aggregate["spearman"]["rho_vs_id"]["r"],
            "rho_vs_lpips_spearman": aggregate["spearman"]["rho_vs_lpips"]["r"],
            "id_beta_rho": aggregate["regression"]["id"]["beta_rho"],
            "lpips_beta_rho": aggregate["regression"]["lpips"]["beta_rho"],
            "matched_pair_id_low_minus_high": (
                aggregate["matched_pairs"]
                ["mean_id_cos_diff_low_minus_high"].get("mean", float("nan"))
            ),
            "matched_pair_lpips_low_minus_high": (
                aggregate["matched_pairs"][pair_key].get("mean", float("nan"))
            ),
        }
        add_dino_row_metrics(row, aggregate)
        seed_rows.append(row)
        for attr, attr_payload in payload.get("per_attr", {}).items():
            attr_agg = attr_payload["aggregate"]
            attr_pair_key = lpips_pair_metric(attr_agg)
            attr_row = {
                **row,
                "attr": attr,
                "attr_n_rows": attr_agg["n_rows"],
                "attr_n_matched_pairs": attr_agg["n_matched_pairs"],
                "attr_rho_vs_id_spearman": attr_agg["spearman"]["rho_vs_id"]["r"],
                "attr_rho_vs_lpips_spearman": attr_agg["spearman"]["rho_vs_lpips"]["r"],
                "attr_id_beta_rho": attr_agg["regression"]["id"]["beta_rho"],
                "attr_lpips_beta_rho": attr_agg["regression"]["lpips"]["beta_rho"],
                "attr_matched_pair_id_low_minus_high": (
                    attr_agg["matched_pairs"]
                    ["mean_id_cos_diff_low_minus_high"].get("mean", float("nan"))
                ),
                "attr_matched_pair_lpips_low_minus_high": (
                    attr_agg["matched_pairs"][attr_pair_key].get("mean", float("nan"))
                ),
            }
            if "rho_vs_dino" in attr_agg.get("spearman", {}):
                attr_row["attr_rho_vs_dino_spearman"] = (
                    attr_agg["spearman"]["rho_vs_dino"]["r"]
                )
                attr_row["attr_dino_beta_rho"] = (
                    attr_agg["regression"]["dino"]["beta_rho"]
                )
                attr_row["attr_matched_pair_dino_low_minus_high"] = (
                    attr_agg["matched_pairs"]["mean_dino_cos_diff_low_minus_high"]
                    .get("mean", float("nan"))
                )
            rows.append(attr_row)

    seed_level = {
        "rho_vs_id_spearman": summarize(
            [r["rho_vs_id_spearman"] for r in seed_rows], "lower"),
        "rho_vs_lpips_spearman": summarize(
            [r["rho_vs_lpips_spearman"] for r in seed_rows], "higher"),
        "id_beta_rho": summarize(
            [r["id_beta_rho"] for r in seed_rows], "lower"),
        "lpips_beta_rho": summarize(
            [r["lpips_beta_rho"] for r in seed_rows], "higher"),
        "matched_pair_id_low_minus_high": summarize(
            [r["matched_pair_id_low_minus_high"] for r in seed_rows],
            "higher",
        ),
        "matched_pair_lpips_low_minus_high": summarize(
            [r["matched_pair_lpips_low_minus_high"] for r in seed_rows],
            "lower",
        ),
    }
    add_seed_level_if_present(
        seed_level, seed_rows, "rho_vs_dino_spearman", "lower")
    add_seed_level_if_present(
        seed_level, seed_rows, "dino_beta_rho", "lower")
    add_seed_level_if_present(
        seed_level, seed_rows, "matched_pair_dino_low_minus_high", "higher")

    result = {
        "inputs": [str(resolve(p)) for p in args.inputs],
        "n_runs": len(args.inputs),
        "seed_level": seed_level,
        "seed_rows": seed_rows,
        "attr_rows": rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_risk_predictive_summary.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    write_json_atomic(out, result)
    print(f"saved {out}")
    for key, summary in result["seed_level"].items():
        print(f"{key}: mean={summary.get('mean', float('nan')):+.4f} "
              f"wins={summary.get('wins', '-')}/{summary.get('n', '?')}")


if __name__ == "__main__":
    main()
