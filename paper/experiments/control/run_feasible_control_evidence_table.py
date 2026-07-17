"""Reviewer-facing evidence table for feasible minimum-risk control."""
from __future__ import annotations

import argparse
import json
import math
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


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text())


def sign_test_p(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * prob))


def bootstrap_ci(values: list[float], rng: np.random.Generator,
                 n_boot: int) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def summarize_values(values: list[float], expected: str,
                     rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "wins": 0,
            "sign_test_p": float("nan"),
            "bootstrap_ci95": [float("nan"), float("nan")],
        }
    if expected == "positive":
        wins = int((arr > 0).sum())
        passed_mean = float(arr.mean()) > 0
    elif expected == "negative":
        wins = int((arr < 0).sum())
        passed_mean = float(arr.mean()) < 0
    else:
        raise ValueError(f"unknown expected direction: {expected}")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "wins": wins,
        "win_rate": float(wins / arr.size),
        "sign_test_p": sign_test_p(wins, int(arr.size)),
        "bootstrap_ci95": bootstrap_ci(arr.tolist(), rng, n_boot),
        "expected": expected,
        "pass_direction": bool(
            passed_mean and wins >= max(1, math.ceil(0.8 * arr.size))
        ),
    }


def seed_values_from_repeat(summary: dict[str, Any],
                            comparison: str,
                            metric: str) -> list[float]:
    vals = []
    per_seed = summary.get("per_seed", {})
    for seed in sorted(per_seed, key=lambda x: int(x)):
        cur = per_seed[seed].get(comparison, {})
        if metric in cur:
            vals.append(float(cur[metric]))
    return vals


def seed_values_from_negative(summary: dict[str, Any],
                              comparison: str,
                              metric: str) -> list[float]:
    vals_by_seed: dict[int, list[float]] = {}
    for row in summary.get("diff_rows", []):
        if row.get("comparison") != comparison or row.get("metric") != metric:
            continue
        vals_by_seed.setdefault(int(row["seed"]), []).append(float(row["diff"]))
    return [
        float(np.mean(vals_by_seed[seed]))
        for seed in sorted(vals_by_seed)
        if vals_by_seed[seed]
    ]


def seed_values_from_predictive(summary: dict[str, Any],
                                metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in sorted(summary.get("seed_rows", []), key=lambda r: int(r["seed"]))
        if metric in row
    ]


def evidence_row(section: str, claim: str, values: list[float],
                 expected: str, rng: np.random.Generator,
                 n_boot: int, *, required: bool = True) -> dict[str, Any]:
    return {
        "section": section,
        "claim": claim,
        "required_for_feasible_control": required,
        **summarize_values(values, expected, rng, n_boot),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Section | Claim | Mean | 95% bootstrap CI | Wins | p(sign) | Required | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        ci = row["bootstrap_ci95"]
        lines.append(
            f"| {row['section']} | {row['claim']} | "
            f"{row['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{row['wins']}/{row['n']} | {row['sign_test_p']:.4f} | "
            f"{'yes' if row['required_for_feasible_control'] else 'no'} | "
            f"{'yes' if row['pass_direction'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def add_predictive_rows(rows: list[dict[str, Any]], label: str,
                        summary: dict[str, Any],
                        rng: np.random.Generator, n_boot: int) -> None:
    for metric, expected in [
        ("rho_vs_lpips_spearman", "positive"),
        ("lpips_beta_rho", "positive"),
        ("matched_pair_lpips_low_minus_high", "negative"),
        ("matched_pair_id_low_minus_high", "positive"),
    ]:
        rows.append(evidence_row(
            label,
            metric,
            seed_values_from_predictive(summary, metric),
            expected,
            rng,
            n_boot,
        ))


def add_feasible_controller_rows(rows: list[dict[str, Any]], label: str,
                                 summary: dict[str, Any],
                                 rng: np.random.Generator, n_boot: int) -> None:
    for comp in [
        "risk_minus_gain_only",
        "risk_minus_random",
        "risk_minus_high_risk",
    ]:
        rows.append(evidence_row(
            label,
            f"{comp} ID",
            seed_values_from_repeat(summary, comp, "mean_id_cos"),
            "positive",
            rng,
            n_boot,
        ))
        rows.append(evidence_row(
            label,
            f"{comp} true-LPIPS",
            seed_values_from_repeat(summary, comp, "mean_lpips_true"),
            "negative",
            rng,
            n_boot,
        ))


def add_low_risk_boundary_rows(rows: list[dict[str, Any]], label: str,
                               summary: dict[str, Any],
                               rng: np.random.Generator, n_boot: int) -> None:
    for metric, expected in [
        ("mean_id_cos", "positive"),
        ("mean_lpips_true", "negative"),
        ("mean_probe_gain", "positive"),
        ("mean_abs_alpha", "negative"),
    ]:
        rows.append(evidence_row(
            label,
            f"risk_minus_low_risk {metric}",
            seed_values_from_repeat(summary, "risk_minus_low_risk", metric),
            expected,
            rng,
            n_boot,
            required=False,
        ))


def add_negative_rows(rows: list[dict[str, Any]], label: str,
                      summary: dict[str, Any],
                      rng: np.random.Generator, n_boot: int) -> None:
    for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
        rows.append(evidence_row(
            label,
            f"{comp} ID",
            seed_values_from_negative(summary, comp, "mean_id_cos"),
            "positive",
            rng,
            n_boot,
        ))
        rows.append(evidence_row(
            label,
            f"{comp} true-LPIPS",
            seed_values_from_negative(summary, comp, "mean_lpips_true"),
            "negative",
            rng,
            n_boot,
        ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--church-predictive", required=True)
    ap.add_argument("--bedroom-predictive", required=True)
    ap.add_argument("--church-actual", required=True)
    ap.add_argument("--church-negative", required=True)
    ap.add_argument("--bedroom-actual", required=True)
    ap.add_argument("--bedroom-negative", required=True)
    ap.add_argument("--readiness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4131)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="feasible_control_evidence_table")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    set_deterministic(args.seed)
    rows: list[dict[str, Any]] = []

    add_predictive_rows(rows, "church predictive", load(args.church_predictive),
                        rng, args.n_boot)
    add_predictive_rows(rows, "bedroom predictive", load(args.bedroom_predictive),
                        rng, args.n_boot)

    church_actual = load(args.church_actual)
    church_negative = load(args.church_negative)
    bedroom_actual = load(args.bedroom_actual)
    bedroom_negative = load(args.bedroom_negative)
    add_feasible_controller_rows(rows, "church feasible controller",
                                 church_actual, rng, args.n_boot)
    add_negative_rows(rows, "church feasible negatives",
                      church_negative, rng, args.n_boot)
    add_low_risk_boundary_rows(rows, "church low-risk boundary",
                               church_actual, rng, args.n_boot)
    add_feasible_controller_rows(rows, "bedroom feasible controller",
                                 bedroom_actual, rng, args.n_boot)
    add_negative_rows(rows, "bedroom feasible negatives",
                      bedroom_negative, rng, args.n_boot)
    add_low_risk_boundary_rows(rows, "bedroom low-risk boundary",
                               bedroom_actual, rng, args.n_boot)

    readiness = load(args.readiness)
    required_rows = [r for r in rows if r["required_for_feasible_control"]]
    result = {
        "rows": rows,
        "readiness": readiness.get("readiness"),
        "required_pass_count": int(sum(r["pass_direction"] for r in required_rows)),
        "required_total": len(required_rows),
        "all_pass_count": int(sum(r["pass_direction"] for r in rows)),
        "all_total": len(rows),
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_feasible_control_evidence_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    write_json_atomic(out, result)
    (out.parent / "feasible_control_evidence_table.md").write_text(
        markdown_table(rows))
    print(f"saved {out}")
    print(
        f"readiness={result['readiness']} "
        f"required={result['required_pass_count']}/{result['required_total']} "
        f"all={result['all_pass_count']}/{result['all_total']}"
    )


if __name__ == "__main__":
    main()
