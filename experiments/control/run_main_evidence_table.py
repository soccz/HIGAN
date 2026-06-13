"""Build the final main-paper evidence table from locked summaries.

This reducer creates reviewer-facing evidence rows with seed-level effects,
bootstrap confidence intervals, and exact sign-test p-values.  It is a pure
post-hoc table builder over predeclared outputs; it does not tune or filter
experiments based on observed results.
"""
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
    return [float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975))]


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
        "pass_direction": bool(passed_mean and wins >= max(1, math.ceil(0.8 * arr.size))),
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
                 n_boot: int) -> dict[str, Any]:
    return {
        "section": section,
        "claim": claim,
        **summarize_values(values, expected, rng, n_boot),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Section | Claim | Mean | 95% bootstrap CI | Wins | p(sign) | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        ci = row["bootstrap_ci95"]
        lines.append(
            f"| {row['section']} | {row['claim']} | "
            f"{row['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{row['wins']}/{row['n']} | {row['sign_test_p']:.4f} | "
            f"{'yes' if row['pass_direction'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def add_controller_rows(rows: list[dict[str, Any]],
                        label: str,
                        summary: dict[str, Any],
                        rng: np.random.Generator,
                        n_boot: int) -> None:
    for comp in [
        "risk_minus_gain_only",
        "risk_minus_random",
        "risk_minus_high_risk",
        "risk_minus_low_risk",
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


def add_negative_rows(rows: list[dict[str, Any]],
                      label: str,
                      summary: dict[str, Any],
                      rng: np.random.Generator,
                      n_boot: int) -> None:
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


def add_predictive_rows(rows: list[dict[str, Any]],
                        label: str,
                        summary: dict[str, Any],
                        rng: np.random.Generator,
                        n_boot: int) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bedroom-actual", required=True)
    ap.add_argument("--bedroom-negative", required=True)
    ap.add_argument("--church-structured", required=True)
    ap.add_argument("--church-structured-negative", required=True)
    ap.add_argument("--church-confirmatory", required=True)
    ap.add_argument("--church-confirmatory-negative", required=True)
    ap.add_argument("--church-predictive", required=True)
    ap.add_argument("--bedroom-predictive", required=True)
    ap.add_argument("--tiebreak-actual", required=True)
    ap.add_argument("--tiebreak-negative", required=True)
    ap.add_argument("--readiness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4111)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="main_evidence_table")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    set_deterministic(args.seed)
    rows: list[dict[str, Any]] = []

    add_controller_rows(rows, "bedroom controller", load(args.bedroom_actual),
                        rng, args.n_boot)
    add_negative_rows(rows, "bedroom negatives", load(args.bedroom_negative),
                      rng, args.n_boot)
    add_controller_rows(rows, "church structured controller",
                        load(args.church_structured), rng, args.n_boot)
    add_negative_rows(rows, "church structured negatives",
                      load(args.church_structured_negative), rng, args.n_boot)
    add_controller_rows(rows, "church confirmatory controller",
                        load(args.church_confirmatory), rng, args.n_boot)
    add_negative_rows(rows, "church confirmatory negatives",
                      load(args.church_confirmatory_negative), rng, args.n_boot)
    add_predictive_rows(rows, "church predictive", load(args.church_predictive),
                        rng, args.n_boot)
    add_predictive_rows(rows, "bedroom predictive", load(args.bedroom_predictive),
                        rng, args.n_boot)
    add_controller_rows(rows, "gain-first risk-tiebreak controller",
                        load(args.tiebreak_actual), rng, args.n_boot)
    add_negative_rows(rows, "gain-first risk-tiebreak negatives",
                      load(args.tiebreak_negative), rng, args.n_boot)

    readiness = load(args.readiness)
    result = {
        "rows": rows,
        "readiness": readiness.get("readiness"),
        "pass_count": int(sum(1 for row in rows if row["pass_direction"])),
        "total": len(rows),
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_main_evidence_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    write_json_atomic(out, result)
    md = markdown_table(rows)
    (out.parent / "main_evidence_table.md").write_text(md)
    print(f"saved {out}")
    print(f"readiness={result['readiness']} evidence={result['pass_count']}/{result['total']}")


if __name__ == "__main__":
    main()
