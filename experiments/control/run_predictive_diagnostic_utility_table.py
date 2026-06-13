"""Evaluate rho as a diagnostic score for high-damage edit candidates.

This table reframes the strongest defensible claim: rho should be useful as a
pre-edit risk diagnostic even when it is not a universal controller.  For each
seed-level predictive run, it ranks candidates within each attribute and asks
whether rho separates high-damage candidates from lower-damage candidates.
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


PREDICTORS = [
    ("rho", "rho"),
    ("probe_gain", "probe gain"),
    ("abs_alpha", "|alpha|"),
    ("mean_abs_delta_attr", "|semantic delta|"),
    ("calib_max_abs_delta_attr", "calib max |semantic delta|"),
]

TARGETS = [
    ("clip_id_damage", "CLIP-ID damage"),
    ("lpips_damage", "LPIPS damage"),
    ("dino_damage", "DINO damage"),
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def output_metrics_path(path: str | Path) -> Path:
    p = resolve(path)
    return p if p.suffix == ".json" else p / "metrics.json"


def parse_summary_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--summaries entries must be label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty summary label in: {raw}")
    return label, resolve(path.strip())


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def percentile_scores(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if int(mask.sum()) == 1:
        out[mask] = 0.5
    elif int(mask.sum()) > 1:
        ranks = rankdata(arr[mask])
        out[mask] = ranks / (mask.sum() - 1)
    return [float(v) for v in out]


def auroc(labels: list[int], scores: list[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def exact_sign_p(wins: int, n: int) -> float | None:
    if n <= 0:
        return None
    lo = sum(math.comb(n, k) for k in range(0, wins + 1)) / (2 ** n)
    hi = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(lo, hi)))


def raw_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attr, attr_payload in payload.get("per_attr", {}).items():
        for row in attr_payload.get("rows", []):
            if row.get("gain_eligible") is not True:
                continue
            enriched = dict(row)
            enriched.setdefault("attr", attr)
            enriched["abs_alpha"] = abs(float(enriched.get("alpha", 0.0)))
            enriched["clip_id_damage"] = 1.0 - float(enriched["mean_id_cos"])
            lpips_key = (
                "mean_lpips_true"
                if "mean_lpips_true" in enriched
                else "mean_lpips_proxy"
            )
            enriched["lpips_damage"] = float(enriched[lpips_key])
            if "mean_dino_cos" in enriched:
                enriched["dino_damage"] = 1.0 - float(enriched["mean_dino_cos"])
            rows.append(enriched)
    return rows


def run_diagnostics(rows: list[dict[str, Any]],
                    target: str, high_quantile: float) -> dict[str, float]:
    labels: list[int] = []
    scores_by_predictor: dict[str, list[float]] = {key: [] for key, _ in PREDICTORS}
    by_attr: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if target in row:
            by_attr.setdefault(str(row.get("attr", "")), []).append(row)

    for attr_rows in by_attr.values():
        target_values = [float(row[target]) for row in attr_rows]
        finite_targets = np.asarray(
            [v for v in target_values if math.isfinite(v)], dtype=float)
        if finite_targets.size < 4:
            continue
        threshold = float(np.quantile(finite_targets, high_quantile))
        attr_labels = [
            int(math.isfinite(float(row[target])) and float(row[target]) >= threshold)
            for row in attr_rows
        ]
        if sum(attr_labels) == 0 or sum(attr_labels) == len(attr_labels):
            continue
        labels.extend(attr_labels)
        for key, _ in PREDICTORS:
            values = [float(row.get(key, float("nan"))) for row in attr_rows]
            scores_by_predictor[key].extend(percentile_scores(values))

    result = {}
    for key, _ in PREDICTORS:
        result[f"{key}_auroc"] = auroc(labels, scores_by_predictor[key])
    baseline_keys = [key for key, _ in PREDICTORS if key != "rho"]
    baseline_aurocs = [
        result[f"{key}_auroc"]
        for key in baseline_keys
        if math.isfinite(result.get(f"{key}_auroc", float("nan")))
    ]
    result["best_baseline_auroc"] = max(baseline_aurocs) if baseline_aurocs else float("nan")
    result["rho_minus_best_baseline_auroc"] = (
        result["rho_auroc"] - result["best_baseline_auroc"]
        if math.isfinite(result.get("rho_auroc", float("nan")))
        and math.isfinite(result.get("best_baseline_auroc", float("nan")))
        else float("nan")
    )
    result["n_candidates"] = float(len(labels))
    result["n_high_damage"] = float(sum(labels))
    return result


def summarize(values: list[float], *, expected: str,
              required: bool, threshold: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "wins": None,
            "n": 0,
            "p_sign_two_sided": None,
            "expected": expected,
            "required": required,
            "pass": False,
        }
    mean = float(arr.mean())
    if expected == "above":
        wins = int((arr > threshold).sum())
        passed = mean > threshold and wins >= math.ceil(0.8 * arr.size)
    elif expected == "positive":
        wins = int((arr > threshold).sum())
        passed = mean > threshold and wins >= math.ceil(0.8 * arr.size)
    else:
        raise ValueError(expected)
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "wins": wins,
        "n": int(arr.size),
        "p_sign_two_sided": exact_sign_p(wins, int(arr.size)),
        "expected": expected,
        "threshold": threshold,
        "required": required,
        "pass": bool(passed),
    }


def readiness_label(required_passes: int, required_total: int) -> str:
    if required_passes == required_total:
        return "predictive_diagnostic_utility_ready"
    if required_passes >= required_total - 4:
        return "predictive_diagnostic_utility_borderline"
    return "predictive_diagnostic_utility_sensitive"


def format_float(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):+.{precision}f}"
    return "n/a"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Summary | Metric | Mean | Wins | p(sign) | Required | Pass |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        p_sign = row.get("p_sign_two_sided")
        p_text = f"{p_sign:.4f}" if isinstance(p_sign, float) else "n/a"
        lines.append(
            "| {summary} | {metric} | {mean} | {wins}/{n} | {p} | {required} | {passed} |".format(
                summary=row["summary"],
                metric=row["display"],
                mean=format_float(row.get("mean")),
                wins=row.get("wins", "n/a"),
                n=row.get("n", "n/a"),
                p=p_text,
                required="yes" if row["required"] else "no",
                passed="yes" if row["pass"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--high-quantile", type=float, default=0.75)
    ap.add_argument("--rho-auroc-threshold", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=4226)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="predictive_diagnostic_utility_table")
    args = ap.parse_args()

    if not 0.0 < args.high_quantile < 1.0:
        raise ValueError("--high-quantile must be in (0, 1)")
    if not 0.5 <= args.rho_auroc_threshold < 1.0:
        raise ValueError("--rho-auroc-threshold must be in [0.5, 1)")

    set_deterministic(args.seed)
    detail_rows = []
    rows = []
    for label, summary_path in [parse_summary_arg(raw) for raw in args.summaries]:
        summary = json.loads(summary_path.read_text())
        per_target: dict[str, list[dict[str, float]]] = {
            target: [] for target, _ in TARGETS
        }
        for raw_path in summary.get("inputs", []):
            payload = json.loads(resolve(raw_path).read_text())
            run_rows = raw_rows(payload)
            for target, _ in TARGETS:
                if not any(target in row for row in run_rows):
                    continue
                metrics = run_diagnostics(run_rows, target, args.high_quantile)
                if not math.isfinite(metrics.get("rho_auroc", float("nan"))):
                    continue
                metrics["seed"] = float(payload.get("config", {}).get("seed", -1))
                metrics["source"] = str(resolve(raw_path))
                per_target[target].append(metrics)
                detail_rows.append({
                    "summary": label,
                    "target": target,
                    **metrics,
                })
        for target, display in TARGETS:
            runs = per_target[target]
            if not runs:
                continue
            rho_values = [run["rho_auroc"] for run in runs]
            delta_values = [run["rho_minus_best_baseline_auroc"] for run in runs]
            rows.append({
                "summary": label,
                "metric": f"rho_auroc_{target}",
                "display": f"rho AUROC for {display}",
                **summarize(
                    rho_values,
                    expected="above",
                    required=True,
                    threshold=args.rho_auroc_threshold,
                ),
            })
            rows.append({
                "summary": label,
                "metric": f"rho_minus_best_baseline_auroc_{target}",
                "display": f"rho AUROC minus best simple baseline for {display}",
                **summarize(
                    delta_values,
                    expected="positive",
                    required=False,
                    threshold=0.0,
                ),
            })

    required_rows = [row for row in rows if row["required"]]
    required_passes = sum(1 for row in required_rows if row["pass"])
    pass_count = sum(1 for row in rows if row["pass"])
    result = {
        "readiness": readiness_label(required_passes, len(required_rows)),
        "pass_count": pass_count,
        "total": len(rows),
        "required_pass_count": required_passes,
        "required_total": len(required_rows),
        "rows": rows,
        "detail_rows": detail_rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_predictive_diagnostic_utility_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    table_path = out.parent / "predictive_diagnostic_utility_table.md"
    write_markdown(table_path, rows)
    print(f"saved {out}")
    print(f"table={table_path}")
    print(
        f"readiness={result['readiness']} "
        f"required_pass={required_passes}/{len(required_rows)} "
        f"all_pass={pass_count}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
