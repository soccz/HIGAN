"""Permutation-null test for predictive-validity summaries.

This table asks whether the observed rho/damage relationship is stronger than
one obtained by shuffling rho within each attribute.  It is a non-GPU sanity
check for the concern that the result is a bookkeeping artifact of the summary
or the candidate pool.
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


METRICS = [
    ("rho_vs_id_spearman", "rho vs ID Spearman", "negative", False),
    ("rho_vs_lpips_spearman", "rho vs LPIPS Spearman", "positive", True),
    ("id_beta_rho", "rho beta for ID", "negative", False),
    ("lpips_beta_rho", "rho beta for LPIPS", "positive", True),
    ("matched_pair_id_low_minus_high", "low-risk minus high-risk ID", "positive", True),
    (
        "matched_pair_lpips_low_minus_high",
        "low-risk minus high-risk LPIPS",
        "negative",
        True,
    ),
    ("rho_vs_dino_spearman", "rho vs DINO Spearman", "negative", True),
    ("dino_beta_rho", "rho beta for DINO", "negative", True),
    (
        "matched_pair_dino_low_minus_high",
        "low-risk minus high-risk DINO",
        "positive",
        True,
    ),
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


def spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = rankdata(xa[mask])
    yr = rankdata(ya[mask])
    if float(xr.std()) == 0.0 or float(yr.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def standardized_beta(rows: list[dict[str, Any]], y_key: str) -> float:
    x_keys = ["mean_abs_delta_attr", "probe_gain", "rho"]
    x = np.asarray([[float(r[k]) for k in x_keys] for r in rows], dtype=float)
    y = np.asarray([float(r[y_key]) for r in rows], dtype=float)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[mask]
    y = y[mask]
    if x.shape[0] <= len(x_keys):
        return float("nan")
    x_std = x.std(axis=0)
    y_std = float(y.std())
    keep = x_std > 1e-12
    if y_std <= 1e-12 or not keep[-1]:
        return float("nan")
    xz = (x[:, keep] - x[:, keep].mean(axis=0)) / x_std[keep]
    yz = (y - y.mean()) / y_std
    design = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    kept_keys = [k for k, ok in zip(x_keys, keep) if ok]
    beta_by_key = {k: float(v) for k, v in zip(kept_keys, coef[1:])}
    return beta_by_key.get("rho", float("nan"))


def lpips_key(rows: list[dict[str, Any]]) -> str:
    if rows and all("mean_lpips_true" in r for r in rows):
        return "mean_lpips_true"
    return "mean_lpips_proxy"


def eligible_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attr, attr_payload in payload.get("per_attr", {}).items():
        for row in attr_payload.get("rows", []):
            if row.get("gain_eligible") is True:
                enriched = dict(row)
                enriched.setdefault("attr", attr)
                rows.append(enriched)
    return rows


def build_matched_pairs(rows: list[dict[str, Any]],
                        gain_match_rel: float) -> list[dict[str, Any]]:
    pairs = []
    for i, lhs in enumerate(rows):
        for rhs in rows[i + 1:]:
            if lhs.get("attr") != rhs.get("attr"):
                continue
            gain_scale = max(
                abs(float(lhs["probe_gain"])),
                abs(float(rhs["probe_gain"])),
                1e-8,
            )
            if abs(float(lhs["probe_gain"]) - float(rhs["probe_gain"])) > (
                    gain_match_rel * gain_scale):
                continue
            if float(lhs["rho"]) == float(rhs["rho"]):
                continue
            low, high = (lhs, rhs) if float(lhs["rho"]) < float(rhs["rho"]) else (rhs, lhs)
            pair = {
                "mean_id_cos_diff_low_minus_high": (
                    float(low["mean_id_cos"]) - float(high["mean_id_cos"])
                ),
                "mean_lpips_proxy_diff_low_minus_high": (
                    float(low["mean_lpips_proxy"]) -
                    float(high["mean_lpips_proxy"])
                ),
            }
            if "mean_lpips_true" in low and "mean_lpips_true" in high:
                pair["mean_lpips_true_diff_low_minus_high"] = (
                    float(low["mean_lpips_true"]) -
                    float(high["mean_lpips_true"])
                )
            if "mean_dino_cos" in low and "mean_dino_cos" in high:
                pair["mean_dino_cos_diff_low_minus_high"] = (
                    float(low["mean_dino_cos"]) - float(high["mean_dino_cos"])
                )
            pairs.append(pair)
    return pairs


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def compute_metrics(rows: list[dict[str, Any]],
                    gain_match_rel: float) -> dict[str, float]:
    if not rows:
        return {}
    lp_key = lpips_key(rows)
    pairs = build_matched_pairs(rows, gain_match_rel)
    metrics = {
        "rho_vs_id_spearman": spearman(
            [float(r["rho"]) for r in rows],
            [float(r["mean_id_cos"]) for r in rows],
        ),
        "rho_vs_lpips_spearman": spearman(
            [float(r["rho"]) for r in rows],
            [float(r[lp_key]) for r in rows],
        ),
        "id_beta_rho": standardized_beta(rows, "mean_id_cos"),
        "lpips_beta_rho": standardized_beta(rows, lp_key),
        "matched_pair_id_low_minus_high": finite_mean(
            [p["mean_id_cos_diff_low_minus_high"] for p in pairs]
        ),
        "matched_pair_lpips_low_minus_high": finite_mean(
            [p[f"{lp_key}_diff_low_minus_high"] for p in pairs]
        ),
    }
    if all("mean_dino_cos" in r for r in rows):
        metrics["rho_vs_dino_spearman"] = spearman(
            [float(r["rho"]) for r in rows],
            [float(r["mean_dino_cos"]) for r in rows],
        )
        metrics["dino_beta_rho"] = standardized_beta(rows, "mean_dino_cos")
        metrics["matched_pair_dino_low_minus_high"] = finite_mean(
            [p["mean_dino_cos_diff_low_minus_high"]
             for p in pairs
             if "mean_dino_cos_diff_low_minus_high" in p]
        )
    return metrics


def shuffled_rows(rows: list[dict[str, Any]],
                  rng: np.random.Generator) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    by_attr: dict[str, list[int]] = {}
    for idx, row in enumerate(out):
        by_attr.setdefault(str(row.get("attr", "")), []).append(idx)
    for indices in by_attr.values():
        if len(indices) < 2:
            continue
        rhos = np.asarray([float(out[idx]["rho"]) for idx in indices], dtype=float)
        rng.shuffle(rhos)
        for idx, rho in zip(indices, rhos):
            out[idx]["rho"] = float(rho)
    return out


def stronger(obs: float, null_value: float, expected: str) -> bool:
    if not (math.isfinite(obs) and math.isfinite(null_value)):
        return False
    if expected == "positive":
        return obs > null_value
    if expected == "negative":
        return obs < null_value
    raise ValueError(expected)


def empirical_p(obs: float, nulls: list[float], expected: str) -> float | None:
    arr = np.asarray(nulls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not math.isfinite(obs) or arr.size == 0:
        return None
    if expected == "positive":
        return float((1 + np.sum(arr >= obs)) / (arr.size + 1))
    if expected == "negative":
        return float((1 + np.sum(arr <= obs)) / (arr.size + 1))
    raise ValueError(expected)


def summarize_metric(metric_runs: list[dict[str, Any]],
                     expected: str, required: bool, alpha: float) -> dict[str, Any]:
    observed = [
        float(run["observed"])
        for run in metric_runs
        if isinstance(run.get("observed"), (int, float)) and math.isfinite(run["observed"])
    ]
    null_medians = [
        float(run["null_median"])
        for run in metric_runs
        if isinstance(run.get("null_median"), (int, float))
        and math.isfinite(run["null_median"])
    ]
    p_values = [
        float(run["empirical_p"])
        for run in metric_runs
        if isinstance(run.get("empirical_p"), (int, float))
        and math.isfinite(run["empirical_p"])
    ]
    n = min(len(observed), len(null_medians))
    wins = sum(
        1 for run in metric_runs
        if stronger(
            float(run.get("observed", float("nan"))),
            float(run.get("null_median", float("nan"))),
            expected,
        )
    )
    mean_observed = finite_mean(observed)
    mean_null_median = finite_mean(null_medians)
    median_p = float(np.median(p_values)) if p_values else None
    direction_ok = (
        n >= 5
        and wins >= math.ceil(0.8 * n)
        and stronger(mean_observed, mean_null_median, expected)
    )
    p_ok = median_p is not None and median_p <= alpha
    return {
        "mean": mean_observed,
        "null_median_mean": mean_null_median,
        "wins": wins,
        "n": n,
        "median_empirical_p": median_p,
        "expected": expected,
        "required": required,
        "pass": bool(direction_ok and p_ok),
    }


def format_float(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):+.{precision}f}"
    return "n/a"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Summary | Metric | Observed | Null median | Wins | median p(null) | Required | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        p_null = row.get("median_empirical_p")
        p_text = f"{p_null:.4f}" if isinstance(p_null, float) else "n/a"
        lines.append(
            "| {summary} | {metric} | {mean} | {null} | {wins}/{n} | {p} | {required} | {passed} |".format(
                summary=row["summary"],
                metric=row["display"],
                mean=format_float(row.get("mean")),
                null=format_float(row.get("null_median_mean")),
                wins=row.get("wins", "n/a"),
                n=row.get("n", "n/a"),
                p=p_text,
                required="yes" if row["required"] else "no",
                passed="yes" if row["pass"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def readiness_label(required_passes: int, required_total: int) -> str:
    if required_passes == required_total:
        return "predictive_permutation_null_ready"
    if required_passes >= required_total - 4:
        return "predictive_permutation_null_borderline"
    return "predictive_permutation_null_sensitive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=4221)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="predictive_permutation_null_table")
    args = ap.parse_args()

    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1)")

    set_deterministic(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = []
    detail_rows = []
    for label, summary_path in [parse_summary_arg(raw) for raw in args.summaries]:
        summary = json.loads(summary_path.read_text())
        metric_runs: dict[str, list[dict[str, Any]]] = {}
        for raw_path in summary.get("inputs", []):
            payload = json.loads(resolve(raw_path).read_text())
            run_rows = eligible_rows(payload)
            gain_match_rel = float(
                payload.get("config", {}).get("gain_match_rel", 0.25)
            )
            observed = compute_metrics(run_rows, gain_match_rel)
            null_values: dict[str, list[float]] = {key: [] for key in observed}
            for _ in range(args.permutations):
                shuffled = shuffled_rows(run_rows, rng)
                permuted = compute_metrics(shuffled, gain_match_rel)
                for key, value in permuted.items():
                    if key in null_values:
                        null_values[key].append(value)
            for key, value in observed.items():
                nulls = null_values.get(key, [])
                _, _, expected, _ = next(m for m in METRICS if m[0] == key)
                run_detail = {
                    "summary": label,
                    "source": str(resolve(raw_path)),
                    "metric": key,
                    "observed": value,
                    "null_median": finite_mean([float(np.median(nulls))])
                    if nulls else float("nan"),
                    "empirical_p": empirical_p(value, nulls, expected),
                }
                metric_runs.setdefault(key, []).append(run_detail)
                detail_rows.append(run_detail)

        for key, display, expected, required in METRICS:
            runs = metric_runs.get(key, [])
            if not runs:
                continue
            row = {
                "summary": label,
                "metric": key,
                "display": display,
                **summarize_metric(runs, expected, required, args.alpha),
            }
            rows.append(row)

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
            "script": "experiments/control/run_predictive_permutation_null_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    table_path = out.parent / "predictive_permutation_null_table.md"
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
