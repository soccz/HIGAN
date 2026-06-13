"""Build a generic evidence table for predictive-validity stress tests."""
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


def direction_pass(mean: Any, wins: Any, n: Any, expected: str) -> bool:
    if not isinstance(mean, (int, float)) or not isinstance(wins, int):
        return False
    if not isinstance(n, int) or n < 5:
        return False
    if expected == "positive":
        return mean > 0 and wins >= math.ceil(0.8 * n)
    if expected == "negative":
        return mean < 0 and wins >= math.ceil(0.8 * n)
    raise ValueError(expected)


def exact_sign_p(wins: Any, n: Any) -> float | None:
    if not isinstance(wins, int) or not isinstance(n, int) or n <= 0:
        return None
    lo = sum(math.comb(n, k) for k in range(0, wins + 1)) / (2 ** n)
    hi = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(lo, hi)))


def bootstrap_ci(values: list[float], seed: int) -> list[float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return [None, None]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(5000, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def row_from_summary(label: str, summary: dict[str, Any],
                     metric_key: str, display: str, expected: str,
                     required: bool, seed: int) -> dict[str, Any]:
    metric = summary.get("seed_level", {}).get(metric_key, {})
    values = []
    seed_rows = summary.get("seed_rows", [])
    if isinstance(seed_rows, list):
        for seed_row in seed_rows:
            value = seed_row.get(metric_key)
            if isinstance(value, (int, float)):
                values.append(float(value))
    mean = metric.get("mean")
    wins = metric.get("wins")
    n = metric.get("n")
    passed = direction_pass(mean, wins, n, expected)
    return {
        "summary": label,
        "metric": metric_key,
        "display": display,
        "mean": mean,
        "ci95_bootstrap": bootstrap_ci(values, seed),
        "wins": wins,
        "n": n,
        "p_sign_two_sided": exact_sign_p(wins, n),
        "expected": expected,
        "required": required,
        "pass": passed,
    }


def readiness_label(required_passes: int, required_total: int) -> str:
    if required_passes == required_total:
        return "extended_predictive_stress_ready"
    if required_passes >= required_total - 4:
        return "extended_predictive_stress_borderline"
    return "extended_predictive_stress_sensitive"


def format_float(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):+.{precision}f}"
    return "n/a"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Summary | Metric | Mean | 95% bootstrap CI | Wins | p(sign) | Required | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        ci = row.get("ci95_bootstrap") or [None, None]
        ci_text = f"[{format_float(ci[0])}, {format_float(ci[1])}]"
        p_sign = row.get("p_sign_two_sided")
        p_text = f"{p_sign:.4f}" if isinstance(p_sign, float) else "n/a"
        lines.append(
            "| {summary} | {metric} | {mean} | {ci} | {wins}/{n} | {p} | {required} | {passed} |".format(
                summary=row["summary"],
                metric=row["display"],
                mean=format_float(row.get("mean")),
                ci=ci_text,
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
    ap.add_argument("--seed", type=int, default=4209)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="predictive_stress_evidence_table")
    args = ap.parse_args()

    set_deterministic(args.seed)
    rows: list[dict[str, Any]] = []
    parsed = [parse_summary_arg(raw) for raw in args.summaries]
    for label, path in parsed:
        summary = json.loads(path.read_text())
        for offset, (key, display, expected, required) in enumerate(METRICS):
            rows.append(
                row_from_summary(
                    label,
                    summary,
                    key,
                    display,
                    expected,
                    required,
                    args.seed + len(rows) + offset,
                )
            )

    required_rows = [r for r in rows if r["required"]]
    required_passes = sum(1 for r in required_rows if r["pass"])
    all_passes = sum(1 for r in rows if r["pass"])
    result = {
        "readiness": readiness_label(required_passes, len(required_rows)),
        "required_pass_count": required_passes,
        "required_total": len(required_rows),
        "all_pass_count": all_passes,
        "all_total": len(rows),
        "rows": rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_predictive_stress_evidence_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    table_path = out.parent / "predictive_stress_evidence_table.md"
    write_markdown(table_path, rows)
    print(f"saved {out}")
    print(f"table={table_path}")
    print(
        f"readiness={result['readiness']} "
        f"required_pass={required_passes}/{len(required_rows)} "
        f"all_pass={all_passes}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
