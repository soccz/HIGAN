"""Predeclared readiness checks for predictive-validity assumption stress tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def get_metric(summary: dict[str, Any], key: str) -> dict[str, Any]:
    out = summary.get("seed_level", {}).get(key, {})
    return out if isinstance(out, dict) else {}


def passes_direction(metric: dict[str, Any], expected: str,
                     *, min_wins: int = 4, min_n: int = 5) -> bool:
    if metric.get("n", 0) < min_n:
        return False
    mean = metric.get("mean")
    wins = metric.get("wins")
    if not isinstance(mean, (int, float)) or not isinstance(wins, int):
        return False
    if expected == "positive":
        return mean > 0 and wins >= min_wins
    if expected == "negative":
        return mean < 0 and wins >= min_wins
    raise ValueError(f"unknown expected direction: {expected}")


def row(name: str, metric: dict[str, Any], expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "mean": metric.get("mean"),
        "wins": metric.get("wins"),
        "n": metric.get("n"),
        "expected": expected,
        "min_wins": 4,
        "pass": passes_direction(metric, expected),
    }


def predictive_checks(label: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row(f"{label}: rho_vs_lpips_spearman",
            get_metric(summary, "rho_vs_lpips_spearman"), "positive"),
        row(f"{label}: lpips_beta_rho",
            get_metric(summary, "lpips_beta_rho"), "positive"),
        row(f"{label}: matched_pair_lpips_low_minus_high",
            get_metric(summary, "matched_pair_lpips_low_minus_high"), "negative"),
        row(f"{label}: matched_pair_id_low_minus_high",
            get_metric(summary, "matched_pair_id_low_minus_high"), "positive"),
    ]


def readiness_label(checks: list[dict[str, Any]]) -> str:
    passes = sum(1 for r in checks if r["pass"])
    if passes == len(checks):
        return "assumption_stress_predictive_ready"
    if passes >= len(checks) - 2:
        return "assumption_stress_predictive_borderline"
    return "assumption_sensitive_predictive_claim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-church", required=True)
    ap.add_argument("--fd-bedroom", required=True)
    ap.add_argument("--prompt-church", required=True)
    ap.add_argument("--prompt-bedroom", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4191)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="predictive_assumption_readiness")
    args = ap.parse_args()

    set_deterministic(args.seed)
    checks: list[dict[str, Any]] = []
    checks.extend(predictive_checks("fd_church", load(args.fd_church)))
    checks.extend(predictive_checks("fd_bedroom", load(args.fd_bedroom)))
    checks.extend(predictive_checks("prompt_church", load(args.prompt_church)))
    checks.extend(predictive_checks("prompt_bedroom", load(args.prompt_bedroom)))
    result = {
        "readiness": readiness_label(checks),
        "passes": sum(1 for r in checks if r["pass"]),
        "total": len(checks),
        "checks": checks,
        "claim_guidance": {
            "assumption_stress_predictive_ready": (
                "Predictive-validity claim survived finite-difference risk "
                "estimation and prompt-template stress tests."
            ),
            "assumption_stress_predictive_borderline": (
                "Predictive-validity claim is mostly robust, but paper must "
                "report sensitivity boundaries."
            ),
            "assumption_sensitive_predictive_claim": (
                "Predictive-validity claim depends materially on estimator or "
                "prompt assumptions; narrow the claim."
            ),
        },
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_predictive_assumption_readiness.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    write_json_atomic(out, result)
    print(f"saved {out}")
    print(f"readiness={result['readiness']} passes={result['passes']}/{result['total']}")


if __name__ == "__main__":
    main()
