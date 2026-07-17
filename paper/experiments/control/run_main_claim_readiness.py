"""Summarize whether the locked control campaign supports a main-paper claim.

This script does not tune, reweight, or rerun experiments.  It reads locked
summary outputs and applies predeclared decision rules so the final paper claim
is determined by evidence strength rather than preference.
"""
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


def get_comp_metric(summary: dict[str, Any], comp: str,
                    metric: str) -> dict[str, Any]:
    out = summary.get("seed_level", {}).get(comp, {}).get(metric, {})
    return out if isinstance(out, dict) else {}


def get_negative_metric(summary: dict[str, Any], comp: str,
                        metric: str) -> dict[str, Any]:
    out = (
        summary.get("paired_controller_diffs", {})
        .get("seed_level", {})
        .get(comp, {})
        .get(metric, {})
    )
    return out if isinstance(out, dict) else {}


def passes_direction(metric: dict[str, Any],
                     expected: str,
                     *,
                     min_wins: int = 4,
                     min_n: int = 5) -> bool:
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


def row(name: str, metric: dict[str, Any], expected: str,
        *, min_wins: int = 4) -> dict[str, Any]:
    ok = passes_direction(metric, expected, min_wins=min_wins)
    return {
        "name": name,
        "mean": metric.get("mean"),
        "wins": metric.get("wins"),
        "n": metric.get("n"),
        "expected": expected,
        "min_wins": min_wins,
        "pass": ok,
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


def controller_checks(label: str, actual_summary: dict[str, Any],
                      negative_summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for comp in [
        "risk_minus_gain_only",
        "risk_minus_random",
        "risk_minus_high_risk",
        "risk_minus_low_risk",
    ]:
        checks.append(row(
            f"{label}: {comp} ID",
            get_comp_metric(actual_summary, comp, "mean_id_cos"),
            "positive",
        ))
        checks.append(row(
            f"{label}: {comp} true-LPIPS",
            get_comp_metric(actual_summary, comp, "mean_lpips_true"),
            "negative",
        ))
    for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
        checks.append(row(
            f"{label}: {comp} ID",
            get_negative_metric(negative_summary, comp, "mean_id_cos"),
            "positive",
        ))
        checks.append(row(
            f"{label}: {comp} true-LPIPS",
            get_negative_metric(negative_summary, comp, "mean_lpips_true"),
            "negative",
        ))
    return checks


def readiness_label(predictive: list[dict[str, Any]],
                    controller: list[dict[str, Any]]) -> str:
    predictive_pass = sum(1 for r in predictive if r["pass"])
    controller_pass = sum(1 for r in controller if r["pass"])
    if predictive_pass == len(predictive) and controller_pass == len(controller):
        return "strong_main_control_claim_ready"
    if predictive_pass >= len(predictive) - 2 and controller_pass >= len(controller) - 3:
        return "borderline_main_control_claim"
    if predictive_pass >= len(predictive) - 2:
        return "main_predictive_claim_only"
    return "not_main_ready_without_claim_narrowing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--church-predictive", required=True)
    ap.add_argument("--bedroom-predictive", required=True)
    ap.add_argument("--tiebreak-actual", required=True)
    ap.add_argument("--tiebreak-negative", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="main_claim_readiness")
    args = ap.parse_args()

    set_deterministic(args.seed)
    church_pred = load(args.church_predictive)
    bedroom_pred = load(args.bedroom_predictive)
    tiebreak_actual = load(args.tiebreak_actual)
    tiebreak_negative = load(args.tiebreak_negative)

    predictive = []
    predictive.extend(predictive_checks("church", church_pred))
    predictive.extend(predictive_checks("bedroom", bedroom_pred))
    controller = controller_checks(
        "church_gain_first_risk_tiebreak",
        tiebreak_actual,
        tiebreak_negative,
    )
    result = {
        "readiness": readiness_label(predictive, controller),
        "predictive_passes": sum(1 for r in predictive if r["pass"]),
        "predictive_total": len(predictive),
        "controller_passes": sum(1 for r in controller if r["pass"]),
        "controller_total": len(controller),
        "predictive_checks": predictive,
        "controller_checks": controller,
        "claim_guidance": {
            "strong_main_control_claim_ready": (
                "Claim risk is both predictive and actionable as a controller."
            ),
            "borderline_main_control_claim": (
                "Claim risk as actionable but emphasize failure boundaries."
            ),
            "main_predictive_claim_only": (
                "Make predictive-validity the main claim; controller becomes "
                "a conditional application."
            ),
            "not_main_ready_without_claim_narrowing": (
                "Narrow to failure characterization or run a different method; "
                "do not claim a general controller."
            ),
        },
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_main_claim_readiness.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    write_json_atomic(out, result)
    print(f"saved {out}")
    print(
        f"readiness={result['readiness']} "
        f"predictive={result['predictive_passes']}/{result['predictive_total']} "
        f"controller={result['controller_passes']}/{result['controller_total']}"
    )


if __name__ == "__main__":
    main()
