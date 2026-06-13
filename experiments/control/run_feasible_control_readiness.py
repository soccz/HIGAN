"""Decision rules for the semantic-feasible minimum-risk controller.

The earlier controller experiments showed that an unconstrained "beat every
baseline" claim is too broad.  This reducer tests a narrower control claim:
after candidates are filtered for semantic feasibility by probe gain, curvature
risk should be useful for choosing lower-damage edits.  The low-risk baseline is
reported as a boundary/ablation, not as a required competitor to beat.
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


def row(name: str, metric: dict[str, Any], expected: str,
        *, min_wins: int = 4, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "mean": metric.get("mean"),
        "wins": metric.get("wins"),
        "n": metric.get("n"),
        "expected": expected,
        "min_wins": min_wins,
        "required": required,
        "pass": passes_direction(metric, expected, min_wins=min_wins),
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


def feasible_controller_checks(label: str, actual_summary: dict[str, Any],
                               negative_summary: dict[str, Any]
                               ) -> list[dict[str, Any]]:
    checks = []
    for comp in [
        "risk_minus_gain_only",
        "risk_minus_random",
        "risk_minus_high_risk",
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


def boundary_checks(label: str, actual_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row(
            f"{label}: risk_minus_low_risk ID boundary",
            get_comp_metric(actual_summary, "risk_minus_low_risk", "mean_id_cos"),
            "positive",
            required=False,
        ),
        row(
            f"{label}: risk_minus_low_risk true-LPIPS boundary",
            get_comp_metric(actual_summary, "risk_minus_low_risk", "mean_lpips_true"),
            "negative",
            required=False,
        ),
        row(
            f"{label}: risk_minus_low_risk probe-gain boundary",
            get_comp_metric(actual_summary, "risk_minus_low_risk", "mean_probe_gain"),
            "positive",
            required=False,
        ),
        row(
            f"{label}: risk_minus_low_risk alpha-efficiency boundary",
            get_comp_metric(actual_summary, "risk_minus_low_risk", "mean_abs_alpha"),
            "negative",
            required=False,
        ),
    ]


def readiness_label(predictive: list[dict[str, Any]],
                    controller: list[dict[str, Any]]) -> str:
    predictive_pass = sum(1 for r in predictive if r["pass"])
    controller_pass = sum(1 for r in controller if r["pass"])
    if predictive_pass == len(predictive) and controller_pass == len(controller):
        return "strong_main_feasible_control_ready"
    if predictive_pass >= len(predictive) - 2 and controller_pass >= len(controller) - 4:
        return "borderline_feasible_control_claim"
    if predictive_pass >= len(predictive) - 2:
        return "predictive_claim_with_feasible_control_boundary"
    return "not_main_ready_without_claim_narrowing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--church-predictive", required=True)
    ap.add_argument("--bedroom-predictive", required=True)
    ap.add_argument("--church-actual", required=True)
    ap.add_argument("--church-negative", required=True)
    ap.add_argument("--bedroom-actual", required=True)
    ap.add_argument("--bedroom-negative", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4121)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="feasible_control_readiness")
    args = ap.parse_args()

    set_deterministic(args.seed)
    church_pred = load(args.church_predictive)
    bedroom_pred = load(args.bedroom_predictive)
    church_actual = load(args.church_actual)
    church_negative = load(args.church_negative)
    bedroom_actual = load(args.bedroom_actual)
    bedroom_negative = load(args.bedroom_negative)

    predictive: list[dict[str, Any]] = []
    predictive.extend(predictive_checks("church", church_pred))
    predictive.extend(predictive_checks("bedroom", bedroom_pred))

    controller: list[dict[str, Any]] = []
    controller.extend(feasible_controller_checks(
        "church_feasible_low_risk", church_actual, church_negative))
    controller.extend(feasible_controller_checks(
        "bedroom_feasible_low_risk", bedroom_actual, bedroom_negative))

    boundaries: list[dict[str, Any]] = []
    boundaries.extend(boundary_checks("church_feasible_low_risk", church_actual))
    boundaries.extend(boundary_checks("bedroom_feasible_low_risk", bedroom_actual))

    result = {
        "readiness": readiness_label(predictive, controller),
        "predictive_passes": sum(1 for r in predictive if r["pass"]),
        "predictive_total": len(predictive),
        "controller_passes": sum(1 for r in controller if r["pass"]),
        "controller_total": len(controller),
        "boundary_passes": sum(1 for r in boundaries if r["pass"]),
        "boundary_total": len(boundaries),
        "predictive_checks": predictive,
        "controller_checks": controller,
        "boundary_checks": boundaries,
        "claim_guidance": {
            "strong_main_feasible_control_ready": (
                "Claim curvature risk is predictive and actionable when used "
                "as a semantic-feasibility-constrained edit controller."
            ),
            "borderline_feasible_control_claim": (
                "Claim feasible control, but foreground domain/baseline "
                "failure boundaries."
            ),
            "predictive_claim_with_feasible_control_boundary": (
                "Make predictive validity the main claim and present feasible "
                "control as a conditional application."
            ),
            "not_main_ready_without_claim_narrowing": (
                "Narrow to predictive/failure analysis; do not claim a "
                "general controller."
            ),
        },
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_feasible_control_readiness.py",
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
        f"controller={result['controller_passes']}/{result['controller_total']} "
        f"boundary={result['boundary_passes']}/{result['boundary_total']}"
    )


if __name__ == "__main__":
    main()
