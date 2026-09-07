"""Apply the frozen v2 practical-effect boundaries without tuning any model."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_bedroom import sha, write_json


def decision(summary, p, final_confirmation=False):
    overall = summary["overall"]
    reasons = []
    for baseline in ("near_ridge", "two_amplitude"):
        result = overall["two_direction_comparisons"][baseline]
        lo, hi = result["mae_relative_gain_95"]
        if lo is not None and lo < p["minimum_mae_relative_improvement"] <= hi:
            reasons.append(
                f"MAE practical-effect boundary unresolved against {baseline}"
            )
        lo, hi = result["matched_risk_improvement_95"]
        if lo < p["minimum_matched_risk_improvement"] <= hi:
            reasons.append(
                f"Matched-risk practical-effect boundary unresolved against {baseline}"
            )
    empirical = {}
    for method in ("near_ridge", "two_amplitude", "two_direction"):
        r = overall["methods"][method]
        a = np.asarray([v["accepted"] for v in r["by_anchor"].values()])
        w = np.asarray([v["wrong"] for v in r["by_anchor"].values()])
        rng = np.random.default_rng(p["bootstrap_seed"])
        draws = rng.integers(len(a), size=(p["bootstrap_repetitions"], len(a)))
        counts = a[draws].sum(axis=1)
        valid = counts > 0
        interval = (
            np.quantile(
                w[draws].sum(axis=1)[valid] / counts[valid], [0.025, 0.975]
            ).tolist()
            if valid.any()
            else [None, None]
        )
        empirical[method] = {
            "coverage": r["coverage"],
            "false_adoption": r["false_adoption"],
            "risk_95": interval,
            "practical_gate": r["empirical_practical_gate"],
        }
        if r["coverage"] >= p["minimum_coverage"] and interval[0] is not None:
            if interval[0] <= p["evaluation_false_adoption_target"] <= interval[1]:
                reasons.append(f"Practical risk boundary unresolved for {method}")
    return {
        "scope": "predefined finite experiment decision; no universal accuracy or novelty claim",
        "confirmation_required": bool(reasons) and not final_confirmation,
        "confirmation_limit_reached": bool(reasons) and final_confirmation,
        "unresolved_boundaries": reasons,
        "empirical_practical_results": empirical,
        "interpretation": "Use independent confirmation if a specified boundary is unresolved; never tune on evaluation. Residual uncertainty is reported, not converted to success.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    result = decision(summary, manifest["protocol"], manifest["mode"] == "confirm")
    result.update(
        summary_sha256=sha(args.run / "summary.json"),
        decision_source_sha256=sha(__file__),
    )
    write_json(args.run / "decision.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
