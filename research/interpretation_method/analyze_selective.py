"""Secondary hierarchy/view controls, specified before v2 evaluation.

This script never fits a predictor or chooses an acceptance threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from run_bedroom import sha, write_json
from selective import configure, summarize


def compare(overview, p, richer, baseline):
    methods = overview["methods"]
    anchors = sorted(methods[richer]["by_anchor"])
    r = np.asarray([methods[richer]["by_anchor"][s]["mae"] for s in anchors])
    b = np.asarray([methods[baseline]["by_anchor"][s]["mae"] for s in anchors])
    risk_delta = np.asarray(
        [
            methods[baseline]["by_anchor"][s]["fixed_coverage_risk"]
            - methods[richer]["by_anchor"][s]["fixed_coverage_risk"]
            for s in anchors
        ]
    )
    rng = np.random.default_rng(p["bootstrap_seed"])
    draws = rng.integers(len(anchors), size=(p["bootstrap_repetitions"], len(anchors)))
    gain = 1 - r[draws].mean(axis=1) / b[draws].mean(axis=1)
    return {
        "richer": richer,
        "baseline": baseline,
        "mae_relative_gain": float(1 - r.mean() / b.mean()),
        "mae_relative_gain_95": np.quantile(gain, [0.025, 0.975]).tolist(),
        "matched_risk_improvement": float(risk_delta.mean()),
        "matched_risk_improvement_95": np.quantile(
            risk_delta[draws].mean(axis=1), [0.025, 0.975]
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    configure()
    p = json.loads((args.run / "manifest.json").read_text())["protocol"]
    rows = [json.loads(s) for s in (args.run / "rows.jsonl").read_text().splitlines()]
    thresholds = json.loads((args.run / "thresholds.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    result = {
        "scope": "secondary information hierarchy and view exclusion; all models/thresholds remain frozen",
        "rows_sha256": sha(args.run / "rows.jsonl"),
        "analysis_source_sha256": sha(__file__),
        "information_increments": [
            compare(summary["overall"], p, r, b)
            for r, b in [
                ("near_ridge", "overlap"),
                ("two_amplitude", "near_ridge"),
                ("two_direction", "two_amplitude"),
            ]
        ],
        "without_view": summarize(
            [r for r in rows if "view" not in (r["a"], r["b"])], p, thresholds
        ),
    }
    write_json(args.run / "secondary_analysis.json", result)
    print(
        json.dumps(
            {
                "information_increments": result["information_increments"],
                "without_view_n": result["without_view"]["n"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
