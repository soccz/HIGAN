"""Post-evaluation diagnostic, never a predictor or a confirmatory success gate.

Recover the probe/target inner product from three stored squared norms. The
oracle projection uses the observed target and is only a representational limit.
"""

import argparse
import json
from pathlib import Path
import statistics

from run_bedroom import sha, write_json


def projection_diagnostic(row):
    actual = row["actual_relative"]
    probe = row["risk_predictions"]["probe_constant"]
    error = row["relative_errors"]["probe_constant"]
    if min(actual, probe) <= 0:
        return {"cosine": None, "positive_projection_error": actual}
    dot = (actual**2 + probe**2 - error**2) / 2
    cosine = dot / (actual * probe)
    if abs(cosine) > 1 + 1e-9:
        raise ValueError("Stored norms violate the inner-product bound")
    cosine = max(-1.0, min(1.0, cosine))
    residual = actual * (1 - max(0.0, cosine) ** 2) ** 0.5
    if residual > row["relative_errors"]["probe_calibrated"] + 1e-10:
        raise ValueError("Oracle projection worse than a permitted fixed coefficient")
    return {"cosine": cosine, "positive_projection_error": residual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows_path = args.run / "rows.jsonl"
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()]
    p = json.loads((args.run / "manifest.json").read_text())["protocol"]

    def group(group_rows):
        diagnostic = [projection_diagnostic(row) for row in group_rows]
        return {
            "n": len(group_rows),
            "median_probe_target_cosine": statistics.median(
                d["cosine"] for d in diagnostic if d["cosine"] is not None
            ),
            "mean_oracle_positive_projection_error": statistics.mean(
                d["positive_projection_error"] for d in diagnostic
            ),
            "mean_frozen_calibrated_error": statistics.mean(
                r["relative_errors"]["probe_calibrated"] for r in group_rows
            ),
        }

    result = {
        "scope": "post-evaluation observed-response diagnosis, not a prediction; oracle uses target labels; not part of preregistered success gate",
        "rows_sha256": sha(rows_path),
        "analysis_source_sha256": sha(__file__),
        "overall": group(rows),
        "by_scale": {
            str(s): group([r for r in rows if r["scale"] == s])
            for s in p["target_scales"]
        },
    }
    write_json(args.run / "projection_diagnostic.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
