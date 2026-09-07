"""Apply the validated nearest-probe rule without observing target joints.

Scope: frozen bedroom model, eight existing directions, and three locked scales.
The output is an empirical adopt/abstain decision, not a per-case certificate.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from run_bedroom import ROOT, load_generator, render, rms, sample, sha, write_json
from measure import components
from prediction import single_features
from selective import configure


def score_near(a, b, near_c, scale, model):
    denominator, singles = single_features(a, b, scale)
    vector = np.asarray(
        singles + [math.log(max(float(rms(near_c)) / denominator, 1e-12))]
    )
    fitted = model["by_scale"][str(scale)]["near_ridge"]
    x = (vector - fitted["mean"]) / fitted["std"]
    value = float(np.dot(np.r_[1.0, x], fitted["weights"]))
    return math.exp(max(-27.0, min(5.0, value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).parent / "runs/selective_v2_calibration",
    )
    args = parser.parse_args()
    configure(gpu=True)
    manifest = json.loads((args.calibration / "manifest.json").read_text())
    complete = json.loads((args.calibration / "complete.json").read_text())
    p = manifest["protocol"]
    if manifest["mode"] != "calibrate" or complete["status"] != "COMPLETE":
        raise ValueError("A frozen calibration bundle is required")
    for name in ("manifest.json", "model.json", "thresholds.json"):
        if sha(args.calibration / name) != complete["artifact_hashes"][name]:
            raise ValueError(f"Calibration artifact changed: {name}")
    for name, digest in {
        **manifest["source_hashes"],
        **manifest["input_hashes"],
    }.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Validated source/input changed: {name}")
    model = json.loads((args.calibration / "model.json").read_text())
    threshold = json.loads((args.calibration / "thresholds.json").read_text())[
        "methods"
    ]["near_ridge"]["threshold"]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "manifest.json",
        {
            "scope": "Nearest-probe empirical additive-explanation selection; target singles known; no target joint output queried",
            "seed": args.seed,
            "protocol": p,
            "threshold": threshold,
            "calibration_path": str(args.calibration.resolve()),
            "calibration_manifest_sha256": sha(args.calibration / "manifest.json"),
            "model_sha256": sha(args.calibration / "model.json"),
            "thresholds_sha256": sha(args.calibration / "thresholds.json"),
            "application_source_sha256": sha(__file__),
        },
    )
    generator, directions, _ = load_generator(p)
    calls = 0
    started = time.monotonic()

    def forward(w):
        nonlocal calls
        calls += 1
        return render(generator, w).cpu()

    wp = sample(generator, args.seed)
    base = forward(wp)
    probe_scale = p["probe_scales"][1]
    scales = [probe_scale, *p["target_scales"]]
    singles = {
        (a, scale, sign): forward(wp + sign * scale * direction)
        for a, direction in directions.items()
        for scale in scales
        for sign in p["signs"]
    }
    rows = []
    for a, b in itertools.combinations(p["attributes"], 2):
        for sa, sb in itertools.product(p["signs"], repeat=2):
            joint = forward(
                wp + sa * probe_scale * directions[a] + sb * probe_scale * directions[b]
            )
            near_c = components(
                base, singles[a, probe_scale, sa], singles[b, probe_scale, sb], joint
            )[3]
            for scale in p["target_scales"]:
                risk = score_near(
                    singles[a, scale, sa].double() - base.double(),
                    singles[b, scale, sb].double() - base.double(),
                    near_c,
                    scale,
                    model,
                )
                rows.append(
                    {
                        "seed": args.seed,
                        "a": a,
                        "b": b,
                        "sign_a": sa,
                        "sign_b": sb,
                        "scale": scale,
                        "predicted_relative_interaction": risk,
                        "decision": (
                            "adopt_additive" if risk <= threshold else "abstain"
                        ),
                    }
                )
    n = len(p["attributes"])
    expected = 1 + n * len(scales) * 2 + n * (n - 1) // 2 * 4
    if calls != expected or len(rows) != p["expected_rows_per_anchor"]:
        raise RuntimeError("Unexpected prediction grid or call count")
    path = args.out / "predictions.jsonl"
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    result = {
        "status": "COMPLETE",
        "predictions": len(rows),
        "forward_calls": calls,
        "target_joint_calls": 0,
        "probe_scales_used": [probe_scale],
        "adopted": sum(r["decision"] == "adopt_additive" for r in rows),
        "elapsed_seconds": time.monotonic() - started,
        "predictions_sha256": sha(path),
    }
    write_json(args.out / "complete.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
