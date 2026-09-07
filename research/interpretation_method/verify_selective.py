"""Audit every frozen phase and optionally independently regenerate one case."""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

from run_bedroom import ROOT, rms, sha, torch, write_json
from selective import METHODS, calibrate, configure, features, fit, key, predict, report
from run_selective import validate


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, tolerance=1e-12):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and set(a) == set(b)
            and all(close(a[k], b[k], tolerance) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(close(x, y, tolerance) for x, y in zip(a, b))
        )
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and math.isfinite(a)
            and math.isfinite(b)
            and abs(a - b) <= tolerance * max(1, abs(a), abs(b))
        )
    return a == b


def audit(run):
    configure()
    manifest = json.loads((run / "manifest.json").read_text())
    p = manifest["protocol"]
    validate(p)
    complete = json.loads((run / "complete.json").read_text())
    check(complete["status"] == "COMPLETE", "Run not complete")
    for name, digest in complete["artifact_hashes"].items():
        check(sha(run / name) == digest, f"Artifact changed: {name}")
    check(
        sha(manifest["protocol_path"]) == manifest["protocol_sha256"],
        "Protocol changed",
    )
    check(
        p == json.loads(Path(manifest["protocol_path"]).read_text()),
        "Embedded protocol mismatch",
    )
    for name, digest in manifest["source_hashes"].items():
        check(
            sha(ROOT / name) == digest == sha(run / "source_snapshot" / name),
            f"Source changed: {name}",
        )
    for name, digest in manifest["input_hashes"].items():
        check(sha(ROOT / name) == digest, f"Input changed: {name}")
    previous_result = None
    if manifest["mode"] != "preflight":
        previous = Path(manifest["previous_path"])
        _, previous_result = audit(previous)
        previous_manifest = json.loads((previous / "manifest.json").read_text())
        for field in ("source_hashes", "input_hashes", "protocol_sha256"):
            check(
                manifest[field] == previous_manifest[field],
                f"Prerequisite mismatch: {field}",
            )
        for name in ("model", "thresholds"):
            if name + "_sha256" in manifest:
                check(
                    sha(run / (name + ".json"))
                    == sha(previous / (name + ".json"))
                    == manifest[name + "_sha256"],
                    f"Frozen {name} changed",
                )
    else:
        check(
            json.loads((run / "preflight.json").read_text())["status"] == "PASS",
            "Preflight failed",
        )
        return manifest, {"status": "PASS", "phase": "preflight"}

    phase = {
        "train": "training",
        "calibrate": "calibration",
        "evaluate": "evaluation",
        "confirm": "confirmation",
    }[manifest["mode"]]
    seeds = p[phase + "_seeds"]
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    pairs = list(itertools.combinations(p["attributes"], 2))
    expected = {
        (seed, a, b, s, sa, sb)
        for seed in seeds
        for a, b in pairs
        for s in p["target_scales"]
        for sa, sb in itertools.product(p["signs"], repeat=2)
    }
    check(
        len(rows) == len(expected) and {key(r) for r in rows} == expected,
        "Coverage/duplicate mismatch",
    )
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    index = {key(r): r for r in rows}
    model = json.loads((run / "model.json").read_text())
    for seed in seeds:
        e = [v for v in events if v["seed"] == seed]
        first = (
            "training_features_locked" if phase == "training" else "predictions_locked"
        )
        check(
            [v["event"] for v in e]
            == [first, "target_joint_observation_begins", "anchor_complete"],
            "Prediction order mismatch",
        )
        check(
            e[0]["elapsed_seconds"]
            <= e[1]["elapsed_seconds"]
            < e[2]["elapsed_seconds"],
            "Invalid event times",
        )
        check(e[0]["forward_calls"] == e[1]["forward_calls"], "Joint call before lock")
        check(
            e[2]["forward_calls"] - e[1]["forward_calls"]
            == p["expected_rows_per_anchor"],
            "Target calls mismatch",
        )
        path = run / "predictions" / f"{seed}.json"
        check(sha(path) == e[0]["sha256"], "Locked prediction file changed")
        predictions = json.loads(path.read_text())
        check(
            len(predictions) == p["expected_rows_per_anchor"],
            "Prediction count mismatch",
        )
        for pred in predictions:
            row = index[key(pred)]
            check(
                all(row[k] == v for k, v in pred.items()),
                "Scored result changed prediction",
            )
            check(
                math.isfinite(row["actual_relative"]) and row["actual_relative"] >= 0,
                "Invalid target risk",
            )
            if phase != "training":
                check(
                    set(pred["scores"]) == set(METHODS)
                    and close(predict(pred, model, p), pred["scores"]),
                    "Prediction does not reproduce",
                )
        saved = torch.load(run / "signed_examples" / f"{seed}.pt", map_location="cpu")
        record = index[key(saved)]
        measured_features = features(
            saved["response_a"],
            saved["response_b"],
            saved["small_c"],
            saved["near_c"],
            saved["scale"],
        )
        check(
            all(close(v, record[k]) for k, v in measured_features.items()),
            "Saved features do not reproduce",
        )
        actual = float(rms(saved["interaction"])) / measured_features["denominator"]
        check(close(actual, record["actual_relative"]), "Saved target risk mismatch")
    latents = torch.load(run / "latents.pt", map_location="cpu")
    check(set(latents) == set(seeds), "Latent coverage mismatch")
    measured = json.loads((run / "measurement.json").read_text())
    check(
        measured["forward_calls"] == len(seeds) * p["expected_calls_per_anchor"],
        "Total calls mismatch",
    )
    if phase == "training":
        check(close(fit(rows, p), model), "Training fit mismatch")
    elif phase == "calibration":
        check(
            calibrate(rows, p) == json.loads((run / "thresholds.json").read_text()),
            "Independent calibration does not reproduce",
        )
    else:
        thresholds = json.loads((run / "thresholds.json").read_text())
        check(
            close(
                report(rows, p, thresholds),
                json.loads((run / "summary.json").read_text()),
            ),
            "Summary/bootstrap mismatch",
        )
    return manifest, {
        "status": "PASS",
        "phase": phase,
        "rows": len(rows),
        "independent_anchors": len(seeds),
        "sources": len(manifest["source_hashes"]),
        "inputs": len(manifest["input_hashes"]),
        "prediction_locks": len(seeds),
        "saved_tensor_cases": len(seeds),
        "previous": previous_result,
    }


def rerender(run, manifest, seed):
    configure(gpu=True)
    p = manifest["protocol"]
    sys.path.insert(0, str(ROOT / "higan_dev"))
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary, manipulate_wp

    generator = HiGANGenerator(ROOT / p["higan_repo"], device="cuda")
    saved = torch.load(run / "signed_examples" / f"{seed}.pt", map_location="cpu")
    wp = torch.load(run / "latents.pt", map_location="cpu")[seed].cuda()
    directory = ROOT / p["higan_repo"] / "boundaries/stylegan_bedroom"
    a, b = [load_boundary(directory, saved[k]) for k in ("a", "b")]
    synth = generator._net.synthesis

    def original(w):
        with torch.no_grad():
            return type(synth).forward(synth, w).double().cpu()

    y0 = original(wp)

    def response(scale):
        wa, wb = [manipulate_wp(wp, bound, [scale])[:, 0] for bound in (a, b)]
        wab = manipulate_wp(wa, b, [scale])[:, 0]
        ya, yb, yab = [original(w) for w in (wa, wb, wab)]
        return ya - y0, yb - y0, yab - ya - yb + y0

    ra, rb, rc = response(saved["scale"])
    c1, c2 = [response(s)[2] for s in p["probe_scales"]]
    errors = {
        name: float((value - saved[name]).abs().max())
        for name, value in (
            ("base", y0),
            ("response_a", ra),
            ("response_b", rb),
            ("interaction", rc),
            ("small_c", c1),
            ("near_c", c2),
        )
    }
    check(max(errors.values()) <= 1e-6, "Independent original-generator mismatch")
    return {
        "status": "PASS",
        "seed": seed,
        "scale": saved["scale"],
        "original_forward_calls": 10,
        "max_abs_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--rerender-seed", type=int)
    args = parser.parse_args()
    manifest, result = audit(args.run)
    if args.rerender_seed is not None:
        result["independent_rerender"] = rerender(
            args.run, manifest, args.rerender_seed
        )
    write_json(
        args.run
        / (
            "verification_rerender.json"
            if args.rerender_seed is not None
            else "verification.json"
        ),
        result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
