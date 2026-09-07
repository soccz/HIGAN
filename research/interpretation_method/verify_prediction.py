"""Audit the frozen prediction experiment, with optional independent GPU rerender."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

from run_bedroom import ROOT, sha, torch, write_json
from prediction import (
    RISK_METHODS,
    VECTOR_METHODS,
    evaluate_response,
    fit_model,
    risk_predictions,
    single_features,
    summarize_rows,
    vector_predictions,
)
from run_prediction import configure, mixed_jvp, validate_protocol


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def row_key(row):
    return tuple(row[k] for k in ("seed", "a", "b", "scale", "sign_a", "sign_b"))


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    p = manifest["protocol"]
    validate_protocol(p)
    check(manifest["mode"] == "evaluate", "An evaluation run is required")
    for name, digest in manifest["source_hashes"].items():
        check(
            sha(run / "source_snapshot" / name) == digest == sha(ROOT / name),
            f"Source changed: {name}",
        )
    for name, digest in manifest["input_hashes"].items():
        check(sha(ROOT / name) == digest, f"Input changed: {name}")
    protocol_path = Path(__file__).with_name("protocol_prediction_v1.json")
    check(sha(protocol_path) == manifest["protocol_sha256"], "Protocol changed")
    training = Path(manifest["training_path"])
    check(
        sha(training / "training_rows.jsonl") == manifest["training_rows_sha256"],
        "Training rows changed",
    )
    check(
        sha(training / "model.json")
        == sha(run / "model.json")
        == manifest["model_sha256"],
        "Fitted model changed",
    )
    train_manifest = json.loads((training / "manifest.json").read_text())
    for k in ("protocol_sha256", "source_hashes", "input_hashes"):
        check(train_manifest[k] == manifest[k], f"Training/evaluation mismatch: {k}")
    train_rows = [
        json.loads(s)
        for s in (training / "training_rows.jsonl").read_text().splitlines()
    ]
    model = json.loads((run / "model.json").read_text())
    check(fit_model(train_rows, p) == model, "Development-only fit does not reproduce")
    pairs = list(itertools.combinations(p["attributes"], 2))

    def grid(seeds):
        return {
            (seed, a, b, s, sa, sb)
            for seed in seeds
            for a, b in pairs
            for s in p["target_scales"]
            for sa, sb in itertools.product(p["signs"], repeat=2)
        }

    check(
        len(train_rows) == p["expected_training_rows"]
        and {row_key(r) for r in train_rows} == grid(p["development_seeds"]),
        "Training coverage mismatch",
    )
    rows = [json.loads(s) for s in (run / "rows.jsonl").read_text().splitlines()]
    check(
        len(rows) == p["expected_evaluation_rows"]
        and {row_key(r) for r in rows} == grid(p["evaluation_seeds"]),
        "Evaluation coverage mismatch",
    )
    check(finite(rows) and finite(train_rows), "Nonfinite metrics")
    events = [json.loads(s) for s in (run / "events.jsonl").read_text().splitlines()]
    predictions = {}
    for seed in p["evaluation_seeds"]:
        group = [e for e in events if e["seed"] == seed]
        check(
            [e["event"] for e in group]
            == [
                "predictions_locked",
                "target_joint_observation_begins",
                "anchor_complete",
            ],
            "Prediction/observation order mismatch",
        )
        check(
            group[0]["elapsed_seconds"]
            <= group[1]["elapsed_seconds"]
            < group[2]["elapsed_seconds"],
            "Event times out of order",
        )
        check(
            group[0]["forward_calls"] == group[1]["forward_calls"],
            "Target calls before lock",
        )
        check(
            group[2]["forward_calls"] - group[1]["forward_calls"]
            == len(pairs) * len(p["target_scales"]) * 4,
            "Target-call count mismatch",
        )
        path = run / "predictions" / f"{seed}.json"
        check(sha(path) == group[0]["sha256"], "Locked predictions changed")
        saved = json.loads(path.read_text())
        check(len(saved) == group[0]["rows"], "Prediction count mismatch")
        predictions.update({row_key(r): r for r in saved})
    check(
        set(predictions) == grid(p["evaluation_seeds"]), "Prediction coverage mismatch"
    )
    for row in rows:
        prediction = predictions[row_key(row)]
        check(
            all(row[k] == v for k, v in prediction.items()),
            "Result differs from saved prediction",
        )
        threshold = (
            p["relative_tolerance"] + p["absolute_tolerance"] / row["denominator"]
        )
        check(
            row["actual_additive"] == (row["actual_relative"] <= threshold),
            "Truth label mismatch",
        )
        check(
            set(row["relative_errors"]) == set(VECTOR_METHODS), "Missing vector method"
        )
        check(set(row["adopt"]) == set(RISK_METHODS), "Missing risk method")
        for name in RISK_METHODS:
            risk = prediction["risk_predictions"][name]
            check(row["adopt"][name] == (risk <= threshold), "Decision mismatch")
            check(
                row["risk_absolute_errors"][name] == abs(risk - row["actual_relative"]),
                "Risk error mismatch",
            )
        check(
            row["relative_errors"]["additive"] == row["actual_relative"],
            "Additive error mismatch",
        )
    summary = json.loads((run / "summary.json").read_text())
    check(summarize_rows(rows, p) == summary, "Summary/bootstrap does not reproduce")
    progress = json.loads((run / "progress.json").read_text())
    check(
        progress["status"] == "COMPLETE"
        and progress["forward_calls"] == p["expected_evaluation_forward_calls"]
        and progress["mixed_jvps"] == p["expected_evaluation_mixed_jvps"],
        "Incomplete run",
    )
    latents = torch.load(run / "latents.pt", map_location="cpu")
    check(set(latents) == set(p["evaluation_seeds"]), "Latent coverage mismatch")
    max_error = 0.0
    row_index = {row_key(r): r for r in rows}
    for seed in p["evaluation_seeds"]:
        saved = torch.load(run / "signed_examples" / f"{seed}.pt", map_location="cpu")
        denom, features = single_features(
            saved["response_a"], saved["response_b"], saved["scale"]
        )
        fitted = model["by_scale"][str(saved["scale"])]
        vectors = vector_predictions(
            saved["probe_c"],
            saved["mixed"],
            saved["scale"],
            p["probe_scale"],
            (saved["sign_a"], saved["sign_b"]),
            fitted["coefficient"],
        )
        risks = risk_predictions(vectors, denom, features, fitted)
        measured = evaluate_response(
            saved["interaction"],
            vectors,
            risks,
            denom,
            p["absolute_tolerance"],
            p["relative_tolerance"],
        )
        row = row_index[row_key(saved)]
        for method in VECTOR_METHODS:
            max_error = max(
                max_error,
                abs(
                    measured["relative_errors"][method] - row["relative_errors"][method]
                ),
            )
        check(risks == row["risk_predictions"], "Saved tensor prediction mismatch")
    check(max_error < 1e-12, "Saved tensor error mismatch")
    return manifest, {
        "status": "PASS",
        "rows": len(rows),
        "independent_anchors": len(p["evaluation_seeds"]),
        "source_files_verified": len(manifest["source_hashes"]),
        "input_files_verified": len(manifest["input_hashes"]),
        "prediction_locks_verified": len(p["evaluation_seeds"]),
        "saved_tensor_error_max": max_error,
        "frozen_fit_and_cluster_bootstrap_reproduced": True,
    }


def rerender(run, manifest, seed):
    configure()
    p = manifest["protocol"]
    sys.path.insert(0, str(ROOT / "higan_dev"))
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary, manipulate_wp

    generator = HiGANGenerator(ROOT / p["higan_repo"], device="cuda")
    saved = torch.load(run / "signed_examples" / f"{seed}.pt", map_location="cpu")
    wp = torch.load(run / "latents.pt", map_location="cpu")[seed].cuda()
    boundaries = ROOT / p["higan_repo"] / "boundaries/stylegan_bedroom"
    a, b = (load_boundary(boundaries, saved[name]) for name in ("a", "b"))
    synth = generator._net.synthesis

    def original(w):
        with torch.no_grad():
            return type(synth).forward(synth, w).double().cpu()

    y0 = original(wp)

    def responses(scale):
        wa, wb = (manipulate_wp(wp, bound, [scale])[:, 0] for bound in (a, b))
        wab = manipulate_wp(wa, b, [scale])[:, 0]
        ya, yb, yab = original(wa), original(wb), original(wab)
        return ya - y0, yb - y0, yab - ya - yb + y0

    ra, rb, rc = responses(saved["scale"])
    pc = responses(p["probe_scale"])[2]
    # Build derivative directions from boundary metadata independently of the runner.
    da, db = torch.zeros_like(wp).cpu(), torch.zeros_like(wp).cpu()
    for bound, direction in ((a, da), (b, db)):
        for layer in bound.manipulate_layers:
            direction[0, layer] = bound.direction
    mixed = mixed_jvp(generator, wp, da.cuda(), db.cuda())
    errors = {
        name: float((value - saved[name]).abs().max())
        for name, value in (
            ("base", y0),
            ("response_a", ra),
            ("response_b", rb),
            ("interaction", rc),
            ("probe_c", pc),
            ("mixed", mixed),
        )
    }
    check(max(errors.values()) <= 1e-6, "Independent rerender mismatch")
    return {
        "status": "PASS",
        "seed": seed,
        "scale": saved["scale"],
        "max_abs_errors": errors,
        "original_forward_calls": 7,
        "mixed_jvps": 1,
        "scope": "Original manipulation and unpatched forward for finite responses; existing AD wrapper for mixed derivative",
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
    name = (
        "verification_rerender.json"
        if args.rerender_seed is not None
        else "verification.json"
    )
    write_json(args.run / name, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
