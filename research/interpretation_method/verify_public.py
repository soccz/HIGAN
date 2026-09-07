"""Recompute the published v2 tables without GPU assets or private run tensors.

This checks the public numerical record, not the original generator outputs.
Use verify_selective.py on a complete local run for the latter audit.
"""

import itertools
import json
from pathlib import Path

from analyze_selective import compare
from run_bedroom import ROOT, sha
from run_selective import validate
from selective import calibrate, configure, fit, key, predict, report, summarize
from verify_selective import check, close


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    configure()
    directory = Path(__file__).resolve().parent
    protocol = directory / "protocol_selective_v2.json"
    p = read(protocol)
    validate(p)
    previous = None
    counts = {}
    checked_sources = set()
    for phase, mode in [
        ("preflight", "preflight"),
        ("training", "train"),
        ("calibration", "calibrate"),
        ("evaluation", "evaluate"),
    ]:
        run = directory / "runs" / f"selective_v2_{phase}"
        manifest, complete = read(run / "manifest.json"), read(run / "complete.json")
        check(manifest["mode"] == mode, "Wrong phase")
        check(complete["status"] == "COMPLETE", "Incomplete run")
        check(manifest["protocol"] == p, "Embedded protocol mismatch")
        check(manifest["protocol_sha256"] == sha(protocol), "Protocol hash mismatch")
        for name, digest in complete["artifact_hashes"].items():
            if name != "latents.pt":
                check(sha(run / name) == digest, f"Artifact changed: {phase}/{name}")
        for name, digest in manifest["source_hashes"].items():
            if not name.startswith(p["higan_repo"] + "/"):
                check(sha(ROOT / name) == digest, f"Published source changed: {name}")
                checked_sources.add(name)
        if previous is not None:
            old = read(previous / "manifest.json")
            for field in ("source_hashes", "input_hashes", "protocol_sha256"):
                check(manifest[field] == old[field], f"Phase mismatch: {field}")
            for name in ("model", "thresholds"):
                if name + "_sha256" in manifest:
                    check(
                        sha(run / f"{name}.json")
                        == sha(previous / f"{name}.json")
                        == manifest[name + "_sha256"],
                        f"Frozen {name} changed",
                    )
        previous = run
        if phase == "preflight":
            check(read(run / "preflight.json")["status"] == "PASS", "Preflight failed")
            continue
        rows, events, model = (
            lines(run / "rows.jsonl"),
            lines(run / "events.jsonl"),
            read(run / "model.json"),
        )
        expected = {
            (seed, a, b, scale, sa, sb)
            for seed in p[phase + "_seeds"]
            for a, b in itertools.combinations(p["attributes"], 2)
            for scale in p["target_scales"]
            for sa, sb in itertools.product(p["signs"], repeat=2)
        }
        check(
            len(rows) == len(expected) and {key(r) for r in rows} == expected,
            "Grid mismatch",
        )
        index = {key(r): r for r in rows}
        for seed in p[phase + "_seeds"]:
            event = [e for e in events if e["seed"] == seed]
            first = (
                "training_features_locked"
                if phase == "training"
                else "predictions_locked"
            )
            check(
                [e["event"] for e in event]
                == [first, "target_joint_observation_begins", "anchor_complete"],
                "Lock order changed",
            )
            check(
                event[0]["elapsed_seconds"]
                <= event[1]["elapsed_seconds"]
                < event[2]["elapsed_seconds"],
                "Event times changed",
            )
            check(
                event[0]["forward_calls"] == event[1]["forward_calls"],
                "Calls before lock",
            )
            path = run / "predictions" / f"{seed}.json"
            check(sha(path) == event[0]["sha256"], "Locked predictions changed")
            predictions = read(path)
            check(
                len(predictions) == p["expected_rows_per_anchor"],
                "Prediction count mismatch",
            )
            check(
                len({key(r) for r in predictions}) == len(predictions),
                "Duplicate predictions",
            )
            for pred in predictions:
                check(
                    all(index[key(pred)][k] == v for k, v in pred.items()),
                    "Prediction/row mismatch",
                )
                if phase != "training":
                    check(
                        close(predict(pred, model, p), pred["scores"]), "Score mismatch"
                    )
        if phase == "training":
            check(close(fit(rows, p), model), "Training fit mismatch")
        else:
            thresholds = read(run / "thresholds.json")
            if phase == "calibration":
                check(close(calibrate(rows, p), thresholds), "Calibration mismatch")
            else:
                summary = read(run / "summary.json")
                check(
                    close(report(rows, p, thresholds), summary),
                    "Summary/bootstrap mismatch",
                )
                secondary = read(run / "secondary_analysis.json")
                check(
                    secondary["rows_sha256"] == sha(run / "rows.jsonl"),
                    "Secondary rows changed",
                )
                check(
                    secondary["analysis_source_sha256"]
                    == sha(directory / "analyze_selective.py"),
                    "Analysis source changed",
                )
                for result in secondary["information_increments"]:
                    check(
                        close(
                            compare(
                                summary["overall"],
                                p,
                                result["richer"],
                                result["baseline"],
                            ),
                            result,
                        ),
                        "Increment mismatch",
                    )
                check(
                    close(
                        summarize(
                            [r for r in rows if "view" not in (r["a"], r["b"])],
                            p,
                            thresholds,
                        ),
                        secondary["without_view"],
                    ),
                    "View exclusion mismatch",
                )
                decision = read(run / "decision.json")
                check(
                    decision["summary_sha256"] == sha(run / "summary.json"),
                    "Decision summary changed",
                )
                check(
                    decision["decision_source_sha256"]
                    == sha(directory / "decide_selective.py"),
                    "Decision source changed",
                )
        counts[phase] = len(rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "Published v2 artifacts, prediction locks, fitted models, calibrated thresholds, primary/secondary tables and anchor bootstrap",
                "rows": counts,
                "published_sources_checked": len(checked_sources),
                "not_checked": [
                    "GPU output regeneration",
                    "upstream source files and model weights",
                    "latent and signed-output tensors",
                    "historical v1 runs",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
