"""Train -> independent threshold calibration -> fresh selective evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
import traceback

from run_bedroom import (
    ROOT,
    load_generator,
    preflight,
    provenance,
    render,
    rms,
    sample,
    sha,
    torch,
    write_json,
)
from measure import components
from selective import calibrate, configure, features, fit, predict, report


def validate(p):
    groups = [
        p[k + "_seeds"]
        for k in ("training", "calibration", "evaluation", "confirmation")
    ]
    seeds = sum(groups, []) + [p["preflight_seed"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("All anchor phases must be disjoint")
    if p["signs"] != [-1, 1] or len(set(p["attributes"])) != len(p["attributes"]):
        raise ValueError("Distinct attributes and fixed signs required")
    probes, targets = p["probe_scales"], p["target_scales"]
    if (
        len(probes) != 2
        or not (0 < probes[0] < probes[1] < min(targets))
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Require two smaller probes and distinct target scales")
    n = len(p["attributes"])
    pairs = n * (n - 1) // 2
    calls = 1 + (n * 2 + pairs * 4) * (2 + len(targets))
    if (
        calls != p["expected_calls_per_anchor"]
        or pairs * 4 * len(targets) != p["expected_rows_per_anchor"]
    ):
        raise ValueError("Grid/call budget mismatch")
    if p["model"] != "stylegan_bedroom" or p["forward_dtype"] != "float32":
        raise ValueError("Only validated bedroom fp32 mode is supported")


def collect(generator, directions, p, out, phase, model=None):
    seed_phase = {
        "train": "training",
        "calibrate": "calibration",
        "evaluate": "evaluation",
        "confirm": "confirmation",
    }[phase]
    seeds = p[seed_phase + "_seeds"]
    pairs = list(itertools.combinations(p["attributes"], 2))
    signs = list(itertools.product(p["signs"], repeat=2))
    scales = p["probe_scales"] + p["target_scales"]
    rows, latents = [], {}
    calls = 0
    started = time.monotonic()
    (out / "predictions").mkdir()
    (out / "signed_examples").mkdir()

    def forward(wp):
        nonlocal calls
        if time.monotonic() - started > p["max_wall_seconds"]:
            raise RuntimeError("Frozen wall time limit reached")
        result = render(generator, wp).cpu()
        calls += 1
        return result

    with (out / "rows.jsonl").open("x") as stream, (out / "events.jsonl").open(
        "x"
    ) as events:

        def event(kind, **fields):
            events.write(
                json.dumps(
                    {
                        "event": kind,
                        "forward_calls": calls,
                        "elapsed_seconds": time.monotonic() - started,
                        **fields,
                    }
                )
                + "\n"
            )
            events.flush()

        for seed in seeds:
            wp = sample(generator, seed)
            latents[seed] = wp.cpu()
            base = forward(wp)
            singles = {
                (a, s, sign): forward(wp + sign * s * d)
                for a, d in directions.items()
                for s in scales
                for sign in p["signs"]
            }
            probes = {}
            for a, b in pairs:
                for s in p["probe_scales"]:
                    for sa, sb in signs:
                        joint = forward(
                            wp + sa * s * directions[a] + sb * s * directions[b]
                        )
                        probes[a, b, s, sa, sb] = components(
                            base, singles[a, s, sa], singles[b, s, sb], joint
                        )[3]
            predictions = []
            for a, b in pairs:
                for s in p["target_scales"]:
                    for sa, sb in signs:
                        ra, rb = (
                            singles[a, s, sa].double() - base.double(),
                            singles[b, s, sb].double() - base.double(),
                        )
                        cp = [probes[a, b, t, sa, sb] for t in p["probe_scales"]]
                        row = {
                            "seed": seed,
                            "a": a,
                            "b": b,
                            "scale": s,
                            "sign_a": sa,
                            "sign_b": sb,
                            **features(ra, rb, *cp, s),
                        }
                        if model is not None:
                            row["scores"] = predict(row, model, p)
                        predictions.append(row)
            path = out / "predictions" / f"{seed}.json"
            write_json(path, predictions)
            event(
                (
                    "predictions_locked"
                    if model is not None
                    else "training_features_locked"
                ),
                seed=seed,
                sha256=sha(path),
                rows=len(predictions),
            )
            event("target_joint_observation_begins", seed=seed)
            for record in predictions:
                a, b, s, sa, sb = [
                    record[k] for k in ("a", "b", "scale", "sign_a", "sign_b")
                ]
                joint = forward(wp + sa * s * directions[a] + sb * s * directions[b])
                y0, ra, rb, c = components(
                    base, singles[a, s, sa], singles[b, s, sb], joint
                )
                row = {
                    **record,
                    "actual_relative": float(rms(c)) / record["denominator"],
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if (a, b) == pairs[0] and s == p["target_scales"][1] and sa == sb == 1:
                    torch.save(
                        {
                            **record,
                            "base": y0,
                            "response_a": ra,
                            "response_b": rb,
                            "interaction": c,
                            "small_c": probes[a, b, p["probe_scales"][0], sa, sb],
                            "near_c": probes[a, b, p["probe_scales"][1], sa, sb],
                        },
                        out / "signed_examples" / f"{seed}.pt",
                    )
            stream.flush()
            event("anchor_complete", seed=seed, total_rows=len(rows))
            print(
                f"{phase} seed={seed}: rows={len(rows)}, calls={calls}, {time.monotonic()-started:.1f}s",
                flush=True,
            )
    if (
        calls != len(seeds) * p["expected_calls_per_anchor"]
        or len(rows) != len(seeds) * p["expected_rows_per_anchor"]
    ):
        raise RuntimeError("Incomplete run")
    torch.save(latents, out / "latents.pt")
    write_json(
        out / "measurement.json",
        {
            "status": "COMPLETE",
            "phase": phase,
            "rows": len(rows),
            "forward_calls": calls,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preflight", "train", "calibrate", "evaluate", "confirm"],
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    validate(p)
    configure(gpu=True)
    manifest = provenance(args.protocol, p)
    for name in (
        "prediction.py",
        "selective.py",
        "run_selective.py",
        "verify_selective.py",
        "test_selective.py",
    ):
        path = Path(__file__).with_name(name)
        manifest["source_hashes"][str(path.relative_to(ROOT))] = sha(path)
    model = thresholds = None
    if args.mode != "preflight":
        if args.previous is None:
            raise ValueError("A completed prerequisite run is required")
        previous = json.loads((args.previous / "manifest.json").read_text())
        expected_mode = {
            "train": "preflight",
            "calibrate": "train",
            "evaluate": "calibrate",
            "confirm": "calibrate",
        }[args.mode]
        if previous["mode"] != expected_mode:
            raise ValueError("Wrong prerequisite phase")
        for k in ("protocol_sha256", "source_hashes", "input_hashes"):
            if previous[k] != manifest[k]:
                raise ValueError(f"Frozen prerequisite changed: {k}")
        status = json.loads((args.previous / "complete.json").read_text())
        if status["status"] != "COMPLETE":
            raise ValueError("Prerequisite incomplete")
        manifest["previous_path"] = str(args.previous.resolve())
        if args.mode in ("calibrate", "evaluate", "confirm"):
            model = json.loads((args.previous / "model.json").read_text())
            manifest["model_sha256"] = sha(args.previous / "model.json")
            if args.mode == "calibrate":
                training = [
                    json.loads(s)
                    for s in (args.previous / "rows.jsonl").read_text().splitlines()
                ]
                if fit(training, p) != model:
                    raise ValueError("Training fit changed")
            else:
                thresholds = json.loads((args.previous / "thresholds.json").read_text())
                calibration_rows = [
                    json.loads(s)
                    for s in (args.previous / "rows.jsonl").read_text().splitlines()
                ]
                if calibrate(calibration_rows, p) != thresholds:
                    raise ValueError("Calibration changed")
                manifest["thresholds_sha256"] = sha(args.previous / "thresholds.json")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest.update(
        mode=args.mode,
        protocol=p,
        protocol_path=str(args.protocol.resolve()),
        started_utc=datetime.now(timezone.utc).isoformat(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        reduction_threads=torch.get_num_threads(),
    )
    write_json(args.out / "manifest.json", manifest)
    for relative in manifest["source_hashes"]:
        path = args.out / "source_snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    if model is not None:
        write_json(args.out / "model.json", model)
    if thresholds is not None:
        write_json(args.out / "thresholds.json", thresholds)
    try:
        generator, directions, info = load_generator(p)
        write_json(args.out / "directions.json", info)
        if args.mode == "preflight":
            write_json(args.out / "preflight.json", preflight(generator, directions, p))
        else:
            rows = collect(generator, directions, p, args.out, args.mode, model)
            if args.mode == "train":
                write_json(args.out / "model.json", fit(rows, p))
            elif args.mode == "calibrate":
                write_json(args.out / "thresholds.json", calibrate(rows, p))
            else:
                write_json(args.out / "summary.json", report(rows, p, thresholds))
        artifacts = ["manifest.json"]
        artifacts += [
            name
            for name in (
                "rows.jsonl",
                "model.json",
                "thresholds.json",
                "summary.json",
                "preflight.json",
                "measurement.json",
                "latents.pt",
                "events.jsonl",
            )
            if (args.out / name).exists()
        ]
        write_json(
            args.out / "complete.json",
            {
                "status": "COMPLETE",
                "artifact_hashes": {name: sha(args.out / name) for name in artifacts},
            },
        )
        print(json.dumps({"status": "COMPLETE", "phase": args.mode}), flush=True)
    except Exception:
        write_json(
            args.out / "failure.json",
            {"status": "FAILED", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
