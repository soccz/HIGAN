"""Frozen development fitting, then prediction-before-target joint evaluation."""

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
from torch.func import jvp

from measure import components
from prediction import (
    evaluate_response,
    fit_model,
    risk_predictions,
    single_features,
    summarize_rows,
    vector_predictions,
)


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the approved GPU environment")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_protocol(p):
    dev, test = set(p["development_seeds"]), set(p["evaluation_seeds"])
    if (
        dev & test
        or p["preflight_seed"] in dev | test
        or test & set(p["forbid_prior_evaluation_seeds"])
        or len(dev) != len(p["development_seeds"])
        or len(test) != len(p["evaluation_seeds"])
    ):
        raise ValueError(
            "Development, preflight, and fresh evaluation seeds must be disjoint"
        )
    scales = p["target_scales"]
    if len(set(scales)) != len(scales) or not all(
        s > p["probe_scale"] > 0 for s in scales
    ):
        raise ValueError("Target scales must be distinct and larger than probe")
    if p["model"] != "stylegan_bedroom" or p["forward_dtype"] != "float32":
        raise ValueError("Only validated bedroom fp32 mode is supported")
    n = len(p["attributes"])
    pairs = n * (n - 1) // 2
    calls = 1 + n * (len(scales) + 1) * 2 + pairs * (len(scales) + 1) * 4
    rows = pairs * len(scales) * 4
    expected = {
        "expected_training_calls": len(dev) * calls,
        "expected_evaluation_forward_calls": len(test) * calls,
        "expected_training_rows": len(dev) * rows,
        "expected_evaluation_rows": len(test) * rows,
        "expected_evaluation_mixed_jvps": len(test) * pairs,
    }
    if p["signs"] != [-1, 1] or any(p[k] != v for k, v in expected.items()):
        raise ValueError("Locked grid or call budget mismatch")


def mixed_jvp(generator, wp, a, b):
    with torch.no_grad():
        value = jvp(lambda w: jvp(generator.synthesize, (w,), (a,))[1], (wp,), (b,))[
            1
        ].detach()
    if not torch.isfinite(value).all():
        raise RuntimeError("Nonfinite local mixed JVP")
    return value.double().cpu()


def preflight_prediction(generator, directions, p):
    result = preflight(generator, directions, p)
    wp = sample(generator, p["preflight_seed"])
    a, b = [directions[name] for name in p["attributes"][:2]]
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    ab = mixed_jvp(generator, wp, a, b)
    ba = mixed_jvp(generator, wp, b, a)
    error = float(rms(ab - ba) / rms(ab).clamp_min(1e-12))
    if error > p["mixed_symmetry_relative_tolerance"]:
        raise RuntimeError(f"Mixed-JVP symmetry gate failed: {error}")
    result.update(
        mixed_symmetry_relative_error=error,
        mixed_rms=float(rms(ab)),
        mixed_jvp_seconds=(time.monotonic() - start) / 2,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        local_derivative_scope="AD local mixed derivative; no finite-interval Hessian guarantee",
    )
    return result


def run(generator, directions, p, out, model=None):
    evaluation = model is not None
    seeds = p["evaluation_seeds"] if evaluation else p["development_seeds"]
    pairs = list(itertools.combinations(p["attributes"], 2))
    signs = list(itertools.product(p["signs"], repeat=2))
    scales = [p["probe_scale"], *p["target_scales"]]
    rows, latents = [], {}
    calls = mixed_calls = 0
    timing = {"forward_seconds": 0.0, "mixed_jvp_seconds": 0.0}
    started = time.monotonic()
    if evaluation:
        (out / "predictions").mkdir()
        (out / "signed_examples").mkdir()

    def forward(w):
        nonlocal calls
        if time.monotonic() - started > p["max_wall_seconds"]:
            raise RuntimeError("Preregistered wall-time limit reached")
        start = time.monotonic()
        value = render(generator, w).cpu()
        timing["forward_seconds"] += time.monotonic() - start
        calls += 1
        return value

    row_path = out / ("rows.jsonl" if evaluation else "training_rows.jsonl")
    with row_path.open("x") as stream, (out / "events.jsonl").open("x") as events:

        def event(kind, **fields):
            events.write(
                json.dumps(
                    {
                        "event": kind,
                        "elapsed_seconds": time.monotonic() - started,
                        "forward_calls": calls,
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
                (a, s, sign): forward(wp + sign * s * direction)
                for a, direction in directions.items()
                for s in scales
                for sign in p["signs"]
            }
            probes, mixed = {}, {}
            for a, b in pairs:
                for sa, sb in signs:
                    s = p["probe_scale"]
                    joint = forward(
                        wp + sa * s * directions[a] + sb * s * directions[b]
                    )
                    probes[a, b, sa, sb] = components(
                        base, singles[a, s, sa], singles[b, s, sb], joint
                    )[3]
                if evaluation:
                    start = time.monotonic()
                    mixed[a, b] = mixed_jvp(generator, wp, directions[a], directions[b])
                    timing["mixed_jvp_seconds"] += time.monotonic() - start
                    mixed_calls += 1

            predictions = []
            for a, b in pairs:
                for scale in p["target_scales"]:
                    for sa, sb in signs:
                        ra = singles[a, scale, sa].double() - base.double()
                        rb = singles[b, scale, sb].double() - base.double()
                        denominator, features = single_features(ra, rb, scale)
                        record = {
                            "seed": seed,
                            "a": a,
                            "b": b,
                            "scale": scale,
                            "sign_a": sa,
                            "sign_b": sb,
                            "denominator": denominator,
                            "single_features": features,
                        }
                        if evaluation:
                            fitted = model["by_scale"][str(scale)]
                            vectors = vector_predictions(
                                probes[a, b, sa, sb],
                                mixed[a, b],
                                scale,
                                p["probe_scale"],
                                (sa, sb),
                                fitted["coefficient"],
                            )
                            record["risk_predictions"] = risk_predictions(
                                vectors, denominator, features, fitted
                            )
                        predictions.append(record)
            if evaluation:
                prediction_path = out / "predictions" / f"{seed}.json"
                write_json(prediction_path, predictions)
                event(
                    "predictions_locked",
                    seed=seed,
                    sha256=sha(prediction_path),
                    rows=len(predictions),
                )
            event("target_joint_observation_begins", seed=seed)
            for record in predictions:
                a, b, s, sa, sb = (
                    record[k] for k in ("a", "b", "scale", "sign_a", "sign_b")
                )
                joint = forward(wp + sa * s * directions[a] + sb * s * directions[b])
                y0, ra, rb, target_c = components(
                    base, singles[a, s, sa], singles[b, s, sb], joint
                )
                probe_c = probes[a, b, sa, sb]
                denominator = record["denominator"]
                if evaluation:
                    fitted = model["by_scale"][str(s)]
                    vectors = vector_predictions(
                        probe_c,
                        mixed[a, b],
                        s,
                        p["probe_scale"],
                        (sa, sb),
                        fitted["coefficient"],
                    )
                    measured = evaluate_response(
                        target_c,
                        vectors,
                        record["risk_predictions"],
                        denominator,
                        p["absolute_tolerance"],
                        p["relative_tolerance"],
                    )
                    row = {**record, **measured}
                    if (
                        (a, b) == pairs[0]
                        and s == p["target_scales"][1]
                        and sa == sb == 1
                    ):
                        torch.save(
                            {
                                **record,
                                "base": y0,
                                "response_a": ra,
                                "response_b": rb,
                                "interaction": target_c,
                                "probe_c": probe_c,
                                "mixed": mixed[a, b],
                            },
                            out / "signed_examples" / f"{seed}.pt",
                        )
                else:
                    row = {
                        **record,
                        "actual_relative": float(rms(target_c)) / denominator,
                        "normalized_probe_target_dot": float(
                            (probe_c * target_c).mean()
                        )
                        / denominator**2,
                        "normalized_probe_squared": float(probe_c.square().mean())
                        / denominator**2,
                    }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            event("anchor_complete", seed=seed, total_rows=len(rows))
            print(
                f"{'evaluation' if evaluation else 'development'} seed={seed}: rows={len(rows)}, "
                f"calls={calls}, mixed_jvps={mixed_calls}, {time.monotonic() - started:.1f}s",
                flush=True,
            )

    expected_rows = (
        p["expected_evaluation_rows"] if evaluation else p["expected_training_rows"]
    )
    expected_calls = (
        p["expected_evaluation_forward_calls"]
        if evaluation
        else p["expected_training_calls"]
    )
    if len(rows) != expected_rows or calls != expected_calls:
        raise RuntimeError("Incomplete measurement grid")
    if evaluation and mixed_calls != p["expected_evaluation_mixed_jvps"]:
        raise RuntimeError("Incomplete mixed-JVP grid")
    torch.save(latents, out / "latents.pt")
    if evaluation:
        summary = summarize_rows(rows, p)
        write_json(out / "summary.json", summary)
    else:
        write_json(out / "model.json", fit_model(rows, p))
    result = {
        "status": "COMPLETE",
        "rows": len(rows),
        "forward_calls": calls,
        "mixed_jvps": mixed_calls,
        **timing,
        "elapsed_seconds": time.monotonic() - started,
        "cost_scope": "Forward times include device-to-host transfer; mixed JVP is timed separately. Target joint label calls are included in total, not predictor input budget.",
    }
    write_json(out / "progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["preflight", "train", "evaluate"], required=True
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    validate_protocol(p)
    configure()
    manifest = provenance(args.protocol, p)
    for name in (
        "prediction.py",
        "run_prediction.py",
        "verify_prediction.py",
        "test_prediction.py",
    ):
        path = Path(__file__).with_name(name)
        manifest["source_hashes"][str(path.relative_to(ROOT))] = sha(path)
    model = None
    prerequisite = args.preflight if args.mode == "train" else args.training
    if args.mode != "preflight":
        if prerequisite is None:
            raise ValueError("A passed preflight or frozen training run is required")
        previous = json.loads((prerequisite / "manifest.json").read_text())
        for k in ("source_hashes", "input_hashes", "protocol_sha256"):
            if previous[k] != manifest[k]:
                raise ValueError(f"Frozen prerequisite changed: {k}")
        status_file = "preflight.json" if args.mode == "train" else "progress.json"
        status = json.loads((prerequisite / status_file).read_text())["status"]
        if status != ("PASS" if args.mode == "train" else "COMPLETE"):
            raise ValueError("Prerequisite incomplete")
        if args.mode == "evaluate":
            training_rows = [
                json.loads(s)
                for s in (prerequisite / "training_rows.jsonl").read_text().splitlines()
            ]
            model = json.loads((prerequisite / "model.json").read_text())
            if fit_model(training_rows, p) != model:
                raise ValueError("Model differs from development-only fit")
            manifest.update(
                training_path=str(prerequisite.resolve()),
                model_sha256=sha(prerequisite / "model.json"),
                training_rows_sha256=sha(prerequisite / "training_rows.jsonl"),
            )
    args.out.mkdir(parents=True, exist_ok=False)
    manifest.update(
        mode=args.mode,
        started_utc=datetime.now(timezone.utc).isoformat(),
        protocol=p,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
    )
    write_json(args.out / "manifest.json", manifest)
    if model is not None:
        write_json(args.out / "model.json", model)
    for relative in manifest["source_hashes"]:
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    try:
        generator, directions, info = load_generator(p)
        write_json(args.out / "directions.json", info)
        if args.mode == "preflight":
            result = preflight_prediction(generator, directions, p)
            write_json(args.out / "preflight.json", result)
        else:
            result = run(generator, directions, p, args.out, model)
        print(json.dumps(result), flush=True)
    except Exception:
        write_json(
            args.out / "failure.json",
            {"status": "FAILED", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
