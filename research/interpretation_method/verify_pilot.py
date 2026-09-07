"""Audit complete coverage and provenance; optionally rerender one saved example."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    protocol = manifest["protocol"]
    summary = json.loads((run / "summary.json").read_text())
    check(summary["status"] == "COMPLETE", "Run is incomplete")
    check(
        summary["generator_calls"] == protocol["expected_generator_calls"],
        "Wrong call count",
    )
    for path, expected in manifest["source_hashes"].items():
        check(
            digest(run / "source_snapshot" / path) == expected,
            f"Snapshot mismatch: {path}",
        )
        check(digest(ROOT / path) == expected, f"Working source changed: {path}")
    for path, expected in manifest["input_hashes"].items():
        check(digest(ROOT / path) == expected, f"Input changed: {path}")
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    seeds = protocol["development_seeds"] + protocol["replication_seeds"]
    expected = {
        (seed, a, b, scale, sa, sb)
        for seed in seeds
        for a, b in itertools.combinations(protocol["attributes"], 2)
        for scale in protocol["scales"]
        for sa, sb in itertools.product(protocol["signs"], repeat=2)
    }
    seen = set()
    max_closure = 0.0
    for row in rows:
        key = tuple(row[k] for k in ("seed", "a", "b", "scale", "sign_a", "sign_b"))
        check(key not in seen, f"Duplicate row: {key}")
        seen.add(key)
        check(row["seed"] in protocol[f"{row['cohort']}_seeds"], "Cohort mismatch")
        for name, value in row.items():
            if isinstance(value, float):
                check(math.isfinite(value), f"Nonfinite {name}: {key}")
        max_closure = max(
            max_closure,
            row["square_observer_closure_rms"],
            row["squared_distance_closure_error"],
        )
    check(seen == expected, "Missing or unexpected measurement cells")
    check(len(rows) == protocol["expected_pair_rows"], "Protocol row count mismatch")
    check(max_closure < 1e-10, "Algebraic decomposition does not close")
    check(summary["overall"]["n"] == len(rows), "Summary row count mismatch")
    expected_examples = {f"{seed}.pt" for seed in seeds}
    check(
        {p.name for p in (run / "signed_examples").glob("*.pt")} == expected_examples,
        "Missing signed examples",
    )
    return (
        manifest,
        rows,
        {
            "status": "PASS",
            "rows": len(rows),
            "anchors": len(seeds),
            "generator_calls": summary["generator_calls"],
            "source_files_verified": len(manifest["source_hashes"]),
            "input_files_verified": len(manifest["input_hashes"]),
            "max_algebraic_closure_error": max_closure,
        },
    )


def rerender(run, manifest, rows, seed):
    import torch

    check(torch.cuda.is_available(), "Rerender requires the approved CUDA environment")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT / "higan_dev"))
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary, manipulate_wp

    protocol = manifest["protocol"]
    latents = torch.load(run / "latents.pt", map_location="cpu")
    saved = torch.load(run / "signed_examples" / f"{seed}.pt", map_location="cpu")
    wp = latents[seed].cuda()
    generator = HiGANGenerator(ROOT / protocol["higan_repo"], device="cuda")
    boundaries = ROOT / protocol["higan_repo"] / "boundaries/stylegan_bedroom"
    a = load_boundary(boundaries, saved["a"])
    b = load_boundary(boundaries, saved["b"])
    # Use the original manipulation helper and unpatched synthesis forward,
    # independently of the pilot's direction-placement and render functions.
    wa = manipulate_wp(wp, a, [saved["scale"]])[:, 0]
    wb = manipulate_wp(wp, b, [saved["scale"]])[:, 0]
    wab = manipulate_wp(wa, b, [saved["scale"]])[:, 0]
    synth = generator._net.synthesis
    with torch.no_grad():
        y0, ya, yb, yab = [
            type(synth).forward(synth, w).double().cpu() for w in (wp, wa, wb, wab)
        ]
    ra, rb = ya - y0, yb - y0
    rc = yab - ya - yb + y0
    errors = {
        name: float((value - saved[name]).abs().max())
        for name, value in [
            ("base", y0),
            ("response_a", ra),
            ("response_b", rb),
            ("interaction", rc),
        ]
    }
    check(max(errors.values()) <= 1e-6, f"Independent rerender mismatch: {errors}")
    row = next(
        r
        for r in rows
        if r["seed"] == seed
        and r["a"] == saved["a"]
        and r["b"] == saved["b"]
        and r["scale"] == saved["scale"]
        and r["sign_a"] == r["sign_b"] == 1
    )
    actual_rms = float(rc.square().mean().sqrt())
    check(
        abs(actual_rms - row["interaction_rms"]) <= 1e-6,
        "Reported interaction disagrees with rerender",
    )
    return {
        "seed": seed,
        "generator_calls": 4,
        "max_abs_errors": errors,
        "interaction_rms": actual_rms,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--rerender-seed", type=int)
    args = parser.parse_args()
    manifest, rows, result = audit(args.run)
    if args.rerender_seed is not None:
        result["independent_rerender"] = rerender(
            args.run, manifest, rows, args.rerender_seed
        )
    result["verifier_sha256"] = digest(Path(__file__).resolve())
    filename = (
        "verification_rerender.json"
        if args.rerender_seed is not None
        else "verification.json"
    )
    (args.run / filename).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
