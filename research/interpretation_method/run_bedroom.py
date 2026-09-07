"""Locked, descriptive finite-intervention pilot. No model training or editing UI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.func import jvp

from measure import components, rms, summarize

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def provenance(protocol_path, protocol):
    repo = ROOT / protocol["higan_repo"]
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("measure.py"),
        ROOT / "higan_dev/higan_dev/generator.py",
        ROOT / "higan_dev/higan_dev/manipulate.py",
    ]
    sources.extend(sorted((repo / "models").rglob("*.py")))
    sources = [p.resolve() for p in sources]
    inputs = [repo / "models/pretrain/pytorch/stylegan_bedroom256_generator.pth"]
    inputs.extend(
        repo / "boundaries/stylegan_bedroom" / f"{a}_boundary.npy"
        for a in protocol["attributes"]
    )
    return {
        "protocol_sha256": sha(protocol_path),
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
    }


def load_generator(protocol):
    sys.path.insert(0, str(ROOT / "higan_dev"))
    from higan_dev.generator import HiGANGenerator
    from higan_dev.manipulate import load_boundary

    repo = ROOT / protocol["higan_repo"]
    generator = HiGANGenerator(repo, model_name=protocol["model"], device="cuda")
    directions, info = {}, {}
    for name in protocol["attributes"]:
        boundary = load_boundary(
            repo / "boundaries/stylegan_bedroom", name, num_layers=generator.num_layers
        )
        direction = torch.zeros(1, generator.num_layers, generator.w_dim)
        for layer in boundary.manipulate_layers:
            direction[0, layer] = boundary.direction
        direction = direction.to("cuda")
        directions[name] = direction
        info[name] = {
            "layers": boundary.manipulate_layers,
            "full_wp_norm": float(direction.norm()),
        }
    return generator, directions, info


def sample(generator, seed):
    return generator.sample_wp(
        1, generator=torch.Generator(device="cuda").manual_seed(seed)
    ).detach()


def render(generator, wp):
    with torch.no_grad():
        result = generator.synthesize(wp).detach()
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite generator output")
    return result


def preflight(generator, directions, protocol):
    wp = sample(generator, protocol["preflight_seed"])
    base = render(generator, wp)
    repeat_error = max(
        float((render(generator, wp) - base).abs().max()) for _ in range(3)
    )
    synth = generator._net.synthesis
    with torch.no_grad():
        original = type(synth).forward(synth, wp)
    wrapper_error = float((original - base).abs().max())
    if repeat_error > protocol["preflight_repeat_max_abs"]:
        raise RuntimeError(f"Repeated-forward error exceeds gate: {repeat_error}")
    if (
        not torch.isfinite(original).all()
        or wrapper_error > protocol["preflight_wrapper_max_abs"]
    ):
        raise RuntimeError(f"Wrapper differs from original synthesis: {wrapper_error}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(8):
        render(generator, wp)
    torch.cuda.synchronize()
    forward_ms = 1000 * (time.perf_counter() - start) / 8
    first_order_checks = {}
    for name in ("view", "indoor_lighting", "wood"):
        direction = directions[name]
        _, derivative = jvp(generator.synthesize, (wp,), (direction,))
        if not torch.isfinite(derivative).all():
            raise RuntimeError("Nonfinite first-order JVP")
        reference = derivative.double()
        errors = {}
        for epsilon in (0.01, 0.03, 0.1):
            finite = (
                render(generator, wp + epsilon * direction).double()
                - render(generator, wp - epsilon * direction).double()
            ) / (2 * epsilon)
            errors[str(epsilon)] = float(
                rms(finite - reference) / rms(reference).clamp_min(1e-12)
            )
        first_order_checks[name] = errors
    print(
        f"preflight: repeat={repeat_error:.3g}, wrapper={wrapper_error:.3g}, forward={forward_ms:.1f}ms",
        flush=True,
    )
    return {
        "status": "PASS",
        "gate_scope": "finite forward measurement; JVP checks are diagnostic and not a gate for the finite pilot",
        "repeat_max_abs": repeat_error,
        "wrapper_max_abs": wrapper_error,
        "forward_ms": forward_ms,
        "base_min": float(base.min()),
        "base_max": float(base.max()),
        "jvp_fd_relative_rms": first_order_checks,
    }


def aggregate(rows, protocol):
    def group_summary(group):
        values = [
            r["interaction_relative"]
            for r in group
            if r["interaction_relative"] is not None
        ]
        return {
            "n": len(group),
            "responsive_n": sum(r["responsive"] for r in group),
            "median_interaction_relative": (
                statistics.median(values) if values else None
            ),
            "additive_fraction": statistics.mean(
                r["additive_at_tolerance"] for r in group
            ),
            "median_signed_cosine": statistics.median(
                r["signed_cosine"] for r in group if r["signed_cosine"] is not None
            ),
            "tolerance_sensitivity": {
                str(t): statistics.mean(
                    r["interaction_rms"]
                    <= protocol["absolute_tolerance"] + t * (r["a_rms"] + r["b_rms"])
                    for r in group
                )
                for t in protocol["relative_tolerance_sensitivity"]
            },
        }

    return {
        "scope": "descriptive measured effects; no held-out prediction accuracy",
        "overall": group_summary(rows),
        "by_scale": {
            str(s): group_summary([r for r in rows if r["scale"] == s])
            for s in protocol["scales"]
        },
        "view_pairs": group_summary([r for r in rows if "view" in (r["a"], r["b"])]),
        "without_view": group_summary(
            [r for r in rows if "view" not in (r["a"], r["b"])]
        ),
        "by_cohort": {
            c: group_summary([r for r in rows if r["cohort"] == c])
            for c in ("development", "replication")
        },
        "by_anchor": {
            str(seed): group_summary([r for r in rows if r["seed"] == seed])
            for seed in protocol["development_seeds"] + protocol["replication_seeds"]
        },
        "max_square_observer_closure_rms": max(
            r["square_observer_closure_rms"] for r in rows
        ),
        "max_squared_distance_closure_error": max(
            r["squared_distance_closure_error"] for r in rows
        ),
    }


def pilot(generator, directions, protocol, out):
    start = time.monotonic()
    rows, latents = [], {}
    calls = 0
    pairs = list(itertools.combinations(protocol["attributes"], 2))
    (out / "signed_examples").mkdir()
    with (out / "rows.jsonl").open("x") as stream:
        for cohort in ("development", "replication"):
            for seed in protocol[f"{cohort}_seeds"]:
                wp = sample(generator, seed)
                latents[seed] = wp.cpu()
                base = render(generator, wp)
                calls += 1
                singles = {}
                for name, direction in directions.items():
                    for scale in protocol["scales"]:
                        for sign in protocol["signs"]:
                            singles[name, scale, sign] = render(
                                generator, wp + sign * scale * direction
                            )
                            calls += 1
                for a, b in pairs:
                    for scale in protocol["scales"]:
                        for sign_a, sign_b in itertools.product(
                            protocol["signs"], repeat=2
                        ):
                            if time.monotonic() - start > protocol["max_wall_seconds"]:
                                raise RuntimeError(
                                    "Preregistered wall-time limit reached"
                                )
                            joint = render(
                                generator,
                                wp
                                + sign_a * scale * directions[a]
                                + sign_b * scale * directions[b],
                            )
                            calls += 1
                            single_a, single_b = (
                                singles[a, scale, sign_a],
                                singles[b, scale, sign_b],
                            )
                            row = {
                                "seed": seed,
                                "cohort": cohort,
                                "a": a,
                                "b": b,
                                "scale": scale,
                                "sign_a": sign_a,
                                "sign_b": sign_b,
                                **summarize(
                                    base,
                                    single_a,
                                    single_b,
                                    joint,
                                    atol=protocol["absolute_tolerance"],
                                    rtol=protocol["relative_tolerance"],
                                ),
                            }
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            rows.append(row)
                            if (
                                protocol["store_signed_example_every_anchor"]
                                and (a, b) == pairs[0]
                                and scale == 0.5
                                and sign_a == sign_b == 1
                            ):
                                y0, ra, rb, rc = components(
                                    base, single_a, single_b, joint
                                )
                                torch.save(
                                    {
                                        "seed": seed,
                                        "a": a,
                                        "b": b,
                                        "scale": scale,
                                        "base": y0.cpu(),
                                        "response_a": ra.cpu(),
                                        "response_b": rb.cpu(),
                                        "interaction": rc.cpu(),
                                    },
                                    out / "signed_examples" / f"{seed}.pt",
                                )
                stream.flush()
                elapsed = time.monotonic() - start
                write_json(
                    out / "progress.json",
                    {
                        "status": "RUNNING",
                        "last_seed": seed,
                        "rows": len(rows),
                        "calls": calls,
                        "elapsed_seconds": elapsed,
                    },
                )
                print(
                    f"{cohort} seed={seed}: {len(rows)} rows, {calls} calls, {elapsed:.1f}s",
                    flush=True,
                )
                del singles
    if (
        calls != protocol["expected_generator_calls"]
        or len(rows) != protocol["expected_pair_rows"]
    ):
        raise RuntimeError(f"Incomplete pilot: calls={calls}, rows={len(rows)}")
    torch.save(latents, out / "latents.pt")
    summary = aggregate(rows, protocol)
    summary.update(
        {
            "status": "COMPLETE",
            "generator_calls": calls,
            "elapsed_seconds": time.monotonic() - start,
        }
    )
    write_json(out / "summary.json", summary)
    write_json(
        out / "progress.json", {"status": "COMPLETE", "rows": len(rows), "calls": calls}
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "pilot"], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol["model"] != "stylegan_bedroom"
        or protocol["forward_dtype"] != "float32"
    ):
        raise ValueError("Only the validated bedroom fp32 configuration is supported")
    seeds = protocol["development_seeds"] + protocol["replication_seeds"]
    if len(set(seeds + [protocol["preflight_seed"]])) != len(seeds) + 1:
        raise ValueError("Seeds must be disjoint")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; run in the approved GPU execution environment"
        )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = provenance(args.protocol, protocol)
    if args.mode == "pilot":
        if args.preflight is None:
            raise ValueError("Pilot requires a passed preflight directory")
        old = json.loads((args.preflight / "manifest.json").read_text())
        result = json.loads((args.preflight / "preflight.json").read_text())
        if result["status"] != "PASS":
            raise ValueError("Preflight did not pass")
        for key in ("protocol_sha256", "source_hashes", "input_hashes"):
            if old[key] != manifest[key]:
                raise ValueError(f"Preflight provenance changed: {key}")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest.update(
        {
            "mode": args.mode,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "deterministic": True,
            "protocol": protocol,
        }
    )
    write_json(args.out / "manifest.json", manifest)
    for relative in manifest["source_hashes"]:
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    try:
        generator, directions, info = load_generator(protocol)
        write_json(args.out / "directions.json", info)
        if args.mode == "preflight":
            write_json(
                args.out / "preflight.json", preflight(generator, directions, protocol)
            )
        else:
            result = pilot(generator, directions, protocol, args.out)
            print(
                json.dumps({"status": result["status"], "overall": result["overall"]}),
                flush=True,
            )
    except Exception:
        write_json(
            args.out / "failure.json",
            {"status": "FAILED", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
