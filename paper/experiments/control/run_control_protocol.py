"""Run control experiments from a locked protocol JSON file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def arg_name(name: str) -> str:
    return "--" + name


def append_arg(cmd: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(arg_name(name))
        return
    if isinstance(value, list):
        cmd.append(arg_name(name))
        cmd.extend(str(v) for v in value)
        return
    cmd.extend([arg_name(name), str(value)])


def build_command(python_bin: str, protocol_path: Path,
                  experiment: dict[str, Any]) -> list[str]:
    cmd = [python_bin, "-u", experiment["script"]]
    for key, value in experiment.get("args", {}).items():
        append_arg(cmd, key, value)
    cmd.extend(["--protocol", str(protocol_path),
                "--protocol-key", experiment["key"]])
    return cmd


def validate_protocol(protocol: dict[str, Any], protocol_path: Path,
                      only: list[str] | None) -> None:
    experiments = protocol.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{protocol_path} has no experiments list")

    keys = [str(e.get("key")) for e in experiments]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate experiment keys in {protocol_path}: {duplicates}")

    if only:
        unknown = sorted(set(only) - set(keys))
        if unknown:
            raise ValueError(f"--only contains unknown experiment keys: {unknown}")

    for exp in experiments:
        if "key" not in exp or "order" not in exp or "script" not in exp:
            raise ValueError(f"malformed experiment entry in {protocol_path}: {exp}")
        script_path = resolve(exp["script"])
        if not script_path.exists():
            raise FileNotFoundError(
                f"script for experiment {exp['key']} does not exist: {script_path}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/protocols/control_v1.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--python-bin", default="python3")
    ap.add_argument("--logdir", default="logs/control")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    protocol_path = resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, protocol_path, args.only)
    logdir = resolve(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    selected = sorted(protocol["experiments"], key=lambda e: e["order"])
    if args.only:
        selected = [e for e in selected if e["key"] in set(args.only)]
    if not selected:
        raise ValueError("no experiments selected")

    manifest = {
        "protocol": str(protocol_path),
        "protocol_version": protocol.get("version"),
        "started_at_epoch": time.time(),
        "dry_run": args.dry_run,
        "runs": [],
    }
    for exp in selected:
        cmd = build_command(args.python_bin, protocol_path, exp)
        print(" ".join(cmd))
        run_info = {
            "key": exp["key"],
            "script": exp["script"],
            "command": cmd,
            "log": str(logdir / f"{exp['key']}.log"),
        }
        if not args.dry_run:
            start = time.time()
            with (logdir / f"{exp['key']}.log").open("w") as fp:
                proc = subprocess.run(cmd, cwd=PAPER, stdout=fp,
                                      stderr=subprocess.STDOUT)
            run_info["returncode"] = proc.returncode
            run_info["elapsed_s"] = time.time() - start
            if proc.returncode != 0:
                manifest["runs"].append(run_info)
                (logdir / "protocol_run_manifest.json").write_text(
                    json.dumps(manifest, indent=2))
                raise SystemExit(proc.returncode)
        manifest["runs"].append(run_info)

    manifest["finished_at_epoch"] = time.time()
    (logdir / "protocol_run_manifest.json").write_text(
        json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
