"""Wait for a process, then run only missing outputs from a protocol.

This is a guard for long campaigns that were started interactively before a
detached queue was installed.  After the watched PID exits, the script checks
the locked protocol's declared `args.out` paths and launches only experiments
whose metrics are still absent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def metrics_path(exp: dict[str, Any]) -> Path | None:
    out = exp.get("args", {}).get("out")
    if out is None:
        return None
    p = resolve(out)
    return p if p.suffix == ".json" else p / "metrics.json"


def missing_keys(protocol_path: Path) -> list[str]:
    protocol = json.loads(protocol_path.read_text())
    keys = []
    for exp in sorted(protocol.get("experiments", []), key=lambda e: e["order"]):
        p = metrics_path(exp)
        if p is None or not p.exists():
            keys.append(exp["key"])
    return keys


def detach(args: argparse.Namespace) -> None:
    logdir = PAPER / args.queue_logdir
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", __file__,
        "--wait-pid", str(args.wait_pid),
        "--poll-s", str(args.poll_s),
        "--protocol", args.protocol,
        "--logdir", args.logdir,
        "--queue-logdir", args.queue_logdir,
    ]
    if args.aggregate:
        cmd.append("--aggregate")
    with (logdir / "queue.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=PAPER,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(proc.pid)


def run_aggregate(manifest: dict[str, Any], queue_dir: Path) -> None:
    agg_log = queue_dir / "aggregate.log"
    agg_out = PAPER / "experiments" / "out" / "_aggregate_summary.txt"
    start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START aggregate")
    with agg_log.open("w") as err, agg_out.open("w") as out:
        proc = subprocess.run(
            ["python3", "experiments/aggregate_results.py"],
            cwd=PAPER,
            stdout=out,
            stderr=err,
        )
    manifest["steps"].append({
        "name": "aggregate",
        "command": ["python3", "experiments/aggregate_results.py"],
        "stdout": str(agg_out),
        "stderr": str(agg_log),
        "returncode": proc.returncode,
        "elapsed_s": time.time() - start,
    })
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--queue-logdir", required=True)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()

    if args.detach:
        detach(args)
        return

    queue_dir = PAPER / args.queue_logdir
    queue_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = queue_dir / "queue_manifest.json"
    manifest: dict[str, Any] = {
        "started_at_epoch": time.time(),
        "wait_pid": args.wait_pid,
        "poll_s": args.poll_s,
        "protocol": args.protocol,
        "logdir": args.logdir,
        "steps": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] waiting for pid {args.wait_pid}")
    while pid_alive(args.wait_pid):
        time.sleep(args.poll_s)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] wait pid exited")

    protocol_path = resolve(args.protocol)
    keys = missing_keys(protocol_path)
    manifest["missing_keys_after_wait"] = keys
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if keys:
        cmd = [
            "python3", "-u", "experiments/control/run_control_protocol.py",
            "--protocol", args.protocol,
            "--only", *keys,
            "--logdir", args.logdir,
        ]
        protocol_log = queue_dir / "protocol.log"
        start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START missing protocol")
        with protocol_log.open("w") as fp:
            proc = subprocess.run(cmd, cwd=PAPER, stdout=fp, stderr=subprocess.STDOUT)
        manifest["steps"].append({
            "name": "missing_protocol",
            "command": cmd,
            "log": str(protocol_log),
            "returncode": proc.returncode,
            "elapsed_s": time.time() - start,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if proc.returncode != 0:
            manifest["finished_at_epoch"] = time.time()
            manifest["failed_step"] = "missing_protocol"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise SystemExit(proc.returncode)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] no missing protocol outputs")

    if args.aggregate:
        try:
            run_aggregate(manifest, queue_dir)
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest["finished_at_epoch"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE")


if __name__ == "__main__":
    main()
