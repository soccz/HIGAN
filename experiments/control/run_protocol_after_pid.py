"""Run a locked protocol after an existing process exits.

This keeps long GPU validation campaigns continuous without changing the
experiment definitions at runtime.  The protocol itself remains the source of
truth; this script only waits, launches it, and records a queue manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PAPER = Path(__file__).resolve().parents[2]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


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
        "--python-bin", args.python_bin,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--queue-logdir", required=True)
    ap.add_argument("--python-bin", default="python3")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()

    if args.detach:
        detach(args)
        return

    queue_dir = PAPER / args.queue_logdir
    queue_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = queue_dir / "queue_manifest.json"
    manifest = {
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

    cmd = [
        args.python_bin, "-u", "experiments/control/run_control_protocol.py",
        "--protocol", args.protocol,
        "--logdir", args.logdir,
        "--python-bin", args.python_bin,
    ]
    protocol_log = queue_dir / "protocol.log"
    start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START protocol")
    with protocol_log.open("w") as fp:
        proc = subprocess.run(cmd, cwd=PAPER, stdout=fp, stderr=subprocess.STDOUT)
    manifest["steps"].append({
        "name": "protocol",
        "command": cmd,
        "log": str(protocol_log),
        "returncode": proc.returncode,
        "elapsed_s": time.time() - start,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if proc.returncode != 0:
        manifest["finished_at_epoch"] = time.time()
        manifest["failed_step"] = "protocol"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise SystemExit(proc.returncode)

    if args.aggregate:
        agg_log = queue_dir / "aggregate.log"
        agg_out = PAPER / "experiments" / "out" / "_aggregate_summary.txt"
        start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START aggregate")
        with agg_log.open("w") as err, agg_out.open("w") as out:
            proc = subprocess.run(
                [args.python_bin, "experiments/aggregate_results.py"],
                cwd=PAPER,
                stdout=out,
                stderr=err,
            )
        manifest["steps"].append({
            "name": "aggregate",
                "command": [args.python_bin, "experiments/aggregate_results.py"],
            "stdout": str(agg_out),
            "stderr": str(agg_log),
            "returncode": proc.returncode,
            "elapsed_s": time.time() - start,
        })
        if proc.returncode != 0:
            manifest["finished_at_epoch"] = time.time()
            manifest["failed_step"] = "aggregate"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise SystemExit(proc.returncode)

    manifest["finished_at_epoch"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE")


if __name__ == "__main__":
    main()
