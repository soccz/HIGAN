"""Detached queue for the current control validation campaign.

The queue waits for an already-running seed job, resumes the missing actual-risk
seed runs, launches the 10 negative-control GPU runs, then refreshes the
aggregate summary.  It is deliberately explicit rather than dynamic so the
executed protocol sequence is auditable.
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
        "--queue-logdir", args.queue_logdir,
    ]
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


def run_step(name: str, cmd: list[str], log_path: Path) -> dict[str, Any]:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {name}")
    print("  " + " ".join(cmd))
    start = time.time()
    with log_path.open("w") as fp:
        proc = subprocess.run(cmd, cwd=PAPER, stdout=fp, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END {name} rc={proc.returncode} elapsed_s={elapsed:.1f}")
    return {
        "name": name,
        "command": cmd,
        "log": str(log_path),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--queue-logdir", default="logs/control_v3_v4_queue")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()

    if args.detach:
        detach(args)
        return

    logdir = PAPER / args.queue_logdir
    logdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_epoch": time.time(),
        "wait_pid": args.wait_pid,
        "poll_s": args.poll_s,
        "steps": [],
    }
    manifest_path = logdir / "queue_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] waiting for pid {args.wait_pid}")
    while pid_alive(args.wait_pid):
        time.sleep(args.poll_s)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] wait pid exited")

    steps = [
        (
            "control_v3_resume",
            [
                "python3", "-u", "experiments/control/run_control_protocol.py",
                "--protocol", "experiments/protocols/control_v3_risk_robustness.json",
                "--only", "risk_aware_seed_2030", "risk_aware_seed_2031",
                "risk_aware_robustness_summary",
                "--logdir", "logs/control_v3_risk_robustness_resume",
            ],
            logdir / "control_v3_resume.log",
        ),
        (
            "control_v4_risk_signal_negatives",
            [
                "python3", "-u", "experiments/control/run_control_protocol.py",
                "--protocol", "experiments/protocols/control_v4_risk_signal_negatives.json",
                "--logdir", "logs/control_v4_risk_signal_negatives",
            ],
            logdir / "control_v4_risk_signal_negatives.log",
        ),
    ]

    for name, cmd, log_path in steps:
        info = run_step(name, cmd, log_path)
        manifest["steps"].append(info)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if info["returncode"] != 0:
            manifest["finished_at_epoch"] = time.time()
            manifest["failed_step"] = name
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise SystemExit(info["returncode"])

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START aggregate")
    agg_start = time.time()
    agg_log = logdir / "aggregate.log"
    agg_out = PAPER / "experiments" / "out" / "_aggregate_summary.txt"
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
        "elapsed_s": time.time() - agg_start,
    })
    manifest["finished_at_epoch"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE rc={proc.returncode}")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
