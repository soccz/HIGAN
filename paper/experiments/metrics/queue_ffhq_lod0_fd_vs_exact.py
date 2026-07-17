"""Queue a full-resolution FFHQ finite-difference-vs-exact trilemma run.

The current TMLR package uses FFHQ lod=1/2 robustness because full-resolution
lod=0 needs most of the 8GB GPU. This wrapper waits until the GPU is mostly
free, runs a one-record lod=0 smoke test, and only then runs the full sweep.

It writes outputs under paper/experiments/out/ and does not touch
note/submission or the submission tarball.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "paper" / "experiments" / "metrics" / "run_ffhq_fd_vs_exact.py"
QUEUE_OUT = ROOT / "paper" / "experiments" / "out" / "ffhq_fd_vs_exact_trilemma_lod0_queue"
SMOKE_OUT = ROOT / "paper" / "experiments" / "out" / "ffhq_fd_vs_exact_lod0_smoke"
FULL_OUT = ROOT / "paper" / "experiments" / "out" / "ffhq_fd_vs_exact_trilemma_lod0"

GRID = ["8.0", "5.0", "3.0", "2.0", "1.5", "1.0", "0.7", "0.5",
        "0.3", "0.2", "0.1", "0.05", "0.02", "0.01", "0.005", "0.001"]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_free_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def log_line(log, msg: str) -> None:
    print(f"[{stamp()}] {msg}", file=log, flush=True)


def wait_for_gpu(log, min_free_mib: int, poll_sec: int, stable_polls: int, timeout_sec: int) -> bool:
    start = time.time()
    stable = 0
    while True:
        free = gpu_free_mib()
        if free >= min_free_mib:
            stable += 1
            log_line(log, f"GPU free {free} MiB >= {min_free_mib}; stable {stable}/{stable_polls}")
            if stable >= stable_polls:
                return True
        else:
            stable = 0
            log_line(log, f"GPU free {free} MiB < {min_free_mib}; waiting")

        if timeout_sec and time.time() - start > timeout_sec:
            log_line(log, "timeout reached before GPU became free")
            return False
        time.sleep(poll_sec)


def run_cmd(log, name: str, cmd: list[str]) -> int:
    log_line(log, f"starting {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    log_line(log, f"{name} exit code: {proc.returncode}")
    return proc.returncode


def write_summary(log, status: str, smoke_code: int | None, full_code: int | None) -> None:
    summary = {
        "status": status,
        "timestamp_utc": stamp(),
        "smoke_out": str(SMOKE_OUT.relative_to(ROOT)),
        "full_out": str(FULL_OUT.relative_to(ROOT)),
        "smoke_code": smoke_code,
        "full_code": full_code,
    }
    table = FULL_OUT / "trilemma_table.json"
    if table.exists():
        d = json.loads(table.read_text())
        summary["trilemma_summary"] = d.get("summary")
        rows = d.get("rows", [])
        if rows:
            so = [r["second_order"] for r in rows]
            mag_i = min(range(len(rows)), key=lambda i: so[i]["mean_rel_err"])
            bias_i = min(range(len(rows)), key=lambda i: abs(so[i]["mean_signed_bias"]))
            rank_i = max(range(len(rows)), key=lambda i: so[i]["spearman_vs_exact"])
            summary["headline"] = {
                "magnitude_best": rows[mag_i],
                "bias_best": rows[bias_i],
                "rank_best": rows[rank_i],
            }
    out = QUEUE_OUT / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log_line(log, f"wrote {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-mib", type=int, default=7600)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--stable-polls", type=int, default=2)
    ap.add_argument("--timeout-hours", type=float, default=0.0,
                    help="0 means wait indefinitely.")
    args = ap.parse_args()

    QUEUE_OUT.mkdir(parents=True, exist_ok=True)
    log_path = QUEUE_OUT / "queue.log"
    timeout_sec = int(args.timeout_hours * 3600)
    smoke_code = None
    full_code = None

    with log_path.open("a", buffering=1) as log:
        log_line(log, "queue started")
        log_line(log, f"root={ROOT}")
        log_line(log, f"runner={RUNNER}")
        log_line(log, f"min_free_mib={args.min_free_mib} poll_sec={args.poll_sec}")
        write_summary(log, "waiting_for_gpu", smoke_code, full_code)

        if not wait_for_gpu(log, args.min_free_mib, args.poll_sec, args.stable_polls, timeout_sec):
            write_summary(log, "timeout_waiting_for_gpu", smoke_code, full_code)
            return 2

        smoke_cmd = [
            sys.executable, str(RUNNER),
            "--attrs", "smile",
            "--n-random-dirs", "0",
            "--num-samples", "1",
            "--d-grid", "8.0", "3.0", "0.3", "0.02",
            "--lod-override", "0.0",
            "--out", str(SMOKE_OUT.relative_to(ROOT)),
        ]
        smoke_code = run_cmd(log, "lod0-smoke", smoke_cmd)
        if smoke_code != 0:
            write_summary(log, "smoke_failed", smoke_code, full_code)
            return smoke_code

        full_cmd = [
            sys.executable, str(RUNNER),
            "--attrs", "smile", "age", "pose", "gender", "eyeglasses",
            "--n-random-dirs", "3",
            "--num-samples", "12",
            "--d-grid", *GRID,
            "--lod-override", "0.0",
            "--out", str(FULL_OUT.relative_to(ROOT)),
        ]
        full_code = run_cmd(log, "lod0-full", full_cmd)
        write_summary(log, "complete" if full_code == 0 else "full_failed", smoke_code, full_code)
        return full_code


if __name__ == "__main__":
    raise SystemExit(main())
