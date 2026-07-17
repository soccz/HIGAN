"""Post-process: inject _meta into every metrics.json under
experiments/out/ if missing.

This is the simplest path to paper-grade reproducibility metadata:
instead of modifying 16 scripts to call run_metadata() at save time,
we re-walk the output tree once at the end and inject _meta into
any JSON that doesn't have it.

The _meta we inject is the CURRENT run state (git commit, torch
version, GPU name) plus a placeholder seed=2027 (which matches the
universal --seed default).

Idempotent: doesn't overwrite existing _meta.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import run_metadata


def inject(json_path: Path, seed: int = 2027) -> str:
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        return f"  ✗ {json_path}: unreadable ({e})"
    if not isinstance(data, (dict, list)):
        return f"  ⊘ {json_path}: not dict/list, skip"
    if isinstance(data, dict) and "_meta" in data:
        return f"  ok {json_path}: already has _meta"
    meta = run_metadata(seed=seed,
                         extra={"track": json_path.parent.name,
                                "injected_by": "lib/inject_meta.py"})
    if isinstance(data, list):
        # wrap list in dict
        data = {"results": data, "_meta": meta}
    else:
        data["_meta"] = meta
    json_path.write_text(json.dumps(data, indent=2))
    return f"  + {json_path}: _meta injected"


def main():
    out_dir = PAPER / "experiments" / "out"
    if not out_dir.exists():
        print("No experiments/out — nothing to do")
        return
    n_total = 0
    n_added = 0
    n_already = 0
    n_fail = 0
    for jp in sorted(out_dir.rglob("*.json")):
        if "metrics_partial" in jp.name:
            continue
        n_total += 1
        msg = inject(jp)
        print(msg)
        if msg.startswith("  +"): n_added += 1
        elif msg.startswith("  ok"): n_already += 1
        elif msg.startswith("  ✗"): n_fail += 1
    print(f"\n=== {n_total} JSON files: "
          f"{n_added} injected, {n_already} already, {n_fail} failed ===")


if __name__ == "__main__":
    main()
