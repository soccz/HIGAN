"""Audit control protocols, outputs, and metadata traceability.

This is a lightweight reproducibility check.  It does not rerun GPU work; it
checks whether every locked protocol entry has a corresponding metrics file,
whether summary inputs exist, and whether result metadata points back to the
declared protocol key and protocol hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


DEFAULT_PROTOCOLS = [
    "experiments/protocols/control_v3_risk_robustness.json",
    "experiments/protocols/control_v4_risk_signal_negatives.json",
    "experiments/protocols/control_v5_risk_power_sensitivity.json",
    "experiments/protocols/control_v6_true_lpips_validation.json",
    "experiments/protocols/control_v7_expanded_candidate_universe.json",
    "experiments/protocols/control_v9_fixed_target_negatives.json",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics_path(exp: dict[str, Any]) -> Path | None:
    out = exp.get("args", {}).get("out")
    if out is None:
        return None
    p = resolve(out)
    return p if p.suffix == ".json" else p / "metrics.json"


def input_paths(exp: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key, value in exp.get("args", {}).items():
        if key == "inputs" or key.endswith("-inputs"):
            if isinstance(value, list):
                paths.extend(resolve(v) for v in value)
            else:
                paths.append(resolve(value))
    return paths


def metadata_checks(payload: dict[str, Any], protocol_sha: str,
                    key: str) -> tuple[list[str], list[str]]:
    meta = payload.get("_meta", {})
    execution = meta.get("execution", {}) if isinstance(meta, dict) else {}
    issues = []
    warnings = []
    if execution.get("protocol_key") != key:
        issues.append(
            f"protocol_key mismatch: got {execution.get('protocol_key')!r}")
    if execution.get("protocol_sha256") != protocol_sha:
        issues.append("protocol_sha256 mismatch")
    if meta.get("deterministic_set") is not True:
        issues.append("deterministic_set is not true")
    if meta.get("git_dirty") is not False:
        warnings.append("git_dirty is true or missing")
    return issues, warnings


def audit_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha256_file(protocol_path)
    rows = []
    counts = {"complete": 0, "pending": 0, "failed": 0}
    for exp in sorted(protocol.get("experiments", []), key=lambda e: e["order"]):
        key = exp["key"]
        script = resolve(exp["script"])
        out_path = metrics_path(exp)
        missing_inputs = [str(p) for p in input_paths(exp) if not p.exists()]
        issues = []
        if not script.exists():
            issues.append(f"missing script: {script}")
        if missing_inputs:
            issues.append(f"missing inputs: {missing_inputs}")

        status = "pending"
        payload_keys: list[str] = []
        warnings: list[str] = []
        if out_path is None:
            issues.append("no output path declared")
        elif out_path.exists():
            try:
                payload = json.loads(out_path.read_text())
                payload_keys = sorted(payload.keys())
                meta_issues, meta_warnings = metadata_checks(
                    payload, protocol_sha, key)
                issues.extend(meta_issues)
                warnings.extend(meta_warnings)
                status = "complete" if not issues else "failed"
            except Exception as exc:
                issues.append(f"metrics json unreadable: {exc}")
                status = "failed"
        else:
            issues.append(f"missing metrics: {out_path}")

        counts[status] += 1
        rows.append({
            "key": key,
            "order": exp["order"],
            "script": exp["script"],
            "metrics": str(out_path) if out_path else None,
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "payload_keys": payload_keys,
            "claim_tested": exp.get("claim_tested"),
        })
    return {
        "protocol": str(protocol_path),
        "protocol_version": protocol.get("version"),
        "protocol_sha256": protocol_sha,
        "counts": counts,
        "rows": rows,
    }


def write_text_report(result: dict[str, Any], out_path: Path) -> None:
    lines = ["HIGAN control campaign audit", ""]
    for prot in result["protocols"]:
        lines.append(f"[{prot['protocol_version']}] {prot['counts']}")
        for row in prot["rows"]:
            if row["status"] != "complete":
                lines.append(f"  {row['status']:7s} {row['key']}: {row['issues']}")
            elif row["warnings"]:
                lines.append(f"  warning {row['key']}: {row['warnings']}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    ap.add_argument("--out", default="experiments/out/control_campaign_audit")
    ap.add_argument("--seed", type=int, default=3701)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="control_campaign_audit")
    args = ap.parse_args()

    set_deterministic(args.seed)
    protocol_paths = [resolve(p) for p in args.protocols]
    missing_protocols = [str(p) for p in protocol_paths if not p.exists()]
    if missing_protocols:
        raise FileNotFoundError(
            "missing protocol path(s): " + ", ".join(missing_protocols)
        )
    audited = [audit_protocol(p) for p in protocol_paths]
    result = {
        "protocols": audited,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/audit_control_campaign.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }

    out = resolve(args.out)
    metrics_out = out if out.suffix == ".json" else out / "metrics.json"
    write_json_atomic(metrics_out, result)
    write_text_report(result, metrics_out.parent / "audit.txt")
    print(f"saved {metrics_out}")
    for prot in audited:
        print(f"{prot['protocol_version']}: {prot['counts']}")


if __name__ == "__main__":
    main()
