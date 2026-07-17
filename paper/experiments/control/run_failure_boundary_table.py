"""Summarize robust-pass and failure-boundary patterns across evidence tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def output_metrics_path(path: str | Path) -> Path:
    p = resolve(path)
    return p if p.suffix == ".json" else p / "metrics.json"


def parse_table_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--tables entries must be label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty table label in: {raw}")
    return label, resolve(path.strip())


def boundary_tags(summary: str, metric: str, display: str) -> list[str]:
    text = f"{summary} {metric} {display}".lower()
    tags = []
    if "matched_pair" in text or "low-risk minus high-risk" in text:
        tags.append("matched_pair_boundary")
    if "strict" in text:
        tags.append("strict_gain_boundary")
    if "bedroom" in text:
        tags.append("bedroom_boundary")
    if "ffhq" in text:
        tags.append("ffhq_boundary")
    if "dino" in text:
        tags.append("dino_preservation_boundary")
    if "random" in text:
        tags.append("random_universe_boundary")
    if "beta" in text:
        tags.append("conditional_regression_boundary")
    if "controller" in text:
        tags.append("controller_boundary")
    if not tags:
        tags.append("general_boundary")
    return tags


def normalize_rows(label: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("rows", []):
        summary = str(row.get("summary", label))
        metric = str(row.get("metric", ""))
        display = str(row.get("display", metric))
        passed = bool(row.get("pass"))
        rows.append({
            "table": label,
            "summary": summary,
            "metric": metric,
            "display": display,
            "mean": row.get("mean"),
            "wins": row.get("wins"),
            "n": row.get("n"),
            "p_sign_two_sided": row.get("p_sign_two_sided"),
            "required": bool(row.get("required", True)),
            "pass": passed,
            "tags": boundary_tags(summary, metric, display),
        })
    return rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        vals = row[key]
        if not isinstance(vals, list):
            vals = [vals]
        for val in vals:
            bucket = out.setdefault(str(val), {"total": 0, "pass": 0, "fail": 0})
            bucket["total"] += 1
            if row["pass"]:
                bucket["pass"] += 1
            else:
                bucket["fail"] += 1
    return out


def write_markdown(path: Path, failures: list[dict[str, Any]],
                   tag_counts: dict[str, dict[str, int]]) -> None:
    lines = [
        "# Failure Boundary Table",
        "",
        "## Boundary Counts",
        "",
        "| Boundary | Total | Pass | Fail |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tag, counts in sorted(tag_counts.items()):
        lines.append(
            f"| {tag} | {counts['total']} | {counts['pass']} | {counts['fail']} |"
        )
    lines.extend([
        "",
        "## Failed Rows",
        "",
        "| Table | Summary | Metric | Mean | Wins | p(sign) | Required | Tags |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for row in failures:
        p = row.get("p_sign_two_sided")
        p_text = f"{p:.4f}" if isinstance(p, (int, float)) else "n/a"
        mean = row.get("mean")
        mean_text = f"{mean:+.4f}" if isinstance(mean, (int, float)) else "n/a"
        wins = row.get("wins", "n/a")
        n = row.get("n", "n/a")
        lines.append(
            "| {table} | {summary} | {metric} | {mean} | {wins}/{n} | {p} | {required} | {tags} |".format(
                table=row["table"],
                summary=row["summary"],
                metric=row["display"],
                mean=mean_text,
                wins=wins,
                n=n,
                p=p_text,
                required="yes" if row["required"] else "no",
                tags=", ".join(row["tags"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4219)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="failure_boundary_table")
    args = ap.parse_args()

    set_deterministic(args.seed)
    all_rows: list[dict[str, Any]] = []
    for label, path in [parse_table_arg(raw) for raw in args.tables]:
        payload = json.loads(path.read_text())
        all_rows.extend(normalize_rows(label, payload))
    failures = [row for row in all_rows if not row["pass"]]
    required_rows = [row for row in all_rows if row["required"]]
    required_failures = [row for row in required_rows if not row["pass"]]
    result = {
        "total_rows": len(all_rows),
        "fail_rows": len(failures),
        "required_rows": len(required_rows),
        "required_fail_rows": len(required_failures),
        "pass_rows": sum(1 for row in all_rows if row["pass"]),
        "tag_counts": count_by(all_rows, "tags"),
        "metric_counts": count_by(all_rows, "display"),
        "failures": failures,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_failure_boundary_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    table_path = out.parent / "failure_boundary_table.md"
    write_markdown(table_path, failures, result["tag_counts"])
    print(f"saved {out}")
    print(f"table={table_path}")
    print(
        f"fail={result['fail_rows']}/{result['total_rows']} "
        f"required_fail={result['required_fail_rows']}/{result['required_rows']}"
    )


if __name__ == "__main__":
    main()
