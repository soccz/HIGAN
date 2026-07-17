"""Test whether rho adds predictive value beyond gain/edit covariates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


CONTINUOUS_BASE = [
    "mean_abs_delta_attr",
    "probe_gain",
    "abs_alpha",
    "calib_max_abs_delta_attr",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def output_metrics_path(path: str | Path) -> Path:
    p = resolve(path)
    return p if p.suffix == ".json" else p / "metrics.json"


def parse_summary_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--summaries entries must be label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty summary label in: {raw}")
    return label, resolve(path.strip())


def collect_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attr, payload in metrics.get("per_attr", {}).items():
        for row in payload.get("rows", []):
            if not row.get("gain_eligible", False):
                continue
            out = dict(row)
            out["attr"] = attr
            out["abs_alpha"] = abs(float(row.get("alpha", 0.0)))
            out["target_reached_on_calib_float"] = (
                1.0 if row.get("target_reached_on_calib") else 0.0
            )
            rows.append(out)
    return rows


def one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    cats = sorted(set(values))
    if len(cats) <= 1:
        return np.zeros((len(values), 0), dtype=float), []
    cols = []
    names = []
    for cat in cats[1:]:
        cols.append([1.0 if v == cat else 0.0 for v in values])
        names.append(cat)
    return np.asarray(cols, dtype=float).T, names


def design_matrix(rows: list[dict[str, Any]], include_rho: bool) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    for key in CONTINUOUS_BASE:
        cols.append([float(r.get(key, float("nan"))) for r in rows])
        names.append(key)
    cols.append([float(r.get("target_reached_on_calib_float", 0.0)) for r in rows])
    names.append("target_reached_on_calib")

    source, source_names = one_hot([str(r["source_group"]) for r in rows])
    attr, attr_names = one_hot([str(r["attr"]) for r in rows])
    if source.shape[1]:
        cols.extend(source[:, idx].tolist() for idx in range(source.shape[1]))
        names.extend(f"source={name}" for name in source_names)
    if attr.shape[1]:
        cols.extend(attr[:, idx].tolist() for idx in range(attr.shape[1]))
        names.extend(f"attr={name}" for name in attr_names)
    if include_rho:
        cols.append([float(r["rho"]) for r in rows])
        names.append("rho")
    x = np.asarray(cols, dtype=float).T
    return x, names


def standardize_train_test(x_train: np.ndarray,
                           x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    keep = np.isfinite(mean) & np.isfinite(std) & (std > 1e-12)
    x_train = (x_train[:, keep] - mean[keep]) / std[keep]
    x_test = (x_test[:, keep] - mean[keep]) / std[keep]
    return x_train, x_test, keep


def fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray) -> np.ndarray:
    design_train = np.concatenate(
        [np.ones((x_train.shape[0], 1)), x_train], axis=1)
    design_test = np.concatenate(
        [np.ones((x_test.shape[0], 1)), x_test], axis=1)
    coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    return design_test @ coef


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(((y_true - y_true.mean()) ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    num = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - num / denom


def cv_delta_r2(rows: list[dict[str, Any]], y: np.ndarray,
                seed: int, folds: int) -> dict[str, float]:
    x_base, _ = design_matrix(rows, include_rho=False)
    x_full, _ = design_matrix(rows, include_rho=True)
    finite = np.isfinite(y) & np.isfinite(x_base).all(axis=1) & np.isfinite(x_full).all(axis=1)
    y = y[finite]
    x_base = x_base[finite]
    x_full = x_full[finite]
    if y.size < max(8, folds * 2):
        return {"n": int(y.size), "base_cv_r2": float("nan"),
                "full_cv_r2": float("nan"), "delta_cv_r2": float("nan")}

    rng = np.random.default_rng(seed)
    order = rng.permutation(y.size)
    fold_ids = np.array_split(order, min(folds, y.size))
    base_scores = []
    full_scores = []
    for test_idx in fold_ids:
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[test_idx] = False
        y_train = y[train_mask]
        y_test = y[test_idx]
        if float(y_test.std()) <= 1e-12:
            continue
        xb_train, xb_test, _ = standardize_train_test(
            x_base[train_mask], x_base[test_idx])
        xf_train, xf_test, _ = standardize_train_test(
            x_full[train_mask], x_full[test_idx])
        base_scores.append(r2_score(y_test, fit_predict(xb_train, y_train, xb_test)))
        full_scores.append(r2_score(y_test, fit_predict(xf_train, y_train, xf_test)))
    base_arr = np.asarray(base_scores, dtype=float)
    full_arr = np.asarray(full_scores, dtype=float)
    mask = np.isfinite(base_arr) & np.isfinite(full_arr)
    if not mask.any():
        return {"n": int(y.size), "base_cv_r2": float("nan"),
                "full_cv_r2": float("nan"), "delta_cv_r2": float("nan")}
    base_mean = float(base_arr[mask].mean())
    full_mean = float(full_arr[mask].mean())
    return {
        "n": int(y.size),
        "base_cv_r2": base_mean,
        "full_cv_r2": full_mean,
        "delta_cv_r2": full_mean - base_mean,
    }


def full_model_beta_rho(rows: list[dict[str, Any]], y: np.ndarray) -> float:
    x, names = design_matrix(rows, include_rho=True)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[finite]
    x = x[finite]
    if y.size <= x.shape[1] + 1 or float(y.std()) <= 1e-12:
        return float("nan")
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    keep = x_std > 1e-12
    if "rho" not in names:
        return float("nan")
    rho_idx = names.index("rho")
    if not keep[rho_idx]:
        return float("nan")
    xz = (x[:, keep] - x_mean[keep]) / x_std[keep]
    yz = (y - y.mean()) / y.std()
    design = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    kept_names = [name for name, ok in zip(names, keep) if ok]
    return float(dict(zip(kept_names, coef[1:])).get("rho", float("nan")))


def summarize(values: list[float], orientation: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": std,
        "sem": sem,
        "ci95_normal": [
            float(arr.mean() - 1.96 * sem),
            float(arr.mean() + 1.96 * sem),
        ],
        "orientation": orientation,
    }
    if orientation == "higher":
        out["wins"] = int((arr > 0).sum())
    return out


def exact_sign_p(wins: Any, n: Any) -> float | None:
    if not isinstance(wins, int) or not isinstance(n, int) or n <= 0:
        return None
    lo = sum(math.comb(n, k) for k in range(0, wins + 1)) / (2 ** n)
    hi = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(lo, hi)))


def target_values(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    targets = {
        "lpips_damage": np.asarray([r.get("mean_lpips_true", r.get("mean_lpips_proxy"))
                                    for r in rows], dtype=float),
        "clip_id_damage": np.asarray([1.0 - float(r["mean_id_cos"])
                                      for r in rows], dtype=float),
    }
    if rows and all("mean_dino_cos" in r for r in rows):
        targets["dino_damage"] = np.asarray(
            [1.0 - float(r["mean_dino_cos"]) for r in rows], dtype=float)
    return targets


def label_from_path(path: Path) -> str:
    return str(path.relative_to(PAPER)) if path.is_relative_to(PAPER) else str(path)


def collect_seed_rows(summary_label: str, summary_path: Path,
                      folds: int, seed: int) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    out = []
    for offset, raw_path in enumerate(summary.get("inputs", [])):
        metrics_path = resolve(raw_path)
        metrics = json.loads(metrics_path.read_text())
        rows = collect_rows(metrics)
        for target, y in target_values(rows).items():
            cv = cv_delta_r2(rows, y, seed + offset * 17 + len(out), folds)
            beta = full_model_beta_rho(rows, y)
            out.append({
                "summary": summary_label,
                "source": label_from_path(metrics_path),
                "domain": metrics.get("config", {}).get("domain"),
                "seed": metrics.get("config", {}).get("seed"),
                "target": target,
                "n_rows": cv["n"],
                "base_cv_r2": cv["base_cv_r2"],
                "full_cv_r2": cv["full_cv_r2"],
                "delta_cv_r2": cv["delta_cv_r2"],
                "beta_rho_full": beta,
            })
    return out


def grouped_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(r["summary"], r["domain"], r["target"]) for r in seed_rows})
    out = []
    for summary, domain, target in groups:
        rows = [
            r for r in seed_rows
            if r["summary"] == summary and r["domain"] == domain
            and r["target"] == target
        ]
        delta = summarize([r["delta_cv_r2"] for r in rows], "higher")
        beta = summarize([r["beta_rho_full"] for r in rows], "higher")
        out.append({
            "summary": summary,
            "domain": domain,
            "target": target,
            "delta_cv_r2": {
                **delta,
                "p_sign_two_sided": exact_sign_p(delta.get("wins"), delta.get("n")),
            },
            "beta_rho_full": {
                **beta,
                "p_sign_two_sided": exact_sign_p(beta.get("wins"), beta.get("n")),
            },
            "pass": (
                delta.get("n", 0) >= 10
                and delta.get("mean", 0.0) > 0
                and delta.get("wins", 0) >= math.ceil(0.8 * delta.get("n", 0))
                and beta.get("mean", 0.0) > 0
                and beta.get("wins", 0) >= math.ceil(0.8 * beta.get("n", 0))
            ),
        })
    return out


def format_float(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):+.{precision}f}"
    return "n/a"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Summary | Domain | Target | mean ΔCV-R2 | Δ wins | mean beta_rho | beta wins | Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        delta = row["delta_cv_r2"]
        beta = row["beta_rho_full"]
        lines.append(
            "| {summary} | {domain} | {target} | {delta} | {dw}/{dn} | {beta} | {bw}/{bn} | {passed} |".format(
                summary=row["summary"],
                domain=row["domain"],
                target=row["target"],
                delta=format_float(delta.get("mean")),
                dw=delta.get("wins", "n/a"),
                dn=delta.get("n", "n/a"),
                beta=format_float(beta.get("mean")),
                bw=beta.get("wins", "n/a"),
                bn=beta.get("n", "n/a"),
                passed="yes" if row["pass"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def readiness_label(pass_count: int, total: int) -> str:
    if pass_count == total:
        return "rho_incremental_value_ready"
    if pass_count >= total - 3:
        return "rho_incremental_value_borderline"
    return "rho_incremental_value_sensitive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4215)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="predictive_incremental_value_table")
    args = ap.parse_args()

    set_deterministic(args.seed)
    seed_rows: list[dict[str, Any]] = []
    for label, path in [parse_summary_arg(raw) for raw in args.summaries]:
        seed_rows.extend(collect_seed_rows(label, path, args.folds, args.seed))
    rows = grouped_summary(seed_rows)
    pass_count = sum(1 for row in rows if row["pass"])
    result = {
        "readiness": readiness_label(pass_count, len(rows)),
        "pass_count": pass_count,
        "total": len(rows),
        "rows": rows,
        "seed_rows": seed_rows,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_predictive_incremental_value_table.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    out = output_metrics_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, result)
    table_path = out.parent / "predictive_incremental_value_table.md"
    write_markdown(table_path, rows)
    print(f"saved {out}")
    print(f"table={table_path}")
    print(f"readiness={result['readiness']} pass={pass_count}/{len(rows)}")


if __name__ == "__main__":
    main()
