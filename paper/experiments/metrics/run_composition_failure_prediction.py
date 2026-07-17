"""Main-claim post-process: predict compositional edit failure.

Reads the existing C4 mixed-Hessian outputs and turns them into a binary
"high composition nonlinearity" prediction result. The failure label is derived
from the observed saliency-superposition error; the score is the theoretical
mixed-Hessian predictor. A naive baseline using only univariate C1 curvature is
reported for comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import run_metadata  # noqa: E402

OUT = PAPER / "experiments" / "out"


DOMAIN_INPUTS = {
    "bedroom": {
        "c4_dir": "bedroom_c4",
        "sample_dir": "sample_scaling_bedroom_n256",
    },
    "ffhq": {
        "c4_dir": "ffhq_c4",
        "sample_dir": "sample_scaling_ffhq_n512",
    },
    "church": {
        "c4_dir": "church_c4",
        "sample_dir": "sample_scaling_church_n256",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_pixel_rhos(sample_dir: str) -> dict[str, float]:
    metrics = load_json(OUT / sample_dir / "metrics.json")
    table = metrics["bootstrap_ci"]
    latest_n = max(int(k) for k in table)
    return {attr: float(row["mean"]) for attr, row in table[str(latest_n)].items()}


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += float(p > n) + 0.5 * float(p == n)
    return wins / (len(pos) * len(neg))


def percentile_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    order = np.argsort(np.argsort(np.asarray(values)))
    return (order / (len(values) - 1)).astype(float).tolist()


def safe_corr(fn, x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"r": None, "p": None}
    r, p = fn(x, y)
    return {"r": float(r), "p": float(p)}


def domain_rows(domain: str) -> list[dict[str, Any]]:
    cfg = DOMAIN_INPUTS[domain]
    c4 = load_json(OUT / cfg["c4_dir"] / "metrics.json")
    pixel = latest_pixel_rhos(cfg["sample_dir"])
    rows = []
    for row in c4["pairs"]:
        a, b = row["a"], row["b"]
        max_rho = max(pixel[a], pixel[b])
        sum_rho = pixel[a] + pixel[b]
        rows.append({
            "domain": domain,
            "pair": f"{a}+{b}",
            "a": a,
            "b": b,
            "nonlinearity": float(row["nonlinearity"]),
            "mixed_hessian_predictor": float(row["predictor"]),
            "max_univariate_pixel_rho": float(max_rho),
            "sum_univariate_pixel_rho": float(sum_rho),
            "corr_saliency_sum": float(row["corr"]),
        })
    return rows


def summarize_domain(rows: list[dict[str, Any]], failure_quantile: float) -> dict[str, Any]:
    y = [r["nonlinearity"] for r in rows]
    threshold = float(np.quantile(y, failure_quantile))
    labels = [v >= threshold for v in y]
    pred = [r["mixed_hessian_predictor"] for r in rows]
    max_rho = [r["max_univariate_pixel_rho"] for r in rows]
    sum_rho = [r["sum_univariate_pixel_rho"] for r in rows]
    return {
        "n_pairs": len(rows),
        "failure_quantile": failure_quantile,
        "failure_threshold_nonlinearity": threshold,
        "n_failures": int(sum(labels)),
        "spearman_mixed_hessian": safe_corr(spearmanr, pred, y),
        "pearson_mixed_hessian": safe_corr(pearsonr, pred, y),
        "spearman_max_univariate_pixel_rho": safe_corr(spearmanr, max_rho, y),
        "spearman_sum_univariate_pixel_rho": safe_corr(spearmanr, sum_rho, y),
        "auroc_mixed_hessian": roc_auc(pred, labels),
        "auroc_max_univariate_pixel_rho": roc_auc(max_rho, labels),
        "auroc_sum_univariate_pixel_rho": roc_auc(sum_rho, labels),
    }


def pooled_summary(all_rows: list[dict[str, Any]], failure_quantile: float) -> dict[str, Any]:
    rows = []
    for domain in sorted({r["domain"] for r in all_rows}):
        dr = [r for r in all_rows if r["domain"] == domain]
        y = [r["nonlinearity"] for r in dr]
        pred = percentile_ranks([r["mixed_hessian_predictor"] for r in dr])
        max_rho = percentile_ranks([r["max_univariate_pixel_rho"] for r in dr])
        sum_rho = percentile_ranks([r["sum_univariate_pixel_rho"] for r in dr])
        threshold = float(np.quantile(y, failure_quantile))
        for r, p, m, s in zip(dr, pred, max_rho, sum_rho):
            rr = dict(r)
            rr["failure_label"] = r["nonlinearity"] >= threshold
            rr["mixed_hessian_percentile"] = p
            rr["max_univariate_percentile"] = m
            rr["sum_univariate_percentile"] = s
            rows.append(rr)
    labels = [bool(r["failure_label"]) for r in rows]
    return {
        "n_pairs": len(rows),
        "n_failures": int(sum(labels)),
        "auroc_mixed_hessian_percentile": roc_auc(
            [r["mixed_hessian_percentile"] for r in rows], labels),
        "auroc_max_univariate_percentile": roc_auc(
            [r["max_univariate_percentile"] for r in rows], labels),
        "auroc_sum_univariate_percentile": roc_auc(
            [r["sum_univariate_percentile"] for r in rows], labels),
        "rows": rows,
    }


def write_plot(rows: list[dict[str, Any]], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"bedroom": "#2563eb", "ffhq": "#c2410c", "church": "#16a34a"}
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    for domain in sorted(colors):
        dr = [r for r in rows if r["domain"] == domain]
        if not dr:
            continue
        x = [r["mixed_hessian_predictor"] for r in dr]
        y = [r["nonlinearity"] for r in dr]
        ax.scatter(x, y, s=58, alpha=0.82, label=domain,
                   color=colors[domain], edgecolors="white", linewidths=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("mixed-Hessian predictor")
    ax.set_ylabel("observed composition nonlinearity")
    ax.set_title("Composition failure prediction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "composition_failure_prediction.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failure-quantile", type=float, default=0.5,
                    help="Pairs at or above this within-domain nonlinearity quantile are failures.")
    ap.add_argument("--out", default="experiments/out/main_composition_failure_prediction")
    args = ap.parse_args()

    if not 0.0 < args.failure_quantile < 1.0:
        raise ValueError("--failure-quantile must be in (0, 1)")

    out = Path(args.out)
    if not out.is_absolute():
        out = PAPER / out
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_domain = {}
    for domain in DOMAIN_INPUTS:
        rows = domain_rows(domain)
        all_rows.extend(rows)
        by_domain[domain] = summarize_domain(rows, args.failure_quantile)
    pooled = pooled_summary(all_rows, args.failure_quantile)

    payload = {
        "claim": "The mixed-Hessian predictor identifies edit pairs with high saliency-superposition failure.",
        "failure_label": (
            "within-domain observed nonlinearity >= quantile "
            f"{args.failure_quantile}"
        ),
        "by_domain": by_domain,
        "pooled_within_domain_percentiles": pooled,
        "rows": all_rows,
        "_meta": run_metadata(extra={
            "script": "experiments/metrics/run_composition_failure_prediction.py",
            "inputs": [DOMAIN_INPUTS[d]["c4_dir"] for d in DOMAIN_INPUTS],
        }),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_plot(all_rows, out)

    print("=== composition-failure prediction ===")
    for domain, result in by_domain.items():
        sp = result["spearman_mixed_hessian"]["r"]
        auc = result["auroc_mixed_hessian"]
        base = result["auroc_max_univariate_pixel_rho"]
        print(f"{domain:8s} Spearman={sp:+.3f} "
              f"AUROC={auc:.3f} baseline-max-rho={base:.3f}")
    print("pooled percentile AUROC="
          f"{pooled['auroc_mixed_hessian_percentile']:.3f}")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
