"""Main-claim post-process: predict edit regime from curvature signatures.

This script converts the existing C1/C2 outputs into a paper-facing
classification table:

  structural/global edit regime vs texture/appearance edit regime.

It deliberately reads only completed metrics.json files. No generator forward
passes are run here; the GPU-heavy measurements remain in the original tracks.
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

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import run_metadata  # noqa: E402

OUT = PAPER / "experiments" / "out"


DOMAINS = {
    "bedroom": {
        "sample_dir": "sample_scaling_bedroom_n256",
        "clip_dir": "bedroom_c2_path",
        "structural": {"view"},
    },
    "ffhq": {
        "sample_dir": "sample_scaling_ffhq_n512",
        "clip_dir": "ffhq_c2_path",
        "structural": {"pose", "eyeglasses"},
    },
    "church": {
        "sample_dir": "sample_scaling_church_n256",
        "clip_dir": None,
        "structural": set(),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_bootstrap_row(metrics: dict[str, Any]) -> dict[str, Any]:
    table = metrics["bootstrap_ci"]
    latest_n = max(int(k) for k in table)
    return table[str(latest_n)]


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


def balanced_accuracy(labels: list[bool], preds: list[bool]) -> float | None:
    present = sorted(set(labels))
    if not present:
        return None
    recalls = []
    for cls in present:
        idx = [i for i, y in enumerate(labels) if y == cls]
        if not idx:
            continue
        recalls.append(sum(preds[i] == cls for i in idx) / len(idx))
    return float(np.mean(recalls)) if recalls else None


def best_threshold(scores: list[float], labels: list[bool]) -> dict[str, Any]:
    vals = sorted(set(scores))
    if not vals:
        return {"threshold": None, "balanced_accuracy": None, "accuracy": None}
    candidates = [vals[0] - 1e-9]
    candidates += [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
    candidates += [vals[-1] + 1e-9]

    best = None
    for th in candidates:
        preds = [s >= th for s in scores]
        bacc = balanced_accuracy(labels, preds)
        acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
        item = {"threshold": th, "balanced_accuracy": bacc, "accuracy": acc}
        if best is None:
            best = item
            continue
        key = (bacc if bacc is not None else -1.0, acc)
        best_key = (
            best["balanced_accuracy"] if best["balanced_accuracy"] is not None else -1.0,
            best["accuracy"],
        )
        if key > best_key:
            best = item
    return best or {"threshold": None, "balanced_accuracy": None, "accuracy": None}


def evaluate_feature(points: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    rows = [p for p in points if p.get(feature) is not None]
    scores = [float(p[feature]) for p in rows]
    labels = [bool(p["is_structural"]) for p in rows]
    threshold = best_threshold(scores, labels)

    lodo = []
    for domain in sorted({p["domain"] for p in rows}):
        train = [p for p in rows if p["domain"] != domain]
        test = [p for p in rows if p["domain"] == domain]
        train_labels = [bool(p["is_structural"]) for p in train]
        if len(set(train_labels)) < 2:
            continue
        th = best_threshold([float(p[feature]) for p in train], train_labels)
        preds = [float(p[feature]) >= th["threshold"] for p in test]
        y = [bool(p["is_structural"]) for p in test]
        lodo.append({
            "heldout_domain": domain,
            "threshold": th["threshold"],
            "n_test": len(test),
            "n_structural_test": int(sum(y)),
            "accuracy": sum(p == yy for p, yy in zip(preds, y)) / len(y),
            "balanced_accuracy_present_classes": balanced_accuracy(y, preds),
            "predicted_structural": [test[i]["name"] for i, p in enumerate(preds) if p],
            "missed_structural": [test[i]["name"] for i, p in enumerate(preds)
                                  if y[i] and not p],
            "false_structural": [test[i]["name"] for i, p in enumerate(preds)
                                 if p and not y[i]],
        })

    return {
        "feature": feature,
        "n": len(rows),
        "n_structural": int(sum(labels)),
        "auroc": roc_auc(scores, labels),
        "best_threshold": threshold,
        "leave_one_domain_out": lodo,
        "mean_lodo_accuracy": (
            float(np.mean([x["accuracy"] for x in lodo])) if lodo else None
        ),
        "mean_lodo_balanced_accuracy_present_classes": (
            float(np.mean([x["balanced_accuracy_present_classes"] for x in lodo]))
            if lodo else None
        ),
    }


def collect_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for domain, cfg in DOMAINS.items():
        sample = load_json(OUT / cfg["sample_dir"] / "metrics.json")
        pixel = latest_bootstrap_row(sample)
        clip_map = {}
        if cfg["clip_dir"]:
            clip = load_json(OUT / cfg["clip_dir"] / "metrics.json")
            clip_map = {e["attr"]: float(e["mean_ratio"])
                        for e in clip.get("per_attr", [])}

        for attr, row in pixel.items():
            pixel_rho = float(row["mean"])
            clip_ratio = clip_map.get(attr)
            is_structural = attr in cfg["structural"]
            point = {
                "domain": domain,
                "attr": attr,
                "name": f"{domain}-{attr}",
                "is_structural": is_structural,
                "label": "structural" if is_structural else "texture_or_appearance",
                "pixel_rho": pixel_rho,
                "log10_pixel_rho": math.log10(max(pixel_rho, 1e-12)),
                "clip_path_ratio": clip_ratio,
                "log10_clip_path_ratio": (
                    math.log10(max(clip_ratio, 1e-12)) if clip_ratio is not None
                    else None
                ),
            }
            points.append(point)
    return points


def write_plot(points: list[dict[str, Any]], metrics: dict[str, Any], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(points, key=lambda p: p["pixel_rho"])
    y = np.arange(len(rows))
    colors = ["#c2410c" if p["is_structural"] else "#2563eb" for p in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=150)
    ax.barh(y, [p["pixel_rho"] for p in rows], color=colors, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([p["name"] for p in rows], fontsize=7)
    ax.set_xlabel("pixel curvature ratio rho (log scale)")
    ax.set_title("Edit-regime prediction from generator curvature")
    th = metrics["features"]["log10_pixel_rho"]["best_threshold"]["threshold"]
    if th is not None:
        ax.axvline(10 ** th, color="#111827", linestyle="--", linewidth=1)
        ax.text(10 ** th, len(rows) - 0.5, " threshold", fontsize=8,
                va="top", ha="left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "edit_regime_prediction.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/out/main_edit_regime_prediction")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PAPER / out
    out.mkdir(parents=True, exist_ok=True)

    points = collect_points()
    feature_metrics = {
        "log10_pixel_rho": evaluate_feature(points, "log10_pixel_rho"),
        "clip_path_ratio": evaluate_feature(points, "clip_path_ratio"),
    }

    payload = {
        "claim": "Generator-side curvature predicts structural/global vs texture/appearance edit regimes.",
        "label_policy": {
            "bedroom": "view is structural/global; the other bedroom attributes are texture/appearance.",
            "ffhq": "pose and eyeglasses are structural/global; smile, age, and gender are texture/appearance.",
            "church": "clouds, sunny, and vegetation are treated as texture/appearance stress-test attributes.",
        },
        "points": points,
        "features": feature_metrics,
        "_meta": run_metadata(extra={
            "script": "experiments/metrics/run_edit_regime_prediction.py",
            "inputs": [DOMAINS[d]["sample_dir"] for d in DOMAINS],
        }),
    }

    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_plot(points, payload, out)

    print("=== edit-regime prediction ===")
    for name, result in feature_metrics.items():
        auc = result["auroc"]
        lodo = result["mean_lodo_accuracy"]
        print(f"{name:20s} n={result['n']:2d} "
              f"AUROC={auc:.3f} "
              f"LODO-acc={lodo:.3f}" if auc is not None and lodo is not None
              else f"{name:20s} insufficient labels")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
