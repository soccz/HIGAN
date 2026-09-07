"""Predict finite mixed responses using only permitted observations.

Target singles are permitted; target joint responses are labels, never inputs.
The calibrated secant is a baseline, not a claimed novel estimator.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import torch

from measure import rms

VECTOR_METHODS = (
    "additive",
    "local_mixed",
    "probe_constant",
    "probe_linear",
    "probe_quadratic",
    "probe_calibrated",
)
RISK_METHODS = VECTOR_METHODS + ("scale_prior", "overlap_ridge")


def single_features(a, b, scale):
    an, bn = float(rms(a)), float(rms(b))
    denominator = an + bn
    if denominator <= 1e-5 or min(an, bn) <= 1e-6:
        raise ValueError("Single responses unresolved; no relative prediction claim")
    cosine = max(-1.0, min(1.0, float((a * b).mean()) / (an * bn)))
    return denominator, [
        math.log(denominator / scale),
        cosine,
        abs(cosine),
        2 * min(an, bn) / denominator,
    ]


def vector_predictions(probe_c, mixed, scale, probe_scale, signs, coefficient):
    if not (
        scale > probe_scale > 0 and math.isfinite(coefficient) and coefficient >= 0
    ):
        raise ValueError("Require extrapolation and a finite nonnegative coefficient")
    if any(s not in (-1, 1) for s in signs) or len(signs) != 2:
        raise ValueError("Two signs required")
    if probe_c.shape != mixed.shape or not all(
        torch.isfinite(v).all() for v in (probe_c, mixed)
    ):
        raise ValueError("Finite matching tensors required")
    ratio = scale / probe_scale
    return {
        "additive": torch.zeros_like(probe_c),
        "local_mixed": signs[0] * signs[1] * scale**2 * mixed,
        "probe_constant": probe_c,
        "probe_linear": ratio * probe_c,
        "probe_quadratic": ratio**2 * probe_c,
        "probe_calibrated": coefficient * probe_c,
    }


def fit_model(rows, protocol):
    observed = {row["seed"] for row in rows}
    if observed != set(protocol["development_seeds"]) or observed & set(
        protocol["evaluation_seeds"]
    ):
        raise ValueError("Training rows must contain exactly development anchors")
    result = {"scope": "fitted on development anchors only", "by_scale": {}}
    for scale in protocol["target_scales"]:
        group = [r for r in rows if r["scale"] == scale]
        numerator = sum(r["normalized_probe_target_dot"] for r in group)
        denominator = sum(r["normalized_probe_squared"] for r in group)
        if not group or denominator <= 0:
            raise ValueError("Nonzero probe signal required for calibration")
        x = np.asarray([r["single_features"] for r in group], dtype=np.float64)
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std < 1e-12] = 1
        design = np.column_stack([np.ones(len(x)), (x - mean) / std])
        y = np.log(np.maximum([r["actual_relative"] for r in group], 1e-12))
        penalty = np.eye(design.shape[1]) * protocol["ridge_penalty"]
        penalty[0, 0] = 0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        result["by_scale"][str(scale)] = {
            "coefficient": max(0.0, numerator / denominator),
            "scale_prior": statistics.median(r["actual_relative"] for r in group),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "ridge_weights": weights.tolist(),
            "training_rows": len(group),
        }
    return result


def risk_predictions(vectors, denominator, features, fitted):
    x = (np.asarray(features) - fitted["feature_mean"]) / fitted["feature_std"]
    log_prediction = float(np.dot(np.r_[1.0, x], fitted["ridge_weights"]))
    return {
        **{name: float(rms(value)) / denominator for name, value in vectors.items()},
        "scale_prior": fitted["scale_prior"],
        "overlap_ridge": math.exp(max(-20.0, min(5.0, log_prediction))),
    }


def evaluate_response(actual_c, vectors, risks, denominator, atol, rtol):
    actual = float(rms(actual_c)) / denominator
    threshold = rtol + atol / denominator
    acceptable = actual <= threshold
    return {
        "actual_relative": actual,
        "actual_additive": acceptable,
        "relative_errors": {
            name: float(rms(actual_c - value)) / denominator
            for name, value in vectors.items()
        },
        "adopt": {name: risk <= threshold for name, risk in risks.items()},
        "risk_absolute_errors": {
            name: abs(risk - actual) for name, risk in risks.items()
        },
    }


def summarize_rows(rows, protocol):
    def group_summary(group):
        risk = {}
        for method in RISK_METHODS:
            adopted = [r for r in group if r["adopt"][method]]
            wrong = sum(not r["actual_additive"] for r in adopted)
            risk[method] = {
                "adoption_rate": len(adopted) / len(group),
                "false_adoption_rate_among_adopted": (
                    wrong / len(adopted) if adopted else None
                ),
                "false_adoption_rate_all": wrong / len(group),
                "mean_absolute_risk_error": statistics.mean(
                    r["risk_absolute_errors"][method] for r in group
                ),
            }
        return {
            "n": len(group),
            "actual_additive_rate": statistics.mean(
                r["actual_additive"] for r in group
            ),
            "mean_relative_error": {
                m: statistics.mean(r["relative_errors"][m] for r in group)
                for m in VECTOR_METHODS
            },
            "risk": risk,
        }

    by_anchor = {
        str(seed): group_summary([r for r in rows if r["seed"] == seed])
        for seed in protocol["evaluation_seeds"]
    }
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    count = len(by_anchor)
    draws = rng.integers(count, size=(protocol["bootstrap_repetitions"], count))
    errors = {
        m: np.asarray([g["mean_relative_error"][m] for g in by_anchor.values()])
        for m in VECTOR_METHODS
    }
    comparisons = {}
    candidate = errors["probe_calibrated"]
    for method in VECTOR_METHODS:
        if method == "probe_calibrated":
            continue
        difference = errors[method] - candidate
        interval = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975]).tolist()
        relative_gain = float(difference.mean() / errors[method].mean())
        comparisons[method] = {
            "mean_baseline_minus_calibrated": float(difference.mean()),
            "paired_anchor_bootstrap_95_percent": interval,
            "relative_error_reduction": relative_gain,
            "passes_preregistered_gate": relative_gain
            >= protocol["minimum_relative_improvement"]
            and interval[0] > 0,
        }
    return {
        "status": "COMPLETE",
        "scope": "target singles known; target joint withheld until predictions are saved",
        "overall": group_summary(rows),
        "by_scale": {
            str(s): group_summary([r for r in rows if r["scale"] == s])
            for s in protocol["target_scales"]
        },
        "by_anchor": by_anchor,
        "without_view": group_summary(
            [r for r in rows if "view" not in (r["a"], r["b"])]
        ),
        "comparisons": comparisons,
        "extension_gate_passed": all(
            v["passes_preregistered_gate"] for v in comparisons.values()
        ),
        "uncertainty_scope": f"{count} anchor clusters; unadjusted descriptive 95% intervals; no row independence or safety guarantee",
    }
