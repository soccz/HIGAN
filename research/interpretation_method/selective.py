"""Direct mixed-response risk prediction and selective additive explanations.

All fit/calibration sets are anchor-disjoint. Bootstrap intervals are empirical
cluster intervals, not distribution-free or per-case guarantees.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from run_bedroom import rms, torch
from prediction import single_features

LEARNED = ("overlap", "near_ridge", "two_amplitude", "two_direction")
METHODS = LEARNED + ("near_linear", "near_quadratic")


def configure(gpu=False):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use the approved execution environment"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def key(row):
    return tuple(row[k] for k in ("seed", "a", "b", "scale", "sign_a", "sign_b"))


def features(a, b, c_small, c_near, scale):
    denominator, singles = single_features(a, b, scale)
    n1, n2 = float(rms(c_small)), float(rms(c_near))
    cosine = (
        float((c_small * c_near).mean()) / (n1 * n2) if min(n1, n2) > 1e-12 else 0.0
    )
    return {
        "denominator": denominator,
        "single_features": singles,
        "probe_relative_norms": [n1 / denominator, n2 / denominator],
        "probe_cosine": max(-1.0, min(1.0, cosine)),
    }


def feature_vector(record, method):
    base = list(record["single_features"])
    logs = [math.log(max(v, 1e-12)) for v in record["probe_relative_norms"]]
    if method == "overlap":
        return base
    if method == "near_ridge":
        return base + logs[1:]
    if method == "two_amplitude":
        return base + logs
    if method == "two_direction":
        return base + logs + [record["probe_cosine"]]
    raise ValueError(f"Unknown learned method {method}")


def require_seeds(rows, p, phase):
    observed = {r["seed"] for r in rows}
    if observed != set(p[f"{phase}_seeds"]):
        raise ValueError(f"Rows must contain exactly the {phase} anchors")
    if len(rows) != len(observed) * p["expected_rows_per_anchor"]:
        raise ValueError(f"Incomplete {phase} grid")


def fit(rows, p):
    require_seeds(rows, p, "training")
    model = {"by_scale": {}, "fit_scope": "training anchors only; fixed ridge penalty"}
    for scale in p["target_scales"]:
        group = [r for r in rows if r["scale"] == scale]
        y = np.log(np.maximum([r["actual_relative"] for r in group], 1e-12))
        model["by_scale"][str(scale)] = {}
        for method in LEARNED:
            x = np.asarray([feature_vector(r, method) for r in group])
            mean, std = x.mean(axis=0), x.std(axis=0)
            std[std < 1e-12] = 1
            design = np.column_stack([np.ones(len(group)), (x - mean) / std])
            penalty = np.eye(design.shape[1]) * p["ridge_penalty"]
            penalty[0, 0] = 0
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
            model["by_scale"][str(scale)][method] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "weights": weights.tolist(),
            }
    return model


def predict(record, model, p):
    scores = {}
    for method in LEARNED:
        fitted = model["by_scale"][str(record["scale"])][method]
        x = (np.asarray(feature_vector(record, method)) - fitted["mean"]) / fitted[
            "std"
        ]
        log_risk = float(np.dot(np.r_[1, x], fitted["weights"]))
        scores[method] = math.exp(max(-27.0, min(5.0, log_risk)))
    near = record["probe_relative_norms"][1]
    ratio = record["scale"] / p["probe_scales"][1]
    scores.update(near_linear=ratio * near, near_quadratic=ratio**2 * near)
    if not all(math.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError("Nonfinite risk prediction")
    return scores


def bad(row, p):
    return (
        row["actual_relative"]
        > p["relative_tolerance"] + p["absolute_tolerance"] / row["denominator"]
    )


def calibrate(rows, p):
    require_seeds(rows, p, "calibration")
    thresholds = {}
    for method in METHODS:
        best = {
            "threshold": -1.0,
            "adopted": 0,
            "wrong": 0,
            "coverage": 0.0,
            "false_adoption": None,
        }
        for threshold in p["threshold_grid"]:
            selected = [r for r in rows if r["scores"][method] <= threshold]
            wrong = sum(bad(r, p) for r in selected)
            if (
                selected
                and wrong / len(selected) <= p["calibration_false_adoption_target"]
            ):
                if len(selected) > best["adopted"]:
                    best = {
                        "threshold": threshold,
                        "adopted": len(selected),
                        "wrong": wrong,
                        "coverage": len(selected) / len(rows),
                        "false_adoption": wrong / len(selected),
                    }
        thresholds[method] = best
    return {
        "methods": thresholds,
        "scope": "empirical threshold selection on separate calibration anchors; no conditional-risk guarantee",
    }


def fixed_selection(rows, method, fraction):
    mask = np.zeros(len(rows), dtype=bool)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["seed"], row["scale"]), []).append(i)
    for indices in groups.values():

        def order(i):
            tie = hashlib.sha256(str(key(rows[i])).encode()).hexdigest()
            return rows[i]["scores"][method], tie

        selected = sorted(indices, key=order)[: int(len(indices) * fraction)]
        mask[selected] = True
    return mask


def summarize(rows, p, thresholds):
    seeds = sorted({r["seed"] for r in rows})
    methods = {}
    rng = np.random.default_rng(p["bootstrap_seed"])
    draws = rng.integers(len(seeds), size=(p["bootstrap_repetitions"], len(seeds)))
    errors_by_method, fixed_risk_by_method = {}, {}
    truth = np.asarray([bad(r, p) for r in rows])
    indices = {
        seed: np.asarray([i for i, r in enumerate(rows) if r["seed"] == seed])
        for seed in seeds
    }
    for method in METHODS:
        errors = np.asarray(
            [abs(r["scores"][method] - r["actual_relative"]) for r in rows]
        )
        fixed = fixed_selection(rows, method, p["fixed_coverage"])
        selected = np.asarray(
            [
                r["scores"][method] <= thresholds["methods"][method]["threshold"]
                for r in rows
            ]
        )
        e, f, accepted, wrong = [], [], [], []
        by_anchor = {}
        for seed in seeds:
            idx = indices[seed]
            mae = float(errors[idx].mean())
            fr = float(truth[idx][fixed[idx]].mean())
            a, w = int(selected[idx].sum()), int((truth[idx] & selected[idx]).sum())
            e.append(mae)
            f.append(fr)
            accepted.append(a)
            wrong.append(w)
            by_anchor[str(seed)] = {
                "mae": mae,
                "fixed_coverage_risk": fr,
                "accepted": a,
                "wrong": w,
            }
        e, f, accepted, wrong = [np.asarray(v) for v in (e, f, accepted, wrong)]
        boot_a, boot_w = accepted[draws].sum(axis=1), wrong[draws].sum(axis=1)
        valid = boot_a > 0
        upper = (
            float(np.quantile(boot_w[valid] / boot_a[valid], 0.95))
            if valid.any()
            else None
        )
        count, wrong_count = int(accepted.sum()), int(wrong.sum())
        coverage = count / len(rows)
        methods[method] = {
            "mean_absolute_risk_error": float(e.mean()),
            "fixed_coverage": float(fixed.mean()),
            "fixed_coverage_risk": float(f.mean()),
            "threshold": thresholds["methods"][method]["threshold"],
            "coverage": coverage,
            "accepted": count,
            "wrong": wrong_count,
            "false_adoption": wrong_count / count if count else None,
            "cluster_bootstrap_risk_upper_95": upper,
            "valid_bootstrap_draws": int(valid.sum()),
            "empirical_practical_gate": coverage >= p["minimum_coverage"]
            and upper is not None
            and upper <= p["evaluation_false_adoption_target"],
            "by_anchor": by_anchor,
        }
        errors_by_method[method], fixed_risk_by_method[method] = e, f
    comparisons = {}
    candidate = errors_by_method["two_direction"]
    for baseline in ("overlap", "near_ridge", "two_amplitude"):
        reference = errors_by_method[baseline]
        resolved = bool(np.all(reference > 1e-15))
        gain_draws = (
            1 - candidate[draws].mean(axis=1) / reference[draws].mean(axis=1)
            if resolved
            else None
        )
        gain_interval = (
            np.quantile(gain_draws, [0.025, 0.975]).tolist()
            if resolved
            else [None, None]
        )
        risk_delta = (
            fixed_risk_by_method[baseline] - fixed_risk_by_method["two_direction"]
        )
        risk_interval = np.quantile(
            risk_delta[draws].mean(axis=1), [0.025, 0.975]
        ).tolist()
        comparisons[baseline] = {
            "mae_relative_gain": (
                float(1 - candidate.mean() / reference.mean()) if resolved else None
            ),
            "mae_relative_gain_95": gain_interval,
            "mae_at_least_5_percent_supported": resolved
            and gain_interval[0] >= p["minimum_mae_relative_improvement"],
            "mae_5_percent_ruled_out": resolved
            and gain_interval[1] < p["minimum_mae_relative_improvement"],
            "matched_risk_improvement": float(risk_delta.mean()),
            "matched_risk_improvement_95": risk_interval,
            "risk_at_least_1pp_supported": risk_interval[0]
            >= p["minimum_matched_risk_improvement"],
            "risk_1pp_ruled_out": risk_interval[1]
            < p["minimum_matched_risk_improvement"],
        }
    return {
        "n": len(rows),
        "independent_anchors": len(seeds),
        "actual_nonadditive_rate": float(truth.mean()),
        "methods": methods,
        "two_direction_comparisons": comparisons,
        "uncertainty_scope": "anchor-cluster bootstrap; empirical intervals, not a distribution-free or per-case guarantee",
    }


def report(rows, p, thresholds):
    seeds = sorted({r["seed"] for r in rows})
    middle = len(seeds) // 2
    return {
        "status": "COMPLETE",
        "overall": summarize(rows, p, thresholds),
        "first_half": summarize(
            [r for r in rows if r["seed"] in seeds[:middle]], p, thresholds
        ),
        "second_half": summarize(
            [r for r in rows if r["seed"] in seeds[middle:]], p, thresholds
        ),
        "by_scale": {
            str(s): summarize([r for r in rows if r["scale"] == s], p, thresholds)
            for s in p["target_scales"]
        },
    }
