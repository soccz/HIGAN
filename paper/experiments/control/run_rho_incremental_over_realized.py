"""rho incremental value over the most defensible leakage-free baselines.

For each (domain x damage-metric), pool all gain-eligible candidate rows across
all seeds and compute rho's:
  (1) partial Spearman correlation with damage, controlling for each baseline set
  (2) incremental out-of-fold CV-R2 over each baseline set

Baselines:
  (a) abs_alpha             -- a-priori step size (leakage-free, pre-test)
  (b) mean_abs_delta_attr   -- REALIZED on-test semantic magnitude (fair covariate)
  (c) abs_alpha + mean_abs_delta_attr
  (d) 4-feature: abs_alpha, mean_abs_delta_attr, probe_gain, calib_max_abs_delta_attr

The honest question: once we control for mean_abs_delta_attr (the realized
semantic magnitude measured on the same test forward passes that produce the
damage metrics), does rho retain incremental value?
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

PAPER = Path("/mnt/20t/study/HIGAN/paper")
ROOT = PAPER / "experiments/out/control_predictive_n20"

FEATURES = {
    "abs_alpha": lambda r: abs(float(r["alpha"])),
    "mean_abs_delta_attr": lambda r: float(r["mean_abs_delta_attr"]),
    "probe_gain": lambda r: float(r["probe_gain"]),
    "calib_max_abs_delta_attr": lambda r: float(r["calib_max_abs_delta_attr"]),
}

BASELINE_SETS = {
    "(a) abs_alpha": ["abs_alpha"],
    "(b) realized_mag": ["mean_abs_delta_attr"],
    "(c) alpha+realized": ["abs_alpha", "mean_abs_delta_attr"],
    "(d) 4-feature": ["abs_alpha", "mean_abs_delta_attr",
                      "probe_gain", "calib_max_abs_delta_attr"],
}

DAMAGE = {
    "lpips": lambda r: float(r["mean_lpips_true"]),
    "id": lambda r: 1.0 - float(r["mean_id_cos"]),   # higher = more identity damage
}


def load_rows():
    by_domain: dict[str, list[dict]] = {}
    for fp in sorted(glob.glob(str(ROOT / "*/metrics.json"))):
        d = json.load(open(fp))
        dom = d["config"]["domain"]
        for attr, pl in d["per_attr"].items():
            for r in pl["rows"]:
                if not r.get("gain_eligible", False):
                    continue
                if "mean_lpips_true" not in r:
                    continue
                rr = dict(r)
                rr["attr"] = attr
                by_domain.setdefault(dom, []).append(rr)
    return by_domain


def partial_spearman(x, y, Z):
    """Spearman partial correlation of x and y given covariates Z.

    Rank-transform everything, residualize rank(x) and rank(y) on rank(Z) via OLS
    (with intercept), then Pearson-correlate the residuals. Z may be empty.
    """
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    if Z.shape[1] == 0:
        return float(np.corrcoef(rx, ry)[0, 1]), n
    rz = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(n), rz])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan"), n
    return float(np.corrcoef(ex, ey)[0, 1]), n


def cv_r2(X, y, n_splits=5, n_repeats=20, seed=20260530):
    """Mean out-of-fold R2 of ridge-less OLS with within-train standardization.

    Repeated K-fold. Returns mean over folds*repeats of pooled-style R2 computed
    per fold against the global train-mean baseline. We aggregate by averaging the
    per-fold R2 (each fold's R2 uses that fold's own y variance)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        folds = np.array_split(order, n_splits)
        for f in folds:
            test_idx = f
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            Xtr, Xte = X[train_mask], X[test_idx]
            ytr, yte = y[train_mask], y[test_idx]
            if yte.std() < 1e-12:
                continue
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            keep = sd > 1e-12
            if keep.sum() == 0:
                # intercept-only
                pred = np.full(len(yte), ytr.mean())
            else:
                Xtr_s = (Xtr[:, keep] - mu[keep]) / sd[keep]
                Xte_s = (Xte[:, keep] - mu[keep]) / sd[keep]
                A = np.column_stack([np.ones(len(ytr)), Xtr_s])
                coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
                pred = np.column_stack([np.ones(len(yte)), Xte_s]) @ coef
            denom = ((yte - yte.mean()) ** 2).sum()
            r2 = 1.0 - ((yte - pred) ** 2).sum() / denom
            scores.append(r2)
    arr = np.asarray(scores)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


def build_matrix(rows, feat_names):
    return np.column_stack([[FEATURES[f](r) for r in rows] for f in feat_names]) \
        if feat_names else np.zeros((len(rows), 0))


def main():
    by_domain = load_rows()
    results = []
    print("=" * 110)
    print("rho INCREMENTAL VALUE OVER LEAKAGE-FREE BASELINES (pooled per domain, all seeds)")
    print("=" * 110)
    for dom in sorted(by_domain):
        rows = by_domain[dom]
        n = len(rows)
        rho = np.asarray([float(r["rho"]) for r in rows])
        for dmg_name, dmg_fn in DAMAGE.items():
            y = np.asarray([dmg_fn(r) for r in rows])
            # zero-order spearman rho vs damage
            r0, p0 = spearmanr(rho, y)
            print(f"\n### {dom} x {dmg_name}   (n={n})")
            print(f"  zero-order Spearman(rho, {dmg_name}) = {r0:+.3f}  (p={p0:.2e})")
            print(f"  {'baseline':<22} {'partial_rho':>12} {'base_CV_R2':>12} "
                  f"{'+rho_CV_R2':>12} {'deltaR2':>10} {'delta_sem':>10}")
            for bname, feats in BASELINE_SETS.items():
                Z = build_matrix(rows, feats)
                pr, _ = partial_spearman(rho, y, Z)
                Xbase = Z
                Xrho = np.column_stack([Z, rho]) if Z.shape[1] else rho.reshape(-1, 1)
                base_r2, _ = cv_r2(Xbase, y) if Xbase.shape[1] else (
                    cv_r2(np.zeros((n, 0)), y))
                full_r2, _ = cv_r2(Xrho, y)
                # delta sem via paired fold differences
                d, dsem = paired_delta(Xbase, Xrho, y)
                print(f"  {bname:<22} {pr:>+12.3f} {base_r2:>12.4f} "
                      f"{full_r2:>12.4f} {d:>+10.4f} {dsem:>10.4f}")
                results.append({
                    "domain": dom, "damage": dmg_name, "n": n,
                    "zero_order_spearman": float(r0),
                    "baseline": bname, "partial_spearman": float(pr),
                    "base_cv_r2": float(base_r2), "full_cv_r2": float(full_r2),
                    "delta_cv_r2": float(d), "delta_cv_r2_sem": float(dsem),
                })
    out = ROOT / "rho_incremental_over_realized.json"
    json.dump({"rows": results}, open(out, "w"), indent=2)
    print(f"\nsaved {out}")
    return results


def paired_delta(Xbase, Xrho, y, n_splits=5, n_repeats=20, seed=20260530):
    """Per-fold paired delta of R2 (full - base) using identical splits."""
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        folds = np.array_split(order, n_splits)
        for f in folds:
            test_idx = f
            tm = np.ones(n, dtype=bool); tm[test_idx] = False
            yte = y[test_idx]; ytr = y[tm]
            if yte.std() < 1e-12:
                continue
            r2b = _fold_r2(Xbase[tm], Xbase[test_idx], ytr, yte)
            r2f = _fold_r2(Xrho[tm], Xrho[test_idx], ytr, yte)
            diffs.append(r2f - r2b)
    arr = np.asarray(diffs)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


def _fold_r2(Xtr, Xte, ytr, yte):
    if Xtr.shape[1] == 0:
        pred = np.full(len(yte), ytr.mean())
    else:
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); keep = sd > 1e-12
        if keep.sum() == 0:
            pred = np.full(len(yte), ytr.mean())
        else:
            Xtr_s = (Xtr[:, keep] - mu[keep]) / sd[keep]
            Xte_s = (Xte[:, keep] - mu[keep]) / sd[keep]
            A = np.column_stack([np.ones(len(ytr)), Xtr_s])
            coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            pred = np.column_stack([np.ones(len(yte)), Xte_s]) @ coef
    denom = ((yte - yte.mean()) ** 2).sum()
    return 1.0 - ((yte - pred) ** 2).sum() / denom


if __name__ == "__main__":
    main()
