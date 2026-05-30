"""STEP 3 leakage audit (thesis-determining).

The v3 thesis = "edit magnitude operationally dominates rho; rho adds no
incremental operational value." The prior workflow flagged that the "0/41 rho
beats baseline" used a per-fold MAX over 4 features, one of which
(mean_abs_delta_attr = realized semantic change) is mechanically coupled to the
damage outcome -> possible label leakage inflating the baseline.

Decisive question: does rho add incremental value over a CLEAN, leakage-free
magnitude baseline (abs_alpha = the latent-space step size), as opposed to only
losing to the leaky 4-feature max?

  - If rho fails to beat abs_alpha too -> thesis HOLDS (magnitude dominates).
  - If rho beats clean abs_alpha but only lost to the leaky max -> thesis is
    WRONG; the 0/41 was a leakage artifact and rho carries real operational value.

Per (domain x damage-metric) cell, pooled across 20 seeds, we report rho's
incremental cross-validated R^2 and partial Spearman over each candidate baseline.
Pure reanalysis of control_predictive_n20. No GPU.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np

OUT = Path("/mnt/20t/study/HIGAN/paper/experiments/out/control_predictive_n20")

# candidate baselines
LEAKY = "mean_abs_delta_attr"      # realized semantic change (outcome-coupled)
CLEAN = "abs_alpha"                # latent step size (leakage-free magnitude)
OTHER = ["probe_gain", "calib_max_abs_delta_attr"]


def rankdata(x):
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    return r


def spearman(x, y):
    if len(x) < 4:
        return float("nan")
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def partial_spearman(x, y, z):
    """Spearman(x, y | z): correlate rank-residuals of x and y after removing z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rz1 = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(rz1, rx, rcond=None)
    by, *_ = np.linalg.lstsq(rz1, ry, rcond=None)
    ex, ey = rx - rz1 @ bx, ry - rz1 @ by
    ex = (ex - ex.mean()) / (ex.std() + 1e-12)
    ey = (ey - ey.mean()) / (ey.std() + 1e-12)
    return float((ex * ey).mean())


def cv_r2(X, y, k=5, seed=0):
    """k-fold CV R^2 of OLS y ~ X (X includes no intercept col; we add it)."""
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    sse, sst = 0.0, 0.0
    ybar_all = y.mean()
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
        Xte = np.column_stack([np.ones(len(te)), X[te]])
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        pred = Xte @ beta
        sse += ((y[te] - pred) ** 2).sum()
        sst += ((y[te] - ybar_all) ** 2).sum()
    return 1.0 - sse / (sst + 1e-12)


def collect():
    cells = defaultdict(lambda: defaultdict(list))  # domain -> field -> list
    for f in sorted(OUT.glob("*/metrics.json")):
        domain = f.parent.name.rsplit("_seed_", 1)[0]
        d = json.loads(f.read_text())
        for attr, ad in d.get("per_attr", {}).items():
            for r in ad.get("rows", []):
                if not r.get("gain_eligible", True):
                    continue
                cells[domain]["rho"].append(abs(r["rho"]))
                cells[domain][CLEAN].append(abs(r.get("alpha", np.nan)))
                cells[domain][LEAKY].append(r.get("mean_abs_delta_attr", np.nan))
                cells[domain]["probe_gain"].append(abs(r.get("probe_gain", np.nan)))
                cells[domain]["calib_max_abs_delta_attr"].append(
                    r.get("calib_max_abs_delta_attr", np.nan))
                cells[domain]["dmg_id"].append(1.0 - r.get("mean_id_cos", np.nan))
                cells[domain]["dmg_lpips"].append(r.get("mean_lpips_true", np.nan))
    return cells


def main():
    cells = collect()
    results = {}
    print("=" * 78)
    print("LEAKAGE AUDIT: does rho beat a CLEAN magnitude baseline (abs_alpha)?")
    print("=" * 78)
    for domain in sorted(cells):
        c = cells[domain]
        rho = np.asarray(c["rho"], float)
        for metric, yname in [("dmg_lpips", "LPIPS"), ("dmg_id", "ID")]:
            y = np.asarray(c[metric], float)
            mask = np.isfinite(rho) & np.isfinite(y)
            for b in [CLEAN, LEAKY] + OTHER:
                mask &= np.isfinite(np.asarray(c[b], float))
            if mask.sum() < 20:
                continue
            rho_m = rho[mask]; y_m = y[mask]
            row = {"domain": domain, "metric": yname, "n": int(mask.sum()),
                   "marginal_spearman_rho": spearman(rho_m, y_m)}
            for b in [CLEAN, LEAKY] + OTHER:
                bm = np.asarray(c[b], float)[mask]
                # standardize
                Xb = (bm - bm.mean()) / (bm.std() + 1e-12)
                Xr = (rho_m - rho_m.mean()) / (rho_m.std() + 1e-12)
                r2_base = cv_r2(Xb.reshape(-1, 1), y_m)
                r2_full = cv_r2(np.column_stack([Xb, Xr]), y_m)
                row[f"incr_cvr2_over_{b}"] = r2_full - r2_base
                row[f"partial_spearman_rho_given_{b}"] = partial_spearman(
                    rho_m, y_m, bm)
            results[f"{domain}_{yname}"] = row
            print(f"\n[{domain} / {yname}]  n={row['n']}  "
                  f"marginal Spearman(rho,dmg)={row['marginal_spearman_rho']:+.3f}")
            print(f"    vs CLEAN  abs_alpha:  incr CV-R2={row[f'incr_cvr2_over_{CLEAN}']:+.4f}  "
                  f"partial Spearman={row[f'partial_spearman_rho_given_{CLEAN}']:+.3f}")
            print(f"    vs LEAKY  realized-Δ: incr CV-R2={row[f'incr_cvr2_over_{LEAKY}']:+.4f}  "
                  f"partial Spearman={row[f'partial_spearman_rho_given_{LEAKY}']:+.3f}")
            print(f"    vs probe_gain:        incr CV-R2={row['incr_cvr2_over_probe_gain']:+.4f}")

    # ---- verdict ----
    clean_incr = [r[f"incr_cvr2_over_{CLEAN}"] for r in results.values()]
    clean_partial = [r[f"partial_spearman_rho_given_{CLEAN}"] for r in results.values()]
    n_clean_pos = sum(1 for v in clean_incr if v > 0.01)
    n_clean_partial_sig = sum(1 for v in clean_partial if abs(v) > 0.1)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  cells where rho adds CV-R2 > 0.01 over CLEAN abs_alpha: "
          f"{n_clean_pos}/{len(clean_incr)}")
    print(f"  cells where partial Spearman(rho,dmg|abs_alpha) > 0.1: "
          f"{n_clean_partial_sig}/{len(clean_partial)}")
    print(f"  mean incr CV-R2 over clean abs_alpha: {np.mean(clean_incr):+.4f}")
    print(f"  mean partial Spearman over clean abs_alpha: {np.mean(clean_partial):+.3f}")
    if n_clean_pos >= len(clean_incr) // 2 or n_clean_partial_sig >= len(clean_partial) // 2:
        print("\n  => rho DOES add value over CLEAN magnitude in many cells.")
        print("     The 0/41 was likely a LEAKAGE ARTIFACT of the 4-feature max.")
        print("     THESIS MUST CHANGE: rho carries operational value over clean magnitude.")
        verdict = "THESIS_WRONG_rho_beats_clean_magnitude"
    else:
        print("\n  => rho does NOT meaningfully beat clean abs_alpha either.")
        print("     v3 thesis HOLDS: magnitude dominates; rho adds no operational value.")
        print("     (Report the CLEAN baseline, not the leaky max, in the paper.)")
        verdict = "THESIS_HOLDS_magnitude_dominates"
    results["_verdict"] = verdict

    outp = Path("/mnt/20t/study/HIGAN/paper_refutation/data/leakage_audit.json")
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {outp}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
