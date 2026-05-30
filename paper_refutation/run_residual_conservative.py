"""Definitive, reproducible artifact for the CONSERVATIVE rho-residual claim
and the C4b magnitude-leak finding. Kills the unreproducible-number failure mode:
every number the spine cites about the residual / C4b comes from here.

Findings the 5th audit established (reproduced here):
  - The pooled partial-Spearman(rho, damage | attr-magnitude) "6/6 positive" is a
    BETWEEN-ATTRIBUTE pooling artifact (rho ~ attribute difficulty).
  - WITHIN the operational (seed,attr) candidate-selection unit it collapses
    (bedroom/ffhq ~ +0.08; only church survives ~+0.26).
  - Controlling for total image movement (LPIPS) instead of one-axis attr
    magnitude FLIPS the sign negative in all three domains.
  - C4b: low-rho wins much more when it applies the SMALLER realized edit.

Pure reanalysis of control_predictive_n20. No GPU.
"""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import math

import numpy as np

OUT = Path("/mnt/20t/study/HIGAN/paper/experiments/out/control_predictive_n20")
SAVE = Path("/mnt/20t/study/HIGAN/paper_refutation/data/residual_conservative.json")


def rank(x):
    o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r


def spearman(x, y):
    if len(x) < 4: return float("nan")
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = (rx - rx.mean()) / (rx.std() + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def partial_spearman(x, y, z):
    rx, ry, rz = rank(np.asarray(x, float)), rank(np.asarray(y, float)), rank(np.asarray(z, float))
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex = (ex - ex.mean()) / (ex.std() + 1e-12); ey = (ey - ey.mean()) / (ey.std() + 1e-12)
    return float((ex * ey).mean())


def load():
    by_domain = defaultdict(list)
    groups = defaultdict(list)   # (domain,seed,attr) -> rows
    for f in sorted(OUT.glob("*/metrics.json")):
        domain = f.parent.name.rsplit("_seed_", 1)[0]
        d = json.loads(f.read_text())
        for attr, ad in d.get("per_attr", {}).items():
            for r in ad.get("rows", []):
                if not r.get("gain_eligible", True): continue
                rec = {
                    "seed": r.get("seed"), "attr": attr,
                    "rho": abs(r["rho"]),
                    "attr_mag": abs(r.get("mean_abs_delta_attr", np.nan)),
                    "dmg_id": 1.0 - r.get("mean_id_cos", np.nan),
                    "dmg_lpips": r.get("mean_lpips_true", np.nan),
                }
                by_domain[domain].append(rec)
                groups[(domain, r.get("seed"), attr)].append(rec)
    return by_domain, groups


def within_unit_partial(groups, domain, ykey, zkey):
    """Demean rho,y,z within each (seed,attr) group, then partial-Spearman residuals."""
    xs, ys, zs = [], [], []
    for (dom, seed, attr), rows in groups.items():
        if dom != domain or len(rows) < 4: continue
        rho = np.array([r["rho"] for r in rows], float)
        y = np.array([r[ykey] for r in rows], float)
        z = np.array([r[zkey] for r in rows], float)
        m = np.isfinite(rho) & np.isfinite(y) & np.isfinite(z)
        if m.sum() < 4: continue
        rho, y, z = rho[m], y[m], z[m]
        xs.append(rho - rho.mean()); ys.append(y - y.mean()); zs.append(z - z.mean())
    if not xs: return float("nan")
    return partial_spearman(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))


def main():
    by_domain, groups = load()
    res = {"residual": {}, "c4b_magnitude_leak": {}}

    for domain in sorted(by_domain):
        rows = by_domain[domain]
        rho = np.array([r["rho"] for r in rows], float)
        amag = np.array([r["attr_mag"] for r in rows], float)
        did = np.array([r["dmg_id"] for r in rows], float)
        dlp = np.array([r["dmg_lpips"] for r in rows], float)
        m = np.isfinite(rho) & np.isfinite(amag) & np.isfinite(did) & np.isfinite(dlp)
        rho, amag, did, dlp = rho[m], amag[m], did[m], dlp[m]

        # between-attribute: mean rho vs mean damage per attr
        attr_means = defaultdict(lambda: {"rho": [], "id": []})
        for r in rows:
            if np.isfinite(r["rho"]) and np.isfinite(r["dmg_id"]):
                attr_means[r["attr"]]["rho"].append(r["rho"])
                attr_means[r["attr"]]["id"].append(r["dmg_id"])
        mr = [np.mean(v["rho"]) for v in attr_means.values()]
        mi = [np.mean(v["id"]) for v in attr_means.values()]

        res["residual"][domain] = {
            "n": int(m.sum()), "n_attrs": len(attr_means),
            "pooled_partial_rho_dmgID_given_attrmag": partial_spearman(rho, did, amag),
            "within_unit_partial_rho_dmgID_given_attrmag": within_unit_partial(groups, domain, "dmg_id", "attr_mag"),
            "between_attr_spearman_meanrho_meanID": spearman(mr, mi),
            "partial_rho_dmgID_given_LPIPS": partial_spearman(rho, did, dlp),
            "spearman_rho_realized_attrmag": spearman(rho, amag),
        }

    # ---- C4b: low-rho win conditioned on applying smaller vs larger realized edit ----
    for domain in sorted(by_domain):
        small_w = small_n = large_w = large_n = 0
        gaps, wins = [], []
        for (dom, seed, attr), rows in groups.items():
            if dom != domain: continue
            for a, b in combinations(rows, 2):
                if a["rho"] == b["rho"]: continue
                lo, hi = (a, b) if a["rho"] < b["rho"] else (b, a)
                if not (np.isfinite(lo["attr_mag"]) and np.isfinite(hi["attr_mag"])
                        and np.isfinite(lo["dmg_id"]) and np.isfinite(hi["dmg_id"])):
                    continue
                low_wins = lo["dmg_id"] < hi["dmg_id"]      # low-rho less identity damage
                low_smaller = lo["attr_mag"] < hi["attr_mag"]
                if low_smaller:
                    small_n += 1; small_w += int(low_wins)
                else:
                    large_n += 1; large_w += int(low_wins)
                gaps.append(lo["attr_mag"] - hi["attr_mag"])   # signed realized-mag gap
                wins.append(1.0 if low_wins else 0.0)
        res["c4b_magnitude_leak"][domain] = {
            "low_wins_when_low_applies_SMALLER_edit": small_w / max(1, small_n),
            "low_wins_when_low_applies_LARGER_edit": large_w / max(1, large_n),
            "n_small": small_n, "n_large": large_n,
            "spearman_realizedmaggap_vs_lowwins": spearman(np.array(gaps), np.array(wins)),
        }

    # pooled C4b
    sw = sn = lw = ln = 0; allg, allw = [], []
    for domain, c in res["c4b_magnitude_leak"].items():
        sw += c["low_wins_when_low_applies_SMALLER_edit"] * c["n_small"]; sn += c["n_small"]
        lw += c["low_wins_when_low_applies_LARGER_edit"] * c["n_large"]; ln += c["n_large"]
    res["c4b_magnitude_leak"]["POOLED"] = {
        "low_wins_when_low_applies_SMALLER_edit": sw / max(1, sn),
        "low_wins_when_low_applies_LARGER_edit": lw / max(1, ln),
        "n_small": sn, "n_large": ln,
    }

    SAVE.parent.mkdir(parents=True, exist_ok=True)
    SAVE.write_text(json.dumps(res, indent=2))

    print("=" * 78)
    print("CONSERVATIVE rho-RESIDUAL (the honest floor)")
    print("=" * 78)
    for dom, r in res["residual"].items():
        print(f"\n[{dom}]  n={r['n']}")
        print(f"  pooled partial(rho,dmgID|attrmag)   = {r['pooled_partial_rho_dmgID_given_attrmag']:+.3f}  (looks positive)")
        print(f"  WITHIN-unit partial (seed,attr)     = {r['within_unit_partial_rho_dmgID_given_attrmag']:+.3f}  (operational unit)")
        print(f"  between-attr Spearman(meanRho,meanID)= {r['between_attr_spearman_meanrho_meanID']:+.3f}  (attribute-difficulty proxy)")
        print(f"  partial(rho,dmgID|LPIPS)            = {r['partial_rho_dmgID_given_LPIPS']:+.3f}  (sign under total-movement control)")
    print("\n" + "=" * 78)
    print("C4b MAGNITUDE LEAK (low-rho win-rate by which side took the smaller edit)")
    print("=" * 78)
    p = res["c4b_magnitude_leak"]["POOLED"]
    print(f"  POOLED: low wins {p['low_wins_when_low_applies_SMALLER_edit']:.3f} when SMALLER "
          f"(n={p['n_small']}) vs {p['low_wins_when_low_applies_LARGER_edit']:.3f} when LARGER (n={p['n_large']})")
    print(f"\nsaved {SAVE}")


if __name__ == "__main__":
    main()
