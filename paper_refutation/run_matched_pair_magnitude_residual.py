"""Committed artifact for the baseline-free C4 anchor: low-rho matched-pair
win-rate AT tightly-matched REALIZED edit magnitude.

The settle-workflow established that the raw matched-pair win-rate (~0.587) is
magnitude-confounded: build_matched_pairs matches probe_gain + calibration
target, NOT realized mean_abs_delta_attr. The honest baseline-free number is the
win-rate restricted to pairs whose REALIZED magnitudes are within a tight band.

This reconstructs that number reproducibly from the 60 control_predictive_n20
metrics.json files, across realized-relative-difference bands, with z vs 0.5.
No GPU.
"""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import math

import numpy as np

OUT = Path("/mnt/20t/study/HIGAN/paper/experiments/out/control_predictive_n20")
SAVE = Path("/mnt/20t/study/HIGAN/paper_refutation/data/matched_pair_magnitude_residual.json")

BANDS = [1.00, 0.25, 0.10, 0.05, 0.02]   # realized relative-difference caps


def z_vs_half(wins, n):
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    se = math.sqrt(0.25 / n)            # SE under H0: p=0.5
    return p, (p - 0.5) / se


def main():
    # gather eligible candidate rows
    by_group = defaultdict(list)   # (domain, seed, attr) -> rows
    for f in sorted(OUT.glob("*/metrics.json")):
        domain = f.parent.name.rsplit("_seed_", 1)[0]
        d = json.loads(f.read_text())
        for attr, ad in d.get("per_attr", {}).items():
            for r in ad.get("rows", []):
                if not r.get("gain_eligible", True):
                    continue
                seed = r.get("seed")
                by_group[(domain, seed, attr)].append(r)

    # form within-(domain,seed,attr) pairs; low/high by |rho|
    results = {}
    for band in BANDS:
        agg = defaultdict(lambda: {"id_w": 0, "lp_w": 0, "n": 0})
        for (domain, seed, attr), rows in by_group.items():
            for a, b in combinations(rows, 2):
                ra, rb = abs(a["rho"]), abs(b["rho"])
                if ra == rb:
                    continue
                lo, hi = (a, b) if ra < rb else (b, a)
                mlo = abs(lo.get("mean_abs_delta_attr", float("nan")))
                mhi = abs(hi.get("mean_abs_delta_attr", float("nan")))
                if not (math.isfinite(mlo) and math.isfinite(mhi)):
                    continue
                denom = 0.5 * (mlo + mhi)
                if denom <= 0:
                    continue
                rel = abs(mlo - mhi) / denom
                if rel > band:
                    continue
                # low-rho wins ID if it preserves identity better (higher id_cos)
                id_win = lo["mean_id_cos"] > hi["mean_id_cos"]
                # low-rho wins LPIPS if it has LESS perceptual damage (lower lpips)
                lp_win = lo["mean_lpips_true"] < hi["mean_lpips_true"]
                for key in [domain, "POOLED"]:
                    agg[key]["id_w"] += int(id_win)
                    agg[key]["lp_w"] += int(lp_win)
                    agg[key]["n"] += 1
        band_res = {}
        for key, v in agg.items():
            n = v["n"]
            p_id, z_id = z_vs_half(v["id_w"], n)
            p_lp, z_lp = z_vs_half(v["lp_w"], n)
            band_res[key] = {
                "n_pairs": n,
                "id_win_rate": p_id, "id_z": z_id,
                "lpips_win_rate": p_lp, "lpips_z": z_lp,
            }
        results[f"rel_le_{band}"] = band_res

    SAVE.parent.mkdir(parents=True, exist_ok=True)
    SAVE.write_text(json.dumps(results, indent=2))

    print("=" * 76)
    print("Low-rho matched-pair win-rate vs tightening REALIZED-magnitude band")
    print("=" * 76)
    print(f"{'band':>10} {'pool n':>8} {'ID win':>8} {'ID z':>7} "
          f"{'LPIPS win':>10} {'LPIPS z':>8}")
    for band in BANDS:
        p = results[f"rel_le_{band}"].get("POOLED", {})
        if not p:
            continue
        print(f"  rel<= {band:<4} {p['n_pairs']:>7} "
              f"{p['id_win_rate']:>8.3f} {p['id_z']:>+7.2f} "
              f"{p['lpips_win_rate']:>10.3f} {p['lpips_z']:>+8.2f}")
    print(f"\nsaved {SAVE}")
    print("\nINTERPRETATION: significant (z>~2) at moderate bands, collapsing toward")
    print("NS as the band tightens => rho's residual is REAL BUT WEAK and band-sensitive.")


if __name__ == "__main__":
    main()
