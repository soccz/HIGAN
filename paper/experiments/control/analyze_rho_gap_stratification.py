"""rho-gap stratification: the mechanism behind the prediction-control gap.

Headline question for the paper: curvature/rho predicts damage RANK perfectly
(Spearman 100%), yet picking the low-rho candidate only beats the high-rho one
~67% of the time. WHY?

Hypothesis: a matched pair only resolves in rho's favor when the two candidates
are sufficiently SEPARATED in rho. When |rho_low - rho_high| is tiny, the pair
is a coin flip; when the gap is large, low-rho reliably wins.

If win-rate rises monotonically with |rho gap|, the 67% is not noise and not a
fundamental failure -- it is a *characterized operating regime*: rho is a
usable control signal precisely when candidate rho-separation exceeds a
threshold. That converts the weakness into the paper's sharpest finding.

Pure post-hoc reanalysis of all control_predictive_n20/* runs. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
from lib.reproducibility import run_metadata  # noqa: E402

N20 = PAPER / "experiments" / "out" / "control_predictive_n20"


def collect_pairs():
    """Gather every matched pair across all seeds/domains/attrs."""
    pairs_by_domain = defaultdict(list)
    for mp_path in sorted(N20.glob("*/metrics.json")):
        domain = mp_path.parent.name.rsplit("_seed_", 1)[0]
        d = json.loads(mp_path.read_text())
        for attr, attr_data in d.get("per_attr", {}).items():
            for pair in attr_data.get("matched_pairs", []):
                rho_gap = pair.get("rho_diff_low_minus_high")
                id_diff = pair.get("mean_id_cos_diff_low_minus_high")
                lpips_diff = pair.get("mean_lpips_true_diff_low_minus_high")
                gain_diff = pair.get("mean_abs_delta_attr_diff_low_minus_high")
                if rho_gap is None or id_diff is None or lpips_diff is None:
                    continue
                pairs_by_domain[domain].append({
                    "domain": domain,
                    "attr": attr,
                    # by construction low has lower rho, so rho_gap < 0;
                    # magnitude of separation is abs(rho_gap)
                    "rho_sep": abs(rho_gap),
                    # low-rho "wins" on ID if it preserved identity better
                    # (id_cos higher for low) -> id_diff > 0
                    "low_wins_id": id_diff > 0,
                    # low-rho "wins" on LPIPS if it had LESS perceptual damage
                    # (lpips lower for low) -> lpips_diff < 0
                    "low_wins_lpips": lpips_diff < 0,
                    "id_diff": id_diff,
                    "lpips_diff": lpips_diff,
                    "gain_diff": abs(gain_diff) if gain_diff is not None else None,
                })
    return pairs_by_domain


def winrate_by_quantile(pairs, key, n_bins=4):
    """Bin pairs by rho-separation quantile, report win-rate per bin."""
    if len(pairs) < n_bins:
        return []
    seps = np.array([p["rho_sep"] for p in pairs])
    qs = np.quantile(seps, np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        if b == n_bins - 1:
            mask = (seps >= lo) & (seps <= hi)
        else:
            mask = (seps >= lo) & (seps < hi)
        sub = [pairs[i] for i in range(len(pairs)) if mask[i]]
        if not sub:
            out.append({"bin": b, "n": 0})
            continue
        wins = sum(1 for p in sub if p[key])
        out.append({
            "bin": b,
            "rho_sep_lo": float(lo),
            "rho_sep_hi": float(hi),
            "n": len(sub),
            "win_rate": wins / len(sub),
            "wins": wins,
        })
    return out


def spearman_winrate_vs_sep(pairs, key):
    """Rank correlation between rho-separation and win indicator.
    Positive = larger separation -> more likely low-rho wins.
    """
    if len(pairs) < 5:
        return None
    sep = np.array([p["rho_sep"] for p in pairs])
    win = np.array([1.0 if p[key] else 0.0 for p in pairs])

    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x))
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        return r

    rs, rw = rank(sep), rank(win)
    rs = (rs - rs.mean()) / (rs.std() + 1e-12)
    rw = (rw - rw.mean()) / (rw.std() + 1e-12)
    return float((rs * rw).mean())


def main():
    if not N20.exists():
        print("no control_predictive_n20 results")
        return
    pairs_by_domain = collect_pairs()

    all_pairs = [p for ps in pairs_by_domain.values() for p in ps]
    print(f"collected {len(all_pairs)} matched pairs across "
          f"{len(pairs_by_domain)} domains")

    summary = {"domains": {}, "pooled": {}}

    for domain in sorted(pairs_by_domain) + ["__pooled__"]:
        pairs = (all_pairs if domain == "__pooled__"
                 else pairs_by_domain[domain])
        if not pairs:
            continue
        label = "POOLED" if domain == "__pooled__" else domain
        print(f"\n=== {label}  (n_pairs={len(pairs)}) ===")

        for key, name in [("low_wins_lpips", "LPIPS"),
                          ("low_wins_id", "ID")]:
            overall = np.mean([p[key] for p in pairs])
            bins = winrate_by_quantile(pairs, key, n_bins=4)
            rho_corr = spearman_winrate_vs_sep(pairs, key)
            print(f"  [{name}] overall low-rho win-rate = {overall:.3f}  "
                  f"(Spearman win~rho_sep = "
                  f"{rho_corr:+.3f})" if rho_corr is not None
                  else f"  [{name}] overall = {overall:.3f}")
            for bnd in bins:
                if bnd["n"] == 0:
                    continue
                print(f"      Q{bnd['bin']+1} rho_sep[{bnd['rho_sep_lo']:.4f},"
                      f"{bnd['rho_sep_hi']:.4f}]  n={bnd['n']:3d}  "
                      f"win={bnd['win_rate']:.3f}")

            tgt = (summary["pooled"] if domain == "__pooled__"
                   else summary["domains"].setdefault(domain, {}))
            tgt[name] = {
                "overall_win_rate": float(overall),
                "spearman_win_vs_rho_sep": rho_corr,
                "bins": bins,
            }

    out = N20 / "rho_gap_stratification.json"
    out.write_text(json.dumps(
        {**summary, "n_pairs_total": len(all_pairs),
         "_meta": run_metadata(seed=0)}, indent=2))
    print(f"\nsaved {out}")

    # Headline verdict
    pooled_lpips = summary["pooled"].get("LPIPS", {})
    bins = pooled_lpips.get("bins", [])
    valid = [b for b in bins if b.get("n", 0) > 0]
    if len(valid) >= 2:
        first, last = valid[0]["win_rate"], valid[-1]["win_rate"]
        corr = pooled_lpips.get("spearman_win_vs_rho_sep")
        print("\n--- VERDICT ---")
        print(f"  LPIPS win-rate: lowest rho-sep bin = {first:.3f} -> "
              f"highest rho-sep bin = {last:.3f}")
        if corr is not None and corr > 0.1 and last > first + 0.1:
            print("  => MONOTONE: rho is a usable control signal WHEN "
                  "candidate rho-separation is large. 67% is a regime, not "
                  "noise. This is the paper's sharpest finding.")
        elif corr is not None and abs(corr) <= 0.1:
            print("  => FLAT: win-rate does not depend on rho-separation. "
                  "The control gap is fundamental, not a separation regime. "
                  "Frame as honest hard boundary.")
        else:
            print("  => MIXED: partial dependence; report bins, avoid strong "
                  "regime claim.")


if __name__ == "__main__":
    main()
