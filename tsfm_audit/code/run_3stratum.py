"""Core result: Chronos-minus-RLinear MASE gap across three strata, hourly,
season=24, frequency-matched. The dose-response of contamination.

  in_domain   (Benchmark I)  : definitely trained on -> upper bound on TSFM advantage
  zero_shot   (Benchmark II) : Chronos's nominal zero-shot (temporal-disjointness
                               NOT enforced) -> possibly contaminated
  synthetic   (stratum III)  : contamination impossible by construction -> floor

Readout: gap = median(Chronos MASE) - median(RLinear MASE) per stratum
         (more negative = Chronos beats the trained linear baseline by more).
If |gap| is much larger in zero_shot than in synthetic, the nominal zero-shot
advantage is contamination-inflated.
"""
from __future__ import annotations
import json
import time
import numpy as np
import torch
from chronos import ChronosPipeline

from synthetic_surrogates import make_surrogate_set
from load_strata import load_stratum, SEASON
from eval_harness import evaluate_series

CONTEXT, HORIZON, STRIDE, MAXW = 256, 24, 24, 6


def run_stratum_items(pipe, items):
    rows = {"chronos": [], "rlinear": [], "seasonal_naive": []}
    for it in items:
        r = evaluate_series(pipe, it["series"], context=CONTEXT, horizon=HORIZON,
                            season=SEASON, stride=STRIDE, max_windows=MAXW)
        if not r:
            continue
        for k in rows:
            rows[k].append(r[k])
    return rows


def summarize(name, rows):
    ch = np.array(rows["chronos"]); rl = np.array(rows["rlinear"])
    sn = np.array(rows["seasonal_naive"])
    gap = float(np.median(ch) - np.median(rl))
    paired = float(np.mean(ch < rl))   # Chronos beats RLinear per series
    return {
        "stratum": name, "n_series": len(ch),
        "chronos_median_mase": float(np.median(ch)),
        "rlinear_median_mase": float(np.median(rl)),
        "seasonal_naive_median_mase": float(np.median(sn)),
        "chronos_minus_rlinear_gap": gap,
        "chronos_beats_rlinear_winrate": paired,
    }


def main():
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cuda", torch_dtype=torch.float16)
    t0 = time.time()

    strata = {}
    print("loading + evaluating strata (hourly, season=24)...")

    # in-domain
    items = load_stratum("in_domain", max_series_per_config=15)
    strata["in_domain"] = summarize("in_domain", run_stratum_items(pipe, items))
    print(f"  in_domain   done ({strata['in_domain']['n_series']} series, {time.time()-t0:.0f}s)")

    # nominal zero-shot
    items = load_stratum("zero_shot", max_series_per_config=15)
    strata["zero_shot"] = summarize("zero_shot", run_stratum_items(pipe, items))
    print(f"  zero_shot   done ({strata['zero_shot']['n_series']} series, {time.time()-t0:.0f}s)")

    # synthetic floor (period-24 hourly)
    surr = [{"series": d["series"]} for d in make_surrogate_set(n=20, length=1024, seed=2027)
            if d["period"] == 24]
    strata["synthetic"] = summarize("synthetic", run_stratum_items(pipe, surr))
    print(f"  synthetic   done ({strata['synthetic']['n_series']} series, {time.time()-t0:.0f}s)")

    out = {"config": {"context": CONTEXT, "horizon": HORIZON, "season": SEASON,
                      "model": "chronos-t5-small"}, "strata": strata}
    import os
    os.makedirs("../results", exist_ok=True)
    json.dump(out, open("../results/run_3stratum.json", "w"), indent=2)

    print("\n" + "=" * 74)
    print("3-STRATUM DOSE-RESPONSE (hourly, season=24) — Chronos-t5-small")
    print("=" * 74)
    print(f"{'stratum':12s} {'Chronos':>9s} {'RLinear':>9s} {'snaive':>9s} "
          f"{'gap(Ch-RL)':>11s} {'Ch<RL win':>10s}")
    for name in ["in_domain", "zero_shot", "synthetic"]:
        s = strata[name]
        print(f"{name:12s} {s['chronos_median_mase']:>9.3f} {s['rlinear_median_mase']:>9.3f} "
              f"{s['seasonal_naive_median_mase']:>9.3f} {s['chronos_minus_rlinear_gap']:>+11.3f} "
              f"{s['chronos_beats_rlinear_winrate']:>10.2f}")
    print("\n  Contamination signature: |gap| large in in_domain & zero_shot but")
    print("  collapsing in synthetic => Chronos's nominal zero-shot edge is inflated.")
    print(f"\n  saved ../results/run_3stratum.json  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
