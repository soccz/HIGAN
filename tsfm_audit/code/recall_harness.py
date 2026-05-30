"""Within-series memorization probe — sidesteps the control-fragility that the
synthetic-floor comparison suffered.

Two readouts per series, BOTH using DLinear (strong, non-memorizing reference)
as the within-series control so genuine predictability is absorbed:

1) skill gap: median MASE(Chronos) vs MASE(DLinear).
2) RECALL INDEX: a memorizing model recalls the SPECIFIC continuation, so its
   forecast error should be anomalously FLAT across the horizon (it "knows" step
   h+10 as well as h+1), whereas a generalizing model's error COMPOUNDS with h.
   We fit error(h) ~ slope*h on the mean per-horizon error curve, normalized by
   the curve mean, for BOTH Chronos and DLinear (same series, same windows →
   character-controlled). recall_index = dlinear_slope - chronos_slope.
   Positive & large = Chronos's error curve is anomalously flatter than the
   non-memorizing reference on this series = recall-like.

Compare the recall_index DISTRIBUTION across strata. Memorization/contamination
=> recall_index large on in-corpus, ~0 on provably-uncontaminated strata.
"""
from __future__ import annotations
import numpy as np
import torch

from eval_harness import make_windows, chronos_forecast, mase
from baselines import dlinear_forecast


def _norm_slope(err_h: np.ndarray) -> float:
    """Slope of (normalized) error vs horizon. Small = flat = recall-like."""
    h = np.arange(len(err_h), dtype=float)
    e = err_h / (err_h.mean() + 1e-12)
    h = (h - h.mean())
    return float((h @ (e - e.mean())) / (h @ h + 1e-12))


def evaluate_series_recall(pipe, series: np.ndarray, *, context=256, horizon=24,
                           season=24, stride=24, max_windows=8,
                           train_frac=0.6) -> dict:
    n = len(series)
    train_end = int(n * train_frac)
    train_prefix = series[:train_end]
    test_region = series[train_end - context:]
    wins = make_windows(test_region, context, horizon, stride, max_windows)
    if len(wins) < 2:
        return {}
    ch_mase, dl_mase = [], []
    ch_err_h = np.zeros(horizon); dl_err_h = np.zeros(horizon); nw = 0
    for ctx, tgt in wins:
        chf = chronos_forecast(pipe, ctx, horizon)
        dlf = dlinear_forecast(ctx, horizon, train_prefix, context)
        ch_mase.append(mase(chf, tgt, ctx, season))
        dl_mase.append(mase(dlf, tgt, ctx, season))
        ch_err_h += np.abs(chf - tgt)
        dl_err_h += np.abs(dlf - tgt)
        nw += 1
    ch_err_h /= nw; dl_err_h /= nw
    return {
        "chronos_mase": float(np.median(ch_mase)),
        "dlinear_mase": float(np.median(dl_mase)),
        "chronos_slope": _norm_slope(ch_err_h),
        "dlinear_slope": _norm_slope(dl_err_h),
        "recall_index": _norm_slope(dl_err_h) - _norm_slope(ch_err_h),
        "n_windows": nw,
    }
