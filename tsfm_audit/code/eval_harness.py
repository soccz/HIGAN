"""Core evaluation harness: per-series, per-window forecast error for
  - Chronos (TSFM, zero-shot, median of samples)
  - seasonal-naive (cheap content-blind reference)
  - RLinear (on-target trained linear baseline; the fair "did you need a TSFM" bar)

Metric: MASE (scale-free; standard for TSFM benchmarks) + MSE.
The audit reads the TSFM-minus-RLinear gap across strata.

Leakage discipline: RLinear is fit on the series' TRAIN prefix only; normalization
stats are train-only; the forecast window is strictly future.
"""
from __future__ import annotations
import numpy as np
import torch


def make_windows(series: np.ndarray, context: int, horizon: int,
                 stride: int, max_windows: int = 20):
    """Rolling (context, horizon) windows over the tail of the series."""
    wins = []
    i = len(series) - context - horizon
    while i >= 0 and len(wins) < max_windows:
        ctx = series[i:i + context]
        tgt = series[i + context:i + context + horizon]
        wins.append((ctx, tgt))
        i -= stride
    return list(reversed(wins))


def mase(pred: np.ndarray, true: np.ndarray, ctx: np.ndarray,
         season: int) -> float:
    """Mean Absolute Scaled Error: MAE(pred) / MAE(seasonal-naive on context)."""
    if len(ctx) > season:
        scale = np.mean(np.abs(ctx[season:] - ctx[:-season])) + 1e-8
    else:
        scale = np.mean(np.abs(np.diff(ctx))) + 1e-8
    return float(np.mean(np.abs(pred - true)) / scale)


def seasonal_naive(ctx: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Repeat the last seasonal cycle."""
    if len(ctx) >= season:
        base = ctx[-season:]
    else:
        base = ctx[-1:]
    reps = int(np.ceil(horizon / len(base)))
    return np.tile(base, reps)[:horizon].astype(np.float32)


def rlinear_forecast(ctx: np.ndarray, horizon: int,
                     train_prefix: np.ndarray, context: int) -> np.ndarray:
    """RLinear: train-only ridge from context-window -> horizon, applied to ctx.

    Fit on windows drawn from train_prefix ONLY (no test leakage). Normalization
    (RevIN-style) uses each window's own context stats (instance norm), which is
    leakage-free because it conditions only on observed context.
    """
    X, Y = [], []
    i = 0
    while i + context + horizon <= len(train_prefix):
        c = train_prefix[i:i + context]
        h = train_prefix[i + context:i + context + horizon]
        mu, sd = c.mean(), c.std() + 1e-8
        X.append((c - mu) / sd)
        Y.append((h - mu) / sd)
        i += max(1, horizon // 2)
    if len(X) < 5:
        # too short to fit; fall back to seasonal-naive-like persistence
        return np.repeat(ctx[-1], horizon).astype(np.float32)
    X = np.asarray(X); Y = np.asarray(Y)
    # ridge closed form
    lam = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ Y)            # (context, horizon)
    mu, sd = ctx.mean(), ctx.std() + 1e-8
    pred = ((ctx - mu) / sd) @ W * sd + mu
    return pred.astype(np.float32)


@torch.no_grad()
def chronos_forecast(pipe, ctx: np.ndarray, horizon: int,
                     num_samples: int = 20) -> np.ndarray:
    fc = pipe.predict(torch.tensor(ctx), prediction_length=horizon,
                      num_samples=num_samples)            # (1, S, H)
    return np.median(fc[0].numpy(), axis=0).astype(np.float32)


def evaluate_series(pipe, series: np.ndarray, *, context=512, horizon=24,
                    season=24, stride=24, max_windows=10,
                    train_frac=0.6) -> dict:
    """Return per-method mean MASE over the test windows of one series."""
    n = len(series)
    train_end = int(n * train_frac)
    train_prefix = series[:train_end]
    # test windows live strictly in the held-out tail
    test_region = series[train_end - context:]   # include context lead-in
    wins = make_windows(test_region, context, horizon, stride, max_windows)
    if not wins:
        return {}
    rows = {"chronos": [], "seasonal_naive": [], "rlinear": []}
    for ctx, tgt in wins:
        rows["chronos"].append(mase(chronos_forecast(pipe, ctx, horizon), tgt, ctx, season))
        rows["seasonal_naive"].append(mase(seasonal_naive(ctx, horizon, season), tgt, ctx, season))
        rows["rlinear"].append(mase(rlinear_forecast(ctx, horizon, train_prefix, context), tgt, ctx, season))
    # MEDIAN over windows: robust to occasional near-zero MASE-scale blowups when a
    # context window is near-perfectly periodic (denominator -> 0). The blowup hits
    # all methods in the same window, but a few huge values dominate a MEAN; median
    # is the standard robust choice. Cross-series aggregation should also use median.
    return {k: float(np.median(v)) for k, v in rows.items()} | {"n_windows": len(wins)}
