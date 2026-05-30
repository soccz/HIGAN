"""Strong, leakage-free per-series baselines that a TSFM must beat.

DLinear (Zeng et al. AAAI 2023, "Are Transformers Effective for Time Series
Forecasting?") — the standard strong-but-simple baseline. Decompose the context
into trend (moving average) + seasonal-residual, fit an independent linear map
context->horizon on each, trained on the series' TRAIN prefix only. Instance
normalization (per-context mean/std) is leakage-free (conditions on observed
context only).

Replaces the too-weak ridge that made the first 3-stratum gaps un-interpretable.
"""
from __future__ import annotations
import numpy as np


def _moving_avg(x: np.ndarray, k: int = 25) -> np.ndarray:
    if k <= 1 or len(x) <= k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode="valid")[:len(x)]


def _fit_linear(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def dlinear_forecast(ctx: np.ndarray, horizon: int, train_prefix: np.ndarray,
                     context: int, kernel: int = 25) -> np.ndarray:
    """DLinear (corrected): the TARGET is decomposed into trend+seasonal too, and
    each linear map predicts its OWN component (no double counting)."""
    Xt, Yt, Xs, Ys = [], [], [], []
    i = 0
    while i + context + horizon <= len(train_prefix):
        c = train_prefix[i:i + context]
        h = train_prefix[i + context:i + context + horizon]
        mu, sd = c.mean(), c.std() + 1e-8
        cn, hn = (c - mu) / sd, (h - mu) / sd
        ct = _moving_avg(cn, kernel); cs = cn - ct
        ht = _moving_avg(hn, kernel); hs = hn - ht      # decompose TARGET too
        Xt.append(ct); Yt.append(ht); Xs.append(cs); Ys.append(hs)
        i += max(1, horizon // 2)
    if len(Xt) < 5:
        return np.repeat(ctx[-1], horizon).astype(np.float32)
    Wt = _fit_linear(np.asarray(Xt), np.asarray(Yt))
    Ws = _fit_linear(np.asarray(Xs), np.asarray(Ys))
    mu, sd = ctx.mean(), ctx.std() + 1e-8
    cn = (ctx - mu) / sd
    ct = _moving_avg(cn, kernel); cs = cn - ct
    pred = (ct @ Wt + cs @ Ws) * sd + mu
    return pred.astype(np.float32)


def nlinear_forecast(ctx: np.ndarray, horizon: int, train_prefix: np.ndarray,
                     context: int) -> np.ndarray:
    """NLinear (Zeng 2023): subtract the LAST value (handles level shift /
    non-stationarity), single linear map context->horizon, add it back.
    Often the strongest simple baseline on non-stationary real series."""
    X, Y = [], []
    i = 0
    while i + context + horizon <= len(train_prefix):
        c = train_prefix[i:i + context]
        h = train_prefix[i + context:i + context + horizon]
        last = c[-1]
        X.append(c - last); Y.append(h - last)
        i += max(1, horizon // 2)
    if len(X) < 5:
        return np.repeat(ctx[-1], horizon).astype(np.float32)
    W = _fit_linear(np.asarray(X), np.asarray(Y), lam=0.1)
    last = ctx[-1]
    return ((ctx - last) @ W + last).astype(np.float32)
