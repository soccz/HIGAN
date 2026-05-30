"""Stratum III — synthetic surrogates: the logically airtight generalization floor.

Contamination is IMPOSSIBLE by construction: these series are generated fresh here,
so they cannot be in any model's pretraining corpus. We do NOT use KernelSynth /
TSMixup (corpus-derived). We match the surrogates to real-TS scales (seasonal
periods, amplitude, noise) so a reviewer cannot dismiss them as unrepresentative.

A surrogate = trend + one or two seasonal harmonics + AR(p) coloured noise.
"""
from __future__ import annotations
import numpy as np


def make_surrogate(rng: np.random.Generator, length: int = 1024,
                   period: int = 24) -> np.ndarray:
    """One univariate surrogate with realistic seasonal + AR structure."""
    t = np.arange(length)
    # trend: small linear + occasional level — keep mild
    trend = rng.normal(0, 0.002) * t
    # 1-2 seasonal harmonics at the given period (+ optional weekly multiple)
    season = np.zeros(length)
    for k, P in enumerate([period, period * 7]):
        if P >= length:
            continue
        amp = rng.uniform(0.5, 2.0) / (k + 1)
        phase = rng.uniform(0, 2 * np.pi)
        season += amp * np.sin(2 * np.pi * t / P + phase)
    # AR(2) coloured noise
    phi = rng.uniform(0.2, 0.7, size=2)
    e = rng.normal(0, rng.uniform(0.2, 0.8), size=length)
    ar = np.zeros(length)
    for i in range(2, length):
        ar[i] = phi[0] * ar[i - 1] + phi[1] * ar[i - 2] + e[i]
    series = trend + season + ar
    # standardize to unit-ish scale, then random affine to vary scale
    series = (series - series.mean()) / (series.std() + 1e-8)
    series = series * rng.uniform(0.5, 5.0) + rng.uniform(-3, 3)
    return series.astype(np.float32)


def make_surrogate_set(n: int = 200, length: int = 1024,
                       seed: int = 2027) -> list[dict]:
    """A matched set of surrogates spanning common seasonal periods."""
    rng = np.random.default_rng(seed)
    periods = [24, 24, 24, 168, 7, 12, 96]   # weighted toward hourly/daily
    out = []
    for i in range(n):
        P = periods[i % len(periods)]
        s = make_surrogate(rng, length=length, period=P)
        out.append({"id": f"surrogate_{i:04d}", "period": P, "series": s})
    return out


if __name__ == "__main__":
    ss = make_surrogate_set(n=8, length=512)
    for d in ss:
        s = d["series"]
        print(f"  {d['id']}  P={d['period']:3d}  len={len(s)}  "
              f"mean={s.mean():+.2f} std={s.std():.2f} "
              f"min={s.min():+.2f} max={s.max():+.2f}")
    print("OK: surrogates generated (contamination impossible by construction).")
