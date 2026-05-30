"""Load real Chronos benchmark datasets, grouped by Chronos's documented
in-domain (Benchmark I) vs nominal-zero-shot (Benchmark II) split, FREQUENCY-MATCHED
so the cross-stratum gap is not confounded by sampling frequency.

We restrict the first pass to HOURLY series (season=24) across all strata.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("HF_HOME", "/home/soccz/22tb/.cache/huggingface")
from datasets import load_dataset  # noqa: E402

# Chronos's own split (from repo eval configs), restricted to hourly datasets.
HOURLY = {
    "in_domain": ["m4_hourly", "monash_electricity_hourly"],
    "zero_shot": ["monash_traffic", "monash_kdd_cup_2018"],  # kdd is in-domain; fix below
}
# Correct membership per the verified config lists:
#   in_domain (Benchmark I): m4_hourly, monash_electricity_hourly, monash_kdd_cup_2018
#   zero_shot (Benchmark II): monash_traffic (hourly)
STRATA = {
    "in_domain": ["m4_hourly", "monash_electricity_hourly"],
    "zero_shot": ["monash_traffic"],
}
SEASON = 24  # hourly


def load_series(config: str, max_series: int = 20, min_len: int = 400,
                seed: int = 2027) -> list[np.ndarray]:
    """Return up to max_series target arrays of sufficient length from a config."""
    ds = load_dataset("autogluon/chronos_datasets", config, split="train")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    out = []
    for i in idx:
        tgt = np.asarray(ds[int(i)]["target"], dtype=np.float32)
        tgt = tgt[np.isfinite(tgt)]
        if len(tgt) >= min_len:
            out.append(tgt)
        if len(out) >= max_series:
            break
    return out


def load_stratum(stratum: str, max_series_per_config: int = 15) -> list[dict]:
    """Return [{config, series}] for all configs in a stratum."""
    out = []
    for cfg in STRATA[stratum]:
        for s in load_series(cfg, max_series=max_series_per_config):
            out.append({"config": cfg, "series": s})
    return out


if __name__ == "__main__":
    for strat in STRATA:
        items = load_stratum(strat, max_series_per_config=5)
        print(f"{strat}: {len(items)} series from {STRATA[strat]}")
        for it in items[:2]:
            s = it["series"]
            print(f"  {it['config']:28s} len={len(s)} mean={s.mean():.2f} std={s.std():.2f}")
