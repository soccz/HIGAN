# Provenance Lock — Chronos (Day 1)

Project: how much of open TSFM zero-shot "SOTA" is pretraining contamination vs
generalization, via a publication-date / strict-temporal logical negative control.

## Models (cached, 8GB-trivial)
- `amazon/chronos-t5-small` (~46M) — PRIMARY. Cached at
  `/home/soccz/22tb/.cache/huggingface/hub/models--amazon--chronos-t5-small`.
- `amazon/chronos-t5-tiny` (~8M) — cached, for ablation.
- (optional later: Chronos-Bolt-small, TimesFM-2.0, Moirai-small for the sweep.)

## Chronos pretraining corpus (Ansari et al., arXiv 2403.07815, TMLR 10/2024)
- 55 datasets total (Appendix B), domains: energy, transport, healthcare, retail,
  web, weather, finance; freq 5min→yearly.
- Plus synthetic augmentation: **TSMixup** (convex combos of real series) and
  **KernelSynth** (GP-kernel synthetic). Note: TSMixup mixes REAL training series,
  so "synthetic" augmentation is still corpus-derived — a surrogate built from
  post-cutoff data is cleaner.
- Documented evaluation split (THE running start):
  - Exact names from repo eval configs (`scripts/evaluation/configs/`):
  - **Benchmark I — in-domain (15), used for BOTH training & eval:**
    electricity_15min, monash_electricity_hourly, monash_electricity_weekly,
    monash_kdd_cup_2018, m4_daily, m4_hourly, m4_monthly, m4_weekly,
    monash_pedestrian_counts, taxi_30min, uber_tlc_hourly, uber_tlc_daily,
    monash_rideshare, monash_temperature_rain, monash_london_smart_meters.
  - **Benchmark II — zero-shot (27), "not used during training":**
    monash_traffic, monash_australian_electricity, ercot, **ETTm, ETTh**,
    exchange_rate, nn5, monash_nn5_weekly, monash_weather, monash_covid_deaths,
    monash_fred_md, m4_quarterly, m4_yearly, dominick, m5, monash_tourism_{monthly,
    quarterly,yearly}, monash_car_parts, monash_hospital, monash_cif_2016,
    monash_m1_{yearly,quarterly,monthly}, monash_m3_{monthly,yearly,quarterly}.
  - Datasets hosted at HF `autogluon/chronos_datasets` (downloadable).

> ⚠️ **PROVENANCE CORRECTION:** ETT (ETTh/ETTm) is in Chronos's **ZERO-SHOT**
> (Benchmark II), NOT in-domain (earlier draft had it wrong). Useful: ETT is the
> most-used TS benchmark, so a contamination-inflated ETT "zero-shot" is a
> high-impact headline. The leakage-free ETT loader still applies for our RLinear
> baseline; ETT just sits in the zero-shot stratum, not in-domain.

## THE CONTAMINATION HOOK (verified, from the paper itself)
The paper states the zero-shot temporal-disjointness rule — *"the start time of any
dataset within this category must be after the time stamp of the last observation
from the pretraining dataset"* — but **explicitly admits this constraint "was not
strictly enforced."** So Benchmark II "zero-shot SOTA" can be contaminated by
temporal/structural overlap. No hard calendar cutoff is stated; pretraining
predates the 2024-03 arXiv / 2024-10 TMLR dates.

→ The audit's edge: replace Chronos's loosely-enforced "zero-shot" with a STRICT
logical negative control and measure how much the headline zero-shot advantage
survives.

## Negative-control strata (Day 1-2 supply task)
1. **Synthetic surrogates (logically airtight floor):** ARMA + seasonal series,
   spectrum/scale/length-matched to in-corpus series. Contamination IMPOSSIBLE by
   construction. (Generate fresh; do NOT reuse KernelSynth/TSMixup which are
   corpus-derived.)
2. **Provably post-cutoff real series:** real datasets whose observations begin
   after the pretraining cutoff (recent sensor/market streams, post-2024 UCR/Monash
   additions). Document provenance; EXCLUDE any uncertain series (purity > volume).

## Three-stratum design (sharpened — a dose–response of contamination)
1. **In-domain (Benchmark I):** definitely in training → upper bound on the TSFM
   advantage.
2. **Nominal zero-shot (Benchmark II, incl. ETT):** Chronos's claimed zero-shot,
   temporal-disjointness NOT enforced → possibly contaminated.
3. **STRICT logical control:** synthetic surrogates (contamination impossible) +
   provably post-cutoff real series → the true generalization floor.

Prediction under contamination: gap(I) ≳ gap(II) ≫ gap(III). If gap(II) ≈ gap(I)
and both ≫ gap(III), the "zero-shot" benchmark is as inflated as in-domain — the
headline. If gap(II) ≈ gap(III), the zero-shot claim is clean (refute).
We have the leakage-free ETT loader (`/mnt/20t/RegFiLM/code/ett_data.py`,
train-only mu/sigma ✓) for the RLinear baseline on the ETT part of stratum 2.

## Dependent variable (immune to HIGAN's failure modes)
TSFM-minus-RLinear forecast error (MASE/MSE) per window, compared ACROSS strata:
**in-corpus gap − out-of-corpus gap**, block-bootstrap CI. Internal same-model
two-stratum contrast → no single-seed / selector-tautology / baseline-fragility.
RLinear = on-target trained linear baseline (`regfilm.py` linear branch).

## Pre-registered decision gate (write before looking at results)
- **Contamination confirmed:** out-of-corpus gap collapses toward 0 / flips sign
  while in-corpus gap stays large, AND the difference is bootstrap-stable.
- **Generalization confirmed (clean refute):** gap persists across both strata.
- BOTH publish.

## TODO next
- [ ] Enumerate exact Benchmark I / II dataset names from Appendix B Table 2
      (PDF or amazon-science/chronos-forecasting repo).
- [ ] Confirm TimesFM-2.0 + Moirai cutoffs/corpora for the multi-model sweep.
- [ ] Read TSFMAudit (2605.26161) + leakage (2510.13654) in full to confirm the
      per-model decomposition-via-strict-logical-control is the unclosed gap.

Sources: arXiv [2403.07815](https://arxiv.org/abs/2403.07815) (Chronos, TMLR 10/2024).
