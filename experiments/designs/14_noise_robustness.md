# Design — Noise-seed robustness: variance bands on every C1 metric

**Wave 3, Track 14.** Currently we report only mean ratios. Reviewers
expect variance under independent seeds (different random initialisations
of the noise generator). Test that the entire C1 table is robust to
seed choice.

## Hypothesis

**H16.** For every (attribute, domain) pair, the C1 ratio mean varies
by **less than 10\% (relative)** across 5 independent seeds, each with
N=32 noise samples per measurement. Spearman rank correlation between
attribute orderings under different seeds is ≥ 0.95.

## Prior art

Common reviewer ablation across the ML literature. No specific
precedent paper to cite — standard methodology.

## Method

For each seed $s \in \{2027, 2028, 2029, 2030, 2031\}$:
- Re-run [run_higher_order.py](../domains/ffhq/run_higher_order.py)
  (and the bedroom + church equivalents) with `--seed s`.
- Extract per-attribute mean ratio.

Report:
- mean ± std across seeds per attribute
- Spearman rank Spearman between attribute orderings at seed pairs

## Expected signal

- Within-seed std / mean < 10\% for all 16 attribute-domain pairs.
- All pair-wise Spearman ≥ 0.95.

## Failure modes

If some attribute (e.g. church `clouds`) has high seed-variance, that
indicates the boundary itself is unstable at the measured N — we
report this as a *honest signal of measurement uncertainty* and
recommend a larger N for that attribute.

## Compute budget

5 seeds × (8 + 5 + 3 = 16) attrs × 32 samples × ~3-8 s = ~5-10 hours
total.

## Deliverables

- `experiments/metrics/run_noise_robustness.py`
- `experiments/out/noise_robustness/` — per-domain table.
- §5.exp-c1 sentence "mean ± std across 5 seeds" added.
