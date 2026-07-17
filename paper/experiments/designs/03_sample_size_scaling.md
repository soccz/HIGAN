# Design — Sample size scaling + bootstrap CI on C1 / C3

**Track 3.** Address the reviewer-anticipated "your N is too small"
critique. Show that the headline numbers ($\bar\rho$ ratios, C3 IoU
scores) are stable at $N \geq 64$ and report bootstrap 95% CIs on
the existing tables.

## Hypothesis

**H4.** The bedroom + FFHQ + church $\bar\rho$ rank orderings and the
C3 layer-IoU rank orderings are *invariant* under $N \in \{8, 16, 32,
64, 128\}$ to within bootstrap CI width. The N=16 numbers we currently
report are within the bootstrap CI of the N=128 estimate.

If true, the existing tables are trustworthy with explicit error bars
and we can publish them as-is + the new CIs. If not, the larger-N
numbers replace the published ones.

## Prior art

- **Bootstrap percentile CI** ([Efron 1979](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full))
  — standard for non-parametric mean/median uncertainty.
- **DragGAN N=1000**, **InterFaceGAN N~500–1k**, **StyleSpace 1–10k
  for DCI** — community baselines for adequate N.
- **Standard error of correlation** for Spearman/Pearson: bootstrap-CI
  is the consensus method in the GAN-interpretability literature when
  small-N (n<30) prevents Fisher z.

## Method

For each domain × claim:

**C1.** Re-run $\bar\rho_a$ on N = 8, 16, 32, 64, 128 noise seeds per
attribute. The N=8/16 are read from existing
[run_higher_order.py](../domains/ffhq/run_higher_order.py) outputs
(no recompute). N=32/64/128 are new.

**C3.** Re-run the layer-IoU saliency computation on N = 8, 16, 32,
64 (128 too expensive for FFHQ-1024).

**Bootstrap.** For each fixed N, resample 10 000 times with replacement
from the N per-seed estimates; report mean ± 2.5/97.5-percentile CI.

**Stability metric.** For each N, compute Spearman correlation between
the per-attribute ordering at this N and at N_max. If $\rho \geq 0.95$
for $N \geq 64$, the small-N numbers are stable.

## Expected signal

- $\bar\rho$ CI width shrinks $\propto 1/\sqrt{N}$. At N=64 the CI of
  each attribute's ratio is < 20 % of the mean.
- Rank Spearman between N=16 and N=128 is $\geq 0.95$ for both
  bedroom and FFHQ.
- C3 mean across attributes is within ±0.02 of the N=16 number for
  every $N \geq 32$.

## Failure modes

1. **N=128 is too expensive for FFHQ-1024.** Cap at N=64 for FFHQ.
2. **Composed JVP at large N exhausts VRAM.** Process serially with
   `empty_cache()` after each sample — we already do this.
3. **C3 saliency drifts at large N.** If the saliency map changes
   significantly with N, the IoU at top-20% could shift. We track
   per-pixel saliency correlation across N values; if < 0.95, report
   the failure.

## Compute budget

- C1 N=128 on bedroom (256²): ~1 h.
- C1 N=128 on FFHQ-1024 capped to N=64: ~3 h.
- C3 N=64 on bedroom: ~2 h.
- C3 N=64 on FFHQ: ~4 h.
- **Total ~10 h.** Run in background overnight.

## Deliverables

- `experiments/metrics/run_sample_scaling.py` — unified scaling driver
- `experiments/out/sample_scaling/` — CI tables + ratio-vs-N curves
- Update all per-attribute table values in
  [05_experiments.tex](../../sections/05_experiments.tex) with
  `mean ± 95% CI` formatting.
