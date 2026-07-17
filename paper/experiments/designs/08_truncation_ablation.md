# Design — Truncation-ψ ablation on FFHQ C1/C2

**Wave 2, Track 8.** Show that C1/C2 results on FFHQ are not an
artefact of the truncation parameter ψ. Three ψ values, same
attributes, same metrics.

## Hypothesis

**H10.** The per-attribute curvature ratio $\bar\rho_a$ and the
CLIP-feature path ratio are **invariant** (rank-stable) across
ψ ∈ {0.5, 0.7, 1.0}. The Spearman rank correlation between
attribute orderings at any two ψ values is ≥ 0.9.

## Prior art

- **StyleGAN truncation** Karras et al. CVPR 2019
  [arXiv:1812.04948](https://arxiv.org/abs/1812.04948) §B introduces
  the truncation trick: $W' = W_{\text{avg}} + \psi (W - W_{\text{avg}})$
  to trade off fidelity vs diversity. ψ ∈ [0, 1].
- **StyleGAN2** Karras et al. CVPR 2020 §3.4 keeps the same ψ
  parameterisation. InterFaceGAN FFHQ uses ψ = 0.7 default.
- This is a standard reviewer-anticipated ablation in any
  StyleGAN-interpretation paper (see StyleSpace §4.2 for the
  template).

## Method

For each ψ ∈ {0.5, 0.7, 1.0}, repeat the FFHQ C1/C2 measurement of
[run_higher_order.py](../domains/ffhq/run_higher_order.py) +
[run_c2_path_curvature_ffhq.py](../metrics/run_c2_path_curvature_ffhq.py):

- Override `G.truncation_psi = ψ` in the FFHQGenerator wrapper.
- 5 attributes × $N = 32$ noise seeds.
- Report $\bar\rho_a$ and CLIP-path-ratio per attribute per ψ.
- Spearman rank correlation between ψ=0.5↔1.0, ψ=0.7↔1.0 attribute
  orderings.

## Expected signal

- Absolute $\bar\rho$ values decrease at lower ψ (less diverse
  samples → smaller second derivatives in absolute magnitude).
- **Relative ordering** of attributes preserved: pose >
  eyeglasses > gender > age > smile, all three ψ values.
- Spearman r between any pair of ψ orderings ≥ 0.9.

## Failure modes

1. At ψ = 1.0 (no truncation) the FFHQ generator produces more
   extreme faces, potentially making JVP numerically unstable.
   Mitigation: filter out samples with $|x_0| > 1$ before averaging
   (clip-then-mean is conservative).
2. If ordering changes at ψ=1.0, that would be a *positive* finding
   — the C2 ordering depends on the truncation regime. Report
   either way.

## Compute budget

- 3 ψ × 5 attrs × 32 seeds × ~2 s composed JVP + ~10 s CLIP path
  = ~50 min total.

## Deliverables

- `experiments/domains/ffhq/run_truncation_ablation.py`
- `experiments/out/ffhq_truncation/` — table + plot
- §5.exp-c1 sentence noting truncation invariance.
