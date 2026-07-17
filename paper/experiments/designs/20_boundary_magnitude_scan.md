# Design — Boundary direction magnitude scan

**Wave 4, Track 20.** Currently boundaries are L2-normalised to unit
magnitude before measuring curvature. Test how ρ(α) varies as a
function of α magnitude (i.e. how far along the boundary we go).
This is a "non-linearity onset" diagnostic.

## Hypothesis

**H22.** For texture-class attributes (smile, age in FFHQ), ρ(α)
stays approximately constant across α ∈ [0.5, 3.0] σ — the local
curvature is uniform. For structural attributes (pose, eyeglasses),
ρ(α) **grows roughly linearly** with α — the curvature increases
as we approach a topological transition.

Specifically: slope of $\log\bar\rho$ vs $\log\alpha$ on
texture attributes < 0.3; on structural attributes ≥ 0.5.

## Prior art

- **InterFaceGAN** Shen 2020 § 4.2 ablates α ∈ {1, 2, 3} for editing
  but doesn't measure curvature scaling.
- **StyleGAN-NADA** / **StyleCLIP** report visual artefacts for
  α ≥ 3σ but no quantitative measurement.
- This is the curvature-version of the "edit strength vs identity
  preservation" classic.

## Method

For each (FFHQ attribute, 8 seeds), compute ρ at:
- $\alpha \in \{0.25, 0.5, 1.0, 2.0, 3.0\}$ σ
- Slope of $\log\bar\rho$ vs $\log\alpha$ via linear regression.
- 95% CI on slope via bootstrap.

## Expected signal

- Smile: slope $\approx 0.1$ (essentially constant).
- Pose: slope $\approx 0.7$.
- Eyeglasses: slope $\approx 0.6$.

If slope > 0.5 for an attribute, that attribute is at risk of a
topology transition somewhere in $[1, 3]\sigma$ — paper insight:
**curvature scaling tells you how much you can push a direction
before it breaks**.

## Compute budget

- 5 attrs × 5 α values × 8 seeds × ~12 s = ~40 min.

## Deliverables

- `experiments/domains/ffhq/run_alpha_magnitude_scan.py`
- `experiments/out/ffhq_alpha_scan/` — log-log curves + slope table.
- §6 application: "edit-safety budget per direction".
