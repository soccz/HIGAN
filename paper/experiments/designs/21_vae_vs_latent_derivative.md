# Design — VAE-decoded vs latent-space derivative on SD

**Wave 4, Track 21.** On SD, our Track 1 measurement is
$\partial x_0/\partial v$ where $x_0$ is the decoded image. Separate
the contribution of the VAE decoder from the U-Net by also measuring
$\partial z_0/\partial v$ where $z_0$ is the post-final-DDIM-step
latent (pre-decoder).

## Hypothesis

**H23.** Pixel-space curvature ratio and latent-space curvature
ratio are rank-correlated at Spearman r ≥ +0.85 across the 5 face
attributes — the VAE decoder preserves the qualitative curvature
ranking, contributing only a smooth scaling factor.

If true: the rank claim in C1/C2 is U-Net-driven, not VAE-driven.

## Prior art

- **Latent diffusion (LDM)** Rombach et al. CVPR 2022
  [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) — separates
  the VAE compressor from the diffusion U-Net by design.
- **Park-NeurIPS23** measures Jacobian in latent space $x_t$; doesn't
  cross-check against pixel space.

## Method

For each (attr, seed) in Track 1's setup:
- Save *both* $\dot z_0$ (latent-space tangent) and $\dot x_0$
  (pixel-space tangent after VAE decode).
- Compute pixel-curvature ratio (already done) and latent-curvature
  ratio (using $\|d^2 z_0/d\alpha^2\| / \|d z_0/d\alpha\|$).
- Spearman correlation between attribute orderings in both spaces.

## Expected signal

- Spearman r ≥ +0.85.
- Absolute ratios differ (VAE decoder has its own non-linearity),
  but ordering preserved.

## Compute budget

Reuse Track 1's saved tensors. Pure post-processing, <1 minute.

## Deliverables

- `experiments/diffusion/run_vae_vs_latent_ratio.py`
- `experiments/out/sd_vae_vs_latent/` — table + scatter.
- §5.exp-c1 supplementary paragraph on VAE-isolation.
