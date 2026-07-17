# Design — Intrinsic image-manifold dimension via Jacobian rank

**Wave 3, Track 17.** Estimate the local intrinsic dimension of the
image manifold $\mathcal{M} = G(\mathcal{W})$ at a sampled latent
via the effective rank of the Jacobian $J = dG \in \mathbb{R}^{3HW \times LD}$.
This provides a complementary geometric characterisation to the
curvature measurements.

## Hypothesis

**H19.** For StyleGAN1 FFHQ, the effective rank of $J$ at a typical
latent is **~30-60**, far below the LD = 18 × 512 = 9216 latent
dimensionality and the 3HW = 3 × 1024² ≈ 3M ambient image
dimensionality. The image manifold is locally low-dimensional, as
expected for natural-image generators.

This matches Stanczuk et al. ICML 2024's data-manifold-dimension
finding via the score Jacobian (intrinsic dim 5-50 on natural
image datasets).

## Prior art

- **Stanczuk et al. ICML 2024** [arXiv:2212.12611](https://arxiv.org/abs/2212.12611)
  estimates intrinsic dimension via rank deficiency of the
  score-network's Jacobian — directly analogous.
- **MLE-based dimension estimators** Levina & Bickel 2004
  [paper](https://papers.nips.cc/paper_files/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html) —
  classical baseline.
- **Park-NeurIPS23** computes Jacobian SVD on diffusion h-space;
  effective rank not their headline metric but visible in their plots.

## Method

For each of 16 random latents in FFHQ (or bedroom):

1. Probe $K=64$ random unit directions in $\mathcal{W}$.
2. Compute $J e_k$ via JVP for each probe.
3. SVD the resulting $J \in \mathbb{R}^{3HW \times K}$.
4. Effective rank = $(\sum_i \sigma_i)^2 / \sum_i \sigma_i^2$
   (Roy & Vetterli 2007 entropy-style effective rank).
5. Report median and IQR across the 16 latents.

## Expected signal

- Bedroom: effective rank 20-50.
- FFHQ: effective rank 30-80.
- Both well below the corresponding $LD$ values (7168 / 9216).
- Matches Stanczuk's order-of-magnitude estimates.

## Failure modes

The probe basis is rank-K (= 64) — we cannot estimate effective rank
above 64. If true rank is higher we'll see a plateau at 64.
Mitigation: increase K or report "rank ≥ K" qualitatively.

## Compute budget

16 latents × 64 probes × ~3 s JVP (bedroom) / ~12 s (FFHQ) =
~1 h bedroom + ~3 h FFHQ.

## Deliverables

- `experiments/method/run_intrinsic_dim.py`
- `experiments/out/intrinsic_dim/` — table + spectrum plot.
- §6 application: "intrinsic dim diagnostic".
