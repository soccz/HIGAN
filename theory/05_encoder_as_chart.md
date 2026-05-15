# 05 — The encoder as a coordinate chart

## Setting

Given a generator $G : W^+ \to \mathcal{M}$, an **encoder** is a learned map
$E : \mathcal{I} \to W^+$ aiming for $G \circ E \approx \mathrm{id}_\mathcal{M}$.
Together, $(G, E)$ form an *approximate* coordinate atlas: $G$ is the
parameterisation, $E$ is the chart.

In the language of manifolds, $E$ defines local coordinates on
$\mathcal{M}$. The classical criterion of chart quality is *not* just
that points map back to themselves, but that the **derivative structure**
is preserved:

$$ dE_x \circ dG_{wp} \;\approx\; I_{T W^+}, \quad \text{for } x = G(wp). $$

A chart that recovers points but distorts tangent vectors is locally
homeomorphic but not a good chart for differential analysis.

## Reconstruction vs derivative agreement

The standard metric for encoder quality is the reconstruction loss
$\mathcal{L}_{rec} = \| G(E(x)) - x \|_2^2$ (or LPIPS, $\ell_1$, etc.).
This is a *zeroth-order* criterion.

We propose a stronger **first-order** criterion: agreement of saliency
maps between the encoder's $wp$ and the true $wp$ (when known, e.g., for
synthetic test images). For an attribute direction $b$:

$$ \mathrm{ChartConsistency}(E; b, wp) \;=\; \mathrm{corr}_{(h,w)}\big(\, S^{(1)}(E(G(wp)), b),\; S^{(1)}(wp, b) \,\big). $$

This number is in $[-1, 1]$, $1$ being perfect derivative agreement.

## Empirical observation (claim C5)

Across encoder training checkpoints (1k, 5k, 10k, 20k, 40k iterations on
HiGAN bedroom), we measure:

| iter | recon MSE | saliency correlation w/ GT |
|----|----------|---------------------------|
| 1k  | 0.047    | **+0.008** |
| 5k  | 0.040    | +0.028     |
| 10k | 0.043    | +0.126     |
| 20k | 0.040    | +0.164     |
| 40k | 0.039    | **+0.359** |

**Recon MSE improves by 18% from 1k to 40k. Saliency correlation
improves 45×.** The two are *decoupled*: an encoder can plateau on
reconstruction long before its chart is consistent.

This is claim C5: **chart consistency is the stronger and more
informative training-progress signal**.

## Theoretical implication

If $E$ is a coordinate chart and $G$ is its inverse,
$E_*$ is the inverse of $G_*$ (in the chain rule sense). For our purposes
$E_*$ is computed by reverse-mode autodifferentiation through the
encoder. So the comparison $E_* \cdot G_* b$ vs $b$ is a direct test of
chart consistency.

Section 11 of the empirical report already provides reverse-mode encoder
attention. The natural extension is: for every attribute boundary $b$,
compute both $G_* b$ (generator saliency) and $E_* G_* b$ (the chart
pull-back), and measure the consistency.

In practice, since we don't know $b$ in *image space*, we measure the
weaker but observable proxy: agreement of $S^{(1)}$ maps under $E$'s wp
vs the true wp.

## Why this matters for the paper

1. **Methodological**: a new training-progress signal that doesn't
   collapse for under-fit encoders.
2. **Practical**: tells the practitioner *when to stop* encoder training
   (no improvement in chart consistency for $K$ iterations).
3. **Conceptual**: ties encoder training to differential geometry of the
   manifold, not just pixel-space distance.

## What we do **not** claim

- We do not claim $E$ is globally a homeomorphism (the image manifold is
  topologically non-trivial and a single chart cannot cover it).
- We do not claim chart consistency *implies* perfect editing
  performance; only that it is a *necessary* condition for derivative
  faithful editing.
