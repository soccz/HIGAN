# 04 — Lie brackets and compositional editing failure

## The compositional editing question

Given two attribute directions $X_a, X_b \in T W^+$ and a base $wp$, we
can edit the image either by

(i) **linear superposition**: render $G(wp + \alpha X_a + \beta X_b)$, or

(ii) **sequential composition**: render $G(wp + \alpha X_a)$, then
displace by $\beta X_b$ at *that* point.

If the generator were linear in $wp$ both would give identical results.
Empirically they do not, and the discrepancy is large for some pairs and
small for others. Section 13 of the empirical report measures this as
the correlation between $S^{(1)}(a+b)$ and $S^{(1)}(a) + S^{(1)}(b)$.
We observed:

- *indoor_lighting + wood*: corr $0.977$ — superposition holds.
- *carpet + wood*: corr $0.970$ — superposition holds.
- *cluttered_space + glossy*: corr $0.727$ — small interference.
- *wood + view*: corr $0.547$ — large interference.
- *indoor_lighting + view*: corr $0.518$ — large interference.

The empirical signal: pairs where one direction has high
second-order saliency (view) violate superposition.

## Lie bracket formalism

Treat each attribute boundary as a constant vector field on $W^+$:
$X_a(wp) \equiv b_a$ for all $wp$. The pushforward gives image-space
vector fields $Y_a(wp) := dG_{wp}(b_a)$. These are *not* constant in
image space: $Y_a$ depends on $wp$ through the non-linear $G$.

The **Lie bracket** of two vector fields on the image manifold is

$$ [Y_a, Y_b](G(wp)) \;=\; \frac{\partial Y_a}{\partial \alpha_b}(G(wp)) \;-\; \frac{\partial Y_b}{\partial \alpha_a}(G(wp)). $$

Equivalently, in $W^+$ coordinates (where $X_a$ are constant), the
bracket on $\mathcal{M}$ reduces to

$$ [Y_a, Y_b](G(wp)) \;=\; d^2 G_{wp}(b_a, b_b) \;-\; d^2 G_{wp}(b_b, b_a) \;=\; 0 $$

by symmetry of the Hessian. So the *Lie bracket itself* is zero for
constant vector fields on $W^+$ — which means compositionality
**should** hold linearly.

What gives? The empirical failure of superposition comes from a
*different* fact: even though the Lie bracket of two constant vector
fields is zero, the **second-order Taylor remainder** of the joint
displacement is not. Specifically, for a step
$wp' = wp + \alpha b_a + \beta b_b$,

$$ G(wp') \;=\; G(wp) \;+\; \alpha\, dG\,b_a \;+\; \beta\, dG\,b_b \;+\; \tfrac{1}{2}\, d^2G(\alpha b_a + \beta b_b,\,\alpha b_a + \beta b_b) \;+\; O(\delta^3). $$

The cross term is $\alpha\beta\, d^2G(b_a, b_b)$. This is exactly the
**mixed Hessian**, and it is non-zero whenever the generator is curved
in either direction.

## Predictive form

We propose the following empirical predictor for the pairwise
compositional interference:

$$ \mathrm{Interference}(a, b) \;\approx\; \frac{\mathbb{E}_{wp}\big[\, \| d^2G_{wp}(b_a, b_b) \|_2 \,\big]}{ \mathbb{E}_{wp}\big[\, \| dG_{wp} b_a \|_2 \cdot \| dG_{wp} b_b \|_2 \,\big]}. $$

The mixed Hessian $d^2G(b_a, b_b)$ is again estimable in *one composed
JVP*: differentiate $G(wp + \alpha b_a + \beta b_b)$ first in $\alpha$
then in $\beta$:

```python
def f(α, β):
    return G(wp + α * b_a + β * b_b)
inner = lambda β: jvp(lambda α: f(α, β), (0,), (1,))[1]
mixed = jvp(inner, (0,), (1,))[1]                       # = d²G(b_a, b_b)
```

The denominator is $\mathbb{E}\|dG b_a\| \cdot \mathbb{E}\|dG b_b\|$, both
already known from the first-order pipeline.

## Empirical claim C4 (to validate)

**C4**. Pairs of HiGAN bedroom boundaries $(a, b)$ for which the predictor
$\mathrm{Interference}(a, b)$ is large are exactly the pairs that show
low correlation between $S^{(1)}(a + b)$ and $S^{(1)}(a) + S^{(1)}(b)$
(compositional non-linearity from §13 of the report).

Quantitatively, we expect Spearman $|\rho_{spearman}| > 0.6$ between the
predictor and the empirical interference correlation across all
$\binom{8}{2} = 28$ pairs.

## Connection to second-order saliency

The diagonal case ($a = b$) gives exactly $d^2G(b, b)$, which is the
second-order saliency of §03. So C4 says: **the same composed-JVP
machinery that estimates curvature also predicts how attribute pairs
interfere**. One tool, two payoffs.

## What we cannot conclude

- For *vector fields varying in $W^+$* (e.g., learned ones), the Lie
  bracket is genuinely non-zero and gives stronger non-linearity than
  the Hessian alone. Most current GAN boundaries (HiGAN, InterFaceGAN,
  SeFa) are linear directions, so the Hessian story suffices. EditGAN
  and similar non-linear edits would need the full Lie bracket.
- Our prediction is one-sided: high curvature $\Rightarrow$
  non-superposition. The converse can fail if non-linearity stems from
  *higher-order* terms beyond the Hessian.
