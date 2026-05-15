# Theoretical framework — overview

The trained generator of a GAN, viewed as a smooth map
$G : \mathbb{R}^L \to \mathbb{R}^{3HW}$
between a Euclidean latent space and the ambient image space, induces a
*finite-dimensional image manifold* $\mathcal{M} = G(\mathbb{R}^L)
\subset \mathbb{R}^{3HW}$.

In this paper we view many "GAN interpretation" tasks
— saliency, attribute editing, disentanglement,
unsupervised attribute discovery —
as **classical differential-geometric operations** on this manifold.

| Task in literature                              | Geometric object                                |
|------------------------------------------------|-------------------------------------------------|
| Per-pixel saliency for a latent direction       | Pushforward of a tangent vector through $G$     |
| Curvature of an edit (saturation, looping)      | Second-order pushforward / extrinsic curvature  |
| Compositional editing                           | Composition of vector fields on $\mathcal{M}$   |
| Editing failure when combining two attributes   | Lie bracket of those vector fields              |
| Layer-localised attribute boundary              | Section of a layer-indexed fiber bundle         |
| Encoder $E$                                     | Approximate coordinate chart $\mathcal{M} \to W^+$ |
| Random direction discovery                      | Stratification of $\mathcal{M}$ by tangent-space clustering |

Forward-mode automatic differentiation (`torch.func.jvp`) is the natural
computational tool. It evaluates pushforwards in one forward-with-tangent
pass and *composes*, giving Hessian-vector products by nesting. For our
generators that consume a scalar perturbation along a direction and produce
many output pixels, forward mode is the strictly correct direction —
reverse mode would require ~3HW backward passes for the same information.

The six concrete claims developed in the chapters that follow:

- **C1** ([01_manifold_view.md](01_manifold_view.md), [03_hvp_curvature.md](03_hvp_curvature.md)): the image manifold of a trained GAN has measurable extrinsic curvature along every latent direction, well-estimated by composed forward-mode JVP.
- **C2** ([03_hvp_curvature.md](03_hvp_curvature.md)): high-curvature directions correspond to *topological* image changes (objects appearing/disappearing); low-curvature directions are *recolourings*.
- **C3** ([01_manifold_view.md](01_manifold_view.md) §fiber-bundle): HiGAN attribute boundaries live in a fiber bundle over the W+ layer index; their interpretation is layer-localised.
- **C4** ([04_lie_bracket_composition.md](04_lie_bracket_composition.md)): compositional editing failure is governed by the Lie bracket of attribute vector fields; this is *predictable* from the second-order analysis.
- **C5** ([05_encoder_as_chart.md](05_encoder_as_chart.md)): an encoder approximates a coordinate chart, and its training dynamics are best measured by chart consistency (derivative agreement), not reconstruction error.
- **C6** ([06_stratification_discovery.md](06_stratification_discovery.md)): random tangent direction clustering yields an unsupervised stratification of $\mathcal{M}$ that recovers semantic attribute axes.

These claims are not independent: C4 follows from C1 and the standard formula
$[X,Y] = X \cdot \nabla Y - Y \cdot \nabla X$ applied to attribute vector
fields; C5 is the inverse-map perspective of C3; C6 is the
many-direction extension of C1 + C2.

The core empirical loop, repeated for every claim:

1. Define the geometric quantity formally.
2. Provide a Monte-Carlo estimator (typically the JVP-based one).
3. State the assumptions (smoothness, sample size, support).
4. Validate empirically on at least three generative domains (bedroom,
   FFHQ, LSUN church).
5. Compare to existing methods that approximate the same quantity.
