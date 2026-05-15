# 03 — Second-order pushforward and extrinsic curvature

## Definition

For a direction $b \in T W^+$ and a base latent $wp$, the
**second-order pushforward** is the second derivative

$$ H(wp, b) = \frac{\partial^2 G(wp + \alpha b)}{\partial \alpha^2} \bigg|_{\alpha = 0} \;\in\; \mathbb{R}^{3HW}. $$

Equivalently, $H(wp, b) = \mathrm{Hessian}_G(wp) \cdot (b, b)$, where the
Hessian of $G$ is the bilinear map $T W^+ \times T W^+ \to T \mathcal{I}$.

We define the **second-order saliency**

$$ S^{(2)}_{h,w}(wp, b) = \frac{1}{3}\sum_c | H_{c,h,w}(wp, b) |, $$

and the **non-linearity ratio**

$$ \rho_{h,w}(wp, b) = \frac{S^{(2)}_{h,w}(wp, b)}{S^{(1)}_{h,w}(wp, b) + \epsilon}. $$

The pixel-averaged scalar $\bar\rho(b) = \frac{1}{HW}\sum_{h,w} \mathbb{E}_{wp} \rho_{h,w}(wp, b)$
is the **mean non-linearity of direction $b$**.

## Estimator via composed JVP

Define $f(\alpha) = G(wp + \alpha b)$. Then $f'(0) = dG_{wp}(b)$ and
$f''(0) = H(wp, b)$. Both are computable in *one combined* forward pass:

```python
def f(α):
    return G(wp + α * b)

def df(α):
    return jvp(f, (α,), (1,))[1]                # = f'(α), one extra pass

img0, second = jvp(df, (0,), (1,))              # = (f'(0), f''(0))
```

The inner `jvp` evaluates $f'(\alpha)$ by carrying a dual tangent through
the network. The outer `jvp` differentiates that operation once more, also
in forward mode. PyTorch's `torch.func.jvp` composes correctly because it
returns the *primal+tangent* pair that becomes the input of the outer
JVP — exactly the rule for nested forward-mode autodifferentiation
(Pearlmutter 1994 style, in modern guise).

Wall-clock: two forward graph traversals per sample, roughly $2.2\times$
the cost of one ordinary forward. Memory: $\sim 2 \times$ one forward
(no extra activations stored).

## Connection to extrinsic curvature

The image manifold $\mathcal{M}$ is embedded in $\mathbb{R}^{3HW}$. Its
*second fundamental form* at $G(wp)$, in the direction
$dG_{wp}(b) \in T_{G(wp)} \mathcal{M}$, is the projection of $H(wp, b)$
onto the **normal bundle** of $\mathcal{M}$:

$$ \mathrm{II}_{G(wp)}(b, b) \;=\; \Pi_{N_{G(wp)}\mathcal{M}}\,H(wp, b). $$

The full Hessian has *two components*:
1. A tangential part — change of basis as we move along $b$, i.e.
   reparameterisation.
2. A normal part — the *true* curvature, i.e. how $\mathcal{M}$ bends
   inside the ambient image space.

For our purposes the magnitude $|H|$ (which we compute) upper-bounds
$|\mathrm{II}|$ since
$|H|^2 = |\Pi_T H|^2 + |\Pi_N H|^2 \geq |\Pi_N H|^2$.

This is a meaningful empirical proxy because the tangential component
itself encodes how fast $G$'s parameterisation distorts along $b$, which
also matters for editing.

## Claim C2: high curvature ↔ topological transitions

We observe empirically (§19 of the report) that the *view* direction has
$\bar\rho_{view} \approx 23$, while *indoor_lighting* and *wood* have
$\bar\rho \approx 0.5$. View is a low-frequency, structural direction
(window/door geometry); the others are texture re-paints.

**Hypothesis (C2)**: structural edits inject or remove image-space
*level sets* (i.e., new edges, new regions), which are topologically
discontinuous in pixel space and thus produce large second-order
response. Texture edits perturb pixel values *continuously*, with small
second-order response.

We test this on three domains:
- **bedroom**: view (struct) vs lighting/wood/glossy (texture) — see §19.
- **FFHQ**: pose / smile vs hair colour / skin tone — expected pose to
  dominate ratio.
- **church**: layout / sky vs surface texture — expected layout to
  dominate.

If this pattern replicates across domains, we have a *measurable* signal
for "structural vs textural" attribute without semantic labels.

## Claim C4 precursor

Claim C4 (compositional editing failure ↔ Lie bracket non-zero) follows
because the Lie bracket
$[X_a, X_b] \,=\, X_a \cdot \nabla X_b - X_b \cdot \nabla X_a$
contains exactly the cross second derivatives that are non-negligible
when at least one of $X_a$, $X_b$ has large $H$. Section
[04_lie_bracket_composition.md](04_lie_bracket_composition.md) makes this
quantitative.

## Sample complexity

The Monte-Carlo estimator $\frac{1}{N}\sum_n S^{(2)}_{h,w}(wp_n, b)$ has
$O(\sigma_b N^{-1/2})$ standard error, where $\sigma_b$ is the
direction-specific pushforward variance. Empirically $\sigma_b$ is large
for high-curvature directions, so robust ranking of $\bar\rho_b$ across
attributes requires $N \gtrsim 32$ to separate ratios at the
$\rho = 0.5$ vs $\rho = 23$ scale we observed.

## What this section does *not* claim

- We do **not** estimate the full Hessian (only its action $b^T H b$).
- We do **not** separate tangential from normal components in our
  reported number, only bound $|II|$.
- We do **not** require $G$ to be globally smooth — only $C^2$ at the
  $wp$ samples actually used, which is generic.
