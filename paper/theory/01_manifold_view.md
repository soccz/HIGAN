# 01 — The image manifold and its tangent structure

## Setup

Let $W^+ \cong \mathbb{R}^{L \cdot D}$ be the StyleGAN W+ latent space ($L$
layers, each $D = 512$-dimensional).
Let $\mathcal{I} = [-1, 1]^{3 \times H \times W}$ be the image domain.
The trained generator is a smooth, deterministic map

$$ G : W^+ \to \mathcal{I}, \quad wp \mapsto G(wp). $$

Define the **image manifold** of $G$ as the image set
$\mathcal{M} = G(W^+) \subset \mathcal{I}$.
Throughout this paper, we assume $G$ is $C^2$ at every $wp$ of interest
(true for StyleGAN with its conv/AdaIN/LeakyReLU architecture except on a
measure-zero set where ReLU joints kink).

## Tangent space and pushforward

The differential of $G$ at $wp$ is the linear map

$$ dG_{wp} : T_{wp} W^+ \to T_{G(wp)} \mathcal{M} \subset \mathbb{R}^{3HW}, $$

with $T_{wp} W^+ \cong \mathbb{R}^{L \cdot D}$ since $W^+$ is Euclidean.
Given a tangent vector $v \in T_{wp} W^+$, the **pushforward**
$dG_{wp}(v) \in \mathbb{R}^{3HW}$ is the first-order linear response of
the image to a perturbation along $v$.

For a *unit* tangent vector $v$ this is exactly what forward-mode
autodifferentiation computes in a single pass:

$$ \text{jvp}(G, wp, v) = G(wp) + \varepsilon \cdot dG_{wp}(v) + O(\varepsilon^2). $$

The second tuple returned by `torch.func.jvp(G, (wp,), (v,))` is exactly
$dG_{wp}(v)$.

## Per-pixel pushforward magnitude → first-order saliency

For an attribute direction $b \in T W^+$ and a sample latent $wp$, define
the **first-order saliency** of $b$ at pixel $(c, h, w)$ as

$$ S^{(1)}_{c,h,w}(wp, b) = \left| [dG_{wp}(b)]_{c,h,w} \right|. $$

The single-channel summary $S^{(1)}_{h,w} = \frac{1}{3}\sum_c S^{(1)}_{c,h,w}$
is the per-pixel scalar map plotted in §06 of our empirical report.
Empirical estimators average over $N$ samples drawn from $G$'s prior:

$$ \bar{S}^{(1)}_{h,w}(b) = \frac{1}{N}\sum_{n=1}^{N} S^{(1)}_{h,w}(wp_n, b). $$

This is *not* the same as Grad-CAM. Grad-CAM relies on a class probability
gradient through a classifier; here the geometric object exists
independently of any downstream task, and is computed through the
generator alone.

## Why forward mode is the correct direction

The image space has dimension $3HW$ (e.g., $196{,}608$ for $256\times256$);
the perturbation direction is one scalar (the magnitude $\alpha$ along $b$).
Reverse mode is efficient when the output is low-dimensional (one scalar
loss) and the input is high-dimensional (many parameters). Our problem is
the opposite — single input dimension, many output dimensions — so
**forward mode is exactly right and gives all per-pixel sensitivities
in one pass**.

Reverse mode would require $3HW$ backward passes to obtain the same
information, or $\Theta(\log(3HW))$ with vmap, both quadratically more
expensive in our practical regime.

## Image manifold as a learned hypothesis

$\mathcal{M}$ is a $L D$-dimensional immersed submanifold of $\mathbb{R}^{3HW}$
(generically). Its embedding is *learned*: in particular,
- the choice of layer (W+ vs W) controls how the parameterisation factors;
- the intermediate activations control the local rate of change;
- the AdaIN normalisation introduces conditioning on $b$ that depends on
  the *current* $wp$, so the Jacobian $dG_{wp}$ is genuinely non-linear in
  $wp$.

A key consequence is **claim C3**: an attribute direction $b$ is not an
intrinsic property of $W^+$; it is a *section of a fiber bundle*
$\pi : W^+ \to \{1, \dots, L\}$ that projects to the layer index. The
HiGAN authors trained these directions to lie in particular layers, and
when displaced to a different fiber they generate large but semantically
incoherent pushforwards (see §08 of the report — the 8×14 matrix).

## Sampling and Monte-Carlo regime

The expectations above are taken with respect to the generator's *latent
prior* $\mathcal{N}(0, I)$ on $\mathbb{R}^{z}$ pushed through the mapping
network. In practice we sample $N$ latents from this distribution and
average. Variance scales as $O(N^{-1/2})$. For most claims we use
$N \in \{32, 64\}$; for the disentanglement matrix (§09) and the
random-direction taxonomy (§16) we use $N = 32$ or more directions.

## Notation summary

| symbol | meaning |
|---|---|
| $W^+ = \mathbb{R}^{L \cdot D}$ | StyleGAN W+ latent space |
| $\mathcal{M} \subset \mathbb{R}^{3HW}$ | image manifold |
| $G : W^+ \to \mathcal{I}$ | trained generator |
| $b \in T W^+$ | attribute direction (boundary) |
| $dG_{wp}(b) = \partial G/\partial \alpha$ at $\alpha = 0$ | pushforward |
| $S^{(1)}, S^{(2)}$ | first-/second-order saliency maps |
| $[X, Y]$ | Lie bracket of vector fields |
| $\pi : W^+ \to \{1..L\}$ | layer fibration |
