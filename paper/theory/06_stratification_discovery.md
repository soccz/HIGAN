# 06 — Stratification of the image manifold via tangent-space clustering

## Setup

Consider the tangent space $T W^+ \cong \mathbb{R}^{L \cdot D}$ at the
generator's prior mean (zero-centred). A *direction* is a unit vector
$b \in S^{L\cdot D - 1}$. The associated pushforward
$dG_{wp}(b) \in T_{G(wp)} \mathcal{M}$ depends on the base $wp$, but
its image-space *signature* — the saliency map $S^{(1)}(b)$ averaged
over $wp$ — is a function of $b$ alone (modulo Monte-Carlo noise).

This gives us a map

$$ \Phi : S^{L \cdot D - 1} \to \mathbb{R}_{\geq 0}^{H \times W}, \quad
b \mapsto \mathbb{E}_{wp \sim p(W^+)}\big[\, \tfrac{1}{3} \sum_c | dG_{wp}(b)|_{c} \,\big]. $$

$\Phi$ defines a *stratification* of the unit sphere by what its
directions do in image space: two directions with the same $\Phi(b)$ are
"editing the same region".

## Empirical procedure

1. Sample $N$ random unit directions $b_1, \dots, b_N$ on $S^{L \cdot D - 1}$
   (or, more useful in practice, $L$ separate single-layer spheres
   $S^{D-1}$ per layer).
2. Compute $\Phi(b_n)$ for each direction (one composed JVP each).
3. Cluster $\{\Phi(b_n)\}$ with k-means (after PCA reduction).
4. Inspect cluster centroids visually; assign semantic labels via CLIP
   zero-shot (§17 of the report).

In our experiments with 256 single-layer random directions on the bedroom
generator, 8 k-means clusters produce a clear taxonomy:

- 1 large cluster (n=37) at layer 1 is labelled "a view through a window"
  by CLIP — the unsupervised rediscovery of HiGAN's hand-curated *view*
  boundary.
- 1 cluster (n=21) at layer 7 is labelled "a pillow / a blanket / soft
  texture" — a bedding family not present in HiGAN's 8 boundaries.
- Layer 13 produces multiple small clusters with mixed labels ("clean
  room", "metal surface") — fine-detail noise.

This is empirical C6: **the unsupervised stratification is meaningful and
rediscovers hand-curated boundaries without labels**.

## Why a stratification, not a foliation

A *foliation* would require the directions to organize into smooth
leaves of equal dimension. They don't: layer 0 produces near-zero
response, layer 13 produces high-frequency noise, and the middle layers
have clean structural / textural strata. The right structure is a
**stratification** — a decomposition into pieces of varying dimension,
some of which (layer 0) are degenerate.

Concretely, the **layer index** $\pi : W^+ \to \{1, \dots, L\}$ is the
first stratification, and each fibre $\pi^{-1}(\ell)$ admits a further
stratification by the saliency clustering.

## Relation to existing unsupervised direction methods

- **GANSpace** (Härkönen 2020) finds directions by PCA on intermediate
  layer activations. The top PCs are often interpretable. Our method is
  complementary: we sample random directions and *cluster their
  pushforwards in image space*, not their representations in latent
  space.
- **SeFa** (Shen & Zhou 2021) factorises the affine layer of StyleGAN to
  find directions that maximally affect output. This is a closed-form
  global decomposition; our method is a sample-based local one.
- **StyleSpace** (Wu 2021) enumerates individual style channels and
  measures their localised effects. This is one of $L \cdot D$ axis
  directions; we generalise to arbitrary unit vectors and cluster.

A unified comparison should be quantitative: **how well does each method
recover hand-curated boundaries (precision/recall on rediscovery), per
domain?**

## Claim C6 statement

**C6**. K-means clustering of saliency signatures $\Phi(b)$ for $N$
random unit directions yields an unsupervised stratification of the
image manifold. With $N \gtrsim 128$, the stratification's largest
clusters consistently rediscover hand-curated attribute boundaries
across at least three generative domains, with $\geq 70\%$ semantic
agreement on cluster labelling via CLIP zero-shot.

This is the most fragile of our six claims (depends on $k$ choice,
clustering algorithm, semantic-label tolerance) and we devote part of
the paper's experimental section to ablations.
