# Six claims — one-pager evidence ledger

Each claim has: formal statement, measurement procedure, evidence so
far, and what's missing for paper-grade validation.

---

## C1. Image manifolds of trained GANs have measurable extrinsic curvature

**Statement.** For any latent direction $b$, the second-order saliency
$S^{(2)}(b) = |\partial^2 G/\partial \alpha^2|$ is well-defined and
measurable in one composed forward-mode JVP. Its magnitude relative to
$S^{(1)}(b)$ is finite, bounded, and varies non-trivially across
directions.

**Measurement.** Composed JVP through $G$:
```
img0, first  = jvp(f, (α=0,), (1,))
_,    second = jvp(λα: jvp(f,(α,),(1,))[1], (α=0,), (1,))
```
Take $|second|$ averaged over RGB and $N$ samples.

**Evidence.** §19 of report: indoor_lighting $\bar\rho = 0.495$, wood
$0.624$, view $23.217$.

**Missing for paper.**
- Replicate on FFHQ + church (3 domains required).
- Variance bands across sample sizes $N \in \{8, 16, 32, 64, 128\}$.
- Comparison: finite-difference approximation accuracy as $\delta \to 0$.

---

## C2. High curvature ↔ topological image transitions

**Statement.** Directions with $\bar\rho > \tau$ (threshold to be set
empirically, around $\rho \approx 5$) produce edits that change the set
of image-space objects (e.g., window appears/disappears), whereas
low-$\bar\rho$ directions only re-paint pixel values.

**Measurement.** Either
(a) Visual inspection of sweep at distances $\alpha \in [-3, 3]$.
(b) Quantitative: pretrained Mask2Former segmentation on swept frames;
count number of distinct *region labels* per frame; high-$\rho$
directions should change the count.

**Evidence so far.** §19 + §23 morph videos qualitatively show this for
bedroom view. Not yet quantified.

**Missing for paper.**
- Quantitative version with segmentation count delta.
- Replicate on FFHQ (e.g., glasses on/off vs hair colour shift).

---

## C3. Attribute boundaries are layer-localised tangent sections

**Statement.** A boundary direction $b \in \mathbb{R}^D$ has a canonical
**layer of action**. Applied to a non-canonical layer in W+, its
pushforward magnitude is large but its semantic interpretation is
disrupted (saliency moves to unrelated regions of the image).

**Measurement.** For each (attribute, layer) pair, render the saliency
map of $b$ applied to that single layer only. Plot the 8 × 14 matrix
(bedroom; analogous for other domains). Identify each attribute's
"peak intensity layer" vs its "semantic-coherent layer".

**Evidence so far.** §08 of report: all attributes show peak intensity
at layer 0 (single-layer constant input), but semantic structure is
clearest in layers 6-11 (or 0-4 for view) — the HiGAN authors' canonical
manipulate-layers.

**Missing for paper.**
- Formalise "semantic coherence" with a measurable scalar (e.g., IoU
  with the saliency from a known good wp + boundary).
- Replicate on FFHQ (InterFaceGAN provides analogous canonical layers).

---

## C4. Compositional editing failure is predicted by mixed Hessian

**Statement.** Define the pairwise interference predictor
$P(a,b) = \mathbb{E} \|d^2G(b_a, b_b)\| \;/\; (\mathbb{E}\|dG b_a\| \cdot \mathbb{E}\|dG b_b\|)$.
Pairs with high $P(a,b)$ have compositional editing that violates
superposition, measured by correlation between $S^{(1)}(a+b)$ and
$S^{(1)}(a) + S^{(1)}(b)$.

Quantitative prediction: $\mathrm{Spearman}(\,P(a,b),\, 1 - \mathrm{corr}(a,b)\,) > 0.6$
across all $\binom{8}{2} = 28$ HiGAN pairs.

**Evidence so far.** §13 + §19 of report:
- Texture+texture pairs: corr ≈ 0.97, low $\rho$.
- View + texture pairs: corr ≈ 0.55, high $\rho$ (for view).
- Bedroom Spearman r=+0.48, p=0.01 (n=28); FFHQ r=+0.81, p=0.005 (n=10).

**Robustness caveat (Week 2 robustness pass).**
- Dropping the high-curvature outlier attribute (view in bedroom,
  pose in FFHQ) collapses or destabilises the rank:
  - bedroom no-view: Spearman r=-0.17, p=0.47 (n=21)
  - ffhq no-pose: Spearman r=+0.60, p=0.21 (n=6)
- Conclusion: $P(a,b)$ as a **single scalar predictor**
  discriminates *between* curvature regimes (structural vs texture)
  but does NOT discriminate *within* the texture regime. The
  regime boundary itself is the signal C4 captures.
- Pedagogically this is the same effect that drives C2: there is a
  real curvature-regime boundary, and several single-scalar
  measures see it from different angles.

**Missing for paper.**
- A finer within-regime predictor (per-attribute normalised mixed term).
- Cross-domain replication on church (only 3 pairs there — n too small).

---

## C5. Encoders are coordinate charts; agreement of derivatives is the right quality metric

**Statement.** Reconstruction error is a zeroth-order chart-quality
proxy; agreement of saliency maps $S^{(1)}(E(G(wp)))$ vs $S^{(1)}(wp)$
is the first-order proxy and a strictly stronger signal.

**Measurement.** For 5 checkpoints across training, compute reconstruction
MSE and saliency-vs-GT pixel correlation on a held-out synthetic test set
(where wp is known by construction).

**Evidence so far.** §18 of report: recon MSE 0.047 → 0.039 (18%↓),
saliency correlation 0.008 → 0.359 (45× ↑) across 1k → 40k iter.

**Missing for paper.**
- Decompose what specifically improves: encoder layer? attribute family?
- Replicate on FFHQ (if we re-train an encoder for FFHQ, optional).

---

## C6. Random direction clustering rediscovers attribute taxonomy

**Statement.** K-means clustering of $\Phi(b)$ for $N \geq 128$ random
unit directions on per-layer spheres produces clusters whose centroids
correspond to known attribute axes (rediscovery), validated by CLIP
zero-shot semantic labelling.

**Measurement.**
1. Sample $N$ random unit directions on per-layer $S^{D-1}$.
2. Compute saliency for each.
3. Cluster (k-means after PCA-32).
4. CLIP zero-shot caption cluster centroids.
5. Measure precision/recall of (cluster centroid)→(hand-curated boundary)
   matching.

**Evidence so far.** §16+§17 of report: cluster 2 of 8 (n=37 directions
at layer 1) auto-labelled as "a view through a window" by CLIP, matching
HiGAN's view boundary.

**Missing for paper.**
- Full precision/recall table across $N \in \{64, 128, 256, 512\}$.
- Robustness to $k$ choice (silhouette analysis).
- Replicate on FFHQ — does it rediscover InterFaceGAN's smile/age/glasses
  boundaries from random directions alone?

---

## Quick status summary (updated 2026-05-15 Week 2 mid-day)

| Claim | Theory done | Bedroom evidence | FFHQ evidence | Church evidence | Quantitative metric |
|------|-----------|---------------|--------------|---------------|--------------------|
| C1   | ✓         | ✓             | ✓             | ✓              | mean/median/p95 stats |
| C2   | ✓         | ✓ (view 23.2) | ✓ (pose 49.9, eye 22.8) | partial (no structural attr) | ratio across attributes |
| C3   | ✓         | **✓ (8/8 positive at top-10/20/30/50%)** | **✓ (5/5 positive at top-10/20/30/50%, mean +0.16)** | ⬜             | layer-pair IoU difference, threshold-robust |
| C4   | ✓         | ✓ (Spearman 0.48, p=0.01, n=28; no-view -0.17 n=21 — regime, not within) | ✓ (Spearman 0.81, p=0.005, n=10; no-pose +0.60 n=6) | ⬜ (3 pairs only) | Spearman + scatter plot |
| C5   | ✓         | ✓             | ⬜            | n/a           | recon vs saliency-vs-GT |
| C6   | ✓         | ✓ (P=1.00 R=0.62 @ K=3, P-lift +0.81 vs null) | ✓ (P=1.00 R=1.00 @ K=3, P-lift +0.66 vs null) | ⬜ | precision/recall vs CLIP-vocab-permutation null |

**Baselines (NEW, Week 2)**:

| K | bedroom GANSpace | bedroom SeFa | bedroom random+CLIP | FFHQ GANSpace | FFHQ SeFa | FFHQ random |
|--|---|---|---|---|---|---|
| 4  | 4/8  | 3/8 | **6/8** | 4/5 | 3/5 | 2/5 |
| 8  | 5/8  | 6/8 | 6/8 | **5/5** | 4/5 | 3/5 |
| 16 | 7/8  | 7/8 | 7/8 | 5/5 | 4/5 | 4/5 |

Conclusion: **discovery-method ranking is domain-dependent**.
PCA-based methods (GANSpace) dominate where W has clear principal axes
(FFHQ identity bundle); random+CLIP is competitive or better where the
latent is more isotropic (bedroom). Our JVP framework is the *evaluator*
across all three methods.

**CLIP-Grad-CAM vs JVP saliency comparison (CROSS-DOMAIN)**:

Bedroom (StyleGAN1 LSUN, 256²):
| attribute | pixel-corr | IoU top-20% |
|---|---|---|
| view            | -0.049 | 0.089 |
| indoor_lighting | -0.071 | 0.101 |
| wood            | -0.105 | 0.089 |
| glossy          | -0.097 | 0.094 |

FFHQ (StyleGAN1 face, 1024²):
| attribute  | pixel-corr | IoU top-20% |
|---|---|---|
| pose       | -0.017 | 0.200 |
| smile      | -0.001 | 0.200 |
| eyeglasses | -0.013 | 0.200 |
| age        | -0.020 | 0.200 |
| gender     | -0.013 | 0.200 |

FFHQ shows even *stronger* near-orthogonality (all |corr| < 0.02).
JVP saliency and CLIP-grad-CAM are **cross-domain near-orthogonal**: neither
correlated nor IoU-overlapping above chance. The two answer
*different questions*:
  - JVP = "where do pixels move when the latent moves along this direction?"
  - CLIP-grad = "where in this rendered image does CLIP locate the attribute?"
The first is editing-aligned; the second is recognition-aligned. They
do not measure the same thing, and our framework is the editing-aligned
analogue of Grad-CAM.

**Week 1 progress** (cross-domain validation):
- C1 replicates on FFHQ (1024^2 StyleGAN1) and church (256^2 StyleGAN2).
- **C2** non-linearity ratio replicates with the same structural-vs-textural pattern:
  - Bedroom view 23.2, FFHQ pose 49.9 / eyeglasses 22.8 — same order of magnitude for "structural / topological" attributes across architectures.
  - Note: absolute ratio scales differ between StyleGAN1 and StyleGAN2 (church ~0.04 floor vs bedroom ~0.5 floor) — C2 should be stated as **domain-relative ordering**, not absolute.
- **C4** Spearman positive and significant on both domains where multiple pairs are available.

**Bottom line.** All 6 theory writeups exist (Month 1 ≈ done).
Cross-domain experiments and quantitative metrics are the bulk of Months
2-5. User study is Month 6. Theory rigour (formal proofs, sample
complexity) is Months 7-8.
