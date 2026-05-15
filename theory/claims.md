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
The two qualitatively align but the Spearman number has not been
computed.

**Missing for paper.**
- Compute $P(a,b)$ for all 28 pairs.
- Plot $P$ vs interference and report rank correlation.
- Cross-domain replication.

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

## Quick status summary

| Claim | Theory done | Bedroom evidence | FFHQ evidence | Church evidence | Quantitative metric |
|------|-----------|---------------|--------------|---------------|--------------------|
| C1   | ✓         | ✓             | ⬜            | ⬜             | partial            |
| C2   | ✓         | qualitative   | ⬜            | ⬜             | ⬜                  |
| C3   | ✓         | ✓             | ⬜            | ⬜             | partial            |
| C4   | ✓         | qualitative   | ⬜            | ⬜             | ⬜                  |
| C5   | ✓         | ✓             | ⬜            | n/a           | ✓                  |
| C6   | ✓         | partial       | ⬜            | ⬜             | ⬜                  |

**Bottom line.** All 6 theory writeups exist (Month 1 ≈ done).
Cross-domain experiments and quantitative metrics are the bulk of Months
2-5. User study is Month 6. Theory rigour (formal proofs, sample
complexity) is Months 7-8.
