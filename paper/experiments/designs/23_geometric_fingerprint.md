# Design — geometric fingerprint per attribute (meta-analysis)

**Wave 5, Track 23.** Combine every per-attribute measurement we
have — C1 ratio, CLIP path, C3 IoU, peak-curvature-layer, segmentation
range — into a single multi-dimensional descriptor per attribute,
then cluster in this fingerprint space to ask: do attributes cluster
into discrete *geometric types* across domains?

## Hypothesis

**H25.** In the 5-dimensional space of (C1 ρ, CLIP path, C3 score,
per-layer-argmax, alpha-magnitude-slope), attributes cluster into
**3 distinct types**:
1. *Structural / topology-changing*: high ρ, high CLIP path,
   localised at coarse layers, high alpha-slope.
2. *Mid-level / object-anchored*: moderate ρ, low-moderate CLIP path,
   localised at mid layers.
3. *Textural / pixel-recolouring*: low ρ, low CLIP path, distributed
   across many layers.

We test by hierarchical clustering on z-normalised fingerprint
vectors. The 3-cluster solution should reveal the structural /
mid / textural split.

## Prior art

- **StyleSpace** Wu et al. 2021 implicitly does this at the per-channel
  level (coarse/medium/fine StyleGAN layers).
- **InterFaceGAN** manipulates at canonical layer ranges {0--3,
  4--7, 8--17}.
- We're explicitly *measuring* the type, not assuming it.

## Method

For each (domain, attribute) compile a 5-vector:
- $\bar\rho_{\text{pixel}}$ from C1
- CLIP-path ratio from C2
- C3 score from per-layer IoU
- argmax-layer / L (fraction)
- log-slope of $\bar\rho(\alpha)$ from Track 20 (FFHQ only;
  zero-pad for bedroom and church)

Z-normalise each feature across all (domain, attribute) entries.
Hierarchical agglomerative clustering with Ward linkage; cut at k=3.

## Expected signal

- 3-cluster dendrogram separates structural / mid / textural.
- Adjusted Rand Index vs a-priori labels ≥ 0.6.

## Compute budget

Pure post-processing. <1 minute.

## Deliverables

- `experiments/metrics/run_geometric_fingerprint.py`
- `experiments/out/geometric_fingerprint/` — dendrogram + 3-cluster
  assignment + fingerprint table.
- §5 fingerprint figure + §6 application: "attribute-type diagnostic".
