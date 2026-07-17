# Design — Cross-domain curvature signature transfer

**Wave 5, Track 22.** Does the curvature signature of an attribute
type (e.g. "structural") transfer across architectures? Test whether
bedroom view, FFHQ pose, FFHQ eyeglasses, and SD pose+eyeglasses
cluster in a common curvature signature, separate from textural
attributes across all four.

## Hypothesis

**H24.** When the per-attribute (pixel ρ, CLIP-path ratio) pairs from
**all** (domain, attribute) combinations are 2D-clustered, two
clusters emerge:
- "structural" cluster: {bedroom view, FFHQ pose, FFHQ eyeglasses,
  SD pose, SD eyeglasses}
- "textural" cluster: all other attributes from all domains

Decision boundary separates them at $\bar\rho > 5$.

## Prior art

- This is the geometric analog of *taxonomic* statements in editing
  literature (e.g. StyleSpace's coarse / medium / fine layer split,
  InterFaceGAN's per-attribute manipulation depths).
- Direct analog: van der Maaten 2008 t-SNE clustering of attribute
  representations, but applied here to *geometric* features rather
  than visual ones.

## Method

Compile the (pixel ρ, CLIP-path) pair for every attribute across
all 4 domains. Plot in 2D. K-means with k=2. Check that the cluster
assignment matches our hypothesis labels.

## Expected signal

- Cluster assignment matches structural / textural labels at ≥85% acc.
- Decision-boundary threshold reproducible across cluster runs.

## Compute budget

Trivial — post-processing of existing JSONs. <1 min.

## Deliverables

- `experiments/metrics/run_crossdomain_signature.py`
- `experiments/out/crossdomain_signature/` — 2D scatter + cluster.
- §5 cross-domain figure.
