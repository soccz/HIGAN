# Design — Per-layer C1 decomposition

**Wave 2, Track 10.** Decompose the aggregate $\bar\rho$ from §C1
across the 14 (bedroom) / 18 (FFHQ) W$^+$ layers. Where in W$^+$
does curvature live?

## Hypothesis

**H12.** Curvature is **not** uniformly distributed across W$^+$
layers. Structural attributes (view, pose, eyeglasses) concentrate
their second-order energy in coarse layers (0–3 for FFHQ pose, 0–7
for FFHQ eyeglasses, matching InterFaceGAN canonical layers).
Textural attributes (smile, age, wood) spread across fine layers.

Specifically: per-layer ratio
$\rho_\ell(a) = \frac{|d^2 G(b_a^{(\ell)}, b_a^{(\ell)})|}
                       {|dG\, b_a^{(\ell)}|}$
where $b_a^{(\ell)}$ is the boundary direction applied only to layer
$\ell$.

The argmax-layer of $\rho_\ell(a)$ should agree with InterFaceGAN /
HiGAN canonical manipulate-layers for ≥80% of attributes.

## Prior art

- **StyleSpace** Wu et al. CVPR 2021 §4.3 — per-channel localisation
  but in $\mathcal{S}$ space, not per-W+-layer.
- **InterFaceGAN** Shen et al. CVPR 2020 §4.1 — defines canonical
  manipulate-layers as a hand-tuned hyperparameter (e.g. pose layers
  0–3). We compute it from the data.
- **HiGAN** technical report — same per-attribute canonical-layer
  hand-curation.
- Our C3 results (bedroom_c3_iou, ffhq_c3_iou) show layer-IoU
  structure for the first-order saliency. This proposal is the
  second-order version.

## Method

For each (domain, attribute) pair:

1. Apply $b_a$ to a single layer $\ell$ only (all other layers zero).
2. Compute first and second-order pushforward magnitude over $N=32$
   noise seeds:
   - $F_\ell = \mathbb{E}[\,|dG\, b_a^{(\ell)}|\,]$
   - $S_\ell = \mathbb{E}[\,|d^2 G(b_a^{(\ell)}, b_a^{(\ell)})|\,]$
3. Plot the $L$-vector $\rho_\ell(a) = S_\ell / F_\ell$ for each
   attribute (8 + 5 + 3 = 16 line plots).
4. Compare argmax layer to ground-truth canonical layer; report
   agreement percentage.

## Expected signal

- For FFHQ pose: argmax in {0, 1, 2, 3}.
- For FFHQ eyeglasses: argmax in {0, ..., 7}.
- For FFHQ smile: argmax in {4, 5, 6, 7}.
- For bedroom view: argmax in {0, ..., 5}.
- Texture-class attributes show flatter curves with argmax often in
  mid layers (6–11).
- Mean accuracy: argmax-layer ∈ canonical-set for ≥12/16 attrs.

## Failure modes

1. Per-layer measurements are noisier than aggregate (each $\ell$
   averages fewer samples). $N=32$ may be too low; we can extend to
   $N=64$ if needed.
2. For boundaries that span many layers (e.g. age 0–8), argmax is
   harder to pinpoint. We report top-3 layers too.

## Compute budget

- 16 attrs × 14–18 layers × 32 seeds × second-order JVP (~5 s on
  bedroom, ~12 s on FFHQ 1024²) = ~5–6 h. Sequential after Wave 1.

## Deliverables

- `experiments/metrics/run_per_layer_c1.py`
- `experiments/out/per_layer_c1/` — per-attribute line plots + table
- §5 figure showing the per-layer decomposition for 3–4
  representative attributes.
