# Design — DINOv2 path curvature as C2 alternative

**Wave 4, Track 19.** Replicate the C2 CLIP path-curvature ratio
with DINOv2 self-supervised features instead of CLIP. Tests whether
the C2 result depends on CLIP's contrastive vision-language training
or on the geometric structure of the feature space alone.

## Hypothesis

**H21.** DINOv2 path curvature ratio agrees with pixel-curvature
ratio at Pearson r ≥ +0.9 on FFHQ — same direction, similar magnitude
as the CLIP-based measurement (r = +0.97 from Track 1).

If H21 holds: the C2 "two-measure agreement" is not specific to
contrastive vision-language training; any sufficiently rich visual
feature space exhibits the same coarse-to-fine curvature signature.

## Prior art

- **DINOv2** Oquab et al. 2023
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) — self-supervised
  visual features. Trained without language; orthogonal supervision
  signal from CLIP. Strong general-purpose backbone.
- Our existing **C2 CLIP path** ([Track 1
  result](../out/ffhq_c2_path/metrics.json), Spearman r=+1.00 vs
  pixel ratio).
- **StyleCLIP** used CLIP because it was language-conditioned; we
  don't need language for C2.

## Method

Reuse α-sweep frames from Track 1's CLIP measurement; only the
feature encoder changes. For each (domain, attribute):
- DINOv2 ViT-B/14 from `facebookresearch/dinov2`
  ([HF mirror](https://huggingface.co/facebook/dinov2-base))
- Encode the same 13-frame α-sweep
- Path length / direct distance ratio
- Compare to pixel-curvature ratio

## Expected signal

- FFHQ: Pearson r vs pixel ratio ≥ +0.9. (CLIP got +0.97; DINOv2
  likely similar.)
- Bedroom: similar agreement.
- Absolute path ratios may differ from CLIP because DINOv2 features
  have different scale, but the *ranking* is what matters.

## Compute budget

- DINOv2 ViT-B/14 ≈ 700M params, ~1GB. Easy on 8GB.
- 13 attribute-domain pairs × 13 frames × 8 seeds = ~1500 forwards.
  ~5 minutes.

## Deliverables

- `experiments/metrics/run_dino_path_curvature.py`
- `experiments/out/dino_path_curvature/` — table + correlation plot.
- §5.exp-c2 robustness paragraph.
