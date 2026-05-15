# Design — Multi-CLIP-encoder C2 robustness

**Wave 2, Track 9.** Test whether the C2 CLIP-path curvature result
depends on the specific CLIP variant (ViT-B/32). Replicate on
ViT-L/14 and ViT-H/14.

## Hypothesis

**H11.** The Pearson correlation between pixel-curvature ratio and
CLIP-path curvature ratio is preserved across CLIP encoders:
$r \geq +0.9$ for all three of {ViT-B/32, ViT-L/14, ViT-H/14}.

If true: the C2 "two-measure agreement" headline is not an artefact of
ViT-B/32; it's a robust property of CLIP-style semantic features.

## Prior art

- **CLIP** Radford et al. ICML 2021
  [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) — the original.
- **OpenCLIP** Ilharco et al. 2022
  [arXiv:2212.07143](https://arxiv.org/abs/2212.07143) — public
  reproduction including LAION-trained ViT-L/14 and ViT-H/14
  variants we'll use.
- **StyleCLIP** Patashnik et al. ICCV 2021 used ViT-B/32 by default;
  later StyleGAN-NADA / Style-CLIP-Mapper showed L/14 sometimes
  helps. The C2 measurement being CLIP-variant-invariant is the
  cleaner story.

## Method

For bedroom (8 attrs) + FFHQ (5 attrs) — reuse the rendered
α-sweep frames already saved (no new GPU generation needed
beyond CLIP re-encoding):

For each CLIP variant:
- ViT-B/32, laion2b_s34b_b79k (current)
- ViT-L/14, laion2b_s32b_b82k
- ViT-H/14, laion2b_s32b_b79k

Re-encode the α-sweep frames at 224² (CLIP standard), recompute
path-length / direct-distance ratio per attribute. Report Pearson +
Spearman vs pixel-curvature ratio.

## Expected signal

- Per-attribute CLIP-path ratio absolute values shift between
  variants (larger encoders give larger feature vectors, more
  path length).
- **Rank ordering** preserved: structural attributes have higher
  path ratio than textural ones in all three encoders.
- Pearson and Spearman correlations with pixel-curvature stay
  ≥ 0.9 for all three encoders.

## Failure modes

1. ViT-H/14 model is ~3GB — may not fit alongside generator. Mitigation:
   compute CLIP features in sequential mode (one image at a time)
   and free the encoder after each variant.
2. If different encoders disagree (Pearson < 0.7), the C2 result
   becomes "CLIP-B-specific", weakening the paper. We report
   honestly either way.

## Compute budget

- Reuse existing sweep frames if cached; otherwise regenerate.
- 3 encoders × 13 attribute-domain pairs × 9-13 sweep frames × 1 image
  = ~500 CLIP forward passes. Few minutes total.

## Deliverables

- `experiments/metrics/run_multi_clip_c2.py`
- `experiments/out/multi_clip_c2/` — table per CLIP × per attribute,
  rank correlations.
- §5.exp-c2 robustness paragraph + supplementary table.
