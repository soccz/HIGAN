# Design — Concept-TRAK influence attribution comparison

**Wave 3, Track 16.** Compare our gradient saliency to Concept-TRAK
(Park 2025, arXiv:2507.06547), the most recent influence-attribution
method for diffusion editing. Both compute "where does this concept
act in pixel space?"

## Hypothesis

**H18.** Our JVP-based pixel saliency and Concept-TRAK's influence
maps are **partially correlated** (Pearson $r \in [0.2, 0.5]$) on
the 5 face attributes — they share an editing-aligned axis but
differ on the *direction definition* (h-space direction vs DDIM-
inverted noise perturbation).

This positions our framework as complementary to Concept-TRAK,
similar to the JVP-vs-CLIP-Grad-CAM orthogonality story.

## Prior art

- **Concept-TRAK** Park et al. 2025
  [arXiv:2507.06547](https://arxiv.org/abs/2507.06547). Deterministic
  DDIM inversion + autograd through the sampling chain for concept
  influence attribution. Uses ε-token attention but defines influence
  scores at *test time* via score-difference under text-conditioned
  guidance.

## Method

1. Pick 5 face concept prompts matching our SEGA-style directions.
2. For each of 16 seeds, run Concept-TRAK as published (using their
   public repo if available, else our re-implementation following
   their §3) to get an influence map per concept per seed.
3. Compute pixel-wise Pearson + IoU(top-20%) between Concept-TRAK
   and our JVP saliency.

## Expected signal

- |Pearson| ∈ [0.2, 0.5] on face attributes (partial agreement,
  not fully orthogonal, not fully aligned).
- IoU(top-20%) ≈ 0.2-0.3 (above chance, below identity).

## Failure modes

1. Public code may not be available yet (paper published 2025).
   Fallback: re-implement their score-difference influence
   following the paper's §3.
2. If correlation is high (>0.8), the two methods reduce to each
   other — interesting but reduces our claimed novelty.

## Compute budget

- 5 attrs × 16 seeds × ~30 s per Concept-TRAK run = ~40 min.

## Deliverables

- `experiments/diffusion/run_concept_trak_comparison.py`
- `experiments/out/sd_concept_trak/` — correlation + IoU table.
- §5 baselines table SD row.
