# Design — Cross-resolution C1 invariance (FFHQ lod sweep)

**Wave 2, Track 13.** Test whether the C1 curvature ratios are
**resolution-invariant** by rendering the same FFHQ generator at
256² / 512² / 1024² via StyleGAN's lod (level-of-detail) parameter.

## Hypothesis

**H15.** The rank ordering of the 5 InterFaceGAN attributes by
$\bar\rho$ is preserved across {lod=2 → 256², lod=1 → 512², lod=0 →
1024²}. Spearman r between any pair of resolution orderings ≥ 0.9.

This addresses the reviewer concern "is curvature just an
upscaling-artefact at high resolution?".

## Prior art

- **ProgressiveGAN** Karras et al. ICLR 2018
  [arXiv:1710.10196](https://arxiv.org/abs/1710.10196) — introduces
  the lod parameter; final-resolution outputs are renormalised
  upsamples of mid-resolution. This means the *image content* at
  different lod values is consistent, only the spatial sampling
  differs.
- **StyleGAN** Karras 2019 inherits this lod machinery.
- Our existing FFHQGenerator already supports `lod_override` —
  the wrapper documentation says "fall back to 256² with ~10x less
  memory" via lod=2.

## Method

For each lod ∈ {0, 1, 2}:
- `FFHQGenerator(lod_override=lod)`
- 5 attrs × $N = 32$ noise seeds (fixed across lods for paired
  comparison)
- Compute $\bar\rho_a$ via composed JVP.
- Compare attribute orderings.

## Expected signal

- $\bar\rho$ absolute values likely *scale up* with resolution
  (more pixels → larger absolute second-derivative magnitude), but
  the *ratio* normalises this.
- Ordering preserved: pose > eyeglasses > gender > age > smile.
- Spearman r ≥ 0.9 across all pairs.

## Failure modes

1. Some boundaries (especially eyeglasses, pose) may rely on
   high-frequency detail that's invisible at 256² — could change
   the curvature signature. Honest finding either way.
2. Compute is much cheaper at 256² so this also doubles as a
   validation of the cheaper proxy for larger studies.

## Compute budget

- 3 lods × 5 attrs × 32 seeds × (varies: lod=2 ~1s, lod=1 ~5s,
  lod=0 ~15s per second-order JVP) ≈ 80 min total.

## Deliverables

- `experiments/domains/ffhq/run_resolution_invariance.py`
- `experiments/out/ffhq_resolution/` — table + bar plot per lod.
- §5.exp-c1 robustness paragraph.
