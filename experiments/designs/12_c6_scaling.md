# Design — C6 sample-size scaling (N=512 random directions)

**Wave 2, Track 12.** The existing C6 result uses N=192 random
directions (96 above-median). The paper claim is "$N \gtrsim 128$
directions". Scale to N ∈ {128, 256, 384, 512} and report how
precision / recall / coverage scale.

## Hypothesis

**H14.** Bedroom C6 recall (currently 5/8 at N=192) increases
monotonically with N, reaching $\geq 7/8$ at N=512. The three GT
attributes currently missed (wood, indoor_lighting, cluttered_space)
are recovered at $N \geq 384$, $N \geq 256$, and $N \geq 384$
respectively.

FFHQ C6 already at full recall (5/5) at N=192; expected to stay
at 1.00 as N grows.

## Prior art

- **GANSpace** Härkönen et al. NeurIPS 2020 — uses N=5000 W samples
  for the PCA; their "we can find more axes with more samples" claim
  is well-established but not benchmarked across N for taxonomy
  rediscovery.
- **SeFa** Shen & Zhou CVPR 2021 — closed-form, no sample-size
  question.
- **LatentCLR** Yüksel et al. ICCV 2021 — uses fixed K=100 directions
  on FFHQ; doesn't ablate K.
- This is the unsupervised-discovery analogue of GANSpace's N-ablation,
  with our K-means + CLIP-label pipeline.

## Method

For each N ∈ {128, 192, 256, 384, 512}:
- Sample N random unit directions distributed uniformly across the
  L per-layer spheres in W$^+$.
- Filter to above-median pushforward magnitude (keep $\sim N/2$).
- PCA-32 + k-means with K = 8 clusters.
- CLIP zero-shot label each cluster.
- Re-run the precision / recall pipeline of
  [run_c6_precision_recall.py](../metrics/run_c6_precision_recall.py).

Plot precision, recall, F1 vs N for both bedroom and FFHQ.

## Expected signal

- Bedroom: recall 5/8 at N=192 → 6/8 at N=256 → 7/8 at N=384 → 7/8
  or 8/8 at N=512. Precision stays at 1.00 throughout.
- FFHQ: precision and recall both at 1.00 across all N ≥ 192.
- F1 increases monotonically in bedroom.

## Failure modes

1. Larger K (clusters) might be needed to find finer attributes;
   we keep K = 8 for paired comparison, but secondary table at K ∈
   {8, 16} reports the K-sensitivity.
2. If recall plateaus before 7/8, the missing attributes might be
   genuinely outside the geometric structure that our method captures
   — interpretable as a *limitation* not a failure.

## Compute budget

- Per N: 5 attrs × N saliencies × ~3 s (bedroom 256²) =
  60 / 90 / 150 / 192 / 240 minutes. Plus CLIP labelling ~10 min each.
- Total bedroom: ~12 h.
- FFHQ at 1024² is ~5× slower → 60 h for FFHQ alone.
- Compromise: report bedroom full sweep + FFHQ at N=256, 512 only.

## Deliverables

- `experiments/metrics/run_c6_scaling.py`
- `experiments/out/c6_scaling/` — recall-vs-N curve, table.
- §5.exp-c6 figure replacing the static N=192 statement.
