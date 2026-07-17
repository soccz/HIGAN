# Design — Editing head-to-head (curvature pre-selection)

**Track 2.** Move from descriptive ("we measure curvature") to
prescriptive ("our measurement makes editing better"). Show that
ranking a candidate-direction pool by our C2 curvature ratio produces
edits with *better identity preservation at matched attribute strength*
than ranking by baseline criteria.

## Hypothesis (falsifiable)

**H3.** At matched target-attribute Δ-logit, directions in the
**low-curvature** half of a candidate pool give higher ArcFace identity
cos-similarity (i.e. cleaner, more disentangled edits) than directions
in the **high-curvature** half *or* than random / GANSpace-eigenvalue
ranking. Effect size $d \geq 0.3$ on the per-latent paired difference,
$p < 0.01$, N=1000.

**H3' (anti-hypothesis).** If H3 fails, the C2 ratio is not actionable
— it describes a real geometric property but does not buy editing
quality. We report this honestly and the curvature framework remains a
*descriptive* contribution (still novel via C2 cross-architecture).

## Prior art and our delta

- **InterFaceGAN** Shen et al. CVPR 2020 [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.pdf)
  established the boundary projection + classifier-Δ evaluation
  protocol. Identity not the main axis there.
- **StyleSpace** Wu et al. CVPR 2021 established the **"equal-strength
  comparison"** + **DCI + Attribute Dependency**. *Filters* StyleGAN
  channels by relevance and reports improved DCI for the filtered
  subset — closest precedent for a "rank + re-evaluate" experiment.
- **StyleCLIP** Patashnik et al. ICCV 2021 established the
  (CLIP-Δ vs ArcFace-ID) Pareto-curve format that StyleSpace +
  follow-ups all use.
- **DragGAN** Pan et al. SIGGRAPH 2023 — N=1000 FFHQ paired protocol;
  we adopt their N.
- **LELSD** Khodadadeh et al. WACV 2022 [arXiv:2111.12583](https://arxiv.org/abs/2111.12583)
  optimises directions using segmentation supervision but does not
  rank a candidate pool by a generator-side intrinsic metric.

**Our delta:** as far as we can find, no published method ranks a
candidate pool by an *intrinsic, generator-side, classifier-free*
metric (= our curvature ratio) and shows downstream-task improvement.
This is genuine methodological novelty.

## Method

**Generator.** StyleGAN2-FFHQ-1024 (config-f), W+ space.

**Test set.** $N = 1000$ w-codes from $z \sim \mathcal{N}(0, I)$,
fixed seed 2027 — DragGAN's exact FFHQ protocol size. The same 1000
codes are used for every candidate direction (paired comparison).

**Candidate direction pool.** Three sources, all in W:
- GANSpace top-64 PCA components on 50 k sampled W
- SeFa top-32 eigenvectors
- LatentCLR-trained top-100 (Track 5 dependency)
- DisCo-trained top-100 (Track 5 dependency)

Total **~296 candidate directions** per layer-bucket. Layers grouped
into coarse (0–3), mid (4–11), fine (12–17) per InterFaceGAN.

**Pre-selection criterion.** Our C2 pixel-curvature ratio $\bar\rho_v$
computed on 32 hold-out latents per direction (disjoint from the
N=1000 test set). For each candidate, also compute:
- $\bar\rho_v$ (our)
- GANSpace explained-variance fraction (baseline)
- SeFa eigenvalue (baseline)
- LatentCLR contrastive loss (baseline)
- Random ranking (control)

Define five ranking criteria; for each, take K′=16 top-ranked
directions. Run editing with each top-K′ set independently.

**Edit protocol.** For each direction, symmetric α-sweep in W+:
$\alpha \in \{-3, -2, -1, +1, +2, +3\} \cdot \sigma$ where $\sigma$ is
the per-direction unit (L2-normalised in W+). Apply to all 1000 test
latents.

**Equal-strength matching.** Per attribute and per method, find the
α at which the *mean* CelebA-classifier target-logit Δ matches the
reference α=2σ random-baseline value (StyleSpace convention). Compare
metrics at that matched α.

## Metrics

All computed per-latent and aggregated:

| Metric | Implementation |
|---|---|
| **Target Δ-logit** | ResNet-50 CelebA-40 ([d-li14/face-attribute-prediction](https://github.com/d-li14/face-attribute-prediction)) — used by InterFaceGAN, StyleSpace, EditGAN |
| **Identity preservation** | ArcFace cos-similarity ([insightface](https://github.com/deepinsight/insightface)) — StyleCLIP / EditGAN standard |
| **Perceptual drift** | LPIPS-AlexNet ([richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)) |
| **Distributional shift** | FID-50 k between edited and unedited image populations (StyleSpace) |
| **Attribute Dependency** | mean \|Δ-logit\| on the other 30 CelebA attributes per unit target-Δ (StyleSpace) |

## Expected signal

For the **smile / age / glasses / gender / pose** attributes,
at matched target-Δ:

- **ID-cos**: curvature-low > GANSpace-EV > random > curvature-high.
  Per-latent paired t-test $p < 0.01$, Cohen's $d \geq 0.3$ for
  curvature-low vs random.
- **LPIPS**: curvature-low < random < curvature-high.
- **AD (StyleSpace)**: curvature-low ≤ baselines for textural
  attributes (smile/age); curvature-high best for *structural* tasks
  where topology change is desired (eyeglasses on/off).
- **FID**: ≤ +5 from unedited for all curvature-low sets; ≥ +15 for
  curvature-high on textural attributes.

The interesting prediction is *bi-directional*: curvature-low is best
for clean texture edits, curvature-high is best for structural edits
(glasses on/off). One ranking, two regimes — that's the paper's
prescriptive headline.

## Failure modes

1. **Classifier-direction collision.** CelebA-attribute classifier
   might be sensitive to the same features the direction was *trained
   on*, giving false target-Δ wins. Mitigation: report a parallel
   CLIP-Δ score (open-set) and require both move in the same direction.

2. **Pose breaks ArcFace.** ArcFace cos-sim drops on large pose
   changes, falsely penalising pose directions. Mitigation: report
   pose separately, and additionally report per-pose-bucket ID-cos
   (small/medium/large pose Δ).

3. **N=1000 underpowered for AD per-attribute.** The 30-attribute AD
   requires ~30 paired tests per direction; with 1000 latents per
   direction the per-attribute power is fine, but multi-comparison
   correction (Benjamini-Hochberg q<0.05) is required.

4. **LatentCLR / DisCo training instability.** These are
   re-implementations, not loaded checkpoints. If training fails to
   converge in our compute budget, we report results with the
   baselines that did converge and note the limitation.

## Compute budget

- 1000 test latents × ~80 directions (16 × 5 rankings) × 6 α-values
  × 5 attributes = 2.4M generator forwards. At ~30 ms per generator
  forward on FFHQ-1024: ~20 hours.
- Plus 5 classifier networks evaluated on each: another ~3 hours.
- Plus pre-selection saliency: ~2 hours.
- **Total: ~25 hours wall-clock.** Acceptable.

## Deliverables

- `experiments/baselines/run_editing_head_to_head.py` — main script
- `experiments/out/editing_head_to_head/` — Pareto-curve plots per
  attribute, paired-difference tables, FID/AD tables
- `paper/sections/05_experiments.tex` — prescriptive subsection §5.4
- Single headline figure: 5-attribute Pareto curve (ID-cos vs target-Δ),
  one panel per method, paired-CI envelopes.
