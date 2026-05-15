# Design — C5 on FFHQ (encoder as coordinate chart, cross-domain)

**Track 4.** Replicate the bedroom C5 result (saliency-vs-GT
correlation improves from +0.008 at 1k iter to +0.36 at 40k iter,
45× lift) on FFHQ-1024, giving C5 a second domain.

## Hypothesis

**H5.** When an encoder $E$ is trained to invert a frozen FFHQ
StyleGAN, the *first-order* chart-quality metric
$\text{corr}(S^{(1)}(E(G(wp))), S^{(1)}(wp))$ improves monotonically
across checkpoints while the *zeroth-order* metric (reconstruction
MSE) plateaus earlier. This is the same dissociation as bedroom — a
cross-domain confirmation that saliency-agreement is a strictly
stronger chart-quality signal than recon error.

## Prior art

- **pSp encoder** Richardson et al. CVPR 2021 [arXiv:2008.00951](https://arxiv.org/abs/2008.00951)
  is the standard FFHQ-StyleGAN encoder, reaches LPIPS 0.16 / MSE
  0.027 on FFHQ test in ~150 k iter. We use the same architecture
  scaled to our compute.
- **e4e encoder** Tov et al. SIGGRAPH 2021 [arXiv:2102.02766](https://arxiv.org/abs/2102.02766)
  refines pSp with editability emphasis. Out of scope here — we just
  need a strong-enough chart.
- **GAN inversion benchmark** Xia et al. 2022 [arXiv:2101.05278](https://arxiv.org/abs/2101.05278)
  surveys 30+ FFHQ encoders. LPIPS 0.15–0.20 is the consensus
  "good chart" threshold; below that, derivative-agreement becomes
  meaningful.

**Our delta:** all prior work measures reconstruction. Nobody measures
the agreement of *derivatives at the inverted code* with derivatives
at the ground-truth code. That's the C5 contribution.

## Method

**Encoder.** pSp-style architecture (ResNet-IRSE-50 backbone
[paper](https://arxiv.org/abs/1801.07698), 18-headed MLP outputting
W+ ∈ ℝ^{18×512}), trained from scratch.

**Training data.** 50 k synthetic (wp, image) pairs generated from
the FFHQ generator at fixed seed, regenerated on the fly so the
"ground-truth wp" is known.

**Loss.**
$\mathcal{L} = \lambda_\text{mse} \|G(E(I)) - I\|_2^2
            + \lambda_\text{lpips} \text{LPIPS}(G(E(I)), I)
            + \lambda_w \|E(I) - wp_\text{gt}\|_2^2$
with $\lambda_\text{mse}=1, \lambda_\text{lpips}=0.8, \lambda_w=0.1$
(pSp-style + the explicit wp anchor since we have it). Adam, lr 1e-4,
batch 4, 40 k iter — same schedule as our bedroom encoder.

**Evaluation checkpoints.** 1 k, 5 k, 10 k, 20 k, 40 k iter — five
points (matches bedroom).

**Held-out test set.** 256 (wp, image) pairs from a disjoint seed.
For each test pair:
- Recon MSE / LPIPS.
- Saliency agreement: compute $S^{(1)}(b)$ at the ground-truth wp and
  at the inverted $E(G(wp))$ for each of 5 InterFaceGAN boundaries
  (smile, age, pose, gender, eyeglasses). Per-pixel Pearson correlation.
- Aggregate: mean correlation across the 5 attributes × 256 test pairs.

## Expected signal

- Recon MSE: ~0.05 at 1 k → ~0.025 at 40 k (factor-2 improvement, plateau).
- Saliency-agreement: +0.01 at 1 k → +0.30–0.40 at 40 k (>30× lift,
  same order as bedroom's 45×).
- Dissociation visible as a divergent log-log curve: MSE flat after
  20 k, saliency-agreement still rising.

## Failure modes

1. **40 k iter under-trains FFHQ encoder.** pSp uses 150 k; if 40 k
   is insufficient, the saliency-agreement may not lift. Mitigation:
   monitor at 40 k and either extend to 80 k or report the partial
   curve and the diminishing-returns analysis.

2. **LPIPS-only training collapses derivative-agreement.** If the
   encoder hits low recon MSE via texture-matching while ignoring
   structure, derivatives won't align. Mitigation: the explicit $w$-anchor
   loss $\lambda_w$ pins this; if still bad, increase $\lambda_w$.

3. **Saliency at the inverted wp diverges from gt wp due to a
   single failure mode (e.g. extreme pose).** Mitigation: report
   per-attribute correlation, not just mean. If pose-correlation is
   the only one that fails, that's interpretable, not a refutation.

## Compute budget

- 40 k iter × ~3 s/iter on FFHQ-1024 = ~33 hours of training.
- Per-checkpoint evaluation: ~30 min × 5 = 2.5 h.
- **Total ~36 h.** Background run.

## Deliverables

- `experiments/domains/ffhq/encoder/model.py` — pSp-style encoder
- `experiments/domains/ffhq/encoder/train.py` — training script
- `experiments/out/ffhq_c5/` — checkpoints + log + evaluation plot
- Two-domain C5 curve in [05_experiments.tex](../../sections/05_experiments.tex)
