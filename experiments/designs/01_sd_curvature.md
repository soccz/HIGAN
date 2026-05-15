# Design — Stable Diffusion C1/C2 extension

**Track 1.** Lift HIGAN's pixel-space second-fundamental-form measurement
from StyleGAN's $W$ to Stable Diffusion's $h$-space (U-Net mid-block).

## Hypothesis (falsifiable)

**H1 (C1 generalises).** For any direction $v$ in SD h-space, the pixel-
space second derivative $\partial^2 x_0/\partial \alpha^2$ along $v$ (where
$x_0$ is the decoded image at $\alpha v$ perturbation) is well-defined,
bounded, and varies non-trivially across directions — same statement as
on StyleGAN.

**H2 (C2 generalises).** Within a candidate set of CLIP-anchored
directions, the curvature ratio $\bar\rho = E[|\partial^2 x_0|]/E[|\partial x_0|]$
correlates positively with a *second, independent* curvature measure
— the CLIP-feature path-length / direct-distance ratio along the same
$\alpha$-sweep.

If both hold, the curvature framework is architecture-agnostic across
StyleGAN1/2 *and* latent diffusion.

## Prior art and our delta

- **Park, Kwon et al. NeurIPS 2023** [arXiv:2307.12868](https://arxiv.org/abs/2307.12868)
  construct a pullback Riemannian metric $g(t) = J^\top J$ on h-space
  for SD v1.5 and FFHQ-DDPM, take its SVD for editing bases. **First-order
  only.** No second derivative, no extrinsic-curvature notion, no link
  to disentanglement.
- **Asyrp / h-space** Kwon et al. ICLR 2023 [arXiv:2210.10960](https://arxiv.org/abs/2210.10960)
  establishes h-space as the closest published analog of StyleGAN's W:
  homogeneous, linear, robust, timestep-consistent. We use the same space.
- **GENIE** Dockhorn et al. 2022 [arXiv:2208.05405](https://arxiv.org/abs/2208.05405)
  uses JVPs through $\epsilon_\theta$ as a higher-order ODE solver — the
  established recipe for forward-mode AD through a U-Net sampling chain.
  We borrow their JVP-through-DDIM construction.
- **DAAM** Tang et al. ACL 2023 [arXiv:2210.04885](https://arxiv.org/abs/2210.04885)
  is the saliency baseline (cross-attention attribution). We compare our
  gradient-saliency to DAAM at the same timestep.

**Our delta:** we measure the *second* derivative through the sampling
chain, define an extrinsic-curvature ratio, and ask whether it predicts
disentanglement (composition non-linearity) — exactly the C1+C2+C4
chain we've established on StyleGAN, now on a fundamentally different
architecture.

## Method

**Generator.** Stable Diffusion v1.5 base, fp16 with model offloading.
512×512 output via the VAE decoder.

**Sampler.** DDIM, 50 steps, $\eta=0$, CFG scale 7.5 (matches Asyrp/SEGA
/Park-NeurIPS23/LEDITS defaults).

**Latent.** h-space = output of `unet.mid_block` (deepest mid-block
feature map). This is the W-analog established by Asyrp.

**Directions $v$.** CLIP-text-derived: take a *neutral* prompt P0 and
five *attribute* prompts $P_a$ (analogues of FFHQ attributes — smile,
age, gender, eyeglasses, pose). For each $a$, define $v_a$ = the h-space
"edit direction" given by SEGA-style guidance:
$v_a = \mathbb{E}_t[\epsilon_\theta(z_t, P_a) - \epsilon_\theta(z_t, P_0)]$
averaged at the editing timestep. This is what SEGA / Brack et al. 2023
calls the concept guidance vector.

**Timesteps.** Three editing windows $t \in \{0.7T, 0.5T, 0.3T\}$ on
the 50-step schedule (Asyrp Fig.5 / Park-NeurIPS23 §4.2 convention).

**First derivative.** $\partial x_0/\partial \alpha$ via forward-mode
JVP through the remaining $M$ U-Net evaluations from $t$ down to 0,
GENIE-style:
```
dx0_dα = jvp(lambda α: ddim_sample_from(h_t + α*v_a, t),
             (zero,), (one,))[1]
```
For the 50-step schedule starting from $t_{\text{edit}} = 0.5T$, this
is 25 U-Net evaluations under JVP.

**Second derivative.** Composed JVP — `jvp(jvp(...))` — as we already
do for StyleGAN, applied to the same DDIM chain.

**Saliency map.** Per-pixel $|d x_0/d\alpha|$ averaged across the 3 RGB
channels, just as in StyleGAN.

**Curvature ratio (C1 metric).**
$\bar\rho_a = \mathbb{E}_n[|\partial^2 x_0/\partial \alpha^2|] /
              \mathbb{E}_n[|\partial x_0/\partial \alpha|]$
over $N = 64$ noise seeds per attribute.

**CLIP path ratio (C2 second measure).** Sweep $\alpha \in [-3, +3]$ in
13 steps, extract CLIP-ViT-B-32 image features at each step, compute
path-length / direct-distance ratio. Same code path as
[run_c2_path_curvature_ffhq.py](../metrics/run_c2_path_curvature_ffhq.py).

**Baseline saliency.** DAAM cross-attention maps for the same prompt
swap at the same timestep, aggregated over the chosen DDIM step set.

## Expected signal

- **C1 positive.** $\bar\rho_a$ values spread by at least 5× across the
  5 attributes (matches FFHQ where smile=1.75 and pose=49.9).
- **C2 positive.** Pearson correlation between pixel $\bar\rho_a$ and
  CLIP-path ratio across the 5 attributes $\geq +0.7$.
- **Cross-architecture support.** Rank of "structural vs textural"
  attributes is preserved: pose/eyeglasses high, smile/age moderate,
  pure-texture (skin tone) low.
- **Saliency-baseline contrast.** Pixel correlation between our
  $|\partial x_0/\partial v|$ and DAAM cross-attention map $< 0.3$ —
  consistent with the JVP-vs-Grad-CAM near-orthogonality on StyleGAN.

## Failure modes (what would falsify or contaminate)

1. **Memory blow-up at 512².** JVP through 25 U-Net evaluations may
   OOM on 8 GB. Mitigation: gradient checkpointing on the U-Net mid-block
   only; fall back to 256² via `unet.set_attn_processor`. If even 256²
   fails: drop to $t_\text{edit} = 0.3T$ so the JVP chain is short (~7
   evaluations).

2. **Noise dominance.** At $t = 0.7T$ the rendered $x_0$ is still very
   noisy under DDIM. The second derivative may be drowned by noise from
   $\epsilon_\theta$ — false negative for C1. Mitigation: report all
   three timesteps; require positive C1 at $t \in \{0.5T, 0.3T\}$.

3. **Direction definition contamination.** If $v_a$ is computed at the
   same noise seed used for evaluation, we measure the *self-direction*
   (circular). Mitigation: $v_a$ is averaged over 64 *training* seeds,
   evaluation is on 64 *disjoint* test seeds.

4. **CLIP path ratio dominated by lighting.** If the α-sweep also moves
   global lighting, the CLIP-path ratio inflates spuriously. Mitigation:
   normalise CLIP features and report the relative ratio across
   attributes within the same lighting baseline.

5. **VAE decoder non-linearity confound.** Our derivatives include the
   VAE decoder, not just the U-Net. Mitigation: also report derivatives
   in *latent space* $z_0$ (pre-decoder) as a secondary table; the
   image-space numbers are the primary headline because that matches
   what we report for StyleGAN.

## Compute budget

- N=64 seeds × 5 attributes × 3 timesteps × (1st + 2nd derivative)
  = 1920 JVP runs.
- ~25 U-Net evaluations per JVP, ~0.3 s/eval at 512² fp16 → ~7.5 s/JVP.
- Total ~4 hours, plus VAE decoding and CLIP feature extraction.

## Deliverables

- `experiments/diffusion/generator.py` — SD h-space wrapper with
  `synthesize_from_h(h_t, t_edit)` returning JVP-safe $x_0$.
- `experiments/diffusion/run_c1_c2.py` — main script.
- `experiments/out/sd_c1_c2/` — per-attribute $\bar\rho$, CLIP-path
  ratio, and the corresponding scatter + bar plot.
- `paper/sections/05_experiments.tex` — third-architecture C1/C2 row.
