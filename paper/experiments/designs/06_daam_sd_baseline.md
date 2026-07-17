# Design — DAAM saliency baseline on Stable Diffusion

**Wave 2, Track 6.** Add a diffusion saliency baseline so our SD
C1/C2 results sit in a published-protocol comparison, not in
isolation.

## Hypothesis (falsifiable)

**H7.** On SD v1.5, our gradient-based saliency $|\partial x_0/\partial v|$
along an h-space direction $v$ is **near-orthogonal** to DAAM
cross-attention saliency for the same prompt — replicating the
JVP-vs-Grad-CAM orthogonality finding from §exp-baselines-results on
the diffusion architecture.

If true: editing-aligned (ours) and recognition-aligned (DAAM) are
two distinct measurement axes on diffusion too — strengthens the
paper's "complementary, not competing" framing.

## Prior art

- **DAAM** Tang et al. ACL 2023 [arXiv:2210.04885](https://arxiv.org/abs/2210.04885)
  aggregates per-token cross-attention maps across all U-Net
  attention layers and DDIM steps to get a per-pixel attribution.
  De facto standard for "$\partial I /\partial \text{prompt}$" in SD.
- **Concept-TRAK** Park et al. 2025 [arXiv:2507.06547](https://arxiv.org/abs/2507.06547)
  uses deterministic DDIM inversion + autograd for influence
  attribution; relevant as a secondary point of comparison.
- Our bedroom CLIP-Grad-CAM result (corr ∈ [-0.105, -0.049])
  established the orthogonality pattern on StyleGAN; H7 tests
  whether it carries over.

## Method

**Model.** SD v1.5, same loader as `experiments/diffusion/generator.py`.

**Saliency 1 (ours).** $|\partial x_0/\partial \alpha|$ at $\alpha = 0$
via forward-mode JVP through the DDIM chain starting from $t_{\text{edit}}
= 0.5T$, with $v$ the SEGA-style h-space direction for each of the 5
face attributes (smile, age, pose, gender, eyeglasses). Already
computed in Track 1.

**Saliency 2 (DAAM baseline).** Run the same prompt through diffusers
with a per-layer cross-attention hook. For each U-Net attention
layer, capture the attention probabilities for the attribute-prompt's
specific tokens (e.g. "smile", "wrinkled"). Upsample to 512² and
sum across layers and DDIM steps. This is DAAM eq. (1).

**Metrics.**
- per-pixel Pearson correlation between the two saliency maps
- top-20% mask IoU
- both metrics averaged over $N=32$ noise seeds × 5 attributes

## Expected signal

- $|\text{corr}|$ in $[0, 0.2]$ on all 5 attributes (orthogonal).
- IoU(top-20%) near chance (0.05–0.15).
- Same qualitative pattern as bedroom + FFHQ on StyleGAN.

## Failure modes

1. DAAM's token-targeted attention requires the prompt to literally
   contain the attribute word. For "side profile" or "wide smile",
   tokenisation choice matters. Mitigation: pick prompts whose
   attribute-noun token is unambiguous and report the chosen
   tokens.
2. If correlation is *high* (> 0.3), it would mean our gradient
   saliency reduces to attention attribution on SD — a *less*
   interesting paper. We report this finding either way.

## Compute budget

- 5 attrs × 32 seeds × (1 forward sample + DAAM hook) at 512² ≈
  100 × 12 s = 20 min. Cheap.
- Reuse Track 1's JVP saliencies from `experiments/out/sd_c1_c2/`.

## Deliverables

- `experiments/diffusion/run_daam_comparison.py`
- `experiments/out/sd_daam/` — per-attr corr / IoU + figure
- §5 baseline table updated with the SD row.
