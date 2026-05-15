# Design — LatentCLR + DisCo baselines

**Track 5.** Add two 2021–2022 contrastive direction-discovery
baselines (LatentCLR ICCV 2021, DisCo ICLR 2022) so the paper has
modern competitors beyond the 2020-era GANSpace/SeFa.

## Hypothesis

**H6.** When evaluated by our framework's coverage / spatial-diversity
metric (Track 7 from previous session) and by the editing head-to-head
in [02_editing_head_to_head.md](02_editing_head_to_head.md),
LatentCLR + DisCo are **not strictly better** than GANSpace/SeFa on
FFHQ — *contrastive learning of directions does not subsume the linear
methods* for our metric panel. This is a negative claim with diagnostic
value.

## Prior art

- **LatentCLR** Yüksel et al. ICCV 2021 [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuksel_LatentCLR_A_Contrastive_Learning_Approach_for_Unsupervised_Discovery_of_Interpretable_ICCV_2021_paper.pdf),
  [code](https://github.com/catlab-team/latentclr). Learns K direction
  networks $d_k(z)$ by NT-Xent contrastive loss in feature-space
  differences. K=100 on FFHQ-StyleGAN2.
- **DisCo** Ren et al. ICLR 2022 [paper](https://openreview.net/pdf?id=j-63FSNcO5a),
  [code](https://github.com/xrenaa/DisCo). Same contrastive framing,
  improved disentanglement objectives. Reports FactorVAE / DCI / β-VAE
  on Shapes3D/MPI3D/Cars3D.
- LatentCLR reports **no quantitative scalars on FFHQ** in the main
  paper — so any number we compute is the first published reference.

## Method

**LatentCLR re-implementation.**
- Use the official `catlab-team/latentclr` config as the reference.
- Backbone: a 5-layer MLP per direction $d_k(z) \in \mathbb{R}^{w_\text{dim}}$.
- Feature extractor: the StyleGAN synthesis activations at layer index
  10 (their default).
- Loss: NT-Xent over the K direction outputs, batch 32, K=100, T=0.5.
- Optimizer: Adam, lr 5e-4, 100 epochs of 1000 batches each.
- Generator: the same FFHQ-StyleGAN1 (InterFaceGAN) we use elsewhere.

**DisCo re-implementation.**
- `xrenaa/DisCo` reference; their FFHQ config.
- Same K=100 directions, contrastive in feature differences but with
  the orthogonality regulariser from their §3.3.
- 100 epochs.

**Training compute.** 8h–10h per baseline on RTX 3070 (estimated from
LatentCLR's reported 12h on V100, scaling by FLOPs). Run sequentially.

**Evaluation.** Plug into the existing pipeline:
- coverage (Track 7) and spatial diversity ([run_spatial_diversity.py](../baselines/run_spatial_diversity.py))
- editing head-to-head ([02_editing_head_to_head.md](02_editing_head_to_head.md))
- per-direction $\bar\rho$ ratios

## Expected signal

- **Coverage at K=8.** LatentCLR ≈ GANSpace on FFHQ (both PCA-like).
  DisCo possibly higher due to explicit orthogonality.
- **Spatial diversity.** LatentCLR > GANSpace > HiGAN-GT, roughly
  matching the random-direction baseline (because LatentCLR pushes
  for inter-direction contrast).
- **Editing head-to-head.** Curvature-low ranking still leads on
  ID-cos at matched target-Δ — i.e. our pre-selection metric is
  orthogonal to contrastive-vs-PCA training choice.

## Failure modes

1. **Code rot.** 2021–2022 PyTorch / StyleGAN versions may not load on
   torch 2.2.2. Mitigation: vendor the model files and patch
   incompatibilities, no upstream pip-install.
2. **Training instability.** Contrastive losses sometimes collapse.
   Mitigation: track NT-Xent loss curve; if loss < threshold within
   1 epoch (collapse), reset with different seed.
3. **K=100 too large.** Evaluation on 100 directions × 1000 latents
   is heavy. Mitigation: for the editing head-to-head we take the
   top-16 *anyway* (Track 2 protocol), so most evaluation is on a
   filtered subset.

## Compute budget

- LatentCLR training: ~10 h.
- DisCo training: ~10 h.
- Evaluation (top-16 each on the head-to-head): ~5 h.
- **Total ~25 h.** Background run, sequential.

## Deliverables

- `experiments/baselines/latentclr/` — vendored model + train script
- `experiments/baselines/disco/` — same
- `experiments/out/latentclr_ffhq/` and `disco_ffhq/` — 100 directions
  per method + evaluation table
- Paper §5.5: contrastive-baselines row added to the head-to-head table
