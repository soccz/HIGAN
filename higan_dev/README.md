# higan_dev — HiGAN bedroom: inversion + boundary editing + saliency

A local Python port and extension of the original notebook prototypes
(`HIGAN_encoder.ipynb`, `HiGAN_PSP.ipynb`) developed for the RTX 3070 / 8 GB
class of GPUs. What started as a fix for a broken gradient grew into a
multi-method probe of HiGAN's W+ space.

> Full report: **<https://soccz.github.io/projects/higan/>**

## What this package does

1. **Differentiable generator wrapper.** `genforce/higan`'s
   `easy_synthesize` returns a detached numpy `uint8` array, silently
   breaking autograd — the optimisation-based inversion in v1 was reduced
   to "pick the best of N random inits". `HiGANGenerator.synthesize(wp)`
   calls `G.net.synthesis` directly and keeps gradients alive.
2. **Custom bedroom-domain encoder** trained with synthetic supervision
   (sample wp ~ p(w) → render → regress wp). Replaces the FFHQ-trained
   pSp encoder used in v1.
3. **Two flavours of pixel saliency** for HiGAN's 8 attribute boundaries:
   forward perturbation diff (cheap) and JVP-based backward gradient
   (per-pixel, scene-specific lamp/window/etc localisation).
4. **Eleven follow-up analyses** that exploit the differentiable wrapper:
   per-layer decomposition, full 8 × 14 attribute–layer matrix,
   disentanglement matrix, saliency-guided local edit, encoder attention
   (mirror question), random direction discovery, compositional editing,
   robustness, intermediate-layer (true Grad-CAM-style), unsupervised
   taxonomy via K-means, CLIP zero-shot cluster labelling, encoder ckpt
   evolution, second-order ∂²I/∂α², saliency morph animations, real
   LSUN-bedroom photos.

## Quick start

```bash
# 1. Pin torch + cu121 first
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Rest (some need --no-deps to avoid pulling a newer torch)
pip install --no-deps lpips==0.1.4 open_clip_torch
pip install pyyaml "numpy<2" pillow opencv-python tqdm matplotlib scipy \
            scikit-learn datasets ftfy regex wcwidth huggingface_hub \
            safetensors timm

# 3. Download HiGAN assets (~110 MB)
python scripts/01_download_assets.py
```

## Pipeline (28 numbered scripts)

The numbering encodes the order in which they were *added* to the
project, not necessarily the order of typical use. Most scripts read
the same `configs/default.yaml` and write to `out/`.

### Setup & baselines (1–4, 7)

| script | purpose |
| --- | --- |
| `01_download_assets.py` | clone genforce/higan, fetch bedroom256 generator + boundaries + w_1k |
| `02_invert_optim.py` | optimisation-based W+ inversion, with proper lr ramp |
| `03_train_encoder.py` | train the W+ encoder via synthetic supervision (`--resume` supported) |
| `04_invert_encoder.py` | one-shot encoder-based inversion |
| `07_make_testset.py` | build a fixed 4-image synthetic test set with GT wp |

### Core editing & saliency (5, 6, 8–11)

| script | purpose |
| --- | --- |
| `05_manipulate.py` | sweep along boundary, render distance-grid |
| `06_cam_analysis.py` | forward perturbation diff-map saliency |
| `08_compare_inversion.py` | optim vs encoder inversion table |
| `09_edit_real.py` | encoder + boundary editing on test images |
| `10_final_figure.py` | composite summary figure (compare + cam + edit) |
| `11_grad_saliency.py` | JVP-based gradient saliency (headline) |

### Deeper analyses (12–28)

| script | purpose | adds |
| --- | --- | --- |
| `12_per_layer_saliency.py` | per-layer JVP for 3 attributes | layer-by-layer breakdown |
| `13_morph_gif.py` | animated WebP boundary sweep | smooth morphs |
| `14_full_cycle_figure.py` | one-image full pipeline strip | end-to-end demo |
| `15_disentangle.py` | 8 × 8 saliency correlation matrix | attribute clustering |
| `16_local_edit.py` | saliency-as-mask local editing | precise edits |
| `17_encoder_attention.py` | encoder reverse-mode attention | mirror question |
| `18_random_directions.py` | random W+ direction discovery | non-canonical attributes |
| `19_composition.py` | sal(a+b) vs sal(a)+sal(b) | linear vs non-linear pairs |
| `20_robustness.py` | per-sample saliency consistency | scene-specificity |
| `21_layer_saliency.py` | ∂(activation) / ∂α at intermediate blocks | true Grad-CAM analog |
| `22_taxonomy.py` | K-means on 256 random saliencies | unsupervised attribute families |
| `23_full_matrix.py` | 8 × 14 attribute × layer saliency | layer-specificity of boundaries |
| `24_clip_label_clusters.py` | CLIP zero-shot labels for taxonomy clusters | semantic id |
| `25_real_photo_cycle.py` | LSUN bedroom photo full cycle | OOD validation |
| `26_ckpt_evolution.py` | saliency convergence across encoder ckpts | training dynamics |
| `27_higher_order.py` | first-order vs ∂²I/∂α² | non-linearity ratio |
| `28_saliency_morph.py` | animated WebP with per-frame saliency | sees curvature live |

## Layout

```
higan_dev/
├── higan_dev/
│   ├── config.py              dataclass-loaded YAML
│   ├── generator.py           differentiable HiGAN wrapper + monkey-patch for JVP
│   ├── losses.py              VGG perceptual / LPIPS / TV / combined
│   ├── manipulate.py          torch-native boundary manipulation
│   ├── utils.py               image utilities, AverageMeter, label_bar
│   ├── encoder/
│   │   ├── model.py           ResNet50 backbone + per-scale necks + 14 style heads
│   │   └── train.py           synthetic-supervision training, resume support
│   ├── inversion/
│   │   ├── optim.py           Adam-on-W+ inversion w/ proper lr ramp
│   │   └── encode.py          one-shot inversion via encoder
│   └── cam/
│       ├── diff_map.py        forward perturbation pixel attribution
│       ├── grad_saliency.py   JVP per-pixel ∂I/∂α + per-layer decomposition
│       ├── disentangle.py     8 × 8 attribute correlation
│       ├── local_edit.py      saliency-as-mask local editing
│       ├── encoder_attention.py  reverse-mode encoder attention
│       └── composition.py     pairwise compositional saliency
├── scripts/                   28 CLI entrypoints (see table above)
├── configs/default.yaml
└── data/, out/                downloaded assets / generated outputs (gitignored)
```

## Compute notes (RTX 3070 / 8 GB)

| Job | Memory | Wall time |
| --- | --- | --- |
| Generator forward, batch 8 | ~3 GB | <0.1 s |
| Optim inversion, 1000 steps | ~3 GB | ~30 s per image |
| Encoder train, ResNet50 + B=8 + fp32 | ~5.4 GB | ~0.16 s/iter, 40 k iter ≈ 1 h 50 m |
| JVP saliency, 64 samples × 8 attrs | ~3 GB | ~10 s total |
| K-means taxonomy, 256 directions | ~3 GB | ~3 min |
| CLIP labelling, 8 clusters | ~3.3 GB | ~30 s |

Mixed precision is **off**: the StyleGAN bedroom synthesis op produces
NaN under fp16 autocast. Plenty of headroom in fp32.

## Key results

- **v1 was broken**: gradient flow severed by `easy_synthesize` →
  optim inversion couldn't move the latent. Loss 5.6 → 0.027 after the
  one-line fix.
- **Forward perturbation ≠ Grad-CAM**: the v1-style "CAM" was actually
  finite-difference sensitivity, not a backward-gradient method. Adding
  proper JVP through the generator (`grad_saliency.py`) made it actually
  Grad-CAM-spirit, and per-sample saliency now pinpoints scene-specific
  lamps / windows / wood frames.
- **HiGAN's 8 boundaries cluster into 2**: view (layers 0–4) is
  disentangled from the rest (layers 6–11), which are 0.6–0.83
  correlated with each other.
- **Compositional editing is non-linear when crossing layer clusters**:
  view + texture pairs show corr 0.51–0.55, fully explained by view's
  curvature being 23 × that of texture attributes (`27_higher_order.py`).
- **Saliency converges 45 × faster than reconstruction**: at iter 1k
  the encoder's saliency-vs-GT correlation is ~0.01, at iter 40k it's
  0.36. Recon MSE only improves by 18% over the same range.
- **Boundaries are layer-specific**: applying the indoor_lighting
  direction to layer 0 produces a giant pixel response with no semantic
  meaning. Real attribute behaviour lives at layers 6–11 with smaller
  but cleaner saliency.
- **CLIP zero-shot rediscovers `view`**: 256 random W+ directions
  → K-means → CLIP labelling → cluster 2 (n=37, layer 1) is identified
  as "a view through a window" with no labels at any stage.

## Known limits

- Both saliency methods are **classifier-free**: they tell you *where
  pixels move* under a semantic perturbation, not "where a classifier
  decided the attribute is". The grad version tracks *exact* per-pixel
  sensitivity (forward-mode JVP), but it's still not the same as a
  task-specific classifier's Grad-CAM.
- Encoder under-fits at 40 k iter (LPIPS 0.37). Standard pSp training
  uses 200–500 k. Saliency analysis is robust to this; reconstruction
  fidelity is not.
- Real LSUN photos work but the encoder hits LPIPS 0.41–0.59 on them
  (vs 0.31–0.44 on synthetic) — synthetic-supervision domain gap.
