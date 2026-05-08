# higan_dev — HiGAN bedroom inversion + attribute CAM

A local Python port of the original notebook prototypes
(`HIGAN_encoder.ipynb`, `HiGAN_PSP.ipynb`) developed for the RTX 3070 / 8 GB
class of GPUs. Three things are different from the notebooks:

1. **Differentiable generator wrapper.** `genforce/higan`'s
   `easy_synthesize` returns a detached numpy `uint8` array, which silently
   breaks gradient flow — the optimisation-based inversion in the notebook
   was effectively reduced to "pick the best of N random initialisations".
   `higan_dev.generator.HiGANGenerator` exposes `synthesize(wp) -> tensor`
   so the latent actually moves under Adam.
2. **Custom bedroom-domain encoder.** The notebook used the FFHQ-trained pSp
   encoder against a bedroom generator — a hard domain mismatch the README
   itself flagged. We replace it with a small ResNet-backbone encoder
   trained by synthetic supervision (sample `wp ~ p(w)` from the generator,
   render the image, regress back to `wp`). No real bedroom dataset needed.
3. **CAM-style attribute saliency.** The next step the notebook hinted at
   ("install `ttach`, but never use it") is implemented here as a
   classifier-free perturbation analysis: shift `wp` along each HiGAN
   boundary by ±δ, render, and accumulate pixel-space differences and
   variance to produce per-attribute heatmaps.


## Quick start

```bash
# 1. Install (assumes torch 2.2.2 + cu121 already in your env)
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps lpips==0.1.4
pip install pyyaml numpy "numpy<2" pillow opencv-python tqdm matplotlib scipy

# 2. Download HiGAN assets (generator, boundaries, w1k; ~110 MB)
python scripts/01_download_assets.py

# 3. Optimization-based inversion (sanity test — should converge to near-zero loss)
PYTHONPATH=. python scripts/02_invert_optim.py --self-test --steps 1000 --lr 0.1 \
    --out out/inv_selftest

# 4. Train the custom encoder (~1.8 h on RTX 3070, fp32, batch 8)
PYTHONPATH=. python scripts/03_train_encoder.py

# 5. Encoder inversion (one forward pass)
PYTHONPATH=. python scripts/04_invert_encoder.py \
    --ckpt out/encoder_train/ckpt/enc_020000.pt \
    --self-test --out out/inv_encoder

# 6. Attribute manipulation grid
PYTHONPATH=. python scripts/05_manipulate.py \
    --attrs indoor_lighting wood view --num-samples 4 --steps 5 --delta 3 \
    --out out/manipulate

# 7. CAM / spatial saliency for every attribute
PYTHONPATH=. python scripts/06_cam_analysis.py \
    --num-samples 64 --delta 1.5 --out out/cam
```

## Layout

```
higan_dev/
├── higan_dev/
│   ├── config.py              dataclass-loaded YAML
│   ├── generator.py           differentiable HiGAN wrapper
│   ├── losses.py              VGG perceptual / LPIPS / TV / combined
│   ├── manipulate.py          torch-native boundary manipulation
│   ├── encoder/
│   │   ├── model.py           ResNet50 backbone + multi-scale style heads
│   │   └── train.py           synthetic-supervision training loop
│   ├── inversion/
│   │   ├── optim.py           Adam-on-W+ inversion w/ proper lr ramp
│   │   └── encode.py          one-shot inversion via encoder
│   └── cam/
│       └── diff_map.py        boundary-perturbation pixel attribution
├── scripts/                   CLI entrypoints (numbered by pipeline order)
├── configs/default.yaml
└── data/, out/                downloaded assets / generated outputs (gitignored)
```

## Compute notes (RTX 3070 / 8 GB)

| Job                                        | Memory | Wall time         |
| ------------------------------------------ | ------ | ----------------- |
| Generator forward, batch 8                  | ~3 GB  | <0.1 s            |
| Optim inversion, 1000 steps                 | ~3 GB  | ~30 s per image    |
| Encoder train, ResNet50 + B=8 + fp32        | ~4.4 GB | ~0.32 s/iter, 20 k iter ≈ 1 h 50 m |
| CAM analysis, 64 samples × 8 attrs          | ~3 GB  | ~30 s total        |

Mixed precision is **off by default**: the StyleGAN bedroom synthesis op
produces NaN under fp16 autocast. There is plenty of headroom in fp32.

## Architecture choices

**Encoder**. ResNet50 (ImageNet-pretrained) → three per-scale "necks"
(2-conv stacks reducing to a 512-d global vector at 7×7, 14×14 and 28×28
features) → 14 per-layer linear heads → `(B, 14, 512)` `wp` prediction.
Layers 0–3 read from the deepest features (coarse: structure), 4–9 from
14×14 (mid: object), 10–13 from 28×28 (fine: texture).
~50 M parameters. The encoder learns a residual around `w_avg`
(the StyleGAN running mean) for stability.

**Loss**. `λ_w · MSE(wp, wp_gt) + L2(image) + 0.8·LPIPS + 1e-4·TV`.
The `wp` MSE is the strongest signal because we have ground-truth latents
from synthetic supervision; the image-space terms keep reconstructions
sharp where the latent has slack.

**CAM / diff-map**. For each boundary `b` and N latent samples:

```
wp± = wp ± δ · b   (only on the boundary's own manipulate_layers)
I±  = G(wp±)
abs_diff   ← mean_n |I+ − I−| (mean over RGB)
signed_diff ← mean_n (I+ − I−)
variance   ← var across [I−, I0, I+]
```

The accumulators are saved as raw `.npz` plus colorised PNGs and a final
`mean_image`-overlay so the heatmap can be read against an "average bedroom".

## Known limits

- The generator weights are the same `stylegan_bedroom256` checkpoint from
  `genforce/higan`; we did not retrain it.
- The CAM is **classifier-free**. It tells you *where pixels move* under a
  semantic perturbation, not "where a classifier decided the attribute is".
  If a Grad-CAM-style classifier signal is needed later, a small
  bedroom-attribute classifier can be added under `higan_dev/cam/grad_cam.py`.
- The pSp / FFHQ-encoder code path from the notebook is intentionally
  removed — domain mismatch made it worse than starting from `w_avg`.
