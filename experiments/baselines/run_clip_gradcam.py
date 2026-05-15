"""Real Grad-CAM baseline through a CLIP classifier.

The standard Grad-CAM paradigm requires a classifier whose score we
differentiate w.r.t. activations. We do not have a bedroom-attribute
classifier; instead we use CLIP zero-shot:

  score(image; attribute) = CLIP_image(image) · CLIP_text(attribute)

Then for an image produced by the generator from latent wp, we
backpropagate this score through CLIP back to the input image (this
gives image-space gradient saliency), and also pass through to the
generator (giving wp-direction sensitivity).

Comparison metric: per-pixel correlation between
   our JVP saliency for attribute boundary b_a
   vs CLIP-score Grad-CAM for the attribute text prompt

This is the cleanest apples-to-apples comparison between:
   - our classifier-free, generator-side approach (our paper)
   - classifier-based, classifier-side approach (Grad-CAM 2017 standard)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                  # noqa: E402
from higan_dev.manipulate import load_boundary                  # noqa: E402
from higan_dev.cam.grad_saliency import _layered_direction      # noqa: E402


CLIP_PROMPTS = {
    "indoor_lighting": "a brightly lit bedroom with a lamp",
    "wood":            "a bedroom with wooden furniture",
    "view":            "a bedroom with a window showing an outdoor view",
    "carpet":          "a bedroom with a carpet on the floor",
    "cluttered_space": "a cluttered messy bedroom",
    "glossy":          "a glossy shiny bedroom",
    "dirt":            "a dirty bedroom",
    "scary":           "a scary creepy bedroom",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["view", "indoor_lighting", "wood", "glossy"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out", default="out/clip_gradcam")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading HiGAN bedroom ===")
    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    bdir = PAPER.parent / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan_bedroom"
    boundaries = {a: load_boundary(str(bdir), a, num_layers=L).to(G.device)
                  for a in args.attrs}
    b_layered = {a: _layered_direction(boundaries[a], L, D, G.device)
                 for a in args.attrs}

    print("=== loading CLIP ===")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(G.device)

    # Tokenise and encode text features once
    text_features = {}
    for a in args.attrs:
        toks = tokenizer([CLIP_PROMPTS[a]]).to(G.device)
        with torch.no_grad():
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
        text_features[a] = f.squeeze(0)         # (clip_dim,)

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    # CLIP preprocessing is fixed PIL pipeline; replicate the same
    # normalisation as torchvision transforms on tensor inputs so we
    # can backprop through it.
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=G.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=G.device).view(1, 3, 1, 1)

    def clip_normalise_resize(image_neg1_pos1: torch.Tensor) -> torch.Tensor:
        """[-1,1] tensor (B,3,H,W) → CLIP-normalised, resized to 224^2."""
        x = (image_neg1_pos1.clamp(-1, 1) + 1) / 2.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - clip_mean) / clip_std
        return x

    print(f"=== computing JVP saliency + CLIP Grad-CAM for {len(args.attrs)} attrs ===")
    results = {}
    rows = []
    for attr in args.attrs:
        print(f"\n--- {attr} ---")
        b_la = b_layered[attr]
        text_feat = text_features[attr]

        # accumulate per-pixel JVP saliency and CLIP-grad saliency
        acc_jvp = torch.zeros(H, W, device=G.device)
        acc_cg = torch.zeros(H, W, device=G.device)
        per_sample_jvp = []
        per_sample_cg = []
        per_sample_img = []
        for s in range(args.num_samples):
            wp_base = base_wp[s:s + 1].detach()

            # --- (a) our JVP saliency ---
            def f(alpha):
                return G.synthesize(wp_base + alpha.view(1, 1, 1) * b_la.unsqueeze(0))
            _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            jvp_sal = dimg.abs().mean(dim=1).squeeze(0)
            acc_jvp += jvp_sal

            # --- (b) CLIP-grad saliency ---
            # Make wp require grad and propagate to image-space gradient
            with torch.enable_grad():
                wp = wp_base.detach().clone().requires_grad_(True)
                img = G.synthesize(wp)              # (1, 3, H, W) in ~[-1, 1]
                img_clip = clip_normalise_resize(img)
                img_feat = model.encode_image(img_clip)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                score = (img_feat * text_feat.unsqueeze(0)).sum()
                # gradient on the image w.r.t. CLIP score
                img_grad, = torch.autograd.grad(score, img, retain_graph=False)
                cg_sal = img_grad.abs().mean(dim=1).squeeze(0)
            acc_cg += cg_sal.detach()

            if len(per_sample_img) < 3:
                with torch.no_grad():
                    img_show = G.synthesize(wp_base)
                im_u8 = (((img_show.clamp(-1, 1) + 1) / 2).squeeze(0)
                         .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                per_sample_img.append(im_u8)
                m1 = jvp_sal.max().clamp_min(1e-8)
                m2 = cg_sal.max().clamp_min(1e-8)
                per_sample_jvp.append((jvp_sal / m1).cpu().numpy())
                per_sample_cg.append((cg_sal / m2).detach().cpu().numpy())
            torch.cuda.empty_cache()

        jvp_mean = (acc_jvp / args.num_samples).cpu().numpy()
        cg_mean = (acc_cg / args.num_samples).cpu().numpy()

        # pixel-wise correlation between the two saliency maps
        corr = float(np.corrcoef(jvp_mean.flatten(), cg_mean.flatten())[0, 1])
        # IoU at top-20%
        thr_j = np.quantile(jvp_mean, 0.8)
        thr_c = np.quantile(cg_mean, 0.8)
        mask_j = jvp_mean >= thr_j
        mask_c = cg_mean >= thr_c
        inter = float((mask_j & mask_c).sum())
        union = float((mask_j | mask_c).sum())
        iou = inter / union if union > 0 else 0.0
        print(f"  CLIP prompt: '{CLIP_PROMPTS[attr]}'")
        print(f"  Pixel-correlation(JVP, CLIP-grad) = {corr:+.3f}")
        print(f"  IoU(top-20%) = {iou:.3f}")
        results[attr] = {"corr": corr, "iou": iou,
                          "prompt": CLIP_PROMPTS[attr]}

        # save visualisation row: 3 samples × (image | jvp | clip-grad)
        from higan_dev.cam.diff_map import colorize_heat, overlay
        cells = []
        for k in range(len(per_sample_img)):
            cells.append(per_sample_img[k])
            cells.append(colorize_heat(per_sample_jvp[k]))
            cells.append(colorize_heat(per_sample_cg[k]))
        row = np.concatenate(cells, axis=1)
        # add a header strip
        h_lbl = 24
        header = Image.new("RGB", (row.shape[1], h_lbl), (245, 245, 244))
        from PIL import ImageDraw, ImageFont
        d_lbl = ImageDraw.Draw(header)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        cw = H
        for k in range(len(per_sample_img)):
            d_lbl.text((k * 3 * cw + 6, 4),
                       f"image / JVP saliency / CLIP-grad  (sample {k+1})",
                       fill=(40, 40, 40), font=font)
        eye_h = 28
        eyebrow = Image.new("RGB", (row.shape[1], eye_h), (245, 245, 244))
        d_eye = ImageDraw.Draw(eyebrow)
        try:
            font2 = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font2 = ImageFont.load_default()
        d_eye.text((10, 5),
                   f"━━ {attr.upper()}  ·  pixel-corr(JVP, CLIP-grad)={corr:+.3f}  ·  IoU={iou:.3f} ━━",
                   fill=(40, 40, 40), font=font2)
        rows.append(np.concatenate([np.asarray(eyebrow), np.asarray(header),
                                    row], axis=0))

    final = np.concatenate(rows, axis=0)
    Image.fromarray(final).save(out / "clip_gradcam_comparison.png")
    print(f"\nsaved {out / 'clip_gradcam_comparison.png'}")
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
