"""CLIP-Grad-CAM on FFHQ — domain cross-check of the JVP-vs-Grad-CAM
orthogonality result shown on bedroom.

We expect the same finding: JVP saliency (where pixels move when latent
moves along boundary b) and CLIP-Grad-CAM (where CLIP attends to the
attribute in the rendered image) measure different things even when
they share the same attribute name.

Per-attribute metrics: pixel-correlation and IoU(top-20%).
Resolution is 1024² so we use num-samples=4 to fit on 8 GB.
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
sys.path.insert(0, str(PAPER / "experiments"))

from domains.ffhq.generator import FFHQGenerator               # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}

CLIP_PROMPTS = {
    "smile":      "a smiling face with teeth showing",
    "age":        "an old face with wrinkles",
    "pose":       "a face turned to the side, side profile",
    "gender":     "a male face with a beard",
    "eyeglasses": "a face wearing eyeglasses",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["pose", "smile", "eyeglasses", "age", "gender"])
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--out", default="out/ffhq_clip_gradcam")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading FFHQ StyleGAN ===")
    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"

    b_layered = {}
    for a in args.attrs:
        v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(v).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=G.device)
        for li in LAYERS_FOR[a]:
            bl[li] = b_dir
        b_layered[a] = bl

    print("=== loading CLIP ===")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(G.device)

    text_features = {}
    for a in args.attrs:
        toks = tokenizer([CLIP_PROMPTS[a]]).to(G.device)
        with torch.no_grad():
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
        text_features[a] = f.squeeze(0)

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=G.device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=G.device).view(1, 3, 1, 1)

    def clip_norm_resize(img):
        x = (img.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - clip_mean) / clip_std

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    print(f"=== JVP vs CLIP-Grad-CAM, {len(args.attrs)} attrs × {args.num_samples} samples ===")
    results = {}
    for attr in args.attrs:
        print(f"--- {attr} ---")
        b_la = b_layered[attr]
        text_feat = text_features[attr]
        acc_jvp = torch.zeros(H, W, device=G.device)
        acc_cg = torch.zeros(H, W, device=G.device)
        for s in range(args.num_samples):
            wp_base = base_wp[s:s + 1].detach()

            def f(alpha):
                return G.synthesize(wp_base + alpha.view(1, 1, 1)
                                    * b_la.unsqueeze(0))
            _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            acc_jvp += dimg.abs().mean(dim=1).squeeze(0)
            torch.cuda.empty_cache()

            with torch.enable_grad():
                wp = wp_base.detach().clone().requires_grad_(True)
                img = G.synthesize(wp)
                img_clip = clip_norm_resize(img)
                img_feat = model.encode_image(img_clip)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                score = (img_feat * text_feat.unsqueeze(0)).sum()
                img_grad, = torch.autograd.grad(score, img, retain_graph=False)
                cg_sal = img_grad.abs().mean(dim=1).squeeze(0)
            acc_cg += cg_sal.detach()
            torch.cuda.empty_cache()

        jvp_mean = (acc_jvp / args.num_samples).cpu().numpy()
        cg_mean = (acc_cg / args.num_samples).cpu().numpy()

        corr = float(np.corrcoef(jvp_mean.flatten(), cg_mean.flatten())[0, 1])
        thr_j = np.quantile(jvp_mean, 0.8)
        thr_c = np.quantile(cg_mean, 0.8)
        mask_j = jvp_mean >= thr_j
        mask_c = cg_mean >= thr_c
        inter = float((mask_j & mask_c).sum())
        union = float((mask_j | mask_c).sum())
        iou = inter / union if union > 0 else 0.0
        print(f"  prompt='{CLIP_PROMPTS[attr]}'  corr={corr:+.3f}  IoU={iou:.3f}")
        results[attr] = {"corr": corr, "iou": iou,
                         "prompt": CLIP_PROMPTS[attr]}

    # Summary plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
    attrs_sorted = sorted(results, key=lambda a: results[a]["corr"])
    corrs = [results[a]["corr"] for a in attrs_sorted]
    ious = [results[a]["iou"] for a in attrs_sorted]
    x = np.arange(len(attrs_sorted))
    ax.bar(x - 0.2, corrs, width=0.4, color="#0e7490",
           label="pixel correlation")
    ax.bar(x + 0.2, ious, width=0.4, color="#c2410c",
           label="IoU (top-20%)")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(attrs_sorted, fontsize=10)
    ax.set_ylabel("agreement", fontsize=10)
    ax.set_title("FFHQ — JVP saliency vs CLIP-Grad-CAM "
                 "(near-zero ⇒ orthogonal measurements)",
                 fontsize=11, weight="bold", pad=8)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "ffhq_clip_gradcam.png")
    print(f"\nsaved {out / 'ffhq_clip_gradcam.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
