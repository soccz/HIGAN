"""First FFHQ JVP-saliency figure — sanity check that the pipeline replicates
on a domain other than HiGAN bedroom.

For each of 5 InterFaceGAN attributes (smile/age/pose/gender/eyeglasses),
compute per-pixel ∂I/∂α via forward-mode JVP, accumulate across N base
latents, and render a 5-row × (mean | abs | per-sample) figure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

# allow importing the FFHQ wrapper
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from domains.ffhq.generator import FFHQGenerator     # noqa: E402

# attribute → canonical "manipulate_layers" range (from InterFaceGAN paper)
LAYERS_FOR = {
    "pose":        list(range(0, 4)),     # coarse 4-32
    "gender":      list(range(0, 8)),     # coarse-to-mid
    "age":         list(range(0, 8)),     # coarse-to-mid
    "eyeglasses":  list(range(0, 8)),     # coarse-to-mid
    "smile":       list(range(4, 8)),     # mid (mouth region)
}

# bigger range option for sanity
ALL_LAYERS = list(range(18))


def _label(text: str, w: int, h: int = 22, fs: int = 13) -> np.ndarray:
    img = Image.new("RGB", (w, h), (245, 245, 244))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 3), text, fill=(40, 40, 40), font=font)
    return np.asarray(img)


def colorize(heat: np.ndarray, cmap: str = "magma") -> np.ndarray:
    import matplotlib.cm as cm
    rgba = cm.get_cmap(cmap)(np.clip(heat, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_on(img_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return ((1 - alpha) * img_u8 + alpha * colorize(heat)).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--num-samples", type=int, default=8,
                    help="latent samples per attribute (full 1024^2 is heavy)")
    ap.add_argument("--display-size", type=int, default=256,
                    help="downscale per-cell display size to keep figure size manageable")
    ap.add_argument("--out", default="out/ffhq_saliency")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    boundaries_dir = EXPERIMENTS_DIR / "data" / "interfacegan" / "boundaries"

    # one shared set of base latents so all attributes are evaluated on the
    # same bedrooms (er, faces). This is critical for any cross-attribute
    # comparison.
    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)
    print(f"sampled {args.num_samples} base latents, wp shape {tuple(base_wp.shape)}")

    rows = []
    for attr in args.attrs:
        bpath = boundaries_dir / f"stylegan_ffhq_{attr}_w_boundary.npy"
        if not bpath.exists():
            print(f"[skip] no boundary at {bpath}")
            continue
        b_vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(b_vec).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)

        # build layered direction: place on canonical layers, zero elsewhere
        canonical = LAYERS_FOR.get(attr, ALL_LAYERS)
        b_layered = torch.zeros(L, D, device=G.device)
        for li in canonical:
            b_layered[li] = b_dir

        # accumulate JVP saliency over base latents (sample-by-sample to stay safe)
        acc = torch.zeros(H, W, device=G.device)
        mean_img = torch.zeros(3, H, W, device=G.device)
        per_sample_heat: list[np.ndarray] = []
        per_sample_img: list[np.ndarray] = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            img0, dimg = jvp(
                f, (torch.zeros(1, device=G.device),),
                (torch.ones(1, device=G.device),),
            )
            sal_b = dimg.abs().mean(dim=1).squeeze(0)        # (H, W)
            acc += sal_b
            mean_img += ((img0.clamp(-1, 1) + 1) / 2).squeeze(0)
            if len(per_sample_heat) < 3:
                m = sal_b.max().clamp_min(1e-8)
                per_sample_heat.append((sal_b / m).cpu().numpy().astype(np.float32))
                per_sample_img.append(
                    (((img0.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0)
                     .cpu().numpy() * 255).astype(np.uint8)
                )
            torch.cuda.empty_cache()

        mean_sal = (acc / args.num_samples).cpu().numpy()
        mean_sal = mean_sal / max(mean_sal.max(), 1e-8)
        mean_u8 = ((mean_img / args.num_samples).clamp(0, 1).permute(1, 2, 0)
                   .cpu().numpy() * 255).astype(np.uint8)
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {attr:12s} max|sal|={mean_sal.max():.3f}  peak_gpu={peak_mem_gb:.2f}GB")

        # build the row: mean_img | mean saliency heat | overlay | 3 per-sample overlays
        def shrink(arr: np.ndarray) -> np.ndarray:
            pil = Image.fromarray(arr)
            return np.asarray(pil.resize((args.display_size, args.display_size), Image.BILINEAR))

        heat_rgb = colorize(mean_sal)
        ov = overlay_on(mean_u8, mean_sal)
        cells = [shrink(mean_u8), shrink(heat_rgb), shrink(ov)]
        for h_, im_ in zip(per_sample_heat, per_sample_img):
            cells.append(shrink(overlay_on(im_, h_)))
        row = np.concatenate(cells, axis=1)
        labels = ["mean image", "saliency", "overlay", "sample1", "sample2", "sample3"]
        label_strip = np.concatenate(
            [_label(l, args.display_size) for l in labels], axis=1
        )
        eyebrow = _label(
            f"━━ {attr.upper()}  ·  layers {canonical[0]}–{canonical[-1]}  ·  N={args.num_samples} ━━",
            row.shape[1], h=28, fs=16,
        )
        rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

        np.savez(out / f"{attr}_raw.npz", mean_sal=mean_sal, mean_u8=mean_u8)

    if rows:
        final = np.concatenate(rows, axis=0)
        out_path = out / "ffhq_saliency_grid.png"
        Image.fromarray(final).save(out_path)
        print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")


if __name__ == "__main__":
    main()
