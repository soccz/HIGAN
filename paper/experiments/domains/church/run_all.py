"""Church domain — full battery: saliency, ∂²I/∂α², disentanglement.

Mirrors `domains/ffhq/run_saliency.py + run_higher_order.py +
run_disentangle.py` for StyleGAN2 church256 with the 3 available
HiGAN boundaries (clouds, sunny, vegetation). Single script because the
domain is small.

Default canonical manipulate-layers chosen by analogy with bedroom's
texture range (6–11). For coarser sky/vegetation phenomena we extend.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from domains.church.generator import ChurchGenerator     # noqa: E402

CHURCH_BOUNDARY_DIR = (
    Path(__file__).resolve().parents[4]
    / "higan_dev" / "data" / "higan_repo" / "boundaries" / "stylegan2_church"
)

# Church layer ranges are wider than bedroom; we'll use a coarse-friendly
# range for sky-related attributes and a mid-fine range for vegetation.
LAYERS_FOR = {
    "clouds":     list(range(0, 8)),     # coarse: sky region
    "sunny":      list(range(0, 8)),     # coarse-to-mid: lighting + sky
    "vegetation": list(range(6, 12)),    # mid-fine: textures on church
}


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
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--display-size", type=int, default=192)
    ap.add_argument("--out", default="out/church_all")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = ChurchGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    attrs = ["clouds", "sunny", "vegetation"]
    abs_flat: dict[str, np.ndarray] = {}
    results: list[dict] = []
    saliency_rows: list[np.ndarray] = []
    higher_rows: list[np.ndarray] = []

    for attr in attrs:
        bpath = CHURCH_BOUNDARY_DIR / f"{attr}_boundary.npy"
        b_vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(b_vec).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        canonical = LAYERS_FOR[attr]
        b_layered = torch.zeros(L, D, device=G.device)
        for li in canonical:
            b_layered[li] = b_dir

        first_acc = torch.zeros(H, W, device=G.device)
        second_acc = torch.zeros(H, W, device=G.device)
        mean_img_acc = torch.zeros(3, H, W, device=G.device)
        sample_heats: list[np.ndarray] = []
        sample_imgs: list[np.ndarray] = []
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()

            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))

            def df_dalpha(alpha: torch.Tensor) -> torch.Tensor:
                _, dimg = jvp(f, (alpha,), (torch.ones_like(alpha),))
                return dimg

            a0 = torch.zeros(1, device=G.device)
            ones = torch.ones(1, device=G.device)
            img0, first = jvp(f, (a0,), (ones,))
            _, second = jvp(df_dalpha, (a0,), (ones,))

            sal1 = first.abs().mean(dim=1).squeeze(0)
            first_acc += sal1
            second_acc += second.abs().mean(dim=1).squeeze(0)
            mean_img_acc += ((img0.clamp(-1, 1) + 1) / 2).squeeze(0)
            if len(sample_heats) < 3:
                m = sal1.max().clamp_min(1e-8)
                sample_heats.append((sal1 / m).cpu().numpy().astype(np.float32))
                sample_imgs.append(
                    (((img0.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0)
                     .cpu().numpy() * 255).astype(np.uint8)
                )
            torch.cuda.empty_cache()

        first_map = (first_acc / args.num_samples).cpu().numpy()
        second_map = (second_acc / args.num_samples).cpu().numpy()
        mean_u8 = ((mean_img_acc / args.num_samples).clamp(0, 1).permute(1, 2, 0)
                   .cpu().numpy() * 255).astype(np.uint8)

        ratio = second_map / (first_map + 1e-6)
        results.append({
            "attr": attr,
            "first_max": float(first_map.max()),
            "second_max": float(second_map.max()),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_p95": float(np.percentile(ratio, 95)),
            "ratio_p99": float(np.percentile(ratio, 99)),
        })
        print(f"  {attr:12s}  ratio mean={ratio.mean():7.3f}  "
              f"median={np.median(ratio):7.3f}  p95={np.percentile(ratio, 95):7.3f}")

        # for disentanglement matrix
        mn = first_map.max()
        abs_flat[attr] = (first_map / mn if mn > 1e-8 else first_map).flatten()

        # saliency-row figure
        def shrink(a: np.ndarray) -> np.ndarray:
            return np.asarray(Image.fromarray(a).resize(
                (args.display_size, args.display_size), Image.BILINEAR))

        first_n = first_map / max(first_map.max(), 1e-8)
        cells = [shrink(mean_u8), shrink(colorize(first_n)),
                 shrink(overlay_on(mean_u8, first_n))]
        for h_, im_ in zip(sample_heats, sample_imgs):
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
        saliency_rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

        # higher-order row
        second_n = second_map / max(second_map.max(), 1e-8)
        ratio_n = ratio / max(ratio.max(), 1e-8)
        row2 = np.concatenate([
            shrink(colorize(first_n.astype(np.float32))),
            shrink(colorize(second_n.astype(np.float32))),
            shrink(colorize(ratio_n.astype(np.float32), cmap="viridis")),
        ], axis=1)
        labels2 = ["|∂I/∂α|", "|∂²I/∂α²|", "ratio (viridis)"]
        label_strip2 = np.concatenate(
            [_label(l, args.display_size) for l in labels2], axis=1
        )
        eye2 = _label(
            f"━━ {attr.upper()}  ·  ratio mean={ratio.mean():.2f}  median={np.median(ratio):.2f}  p95={np.percentile(ratio, 95):.2f} ━━",
            row2.shape[1], h=28, fs=16,
        )
        higher_rows.append(np.concatenate([eye2, label_strip2, row2], axis=0))

    # build figures
    Image.fromarray(np.concatenate(saliency_rows, axis=0)).save(
        out / "church_saliency_grid.png"
    )
    Image.fromarray(np.concatenate(higher_rows, axis=0)).save(
        out / "church_higher_order_grid.png"
    )
    print("\nfigures saved:")
    print(f"  {out / 'church_saliency_grid.png'}")
    print(f"  {out / 'church_higher_order_grid.png'}")

    # disentanglement 3x3
    n = len(attrs)
    abs_corr = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(attrs):
        for j, b in enumerate(attrs):
            abs_corr[i, j] = float(np.corrcoef(abs_flat[a], abs_flat[b])[0, 1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
    im = ax.imshow(abs_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(attrs, rotation=20, ha="right", fontsize=10)
    ax.set_yticklabels(attrs, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = abs_corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(v) > 0.55 else "#1c1917")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    ax.set_title("Church |sal| pixel-wise correlation", fontsize=11,
                 pad=10, weight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "church_disentangle.png")

    # save metrics
    with open(out / "metrics.json", "w") as f:
        json.dump({"higher_order": results, "abs_corr": abs_corr.tolist(),
                   "attrs": attrs}, f, indent=2)

    print("\nMost entangled (lower triangle):")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {attrs[i]:12s} ↔ {attrs[j]:12s}  corr={abs_corr[i, j]:+.3f}")


if __name__ == "__main__":
    main()
