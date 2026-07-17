"""FFHQ second-order saliency (∂²I/∂α²) — C2 cross-domain replication.

For each attribute, compute first-order pushforward |∂I/∂α| and
second-order pushforward |∂²I/∂α²| via composed JVP, then report the
non-linearity ratio mean / p95 / p99 per attribute. Hypothesis: pose
(coarse structural) has high ratio; texture-like (smile, age) low ratio.
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

from domains.ffhq.generator import FFHQGenerator     # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--display-size", type=int, default=192)
    ap.add_argument("--out", default="out/ffhq_higher_order")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    boundaries_dir = EXPERIMENTS_DIR / "data" / "interfacegan" / "boundaries"

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    results = []
    rows = []
    for attr in args.attrs:
        bpath = boundaries_dir / f"stylegan_ffhq_{attr}_w_boundary.npy"
        if not bpath.exists():
            continue
        b_vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(b_vec).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        canonical = LAYERS_FOR.get(attr, list(range(L)))
        b_layered = torch.zeros(L, D, device=G.device)
        for li in canonical:
            b_layered[li] = b_dir

        first_acc = torch.zeros(H, W, device=G.device)
        second_acc = torch.zeros(H, W, device=G.device)

        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()

            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))

            def df_dalpha(alpha: torch.Tensor) -> torch.Tensor:
                _, dimg = jvp(f, (alpha,), (torch.ones_like(alpha),))
                return dimg

            a0 = torch.zeros(1, device=G.device)
            ones = torch.ones(1, device=G.device)
            # First-order
            _, first = jvp(f, (a0,), (ones,))
            # Second-order (composed JVP)
            _, second = jvp(df_dalpha, (a0,), (ones,))

            first_acc += first.abs().mean(dim=1).squeeze(0)
            second_acc += second.abs().mean(dim=1).squeeze(0)
            torch.cuda.empty_cache()

        first_map = (first_acc / args.num_samples).cpu().numpy()
        second_map = (second_acc / args.num_samples).cpu().numpy()
        ratio = second_map / (first_map + 1e-6)
        # robust statistics
        ratio_mean = float(ratio.mean())
        ratio_median = float(np.median(ratio))
        ratio_p95 = float(np.percentile(ratio, 95))
        ratio_p99 = float(np.percentile(ratio, 99))
        results.append({
            "attr": attr,
            "ratio_mean": ratio_mean,
            "ratio_median": ratio_median,
            "ratio_p95": ratio_p95,
            "ratio_p99": ratio_p99,
            "first_max": float(first_map.max()),
            "second_max": float(second_map.max()),
        })
        print(f"  {attr:12s}  ratio mean={ratio_mean:7.3f}  median={ratio_median:7.3f}  "
              f"p95={ratio_p95:7.3f}  p99={ratio_p99:7.3f}")

        # render row: first | second | ratio
        first_n = first_map / max(first_map.max(), 1e-8)
        second_n = second_map / max(second_map.max(), 1e-8)
        ratio_n = ratio / max(ratio.max(), 1e-8)

        def shrink(arr: np.ndarray) -> np.ndarray:
            return np.asarray(Image.fromarray(arr).resize(
                (args.display_size, args.display_size), Image.BILINEAR))

        row = np.concatenate([
            shrink(colorize(first_n.astype(np.float32))),
            shrink(colorize(second_n.astype(np.float32))),
            shrink(colorize(ratio_n.astype(np.float32), cmap="viridis")),
        ], axis=1)
        labels = ["|∂I/∂α|", "|∂²I/∂α²|", "ratio (viridis)"]
        label_strip = np.concatenate(
            [_label(l, args.display_size) for l in labels], axis=1
        )
        eyebrow = _label(
            f"━━ {attr.upper()}  ·  ratio mean={ratio_mean:.2f}  median={ratio_median:.2f}  p95={ratio_p95:.2f} ━━",
            row.shape[1], h=28, fs=16,
        )
        rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

    if rows:
        final = np.concatenate(rows, axis=0)
        out_path = out / "ffhq_higher_order_grid.png"
        Image.fromarray(final).save(out_path)
        print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
