"""FFHQ 5×5 disentanglement matrix — pixel-wise correlation of saliency.

Same procedure as bedroom §09 / claim C4 prep: for every pair of
attributes, compute pixel-wise Pearson correlation between their
saliency maps. High correlation = they edit the same image regions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.func import jvp
from PIL import Image

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


def render_matrix(matrix: np.ndarray, names: list[str], title: str = "",
                  vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            v = matrix[i, j]
            color = "white" if abs(v) > 0.55 else "#1c1917"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    if title:
        ax.set_title(title, fontsize=11, pad=10, weight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out", default="out/ffhq_disentangle")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution
    boundaries_dir = EXPERIMENTS_DIR / "data" / "interfacegan" / "boundaries"

    rng = torch.Generator(device=G.device).manual_seed(2027)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    abs_maps: dict[str, np.ndarray] = {}
    for attr in args.attrs:
        bpath = boundaries_dir / f"stylegan_ffhq_{attr}_w_boundary.npy"
        b_vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
        b_dir = torch.from_numpy(b_vec).to(G.device)
        b_dir = b_dir / b_dir.norm().clamp_min(1e-8)
        canonical = LAYERS_FOR.get(attr, list(range(L)))
        b_layered = torch.zeros(L, D, device=G.device)
        for li in canonical:
            b_layered[li] = b_dir

        acc = torch.zeros(H, W, device=G.device)
        for s in range(args.num_samples):
            wp = base_wp[s:s + 1].detach()
            def f(alpha: torch.Tensor) -> torch.Tensor:
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(
                f, (torch.zeros(1, device=G.device),),
                (torch.ones(1, device=G.device),),
            )
            acc += dimg.abs().mean(dim=1).squeeze(0)
            torch.cuda.empty_cache()
        sal = (acc / args.num_samples).cpu().numpy()
        m = sal.max()
        sal = (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)
        abs_maps[attr] = sal.flatten()
        print(f"  {attr}: |sal|.max={m:.3f}")

    n = len(args.attrs)
    abs_corr = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(args.attrs):
        for j, b in enumerate(args.attrs):
            abs_corr[i, j] = float(np.corrcoef(abs_maps[a], abs_maps[b])[0, 1])

    img = render_matrix(abs_corr, args.attrs, title="FFHQ |sal| pixel-wise correlation")
    Image.fromarray(img).save(out / "ffhq_disentangle.png")
    np.savez(out / "raw.npz", abs_corr=abs_corr, names=np.asarray(args.attrs))

    # report top entangled / most disentangled pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs_corr[i, j], args.attrs[i], args.attrs[j]))
    pairs.sort(key=lambda x: -x[0])
    print("\nMost entangled pairs:")
    for c, a, b in pairs[:5]:
        print(f"  {a:12s} ↔ {b:12s}  corr={c:+.3f}")
    print("\nMost disentangled pairs:")
    for c, a, b in pairs[-3:]:
        print(f"  {a:12s} ↔ {b:12s}  corr={c:+.3f}")


if __name__ == "__main__":
    main()
