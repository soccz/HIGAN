"""Unsupervised attribute taxonomy via random-direction clustering.

Sample N random directions in W+ (single-layer perturbations), compute the
saliency map for each, then cluster the saliency maps with K-means. Each
cluster represents a "family of edits that affect similar pixel regions" —
an automatic, label-free attribute taxonomy.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.cam.diff_map import colorize_heat


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-directions", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=8,
                    help="latents averaged per direction")
    ap.add_argument("--num-clusters", type=int, default=8)
    ap.add_argument("--examples-per-cluster", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="out/taxonomy")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    # shared base latents
    rng_t = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng_t)
    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1)

    saliencies: list[np.ndarray] = []
    layer_choice: list[int] = []
    strengths: list[float] = []

    print(f"computing saliency for {args.num_directions} random directions...")
    for d in range(args.num_directions):
        # exclude layer 0 (constant input) and the noisy structure-only layers
        li = int(torch.randint(1, L, (1,), generator=rng_dir, device=G.device).item())
        v = torch.randn(D, generator=rng_dir, device=G.device)
        v = v / v.norm().clamp_min(1e-8)
        b_layered = torch.zeros(L, D, device=G.device)
        b_layered[li] = v

        acc = torch.zeros(H, W, device=G.device)
        chunks = 4
        for s in range(0, args.num_samples, chunks):
            wp = base_wp[s:s + chunks].detach()
            B = wp.shape[0]
            def f(alpha):
                return G.synthesize(wp + alpha.view(B, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(
                f,
                (torch.zeros(B, device=G.device),),
                (torch.ones(B, device=G.device),),
            )
            acc += dimg.abs().mean(dim=1).sum(dim=0)
        sal = (acc / args.num_samples).cpu().numpy()
        strengths.append(float(sal.mean()))
        m = sal.max()
        sal = (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)
        saliencies.append(sal)
        layer_choice.append(li)
        if (d + 1) % 32 == 0:
            print(f"  {d + 1}/{args.num_directions}")

    sal_arr = np.stack(saliencies)                                # (N, H, W)
    print(f"saliencies shape: {sal_arr.shape}")

    # filter weak directions (below median strength)
    str_arr = np.asarray(strengths)
    median_strength = float(np.median(str_arr))
    keep_mask = str_arr >= median_strength
    keep_idx = np.where(keep_mask)[0]
    print(f"kept {keep_idx.size} / {len(strengths)} directions above median strength")

    sal_kept = sal_arr[keep_idx]
    layer_kept = np.asarray(layer_choice)[keep_idx]

    # downsample to 64x64 to reduce clustering cost
    sal_small = np.stack([
        np.asarray(Image.fromarray((s * 255).astype(np.uint8)).resize(
            (64, 64), Image.BILINEAR)) / 255.0
        for s in sal_kept
    ]).astype(np.float32)
    sal_flat = sal_small.reshape(sal_small.shape[0], -1)

    # PCA to 32 dims for cleaner clustering
    pca = PCA(n_components=min(32, sal_flat.shape[0] - 1))
    sal_pca = pca.fit_transform(sal_flat)

    km = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(sal_pca)

    # for each cluster, gather examples and compute centroid (in original space)
    cluster_examples: dict[int, list[int]] = {c: [] for c in range(args.num_clusters)}
    cluster_centroid: dict[int, np.ndarray] = {}
    for c in range(args.num_clusters):
        idx = np.where(labels == c)[0]
        cluster_examples[c] = idx.tolist()
        cluster_centroid[c] = sal_kept[idx].mean(0) if idx.size else np.zeros((H, W))

    # rank clusters by size (largest first)
    sizes = [(c, len(cluster_examples[c])) for c in range(args.num_clusters)]
    sizes.sort(key=lambda x: -x[1])
    print("\ncluster sizes:")
    for c, n in sizes:
        layers_in = [int(layer_kept[i]) for i in cluster_examples[c]]
        layer_str = ",".join(str(l) for l in sorted(set(layers_in)))
        print(f"  cluster {c}: n={n:3d}  layers={layer_str}")

    # build a montage: rows = clusters, cols = [centroid | examples]
    cell_w = H
    rows = []
    for c, n in sizes:
        c_norm = cluster_centroid[c]
        m = c_norm.max()
        if m > 1e-8:
            c_norm = c_norm / m
        cells = [colorize_heat(c_norm.astype(np.float32))]
        # pick top-K examples closest to centroid
        idx = cluster_examples[c]
        if idx:
            ex_arr = sal_kept[idx]
            ex_flat = ex_arr.reshape(ex_arr.shape[0], -1)
            cent_flat = cluster_centroid[c].flatten()
            dists = np.linalg.norm(ex_flat - cent_flat, axis=1)
            order = np.argsort(dists)[: args.examples_per_cluster]
            for o in order:
                cells.append(colorize_heat(ex_arr[o].astype(np.float32)))
        # pad if fewer examples than expected
        while len(cells) < args.examples_per_cluster + 1:
            cells.append(np.zeros((H, cell_w, 3), dtype=np.uint8))
        labels_strip = np.concatenate([
            _label("centroid", cell_w),
            *[_label(f"ex {i + 1}", cell_w) for i in range(args.examples_per_cluster)],
        ], axis=1)
        row_img = np.concatenate(cells, axis=1)
        layers_in = [int(layer_kept[i]) for i in cluster_examples[c]]
        layer_str = ",".join(str(l) for l in sorted(set(layers_in))[:6])
        eyebrow = _label(f"━━ CLUSTER {c}  ·  n={n}  ·  layers={layer_str} ━━",
                         row_img.shape[1], h=28, fs=15)
        rows.append(np.concatenate([eyebrow, labels_strip, row_img], axis=0))

    final = np.concatenate(rows, axis=0)
    out_path = out / f"taxonomy_k{args.num_clusters}.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    np.savez(out / "raw.npz",
             saliencies=sal_kept, layers=layer_kept, labels=labels,
             centroids=np.stack([cluster_centroid[c] for c in range(args.num_clusters)]))


if __name__ == "__main__":
    main()
