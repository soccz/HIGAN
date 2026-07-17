"""C6 cross-domain: unsupervised attribute discovery on FFHQ.

Sample N random unit directions on per-layer spheres in W+, compute
each direction's saliency, K-means cluster the saliency maps, then
CLIP zero-shot label each cluster centroid. Goal: rediscover
human-curated InterFaceGAN boundaries (smile/age/pose/gender/glasses)
from random directions alone.

Reuses the pipeline structure of bedroom §22+§24.
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

CLIP_VOCAB = [
    "a smiling face", "a serious face", "a young face", "an old face",
    "a frontal face", "a tilted face", "a side profile",
    "a face with glasses", "a face without glasses",
    "a male face", "a female face",
    "a beard", "a wrinkled face", "a smooth face",
    "blond hair", "dark hair", "a forehead",
    "open mouth", "closed mouth", "raised eyebrows",
    "narrow eyes", "wide eyes",
    "a teeth", "a chin",
]


def colorize(heat: np.ndarray, cmap: str = "magma") -> np.ndarray:
    import matplotlib.cm as cm
    rgba = cm.get_cmap(cmap)(np.clip(heat, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


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


def _spatial_concentration(p: np.ndarray, frac: float = 0.05) -> float:
    flat = p.flatten()
    s = flat.sum()
    if s < 1e-8:
        return 0.0
    k = max(1, int(flat.size * frac))
    top = np.partition(flat, -k)[-k:]
    return float(top.sum() / s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-directions", type=int, default=192)
    ap.add_argument("--num-samples-per-dir", type=int, default=4)
    ap.add_argument("--num-clusters", type=int, default=8)
    ap.add_argument("--clip-num-dirs-per-cluster", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--display-size", type=int, default=192)
    ap.add_argument("--out", default="out/ffhq_c6")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    # shared base latents
    rng_t = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples_per_dir, generator=rng_t)
    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1)

    saliencies: list[np.ndarray] = []
    layer_idx: list[int] = []
    strengths: list[float] = []

    print(f"computing saliency for {args.num_directions} random directions...")
    for d in range(args.num_directions):
        # skip layer 0 (constant input — noisy); sample 1..L-1
        li = int(torch.randint(1, L, (1,), generator=rng_dir, device=G.device).item())
        v = torch.randn(D, generator=rng_dir, device=G.device)
        v = v / v.norm().clamp_min(1e-8)
        b_layered = torch.zeros(L, D, device=G.device)
        b_layered[li] = v

        acc = torch.zeros(H, W, device=G.device)
        for s in range(args.num_samples_per_dir):
            wp = base_wp[s:s + 1].detach()
            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(f, (torch.zeros(1, device=G.device),),
                          (torch.ones(1, device=G.device),))
            acc += dimg.abs().mean(dim=1).squeeze(0)
            torch.cuda.empty_cache()
        sal = (acc / args.num_samples_per_dir).cpu().numpy()
        strengths.append(float(sal.mean()))
        m = sal.max()
        sal = (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)
        saliencies.append(sal)
        layer_idx.append(li)
        if (d + 1) % 32 == 0:
            print(f"  {d + 1}/{args.num_directions}")

    sal_arr = np.stack(saliencies)
    str_arr = np.asarray(strengths)
    print(f"saliencies shape: {sal_arr.shape}")

    # filter weak directions (below-median strength)
    median = float(np.median(str_arr))
    keep_idx = np.where(str_arr >= median)[0]
    sal_kept = sal_arr[keep_idx]
    layer_kept = np.asarray(layer_idx)[keep_idx]
    print(f"kept {keep_idx.size} above-median directions")

    # downsample for clustering speed
    sal_small = np.stack([
        np.asarray(Image.fromarray((s * 255).astype(np.uint8)).resize(
            (64, 64), Image.BILINEAR)) / 255.0
        for s in sal_kept
    ]).astype(np.float32)
    sal_flat = sal_small.reshape(sal_small.shape[0], -1)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(32, sal_flat.shape[0] - 1))
    sal_pca = pca.fit_transform(sal_flat)
    km = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(sal_pca)

    # CLIP labelling using rendered effect on a fixed base
    print("\nrunning CLIP zero-shot labelling on cluster modal-layer effects...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(G.device)
    text_tokens = tokenizer(CLIP_VOCAB).to(G.device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # fixed base for CLIP visualisation
    rng_t2 = torch.Generator(device=G.device).manual_seed(args.seed + 7)
    bases_wp = G.sample_wp(4, generator=rng_t2)
    with torch.no_grad():
        bases_img = G.synthesize(bases_wp).clamp(-1, 1)
    bases_u8 = ((bases_img + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    bases_u8 = (bases_u8 * 255).astype(np.uint8)
    base_for_show = bases_u8[0]

    def clip_score(image_np: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image_np)
        x = preprocess(pil).unsqueeze(0).to(G.device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            return (f @ text_features.T).squeeze(0).cpu().numpy()

    base_sims = np.mean([clip_score(b) for b in bases_u8], axis=0)
    rng_dir2 = torch.Generator(device=G.device).manual_seed(args.seed + 11)
    rows = []
    cluster_labels: dict[int, list[tuple[str, float]]] = {}
    sizes_sorted = sorted(range(args.num_clusters),
                          key=lambda c: -int((labels == c).sum()))

    for c in sizes_sorted:
        idx = np.where(labels == c)[0]
        n = idx.size
        if n == 0:
            continue
        # modal layer of this cluster
        layers_in_cluster = layer_kept[idx]
        modal_layer = int(np.bincount(layers_in_cluster).argmax())

        # sample fresh random directions on that layer, render perturbations
        accum_imgs = []
        delta = 4.0
        with torch.no_grad():
            for _ in range(args.clip_num_dirs_per_cluster):
                v2 = torch.randn(D, generator=rng_dir2, device=G.device)
                v2 = v2 / v2.norm().clamp_min(1e-8)
                bl = torch.zeros(L, D, device=G.device)
                bl[modal_layer] = v2
                wp_p = bases_wp + delta * bl.unsqueeze(0)
                imgs = G.synthesize(wp_p)
                accum_imgs.append(imgs)
            avg_imgs = torch.stack(accum_imgs).mean(0)
            avg_u8 = ((avg_imgs.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            avg_u8 = (avg_u8 * 255).astype(np.uint8)

        avg_sims = np.mean([clip_score(a) for a in avg_u8], axis=0)
        contrastive = avg_sims - base_sims
        order = np.argsort(-contrastive)
        topk = [(CLIP_VOCAB[i], float(contrastive[i])) for i in order[:4]]
        cluster_labels[c] = topk

        layer_str = ",".join(map(str, sorted(set(int(x) for x in layers_in_cluster))[:6]))
        print(f"cluster {c} (n={n:3d}, modal layer={modal_layer}, "
              f"all layers={layer_str}):")
        for w_, s in topk:
            print(f"    {w_:30s}  Δ={s:+.4f}")

        # cluster row visual
        ts = args.display_size
        def shrink(arr: np.ndarray) -> np.ndarray:
            return np.asarray(Image.fromarray(arr).resize((ts, ts), Image.BILINEAR))

        # centroid (mean saliency)
        centroid = sal_kept[idx].mean(0)
        m = centroid.max()
        cent_n = (centroid / m).astype(np.float32) if m > 1e-8 else centroid
        cent_rgb = colorize(cent_n)
        # build row: centroid | base | avg perturbed | top-k caption
        cap = Image.new("RGB", (ts, ts), (250, 250, 249))
        draw = ImageDraw.Draw(cap)
        try:
            f_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            f_mid = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except OSError:
            f_big = f_mid = ImageFont.load_default()
        draw.text((8, 8), f"cluster {c}", fill=(20, 20, 20), font=f_big)
        draw.text((8, 28),
                  f"n={n} · modal layer {modal_layer}",
                  fill=(70, 70, 70), font=f_mid)
        for j, (w_, s) in enumerate(topk):
            color = (180, 90, 60) if s > 0 else (130, 130, 130)
            draw.text((8, 55 + j * 22), f"{w_}  ({s:+.3f})",
                      fill=color, font=f_mid)
        cap_arr = np.asarray(cap)

        row = np.concatenate([
            shrink(cent_rgb),
            shrink(base_for_show),
            shrink(avg_u8[0]),
            cap_arr,
        ], axis=1)
        labels_strip = np.concatenate([
            _label("centroid", ts),
            _label("base", ts),
            _label(f"+δ on layer {modal_layer}", ts),
            _label("CLIP top-4 (Δ)", ts),
        ], axis=1)
        rows.append(np.concatenate([labels_strip, row], axis=0))

    if rows:
        final = np.concatenate(rows, axis=0)
        Image.fromarray(final).save(out / "ffhq_clusters.png")
        print(f"\nsaved {out / 'ffhq_clusters.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"cluster_labels": {str(k): v for k, v in cluster_labels.items()},
                   "kmeans_inertia": float(km.inertia_),
                   "num_clusters": args.num_clusters,
                   "num_directions_kept": int(keep_idx.size)}, f, indent=2)


if __name__ == "__main__":
    main()
