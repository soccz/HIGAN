"""Label taxonomy clusters with CLIP zero-shot, using *rendered perturbation
effects* (not just saliency overlays) so the scores actually discriminate.

For each cluster:
  1. Pick the modal layer in that cluster.
  2. Sample 8 fresh random unit directions on that layer.
  3. Render each direction's +δ effect on the same base bedroom.
  4. Average the perturbed images → "what this cluster makes happen".
  5. CLIP-score the average perturbed image, the base image, and the diff
     against a vocabulary of bedroom-relevant concepts. Use the diff between
     perturbed and base sim scores so we measure *what changed*.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont
import open_clip

from higan_dev.config import Config, resolve
from higan_dev.utils import label_bar as _label
from higan_dev.generator import HiGANGenerator
from higan_dev.cam.diff_map import colorize_heat


VOCAB = [
    "a lamp", "a bed", "a window", "a door", "a pillow", "a blanket",
    "a curtain", "a frame", "a chair", "a table", "a mirror",
    "wood texture", "metal surface", "carpet", "fabric",
    "a wall", "a ceiling", "a floor",
    "bright lighting", "dim lighting", "warm light",
    "cluttered space", "clean room", "scary atmosphere",
    "a view through a window", "outdoor view",
    "glossy reflective surface", "dirty surface",
    "soft texture", "rough texture",
]



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--taxonomy-npz", default="out/taxonomy/raw.npz")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--num-bases", type=int, default=4,
                    help="number of base bedrooms to average over")
    ap.add_argument("--num-dirs-per-cluster", type=int, default=8)
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="out/cluster_labels")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.taxonomy_npz, allow_pickle=True)
    centroids = raw["centroids"]
    labels = raw["labels"]
    layers = raw["layers"]

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    L, D = G.num_layers, G.w_dim

    # bases for averaging
    rng_t = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_bases, generator=rng_t)
    with torch.no_grad():
        base_imgs = G.synthesize(base_wp).clamp(-1, 1)
    base_u8 = G.to_uint8(base_imgs)               # (B, H, W, 3)
    base_for_show = base_u8[0]

    print("loading CLIP ViT-B-32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(G.device)
    text_tokens = tokenizer(VOCAB).to(G.device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def clip_score(image_np: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image_np)
        x = preprocess(pil).unsqueeze(0).to(G.device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            sim = (f @ text_features.T).squeeze(0).cpu().numpy()
        return sim

    # average base similarity (contrastive baseline)
    base_sims = np.mean([clip_score(b) for b in base_u8], axis=0)

    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1)
    rows = []
    cluster_label_dict: dict[int, list[tuple[str, float]]] = {}

    for c in range(centroids.shape[0]):
        idx = np.where(labels == c)[0]
        n = idx.size
        if n == 0:
            continue
        layer_set = sorted(set(int(layers[i]) for i in idx))
        layer_str = ",".join(str(l) for l in layer_set[:6])
        # pick modal layer (most frequent)
        layer_counts = np.bincount([int(layers[i]) for i in idx], minlength=L)
        modal_layer = int(np.argmax(layer_counts))

        # sample fresh directions on that layer; average their +δ rendered effect
        accum = np.zeros_like(base_imgs[0:1].cpu().numpy(), dtype=np.float32)
        accum_imgs = []
        with torch.no_grad():
            for _ in range(args.num_dirs_per_cluster):
                v = torch.randn(D, generator=rng_dir, device=G.device)
                v = v / v.norm().clamp_min(1e-8)
                b_layered = torch.zeros(L, D, device=G.device)
                b_layered[modal_layer] = v
                wp_pos = base_wp + args.delta * b_layered.unsqueeze(0)
                imgs = G.synthesize(wp_pos)
                accum_imgs.append(imgs)
            avg_imgs = torch.stack(accum_imgs).mean(0)
            avg_u8 = G.to_uint8(avg_imgs)            # (B, H, W, 3)

        avg_sims = np.mean([clip_score(a) for a in avg_u8], axis=0)
        contrastive = avg_sims - base_sims
        order = np.argsort(-contrastive)
        topk = [(VOCAB[i], float(contrastive[i])) for i in order[:args.top_k]]
        cluster_label_dict[c] = topk
        print(f"cluster {c} (n={n:3d}, modal layer={modal_layer}):  "
              + "  ".join(f"{w} ({s:+.3f})" for w, s in topk))

        # build visualisation row: heat | base | perturbed avg | diff | label
        sal = centroids[c]
        m = sal.max()
        sal_n = (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)
        heat_rgb = colorize_heat(sal_n)
        diff_img = (avg_u8[0].astype(np.int16) - base_for_show.astype(np.int16))
        diff_view = (diff_img - diff_img.min())
        if diff_view.max() > 0:
            diff_view = (diff_view / diff_view.max() * 255).astype(np.uint8)
        else:
            diff_view = np.zeros_like(base_for_show)

        cap = Image.new("RGB", (base_for_show.shape[1], base_for_show.shape[0]),
                        (250, 250, 249))
        draw = ImageDraw.Draw(cap)
        try:
            f_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            f_mid = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
            f_sm = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except OSError:
            f_big = f_mid = f_sm = ImageFont.load_default()
        draw.text((10, 8), f"cluster {c}", fill=(20, 20, 20), font=f_big)
        draw.text((10, 32),
                  f"n={n}  layers={layer_str}  modal={modal_layer}",
                  fill=(80, 80, 80), font=f_sm)
        for j, (w_, s) in enumerate(topk):
            draw.text((10, 60 + j * 26), f"{w_}", fill=(40, 40, 40), font=f_mid)
            bar_w = max(2, int(150 * (s + 0.02) * 50))
            color = (180, 90, 60) if s > 0 else (130, 130, 130)
            draw.rectangle([10, 78 + j * 26, 10 + bar_w, 84 + j * 26], fill=color)
        cap_arr = np.asarray(cap)

        labels_strip = np.concatenate([
            _label("centroid sal.", base_for_show.shape[1]),
            _label("base bedroom", base_for_show.shape[1]),
            _label(f"avg perturbed (+{args.delta}, layer {modal_layer})",
                   base_for_show.shape[1]),
            _label("normalised diff", base_for_show.shape[1]),
            _label("CLIP top-k (Δ)", base_for_show.shape[1]),
        ], axis=1)
        row_img = np.concatenate(
            [heat_rgb, base_for_show, avg_u8[0], diff_view, cap_arr], axis=1)
        rows.append(np.concatenate([labels_strip, row_img], axis=0))

    final = np.concatenate(rows, axis=0)
    out_path = out / "cluster_labels.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    with open(out / "labels.txt", "w") as f:
        for c, topk in cluster_label_dict.items():
            f.write(f"cluster {c}: " + "  ".join(f"{w} ({s:+.3f})" for w, s in topk) + "\n")


if __name__ == "__main__":
    main()
