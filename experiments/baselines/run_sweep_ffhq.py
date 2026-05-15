"""Same K-sweep comparison as bedroom, but on FFHQ.

Targets to rediscover: smile, age, pose, gender, eyeglasses
(InterFaceGAN's 5 boundaries on StyleGAN1 FFHQ).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from domains.ffhq.generator import FFHQGenerator               # noqa: E402
from baselines.ganspace import ganspace_directions             # noqa: E402
from baselines.sefa import sefa_directions                     # noqa: E402


CLIP_VOCAB = [
    "a face",
    "a smiling face", "a serious face", "open mouth", "closed mouth",
    "a young face", "an old face", "a wrinkled face",
    "a frontal face", "a tilted face", "a side profile",
    "a face with glasses", "a face without glasses", "eyeglasses",
    "a male face", "a female face", "a beard",
    "blond hair", "dark hair", "a forehead",
    "raised eyebrows", "narrow eyes", "wide eyes", "a teeth", "a chin",
]
ATTR_ALIASES = {
    "smile":      ["a smiling face", "open mouth"],
    "age":        ["an old face", "a wrinkled face"],
    "pose":       ["a tilted face", "a side profile"],
    "gender":     ["a male face", "a female face", "a beard"],
    "eyeglasses": ["a face with glasses", "eyeglasses"],
}


def caption_clip(image_np, model, preprocess, text_features, device):
    pil = Image.fromarray(image_np)
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        return (f @ text_features.T).squeeze(0).cpu().numpy()


def render_and_label(directions, apply_layers, G, bases_wp, delta,
                     model, preprocess, text_features):
    device = G.device
    L, D = G.num_layers, G.w_dim
    bases_u8 = G.to_uint8(G.synthesize(bases_wp))
    base_sims = np.mean([
        caption_clip(b, model, preprocess, text_features, device) for b in bases_u8
    ], axis=0)
    per_dir = []
    for k in range(len(directions)):
        v = torch.from_numpy(directions[k]).to(device).float()
        v = v / v.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=device)
        for li in apply_layers:
            bl[li] = v
        with torch.no_grad():
            imgs = G.synthesize(bases_wp + delta * bl.unsqueeze(0))
            u8 = G.to_uint8(imgs)
        avg = np.mean([
            caption_clip(im, model, preprocess, text_features, device) for im in u8
        ], axis=0)
        contrastive = avg - base_sims
        order = np.argsort(-contrastive)
        per_dir.append([(CLIP_VOCAB[i], float(contrastive[i])) for i in order[:3]])
        torch.cuda.empty_cache()
    return per_dir


def coverage(per_dir, K):
    found = set()
    for k in range(min(K, len(per_dir))):
        labels = [w for w, s in per_dir[k] if s > 0]
        for attr, aliases in ATTR_ALIASES.items():
            if any(a in labels for a in aliases):
                found.add(attr)
    return found


def diversity(per_dir, K):
    top1 = set()
    for k in range(min(K, len(per_dir))):
        if per_dir[k]:
            top1.add(per_dir[k][0][0])
    return len(top1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-k", type=int, default=16)
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--num-bases", type=int, default=4)
    ap.add_argument("--apply-layers", nargs="+", type=int,
                    default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="out/ffhq_baselines_sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    gen = torch.Generator(device=G.device).manual_seed(args.seed)
    bases_wp = G.sample_wp(args.num_bases, generator=gen)

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

    print("=== GANSpace-W ===")
    t0 = time.time()
    gs = ganspace_directions(G, n_samples=3000, n_components=args.max_k, seed=args.seed)
    t_gs = time.time() - t0
    print(f"  wall time: {t_gs:.2f}s")
    gs_labels = render_and_label(gs.components, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)

    print("=== SeFa ===")
    t0 = time.time()
    se = sefa_directions(G, n_components=args.max_k)
    t_se = time.time() - t0
    print(f"  wall time: {t_se:.2f}s")
    se_labels = render_and_label(se.components, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)

    print("=== Random ===")
    rng = torch.Generator(device=G.device).manual_seed(args.seed + 1)
    rd_dirs = []
    for _ in range(args.max_k):
        v = torch.randn(G.w_dim, generator=rng, device=G.device)
        v = (v / v.norm().clamp_min(1e-8)).cpu().numpy().astype(np.float32)
        rd_dirs.append(v)
    rd_dirs = np.stack(rd_dirs)
    rd_labels = render_and_label(rd_dirs, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)

    print(f"\n  {'K':>3} {'method':>10}   {'coverage/5':>10}  {'diversity':>10}")
    sweeps = {}
    for K in [2, 4, 6, 8, 10, 12, 16]:
        for name, labels in [("ganspace", gs_labels),
                             ("sefa", se_labels),
                             ("random", rd_labels)]:
            cov = coverage(labels, K)
            div = diversity(labels, K)
            print(f"  {K:>3} {name:>10}   {len(cov)}/5 {str(sorted(cov))[:50]:>0}  {div:>4}")
            sweeps.setdefault(name, []).append({"K": K, "coverage": sorted(cov),
                                                 "n_coverage": len(cov),
                                                 "diversity": div})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    Ks = [2, 4, 6, 8, 10, 12, 16]
    colors = {"ganspace": "#0e7490", "sefa": "#9333ea", "random": "#a16207"}
    for ax, ylabel, key in [
        (axes[0], "Coverage / 5 attributes", "n_coverage"),
        (axes[1], "Diversity (distinct top-1 labels)", "diversity"),
    ]:
        for name in ["ganspace", "sefa", "random"]:
            ys = [d[key] for d in sweeps[name]]
            ax.plot(Ks, ys, "o-", color=colors[name], label=name, lw=2, ms=7)
        ax.set_xlabel("K (candidate directions)"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("FFHQ unsupervised discovery: K-sweep across methods",
                 fontsize=11, weight="bold", y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "ffhq_baselines_sweep.png")

    with open(out / "metrics.json", "w") as f:
        json.dump({"sweeps": sweeps,
                   "wall_time_s": {"ganspace_pca": t_gs, "sefa_svd": t_se}},
                  f, indent=2)


if __name__ == "__main__":
    main()
