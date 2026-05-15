"""More careful baselines comparison on bedroom — sweep K, measure coverage
and diversity at each.

Coverage(K)  = number of distinct HiGAN attributes rediscovered using the
               method's top-K directions
Diversity(K) = number of distinct top-1 CLIP labels across K directions
Cost         = wall-clock to produce K directions
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
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                  # noqa: E402
from baselines.ganspace import ganspace_directions             # noqa: E402
from baselines.sefa import sefa_directions                     # noqa: E402


CLIP_VOCAB = [
    "a bedroom",
    "a bright bedroom", "a dim bedroom", "warm indoor lighting",
    "a wooden bed frame", "wooden furniture",
    "a carpet", "carpeted floor",
    "a cluttered bedroom", "a messy room", "a tidy bedroom",
    "a glossy surface", "shiny furniture",
    "a dirty room", "a clean room",
    "a scary bedroom", "an empty bedroom",
    "a view through a window", "outdoor view",
    "a window", "a curtain", "a lamp", "a bed", "a pillow",
]
ATTR_ALIASES = {
    "indoor_lighting": ["a bright bedroom", "a dim bedroom", "warm indoor lighting", "a lamp"],
    "wood":            ["a wooden bed frame", "wooden furniture"],
    "carpet":          ["a carpet", "carpeted floor"],
    "cluttered_space": ["a cluttered bedroom", "a messy room"],
    "glossy":          ["a glossy surface", "shiny furniture"],
    "dirt":            ["a dirty room"],
    "scary":           ["a scary bedroom"],
    "view":            ["a view through a window", "outdoor view", "a window"],
}


def caption_clip(image_np, model, preprocess, text_features, device):
    pil = Image.fromarray(image_np)
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        return (f @ text_features.T).squeeze(0).cpu().numpy()


def render_and_label(directions, apply_layers, G, bases_wp, delta,
                     model, preprocess, text_features):
    """Returns per-direction top-3 CLIP labels (contrastive to base)."""
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
    return per_dir


def coverage(per_dir, K: int) -> set[str]:
    """Set of HiGAN attribute names rediscovered using the first K directions."""
    found = set()
    for k in range(min(K, len(per_dir))):
        labels = [w for w, s in per_dir[k] if s > 0]
        for attr, aliases in ATTR_ALIASES.items():
            if any(a in labels for a in aliases):
                found.add(attr)
    return found


def diversity(per_dir, K: int) -> int:
    """Number of distinct top-1 labels in first K directions."""
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
                    default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="out/bedroom_baselines_sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
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

    # === GANSpace ===
    print("=== GANSpace-W ===")
    t0 = time.time()
    gs = ganspace_directions(G, n_samples=5000, n_components=args.max_k, seed=args.seed)
    t_gs = time.time() - t0
    gs_labels = render_and_label(gs.components, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)
    print(f"  PCA wall time: {t_gs:.2f}s")

    # === SeFa ===
    print("=== SeFa ===")
    t0 = time.time()
    se = sefa_directions(G, n_components=args.max_k)
    t_se = time.time() - t0
    se_labels = render_and_label(se.components, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)
    print(f"  SVD wall time: {t_se:.2f}s")

    # === Random+CLIP (no clustering) ===
    print("=== Random (uniform unit dirs) ===")
    rng = torch.Generator(device=G.device).manual_seed(args.seed + 1)
    rd_dirs = []
    for _ in range(args.max_k):
        v = torch.randn(G.w_dim, generator=rng, device=G.device)
        v = (v / v.norm().clamp_min(1e-8)).cpu().numpy().astype(np.float32)
        rd_dirs.append(v)
    rd_dirs = np.stack(rd_dirs)
    rd_labels = render_and_label(rd_dirs, args.apply_layers, G, bases_wp,
                                  args.delta, model, preprocess, text_features)

    # === sweep ===
    print("\n=== Sweep ===")
    print(f"  {'K':>4} {'method':>14}  {'coverage/8':>12}  {'diversity':>10}")
    sweeps = {}
    for K in [2, 4, 6, 8, 10, 12, 16]:
        for name, labels in [("ganspace", gs_labels),
                             ("sefa", se_labels),
                             ("random", rd_labels)]:
            cov = coverage(labels, K)
            div = diversity(labels, K)
            print(f"  {K:>4} {name:>14}  {len(cov):>3}/8 {str(sorted(cov))[:60]:>0}  {div:>5}")
            sweeps.setdefault(name, []).append({"K": K, "coverage": sorted(cov),
                                                 "n_coverage": len(cov),
                                                 "diversity": div})

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    Ks = [2, 4, 6, 8, 10, 12, 16]
    colors = {"ganspace": "#0e7490", "sefa": "#9333ea", "random": "#a16207"}
    for ax, ylabel, key in [
        (axes[0], "Coverage / 8 attributes", "n_coverage"),
        (axes[1], "Diversity (distinct top-1 labels)", "diversity"),
    ]:
        for name in ["ganspace", "sefa", "random"]:
            ys = [d[key] for d in sweeps[name]]
            ax.plot(Ks, ys, "o-", color=colors[name], label=name, lw=2, ms=7)
        ax.set_xlabel("K (candidate directions)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Bedroom unsupervised discovery: K-sweep across methods",
                 fontsize=11, weight="bold", y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "bedroom_baselines_sweep.png")
    print(f"\nsaved {out / 'bedroom_baselines_sweep.png'}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"sweeps": sweeps,
                   "wall_time_s": {"ganspace_pca": t_gs, "sefa_svd": t_se}},
                  f, indent=2)


if __name__ == "__main__":
    main()
