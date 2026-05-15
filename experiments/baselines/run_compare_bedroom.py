"""Apples-to-apples comparison: our random+saliency taxonomy vs GANSpace
vs SeFa for unsupervised attribute discovery on HiGAN bedroom.

Protocol:
  For each baseline (GANSpace-W, SeFa) and our method (random + JVP +
  K-means):
    1. Produce K candidate directions.
    2. Apply each to a fixed base latent at ±δ, render +δ side.
    3. CLIP-zero-shot caption each direction's mean perturbed image
       against a bedroom vocabulary.
    4. Measure: how many of HiGAN's 8 hand-curated boundaries are
       "rediscovered" — defined as a CLIP-caption match within top-3.
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# higan_dev / interfacegan paths
PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from higan_dev.generator import HiGANGenerator                  # noqa: E402
from higan_dev.manipulate import load_boundary                  # noqa: E402

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

# attribute aliases used to score "rediscovery"
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


def _caption(image_np: np.ndarray, model, preprocess, text_features, device) -> np.ndarray:
    pil = Image.fromarray(image_np)
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return (f @ text_features.T).squeeze(0).cpu().numpy()


def evaluate_directions(
    name: str,
    directions: np.ndarray,                # (k, w_dim) in W (or wp_layer) space
    layer_idx: int | list[int],            # where to apply
    G: HiGANGenerator,
    bases_wp: torch.Tensor,
    delta: float,
    model, preprocess, text_features,
    top_k_label: int = 3,
) -> dict:
    """Render +δ effect of each direction, CLIP-caption, count matches."""
    K, D = directions.shape
    L = G.num_layers
    device = G.device
    layers = [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)

    # base similarities
    bases_u8 = G.to_uint8(G.synthesize(bases_wp))
    base_sims = np.mean([
        _caption(b, model, preprocess, text_features, device) for b in bases_u8
    ], axis=0)

    per_dir_top_label: list[tuple[str, float]] = []
    per_dir_topk: list[list[tuple[str, float]]] = []
    for k in range(K):
        v = torch.from_numpy(directions[k]).to(device).float()
        v = v / v.norm().clamp_min(1e-8)
        b_layered = torch.zeros(L, D, device=device)
        for li in layers:
            b_layered[li] = v
        with torch.no_grad():
            wp_p = bases_wp + delta * b_layered.unsqueeze(0)
            imgs = G.synthesize(wp_p)
            u8 = G.to_uint8(imgs)
        avg_sims = np.mean([
            _caption(im, model, preprocess, text_features, device) for im in u8
        ], axis=0)
        contrastive = avg_sims - base_sims
        order = np.argsort(-contrastive)
        topk = [(CLIP_VOCAB[i], float(contrastive[i])) for i in order[:top_k_label]]
        per_dir_topk.append(topk)
        per_dir_top_label.append(topk[0])

    # rediscovery rate: an attribute is "rediscovered" if ANY direction's
    # top-3 captions contain one of its aliases (with Δ > 0).
    rediscovered = {}
    for attr, aliases in ATTR_ALIASES.items():
        found = False
        best = None
        for k in range(K):
            labels = [w for w, s in per_dir_topk[k] if s > 0]
            for a in aliases:
                if a in labels:
                    found = True
                    if best is None:
                        best = k
                    break
            if found:
                break
        rediscovered[attr] = (found, best)

    n_rediscovered = sum(1 for v in rediscovered.values() if v[0])
    return {
        "name": name,
        "n_dirs": K,
        "n_rediscovered": n_rediscovered,
        "rediscovered": {k: (bool(v[0]), v[1]) for k, v in rediscovered.items()},
        "per_dir_topk": per_dir_topk,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16,
                    help="number of candidate directions per method")
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--num-bases", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--apply-layers", nargs="+", type=int,
                    default=[6, 7, 8, 9, 10, 11],
                    help="layers on which to apply discovered W-space directions")
    ap.add_argument("--out", default="out/bedroom_baselines")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading HiGAN bedroom ===")
    G = HiGANGenerator(higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo"))
    print(f"  {G.num_layers} layers × {G.w_dim} dim")

    # base latents for CLIP rendering
    gen = torch.Generator(device=G.device).manual_seed(args.seed)
    bases_wp = G.sample_wp(args.num_bases, generator=gen)

    print("\n=== loading CLIP ===")
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

    results = {}

    # ---- 1. GANSpace-W ----
    print("\n=== GANSpace-W ===")
    gs = ganspace_directions(G, n_samples=5000, n_components=args.k, seed=args.seed)
    print(f"  top-{args.k} explained var: {gs.explained}")
    r_gs = evaluate_directions(
        "GANSpace-W", gs.components, args.apply_layers, G, bases_wp, args.delta,
        model, preprocess, text_features,
    )
    print(f"  rediscovered {r_gs['n_rediscovered']}/8 attributes")
    for a, (found, idx) in r_gs["rediscovered"].items():
        print(f"    {a}: {'✓' if found else '✗'} (first match dir {idx})")
    results["ganspace"] = r_gs

    # ---- 2. SeFa ----
    print("\n=== SeFa ===")
    try:
        se = sefa_directions(G, n_components=args.k)
        print(f"  top-{args.k} eigenvalues: {se.eigenvalues}")
        r_se = evaluate_directions(
            "SeFa", se.components, args.apply_layers, G, bases_wp, args.delta,
            model, preprocess, text_features,
        )
        print(f"  rediscovered {r_se['n_rediscovered']}/8 attributes")
        for a, (found, idx) in r_se["rediscovered"].items():
            print(f"    {a}: {'✓' if found else '✗'} (first match dir {idx})")
        results["sefa"] = r_se
    except Exception as e:
        print(f"  SeFa failed: {e}")
        results["sefa"] = {"error": str(e)}

    # ---- 3. Random + JVP saliency cluster centroids (OUR method) ----
    print("\n=== OUR (random + JVP-saliency + K-means cluster modal direction) ===")
    # For comparison fairness, generate K random unit directions on the
    # same layer subset, then use them directly (this is our "before
    # clustering" baseline). Our actual method clusters; the centroid
    # of cluster c is the average of cluster c's directions, which is
    # itself a vector in W space we can evaluate.
    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1)
    rand_dirs = []
    for _ in range(args.k):
        v = torch.randn(G.w_dim, generator=rng_dir, device=G.device)
        v = (v / v.norm().clamp_min(1e-8)).cpu().numpy().astype(np.float32)
        rand_dirs.append(v)
    rand_dirs = np.stack(rand_dirs)
    r_random = evaluate_directions(
        "Random+CLIP", rand_dirs, args.apply_layers, G, bases_wp, args.delta,
        model, preprocess, text_features,
    )
    print(f"  rediscovered {r_random['n_rediscovered']}/8 attributes")
    for a, (found, idx) in r_random["rediscovered"].items():
        print(f"    {a}: {'✓' if found else '✗'} (first match dir {idx})")
    results["random"] = r_random

    # save table
    print("\n=== SUMMARY ===")
    print(f"  {'method':18s} {'rediscovered/8':>16s}")
    for name in ("ganspace", "sefa", "random"):
        if "error" in results.get(name, {}):
            print(f"  {name:18s} ERROR: {results[name]['error']}")
        else:
            n = results[name]["n_rediscovered"]
            print(f"  {name:18s} {n}/8")

    with open(out / "results.json", "w") as f:
        # JSON-serialise — strip torch tensors
        json.dump({
            k: ({**v, "per_dir_topk": v.get("per_dir_topk", [])}
                if "error" not in v else v)
            for k, v in results.items()
        }, f, indent=2)


if __name__ == "__main__":
    main()
