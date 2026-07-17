"""InterFaceGAN supervised-baseline rerun on FFHQ.

Adds the 5 InterFaceGAN-FFHQ boundaries (age/eyeglasses/gender/pose/smile)
to the unsupervised sweep result for direct head-to-head against
GANSpace/SeFa/Random. InterFaceGAN is supervised (uses attribute labels),
so this is an upper-bound reference for coverage/diversity at K=5.
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

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402
from domains.ffhq.generator import FFHQGenerator                  # noqa: E402
from baselines.run_sweep_ffhq import (                            # noqa: E402
    render_and_label, coverage, diversity, CLIP_VOCAB,
)

# The 5 InterFaceGAN-FFHQ supervised boundaries
IFG_ATTRS = ["age", "eyeglasses", "gender", "pose", "smile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--num-bases", type=int, default=4)
    ap.add_argument("--apply-layers", nargs="+", type=int,
                    default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="experiments/out/ffhq_interfacegan_baseline")
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
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

    # Load the 5 InterFaceGAN boundaries (use the W-space versions)
    bdir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    dirs = []
    for a in IFG_ATTRS:
        # Prefer the *_w_boundary variant when it exists
        for cand in [bdir / f"stylegan_ffhq_{a}_w_boundary.npy",
                     bdir / f"stylegan_ffhq_{a}_boundary.npy"]:
            if cand.exists():
                b = np.load(cand).astype(np.float32)
                b = b.squeeze()
                b = b / (np.linalg.norm(b) + 1e-8)
                dirs.append(b)
                print(f"  loaded {cand.name}  shape={b.shape}")
                break
        else:
            print(f"  !! missing boundary for {a}")
    dirs = np.stack(dirs)
    assert dirs.shape == (5, G.w_dim), f"unexpected shape {dirs.shape}"

    t0 = time.time()
    ifg_labels = render_and_label(dirs, args.apply_layers, G, bases_wp,
                                   args.delta, model, preprocess, text_features)
    t_ifg = time.time() - t0
    print(f"\n  InterFaceGAN wall time: {t_ifg:.2f}s")

    K = 5
    cov = coverage(ifg_labels, K)
    div = diversity(ifg_labels, K)

    print(f"\n  K={K}  coverage={len(cov)}/5 ({sorted(cov)})  diversity={div}")

    result = {
        "method": "interfacegan_supervised",
        "K": K,
        "coverage": sorted(cov),
        "n_coverage": len(cov),
        "diversity": div,
        "wall_time_s": t_ifg,
        "boundaries_used": IFG_ATTRS,
        "_meta": run_metadata(seed=args.seed),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\n  saved {out}/metrics.json")


if __name__ == "__main__":
    main()
