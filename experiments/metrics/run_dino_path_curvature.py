"""Track 19 — DINOv2 path-curvature as C2 alternative feature space.

Same α-sweep + path/direct ratio as Track 1 CLIP, but using DINOv2
ViT-B/14 features instead of CLIP ViT-B/32. Tests whether C2 holds
in a self-supervised (non-language-conditioned) feature space.

See designs/19_dino_path_curvature.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

PIXEL_RATIOS_BEDROOM = {
    "indoor_lighting": 0.495, "wood": 0.624, "carpet": 0.95,
    "cluttered_space": 0.85, "glossy": 0.92, "dirt": 0.93,
    "scary": 1.10, "view": 23.22,
}
PIXEL_RATIOS_FFHQ = {
    "smile": 1.75, "age": 7.62, "gender": 8.71,
    "eyeglasses": 22.82, "pose": 49.87,
}


def load_dino(device):
    """Load DINOv2 ViT-B/14 from HuggingFace."""
    try:
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        return model, processor
    except Exception as e:
        print(f"failed to load DINOv2: {e}")
        return None, None


def dino_feat(model, processor, device, img_neg1_pos1):
    """img: (1, 3, H, W) in [-1, 1] → feature on unit sphere."""
    x = (img_neg1_pos1.clamp(-1, 1) + 1) / 2.0
    # processor expects PIL or numpy uint8 — use direct tensor path
    # via mean/std matching its config
    H = W = 224
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        out = model(pixel_values=x)
        # use the CLS token (pooler_output) or mean of patch tokens
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            f = out.pooler_output
        else:
            f = out.last_hidden_state.mean(dim=1)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return f


@torch.no_grad()
def sweep_attr_bedroom(G, bl_dict, attr, base_wp, alphas):
    bl = bl_dict[attr]
    N = base_wp.size(0)
    out = []
    for s in range(N):
        wp = base_wp[s:s + 1].detach()
        frames = []
        for a in alphas:
            img = G.synthesize(wp + float(a) * bl.unsqueeze(0))
            frames.append(img.cpu().squeeze(0))
        out.append(torch.stack(frames, dim=0))
        torch.cuda.empty_cache()
    return out


def path_ratio(frames_per_sample, model, processor, device):
    ratios = []
    for frames in frames_per_sample:
        feats = []
        for k in range(frames.shape[0]):
            img = frames[k:k + 1].to(device)
            f = dino_feat(model, processor, device, img)
            feats.append(f.squeeze(0))
        F_stack = torch.stack(feats)
        seg = torch.linalg.norm(F_stack[1:] - F_stack[:-1], dim=1)
        path_len = seg.sum().item()
        direct = torch.linalg.norm(F_stack[-1] - F_stack[0]).item()
        ratios.append(path_len / direct if direct > 1e-8 else 0.0)
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["bedroom", "ffhq"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--alpha-range", type=float, nargs=2,
                    default=[-3.0, 3.0])
    ap.add_argument("--alpha-steps", type=int, default=13)
    ap.add_argument("--out", default="experiments/out/dino_path_curvature")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alphas = np.linspace(args.alpha_range[0], args.alpha_range[1],
                          args.alpha_steps)
    device = "cuda"

    print(f"[{time.strftime('%H:%M:%S')}] loading DINOv2...")
    model, processor = load_dino(device)
    if model is None:
        print("DINOv2 load failed — abort")
        return

    results = {}
    for domain in args.domains:
        print(f"\n=== domain={domain} ===")
        if domain == "bedroom":
            from higan_dev.generator import HiGANGenerator
            from higan_dev.manipulate import load_boundary
            from higan_dev.cam.grad_saliency import _layered_direction
            G = HiGANGenerator(higan_repo=str(
                PAPER.parent / "higan_dev" / "data" / "higan_repo"
            ))
            L, D = G.num_layers, G.w_dim
            attrs = list(PIXEL_RATIOS_BEDROOM.keys())
            bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
                    / "boundaries" / "stylegan_bedroom")
            bl_dict = {a: _layered_direction(
                load_boundary(str(bdir), a, num_layers=L).to(G.device),
                L, D, G.device) for a in attrs}
            pixel = PIXEL_RATIOS_BEDROOM
        else:
            from domains.ffhq.generator import FFHQGenerator
            G = FFHQGenerator()
            L, D = G.num_layers, G.w_dim
            attrs = list(PIXEL_RATIOS_FFHQ.keys())
            LAYERS_FOR = {
                "pose":        list(range(0, 4)),
                "gender":      list(range(0, 8)),
                "age":         list(range(0, 8)),
                "eyeglasses":  list(range(0, 8)),
                "smile":       list(range(4, 8)),
            }
            boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                              / "boundaries")
            bl_dict = {}
            for a in attrs:
                v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                            allow_pickle=True).squeeze().astype(np.float32)
                d = torch.from_numpy(v).to(G.device)
                d = d / d.norm().clamp_min(1e-8)
                bl = torch.zeros(L, D, device=G.device)
                for li in LAYERS_FOR[a]:
                    bl[li] = d
                bl_dict[a] = bl
            pixel = PIXEL_RATIOS_FFHQ

        rng = torch.Generator(device=G.device).manual_seed(2027)
        base_wp = G.sample_wp(args.num_samples, generator=rng)

        per_attr = {}
        for attr in attrs:
            t0 = time.time()
            frames = sweep_attr_bedroom(G, bl_dict, attr, base_wp, alphas)
            ratios = path_ratio(frames, model, processor, device)
            per_attr[attr] = {"mean": float(np.mean(ratios)),
                               "std": float(np.std(ratios)),
                               "n": len(ratios)}
            print(f"  {attr:18s} dino_path={np.mean(ratios):.3f}  "
                  f"({time.time()-t0:.0f}s)")

        ord_pix = [pixel[a] for a in per_attr]
        ord_dino = [per_attr[a]["mean"] for a in per_attr]
        pe = pearsonr(ord_pix, ord_dino)
        sp = spearmanr(ord_pix, ord_dino)
        results[domain] = {
            "per_attr": per_attr,
            "pearson": {"r": float(pe.statistic), "p": float(pe.pvalue)},
            "spearman": {"r": float(sp.statistic), "p": float(sp.pvalue)},
        }
        print(f"\n{domain} Pearson r={pe.statistic:+.3f} (p={pe.pvalue:.3g})  "
              f"Spearman r={sp.statistic:+.3f} (p={sp.pvalue:.3g})")
        del G
        torch.cuda.empty_cache()

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
