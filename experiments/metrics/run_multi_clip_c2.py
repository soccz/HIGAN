"""Track 9 — multi-CLIP-encoder robustness for the C2 path-curvature metric.

Re-renders the α-sweep on bedroom + FFHQ (cheap; just G.synthesize at
13 alpha values × N samples × 13 attributes) and re-encodes the
resulting images through three CLIP variants:
  ViT-B/32 (current default), ViT-L/14, ViT-H/14.

For each (variant, attr), report path-length / direct-distance ratio
and compare to the pixel ∂²I ratio. Test whether the Pearson + Spearman
between pixel curvature and CLIP path is preserved across CLIP
variants.

See designs/09_multi_clip_robustness.md.
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

CLIP_VARIANTS = [
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
]


@torch.no_grad()
def sweep_one_attr_bedroom(G, b_layered_dict, attr, base_wp, alphas,
                            device):
    """Returns list of (N, num_alpha, 3, H, W) tensors collapsed to CPU."""
    from higan_dev.cam.grad_saliency import _layered_direction
    bl = b_layered_dict[attr]
    N = base_wp.size(0)
    out = []
    for s in range(N):
        wp = base_wp[s:s + 1].detach()
        frames = []
        for a in alphas:
            img = G.synthesize(wp + float(a) * bl.unsqueeze(0))
            frames.append(img.squeeze(0).cpu())
        out.append(torch.stack(frames, dim=0))   # (num_alpha, 3, H, W)
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def sweep_one_attr_ffhq(G, b_layered_dict, attr, base_wp, alphas):
    bl = b_layered_dict[attr]
    N = base_wp.size(0)
    out = []
    for s in range(N):
        wp = base_wp[s:s + 1].detach()
        frames = []
        for a in alphas:
            img = G.synthesize(wp + float(a) * bl.unsqueeze(0))
            frames.append(img.squeeze(0).cpu())
        out.append(torch.stack(frames, dim=0))
        torch.cuda.empty_cache()
    return out


def clip_path_ratio(frames_per_sample, clip_model, mean, std, device):
    """frames_per_sample: list of (num_alpha, 3, H, W) tensors on CPU.
    Returns per-sample path/direct ratio."""
    ratios = []
    for frames in frames_per_sample:
        feats = []
        for k in range(frames.shape[0]):
            img = frames[k:k+1].to(device)
            x = (img.clamp(-1, 1) + 1) / 2
            x = F.interpolate(x, size=(224, 224), mode="bilinear",
                              align_corners=False)
            x = (x - mean) / std
            with torch.no_grad():
                f = clip_model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.squeeze(0))
        F_stack = torch.stack(feats)
        seg = torch.linalg.norm(F_stack[1:] - F_stack[:-1], dim=1)
        path_len = seg.sum().item()
        direct = torch.linalg.norm(F_stack[-1] - F_stack[0]).item()
        ratios.append(path_len / direct if direct > 1e-8 else 0.0)
    return ratios


def load_clip(name, pretrained, device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(name,
                                                         pretrained=pretrained)
    model = model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                    default=["bedroom", "ffhq"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--alpha-range", type=float, nargs=2,
                    default=[-3.0, 3.0])
    ap.add_argument("--alpha-steps", type=int, default=13)
    ap.add_argument("--out", default="experiments/out/multi_clip_c2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alphas = np.linspace(args.alpha_range[0], args.alpha_range[1],
                          args.alpha_steps)

    print(f"[{time.strftime('%H:%M:%S')}] multi-CLIP C2 — "
          f"domains {args.domains} × {args.alpha_steps} α × "
          f"{args.num_samples} seeds × {len(CLIP_VARIANTS)} CLIP variants")

    sweeps = {}
    # ---- bedroom ----
    if "bedroom" in args.domains:
        from higan_dev.generator import HiGANGenerator
        from higan_dev.manipulate import load_boundary
        from higan_dev.cam.grad_saliency import _layered_direction
        G = HiGANGenerator(higan_repo=str(
            PAPER.parent / "higan_dev" / "data" / "higan_repo"
        ))
        L, D = G.num_layers, G.w_dim
        attrs_b = list(PIXEL_RATIOS_BEDROOM.keys())
        bdir = (PAPER.parent / "higan_dev" / "data" / "higan_repo"
                / "boundaries" / "stylegan_bedroom")
        bl_dict = {
            a: _layered_direction(
                load_boundary(str(bdir), a, num_layers=L).to(G.device),
                L, D, G.device
            ) for a in attrs_b
        }
        rng = torch.Generator(device=G.device).manual_seed(2027)
        base_wp = G.sample_wp(args.num_samples, generator=rng)
        bedroom_frames = {}
        for attr in attrs_b:
            t0 = time.time()
            bedroom_frames[attr] = sweep_one_attr_bedroom(
                G, bl_dict, attr, base_wp, alphas, G.device
            )
            print(f"  bedroom {attr} swept ({time.time()-t0:.1f}s)")
        sweeps["bedroom"] = bedroom_frames
        del G
        torch.cuda.empty_cache()

    # ---- ffhq ----
    if "ffhq" in args.domains:
        from domains.ffhq.generator import FFHQGenerator
        G = FFHQGenerator()
        L, D = G.num_layers, G.w_dim
        boundaries_dir = (PAPER / "experiments" / "data" / "interfacegan"
                          / "boundaries")
        LAYERS_FOR = {
            "pose":        list(range(0, 4)),
            "gender":      list(range(0, 8)),
            "age":         list(range(0, 8)),
            "eyeglasses":  list(range(0, 8)),
            "smile":       list(range(4, 8)),
        }
        attrs_f = list(PIXEL_RATIOS_FFHQ.keys())
        bl_dict = {}
        for a in attrs_f:
            v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                        allow_pickle=True).squeeze().astype(np.float32)
            d = torch.from_numpy(v).to(G.device)
            d = d / d.norm().clamp_min(1e-8)
            bl = torch.zeros(L, D, device=G.device)
            for li in LAYERS_FOR[a]:
                bl[li] = d
            bl_dict[a] = bl
        rng = torch.Generator(device=G.device).manual_seed(2027)
        base_wp = G.sample_wp(args.num_samples, generator=rng)
        ffhq_frames = {}
        for attr in attrs_f:
            t0 = time.time()
            ffhq_frames[attr] = sweep_one_attr_ffhq(G, bl_dict, attr,
                                                     base_wp, alphas)
            print(f"  ffhq {attr} swept ({time.time()-t0:.1f}s)")
        sweeps["ffhq"] = ffhq_frames
        del G
        torch.cuda.empty_cache()

    # ---- CLIP encoding per variant ----
    device = "cuda"
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=device).view(1, 3, 1, 1)

    results = {}
    for name, pretrained in CLIP_VARIANTS:
        print(f"\n=== {name} ({pretrained}) ===")
        clip_model = load_clip(name, pretrained, device)
        for domain in args.domains:
            frames_per_attr = sweeps[domain]
            pixel = (PIXEL_RATIOS_BEDROOM if domain == "bedroom"
                     else PIXEL_RATIOS_FFHQ)
            per_attr = {}
            for attr, fs in frames_per_attr.items():
                ratios = clip_path_ratio(fs, clip_model, clip_mean,
                                          clip_std, device)
                per_attr[attr] = {"mean": float(np.mean(ratios)),
                                  "std": float(np.std(ratios)),
                                  "n": len(ratios)}
                print(f"  {domain:8s} {attr:18s} ratio={np.mean(ratios):.3f}")
            # rank correlation vs pixel ratio
            ord_pix = [pixel[a] for a in per_attr]
            ord_clip = [per_attr[a]["mean"] for a in per_attr]
            pe = pearsonr(ord_pix, ord_clip)
            sp = spearmanr(ord_pix, ord_clip)
            results.setdefault(name, {})[domain] = {
                "per_attr": per_attr,
                "pearson": {"r": float(pe.statistic), "p": float(pe.pvalue)},
                "spearman": {"r": float(sp.statistic), "p": float(sp.pvalue)},
            }
            print(f"  {domain}  Pearson r={pe.statistic:+.3f} (p={pe.pvalue:.3g})  "
                  f"Spearman r={sp.statistic:+.3f} (p={sp.pvalue:.3g})")
        del clip_model
        torch.cuda.empty_cache()

    print("\n=== summary ===")
    for name in [n for n, _ in CLIP_VARIANTS]:
        for d in args.domains:
            r = results[name][d]
            print(f"  {name:10s} {d:8s} Pearson={r['pearson']['r']:+.3f}  "
                  f"Spearman={r['spearman']['r']:+.3f}")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
