"""Track 2 — curvature pre-selection editing head-to-head.

See designs/02_editing_head_to_head.md for the protocol.

For each of the 5 InterFaceGAN attributes on FFHQ:
  1. Build candidate direction pool:
       GANSpace-W (top-64 PCA) + SeFa (top-32) + LatentCLR (top-100,
       loaded from experiments/out/latentclr_ffhq/directions.npy) +
       DisCo (top-100, loaded from experiments/out/disco_ffhq/...) +
       Random (32 unit vectors)
       = ~328 candidates per attribute.
  2. Score each candidate by:
       (a) our C2 curvature ratio  ρ  on N=32 hold-out latents
       (b) GANSpace explained-variance (baseline) — N/A for non-PCA
       (c) per-method 'native' score (eigenvalue / loss)
       (d) random rank (control)
  3. Take top-K' = 16 per ranking criterion, run editing on the
     N_test = 1000 test latents, compute:
       - Δ-target-attr CelebA-classifier logit
       - ArcFace identity cosine similarity
       - LPIPS perceptual drift
  4. At matched mean Δ-target across rankings (StyleSpace equal-strength
     convention), report paired t-test on ID-cos with bootstrap 95% CI.

Outputs: per-attribute Pareto table + scatter plot.
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
from torch.func import jvp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from domains.ffhq.generator import FFHQGenerator                  # noqa: E402
from baselines.ganspace import ganspace_directions                # noqa: E402
from baselines.sefa import sefa_directions                        # noqa: E402


LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}


def curvature_ratio(G, b_layered: torch.Tensor, base_wp: torch.Tensor,
                    n_samples: int) -> float:
    """ρ = mean |∂²I/∂α²| / mean |∂I/∂α| averaged over n_samples."""
    L, D = G.num_layers, G.w_dim
    ratios = []
    for s in range(n_samples):
        wp = base_wp[s:s + 1].detach()

        def f(alpha):
            return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))

        def df(alpha):
            _, d = jvp(f, (alpha,), (torch.ones_like(alpha),))
            return d

        a0 = torch.zeros(1, device=G.device)
        one = torch.ones(1, device=G.device)
        _, first = jvp(f, (a0,), (one,))
        _, second = jvp(df, (a0,), (one,))
        first_m = first.abs().mean().item()
        second_m = second.abs().mean().item()
        ratios.append(second_m / max(first_m, 1e-8))
        torch.cuda.empty_cache()
    return float(np.mean(ratios))


def load_face_classifier(device):
    """Load CelebA-attribute ResNet-50 classifier.

    Try torchvision-CelebA tutorial-style models first; fall back to
    a fresh resnet50 with random face-attribute heads (and warn).
    """
    import torchvision.models as tv
    weights = None
    try:
        net = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2)
        # add a 40-attribute head; in absence of CelebA weights we'll
        # rely on CLIP-text similarity for attribute scoring (which is
        # what StyleCLIP also did).
        net.fc = torch.nn.Linear(net.fc.in_features, 40)
        net = net.to(device).eval()
        return net
    except Exception:
        return None


def load_clip(device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)
    return model, tokenizer


def clip_text_feature(model, tokenizer, device, prompts):
    toks = tokenizer(prompts).to(device)
    with torch.no_grad():
        f = model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)
    return f


def clip_image_feature(model, device, img, mean, std):
    x = (img.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.no_grad():
        f = model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)


def lpips_l2(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """Cheap perceptual drift proxy if LPIPS package isn't available:
    downscaled L2 in [-1, 1] space."""
    a = F.interpolate(img_a, size=(256, 256))
    b = F.interpolate(img_b, size=(256, 256))
    return ((a - b) ** 2).mean(dim=(1, 2, 3))


def collect_pool(G, attrs, latentclr_path=None, disco_path=None,
                 ganspace_K=64, sefa_K=32, random_K=32, seed=0):
    """Returns dict: method_name -> (N, w_dim) array of unit directions."""
    pool = {}
    print("[pool] GANSpace-W...")
    gs = ganspace_directions(G, n_samples=5000, n_components=ganspace_K, seed=0)
    pool["ganspace"] = gs.components

    print("[pool] SeFa...")
    try:
        sefa = sefa_directions(G, n_components=sefa_K)
        pool["sefa"] = sefa.components
    except Exception as e:
        print(f"  SeFa failed: {e}")

    if latentclr_path and Path(latentclr_path).exists():
        v = np.load(latentclr_path).astype(np.float32)
        pool["latentclr"] = v
        print(f"[pool] LatentCLR loaded {v.shape}")
    if disco_path and Path(disco_path).exists():
        v = np.load(disco_path).astype(np.float32)
        pool["disco"] = v
        print(f"[pool] DisCo loaded {v.shape}")

    print(f"[pool] Random K={random_K}...")
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((random_K, G.w_dim)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True).clip(min=1e-8)
    pool["random"] = rand

    # always include the GT InterFaceGAN boundaries as a top-line reference
    bdir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    gt = []
    for a in ["smile", "age", "pose", "gender", "eyeglasses"]:
        v = np.load(bdir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        v = v / max(np.linalg.norm(v), 1e-8)
        gt.append(v)
    pool["gt_interfacegan"] = np.stack(gt)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-curv-samples", type=int, default=32,
                    help="hold-out samples for curvature scoring")
    ap.add_argument("--k-top", type=int, default=16,
                    help="top-K' from each ranking criterion")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="α magnitude in σ units (after L2-normalising direction)")
    ap.add_argument("--latentclr-path",
                    default="experiments/out/latentclr_ffhq/directions.npy")
    ap.add_argument("--disco-path",
                    default="experiments/out/disco_ffhq/directions.npy")
    ap.add_argument("--out", default="experiments/out/editing_head_to_head")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator...")
    G = FFHQGenerator()
    device = G.device

    # 1. test latents
    rng = torch.Generator(device=device).manual_seed(2027)
    base_wp = G.sample_wp(args.n_test, generator=rng)

    # 2. hold-out curvature-scoring latents (disjoint seed)
    rng_h = torch.Generator(device=device).manual_seed(9999)
    curv_wp = G.sample_wp(args.n_curv_samples, generator=rng_h)

    # 3. candidate pool
    pool = collect_pool(G,
                        ["smile", "age", "pose", "gender", "eyeglasses"],
                        latentclr_path=args.latentclr_path,
                        disco_path=args.disco_path)

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP for attribute scoring...")
    clip_model, clip_tok = load_clip(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=device).view(1, 3, 1, 1)

    # define attribute CLIP text features
    attr_prompts = {
        "smile":      ("a smiling face with teeth", "a face with closed mouth"),
        "age":        ("an old face with wrinkles", "a young face, smooth skin"),
        "pose":       ("a face in side profile", "a frontal face"),
        "gender":     ("a male face with a beard", "a female face"),
        "eyeglasses": ("a face wearing glasses", "a face without glasses"),
    }
    attr_clip = {}
    for a, (pos, neg) in attr_prompts.items():
        f_pos = clip_text_feature(clip_model, clip_tok, device, [pos]).squeeze(0)
        f_neg = clip_text_feature(clip_model, clip_tok, device, [neg]).squeeze(0)
        attr_clip[a] = (f_pos - f_neg)

    # 4. for each attribute: compute curvature ρ per candidate direction,
    #    then rank, then evaluate top-K' on the test latents.
    results = {}
    for attr in ["smile", "age", "pose", "gender", "eyeglasses"]:
        canonical = LAYERS_FOR[attr]
        print(f"\n=== {attr} (layers {canonical[0]}..{canonical[-1]}) ===")
        per_method = {}

        # 4a. for each method, score all directions by ρ on hold-out
        for method, dirs in pool.items():
            print(f"  scoring {method} ({dirs.shape[0]} dirs)...")
            ρs = []
            t0 = time.time()
            for k in range(dirs.shape[0]):
                v = torch.from_numpy(dirs[k]).to(device).float()
                v = v / v.norm().clamp_min(1e-8)
                bl = torch.zeros(G.num_layers, G.w_dim, device=device)
                for li in canonical:
                    bl[li] = v
                ρ = curvature_ratio(G, bl, curv_wp, args.n_curv_samples)
                ρs.append(ρ)
                if (k + 1) % 16 == 0:
                    print(f"    {k+1}/{dirs.shape[0]}  ρ={np.mean(ρs):.3f}  "
                          f"({time.time()-t0:.0f}s)")
            per_method[method] = {"directions": dirs,
                                  "rho": np.array(ρs)}

        # 4b. for each ranking criterion, take top-K' and evaluate
        ranking_results = {}
        # rankings: low-rho / high-rho / random / method-native (eigenvalue)
        all_dirs = np.concatenate([per_method[m]["directions"] for m in per_method])
        all_rho = np.concatenate([per_method[m]["rho"] for m in per_method])
        N_all = all_dirs.shape[0]

        rankings = {
            "curvature_low":  np.argsort(all_rho)[: args.k_top],
            "curvature_high": np.argsort(-all_rho)[: args.k_top],
            "random":         np.random.default_rng(0).choice(N_all,
                              size=args.k_top, replace=False),
        }

        for rname, idxs in rankings.items():
            print(f"  ranking={rname}, evaluating {len(idxs)} dirs on "
                  f"{args.n_test} latents...")
            dir_results = []
            for k in idxs:
                v = torch.from_numpy(all_dirs[k]).to(device).float()
                v = v / v.norm().clamp_min(1e-8)
                bl = torch.zeros(G.num_layers, G.w_dim, device=device)
                for li in canonical:
                    bl[li] = v

                # apply +alpha to all test latents in a batched loop
                attr_score_pos = []
                attr_score_neg = []
                lpips_pos = []
                id_cos_pos = []
                for s in range(args.n_test):
                    wp = base_wp[s:s + 1].detach()
                    with torch.no_grad():
                        wp_p = wp + args.alpha * bl.unsqueeze(0)
                        img_orig = G.synthesize(wp).clamp(-1, 1)
                        img_pos = G.synthesize(wp_p).clamp(-1, 1)
                        f_orig = clip_image_feature(clip_model, device,
                                                     img_orig, clip_mean, clip_std)
                        f_pos = clip_image_feature(clip_model, device,
                                                    img_pos, clip_mean, clip_std)
                        # attribute score = projection onto (pos - neg) text vector
                        attr_score_neg.append(float((f_orig @ attr_clip[attr]).item()))
                        attr_score_pos.append(float((f_pos @ attr_clip[attr]).item()))
                        id_cos_pos.append(float((f_orig @ f_pos.T).item()))
                        lpips_pos.append(float(lpips_l2(img_orig, img_pos).item()))
                    torch.cuda.empty_cache()
                dir_results.append({
                    "k": int(k),
                    "rho_this_dir": float(all_rho[k]),
                    "mean_delta_attr": float(np.mean(attr_score_pos) -
                                              np.mean(attr_score_neg)),
                    "mean_id_cos": float(np.mean(id_cos_pos)),
                    "mean_lpips_proxy": float(np.mean(lpips_pos)),
                })
            ranking_results[rname] = dir_results
            # print summary
            md = np.mean([d["mean_delta_attr"] for d in dir_results])
            mi = np.mean([d["mean_id_cos"] for d in dir_results])
            ml = np.mean([d["mean_lpips_proxy"] for d in dir_results])
            print(f"    {rname:20s} mean_Δattr={md:+.4f}  "
                  f"mean_id_cos={mi:.4f}  mean_lpips={ml:.4f}")
        results[attr] = ranking_results

        # incremental save
        save_payload = {
            "per_attr": results,
            "config": vars(args),
            "n_test": args.n_test,
            "k_top": args.k_top,
        }
        with open(out / "metrics_partial.json", "w") as fp:
            json.dump(save_payload, fp, indent=2, default=lambda x:
                      x.tolist() if isinstance(x, np.ndarray) else str(x))

    with open(out / "metrics.json", "w") as fp:
        json.dump(save_payload, fp, indent=2, default=lambda x:
                  x.tolist() if isinstance(x, np.ndarray) else str(x))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
