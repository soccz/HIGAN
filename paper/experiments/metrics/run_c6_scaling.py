"""Track 12 — C6 sample-size scaling: precision/recall vs N random directions.

For each N ∈ {128, 192, 256, 384, 512}: sample N random unit
directions on per-layer W+ spheres, filter to above-median strength,
PCA-32 + K-means K=8, CLIP-label each cluster, then evaluate
precision / recall vs ground-truth attribute taxonomy.

See designs/12_c6_scaling.md.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER.parent / "higan_dev"))
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402

BEDROOM_VOCAB = [
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
BEDROOM_GT = {
    "view":             {"a view through a window", "outdoor view", "a window"},
    "indoor_lighting":  {"bright lighting", "dim lighting", "warm light"},
    "wood":             {"wood texture"},
    "carpet":           {"carpet", "soft texture", "fabric"},
    "cluttered_space":  {"cluttered space"},
    "glossy":           {"glossy reflective surface", "metal surface"},
    "dirt":             {"dirty surface", "rough texture"},
    "scary":            {"scary atmosphere"},
}


def run_for_N(N: int, args, G, vocab_text_feat, clip_preprocess, clip_model,
               bases_wp, bases_u8, base_sims, save_dir):
    L, D = G.num_layers, G.w_dim
    H = W = G.resolution

    rng_dir = torch.Generator(device=G.device).manual_seed(args.seed + 1 + N)
    saliencies, layer_idx, strengths = [], [], []
    for d in range(N):
        li = int(torch.randint(1, L, (1,), generator=rng_dir,
                                device=G.device).item())
        v = torch.randn(D, generator=rng_dir, device=G.device)
        v = v / v.norm().clamp_min(1e-8)
        bl = torch.zeros(L, D, device=G.device)
        bl[li] = v
        acc = torch.zeros(H, W, device=G.device)
        for s in range(args.num_samples_per_dir):
            wp = bases_wp[s:s + 1].detach()
            def f(alpha):
                return G.synthesize(wp + alpha.view(1, 1, 1) * bl.unsqueeze(0))
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
            print(f"    N={N} dir {d+1}/{N}")
    sal_arr = np.stack(saliencies)
    str_arr = np.asarray(strengths)
    median = float(np.median(str_arr))
    keep = np.where(str_arr >= median)[0]
    sal_kept = sal_arr[keep]
    layer_kept = np.asarray(layer_idx)[keep]
    print(f"  N={N}: kept {keep.size} above-median")

    # downsample + PCA + K-means
    sal_small = np.stack([
        np.asarray(Image.fromarray((s * 255).astype(np.uint8)).resize(
            (64, 64), Image.BILINEAR
        )) / 255.0
        for s in sal_kept
    ]).astype(np.float32)
    sal_flat = sal_small.reshape(sal_small.shape[0], -1)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(32, sal_flat.shape[0] - 1))
    sal_pca = pca.fit_transform(sal_flat)
    km = KMeans(n_clusters=args.num_clusters, random_state=0,
                 n_init=10)
    labels = km.fit_predict(sal_pca)

    # CLIP labelling
    rng_dir2 = torch.Generator(device=G.device).manual_seed(args.seed + 11)
    cluster_labels = {}
    for c in range(args.num_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        layers_in_c = layer_kept[idx]
        modal_layer = int(np.bincount(layers_in_c).argmax())
        accum_imgs = []
        delta = 4.0
        with torch.no_grad():
            for _ in range(args.clip_num_dirs_per_cluster):
                v2 = torch.randn(D, generator=rng_dir2, device=G.device)
                v2 = v2 / v2.norm().clamp_min(1e-8)
                bl2 = torch.zeros(L, D, device=G.device)
                bl2[modal_layer] = v2
                wp_p = bases_wp + delta * bl2.unsqueeze(0)
                imgs = G.synthesize(wp_p)
                accum_imgs.append(imgs)
            avg = torch.stack(accum_imgs).mean(0)
            avg_u8 = G.to_uint8(avg)
        # CLIP score
        def clip_score(image_np):
            pil = Image.fromarray(image_np)
            x = clip_preprocess(pil).unsqueeze(0).to(G.device)
            with torch.no_grad():
                f = clip_model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                return (f @ vocab_text_feat.T).squeeze(0).cpu().numpy()
        avg_sims = np.mean([clip_score(a) for a in avg_u8], axis=0)
        contrastive = avg_sims - base_sims
        order = np.argsort(-contrastive)
        topk = [(BEDROOM_VOCAB[i], float(contrastive[i])) for i in order[:4]]
        cluster_labels[c] = topk

    # precision/recall vs GT
    def evaluate(top_k_int):
        matched_clusters = 0
        matched_attrs = set()
        for c, topk in cluster_labels.items():
            labs = {phrase for phrase, _ in topk[:top_k_int]}
            hits = {a for a, proxies in BEDROOM_GT.items() if labs & proxies}
            if hits:
                matched_clusters += 1
                matched_attrs |= hits
        n_cl = len(cluster_labels)
        P = matched_clusters / n_cl if n_cl else 0.0
        R = len(matched_attrs) / len(BEDROOM_GT)
        F1 = 2 * P * R / max(P + R, 1e-8)
        return {"P": P, "R": R, "F1": F1,
                "matched_attrs": sorted(matched_attrs)}

    return {
        "N": N,
        "kept": int(keep.size),
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "evaluation_topk": {f"K={k}": evaluate(k) for k in [1, 2, 3, 4]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", nargs="+", type=int,
                    default=[128, 192, 256, 384, 512])
    ap.add_argument("--num-samples-per-dir", type=int, default=4)
    ap.add_argument("--num-clusters", type=int, default=8)
    ap.add_argument("--clip-num-dirs-per-cluster", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="experiments/out/c6_scaling_bedroom")
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from higan_dev.generator import HiGANGenerator
    G = HiGANGenerator(higan_repo=str(
        PAPER.parent / "higan_dev" / "data" / "higan_repo"
    ))

    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    bases_wp = G.sample_wp(args.num_samples_per_dir, generator=rng)
    with torch.no_grad():
        bases_img = G.synthesize(bases_wp).clamp(-1, 1)
    bases_u8 = G.to_uint8(bases_img)

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.eval().to(G.device)
    text_tokens = tokenizer(BEDROOM_VOCAB).to(G.device)
    with torch.no_grad():
        vocab_text_feat = clip_model.encode_text(text_tokens)
        vocab_text_feat = vocab_text_feat / vocab_text_feat.norm(
            dim=-1, keepdim=True
        )

    def clip_score(image_np):
        pil = Image.fromarray(image_np)
        x = clip_preprocess(pil).unsqueeze(0).to(G.device)
        with torch.no_grad():
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            return (f @ vocab_text_feat.T).squeeze(0).cpu().numpy()

    base_sims = np.mean([clip_score(b) for b in bases_u8], axis=0)

    all_results = {}
    for N in args.Ns:
        print(f"\n=== N = {N} ===")
        t0 = time.time()
        all_results[str(N)] = run_for_N(
            N, args, G, vocab_text_feat, clip_preprocess, clip_model,
            bases_wp, bases_u8, base_sims, out
        )
        print(f"  N={N} done ({time.time()-t0:.1f}s)")
        with open(out / "metrics_partial.json", "w") as fp:
            json.dump(all_results, fp, indent=2)

    with open(out / "metrics.json", "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nsaved {out / 'metrics.json'}")

    # Plot recall vs N at K=3
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = args.Ns
    R_k3 = [all_results[str(N)]["evaluation_topk"]["K=3"]["R"] for N in Ns]
    P_k3 = [all_results[str(N)]["evaluation_topk"]["K=3"]["P"] for N in Ns]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.plot(Ns, R_k3, "s-", color="#c2410c", lw=2, label="Recall @ K=3")
    ax.plot(Ns, P_k3, "o-", color="#0e7490", lw=2, label="Precision @ K=3")
    ax.set_xlabel("N random directions"); ax.set_ylabel("score")
    ax.set_title("C6 — bedroom precision / recall vs N",
                 fontsize=11, weight="bold")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(out / "c6_scaling.png")
    print(f"saved {out / 'c6_scaling.png'}")


if __name__ == "__main__":
    main()
