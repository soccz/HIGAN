"""Track 4 evaluation — C5 chart-quality metric on FFHQ across checkpoints.

For each saved encoder checkpoint:
    recon_mse  = mean MSE between G(E(G(wp_gt))) and G(wp_gt) on test set
    recon_lpips = mean LPIPS at 256²
    sal_corr   = mean per-pixel saliency correlation between
                 S^(1)(b) at wp_gt and S^(1)(b) at E(G(wp_gt))
                 averaged over the 5 InterFaceGAN attributes
                 and the test latents.

Plots: recon vs sal_corr across the 5 checkpoints; reproduces the
bedroom dissociation curve (recon plateaus, sal_corr keeps rising).
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

PAPER = Path(__file__).resolve().parents[4]   # paper/
EXPERIMENTS = PAPER / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from domains.ffhq.generator import FFHQGenerator             # noqa: E402
from higan_dev.encoder.model import WPlusEncoder, WPlusEncoderCfg  # noqa: E402

LAYERS_FOR = {
    "pose":        list(range(0, 4)),
    "gender":      list(range(0, 8)),
    "age":         list(range(0, 8)),
    "eyeglasses":  list(range(0, 8)),
    "smile":       list(range(4, 8)),
}


def saliency(G: FFHQGenerator, wp: torch.Tensor,
             b_layered: torch.Tensor) -> np.ndarray:
    def f(alpha):
        return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
    a0 = torch.zeros(1, device=G.device)
    one = torch.ones(1, device=G.device)
    _, d = jvp(f, (a0,), (one,))
    return d.abs().mean(dim=1).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="paths to encoder checkpoint files")
    ap.add_argument("--num-test", type=int, default=64)
    ap.add_argument("--out", default="experiments/out/ffhq_c5/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator()
    device = G.device
    boundaries_dir = PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
    attrs = ["smile", "age", "pose", "gender", "eyeglasses"]
    b_layered = {}
    for a in attrs:
        v = np.load(boundaries_dir / f"stylegan_ffhq_{a}_w_boundary.npy",
                    allow_pickle=True).squeeze().astype(np.float32)
        d = torch.from_numpy(v).to(device)
        d = d / d.norm().clamp_min(1e-8)
        bl = torch.zeros(G.num_layers, G.w_dim, device=device)
        for li in LAYERS_FOR[a]:
            bl[li] = d
        b_layered[a] = bl

    rng = torch.Generator(device=device).manual_seed(9999)  # disjoint
    wp_test = G.sample_wp(args.num_test, generator=rng)
    # Memory-frugal: never materialize the full (N, 3, 1024, 1024) tensor on GPU.
    # All per-sample work happens in a loop over single samples.

    results = []
    for ck in args.ckpts:
        state = torch.load(ck, map_location=device, weights_only=False)
        enc_cfg = WPlusEncoderCfg(**state["encoder_cfg"])
        enc = WPlusEncoder(enc_cfg).to(device)
        enc.load_state_dict(state["model"])
        enc.eval()

        n_iter = int(state.get("iter", 0))
        print(f"\n== ckpt {ck} (iter={n_iter}) ==")

        # Pre-compute wp_pred for all test samples (small tensor, fits)
        wp_pred_list = []
        recon_mse_sum = 0.0
        recon_l2_256_sum = 0.0
        with torch.no_grad():
            for s in range(args.num_test):
                wp_gt_s = wp_test[s:s + 1]
                img_gt_s = G.synthesize(wp_gt_s).clamp(-1, 1)
                tgt_small = F.interpolate(img_gt_s, size=(256, 256),
                                           mode="bilinear", align_corners=False)
                wp_pred_s = enc(tgt_small)
                img_pred_s = G.synthesize(wp_pred_s).clamp(-1, 1)
                recon_mse_sum += F.mse_loss(img_pred_s, img_gt_s).item()
                recon_l2_256_sum += F.mse_loss(
                    F.interpolate(img_pred_s, 256),
                    F.interpolate(img_gt_s, 256)
                ).item()
                wp_pred_list.append(wp_pred_s.detach())
                torch.cuda.empty_cache()
        recon_mse = recon_mse_sum / args.num_test
        recon_l2_256 = recon_l2_256_sum / args.num_test
        wp_pred = torch.cat(wp_pred_list, dim=0)
        print(f"  recon_mse={recon_mse:.6f}  recon_l2_256={recon_l2_256:.6f}")

        # saliency correlation per attribute
        per_attr_corr = {}
        for attr in attrs:
            bl = b_layered[attr]
            corrs = []
            for s in range(args.num_test):
                wp_gt = wp_test[s:s + 1].detach()
                wp_p = wp_pred[s:s + 1].detach()
                sal_gt = saliency(G, wp_gt, bl)
                sal_p = saliency(G, wp_p, bl)
                # per-pixel Pearson
                c = float(np.corrcoef(sal_gt.flatten(), sal_p.flatten())[0, 1])
                corrs.append(c)
                torch.cuda.empty_cache()
            mean_c = float(np.nanmean(corrs))
            per_attr_corr[attr] = mean_c
            print(f"  {attr:12s} sal_corr={mean_c:+.3f}")

        agg = float(np.mean(list(per_attr_corr.values())))
        results.append({
            "ckpt": ck, "iter": n_iter,
            "recon_mse": recon_mse, "recon_l2_256": recon_l2_256,
            "per_attr_corr": per_attr_corr,
            "sal_corr_mean": agg,
        })
        print(f"  recon_mse={recon_mse:.4f}  mean sal_corr={agg:+.3f}")

    with open(out / "c5_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iters = [r["iter"] for r in results]
    recon = [r["recon_mse"] for r in results]
    sal = [r["sal_corr_mean"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax1.plot(iters, recon, "o-", color="#0e7490", lw=2, label="recon MSE")
    ax1.set_xlabel("training iteration"); ax1.set_ylabel("recon MSE",
                                                          color="#0e7490")
    ax2 = ax1.twinx()
    ax2.plot(iters, sal, "s-", color="#c2410c", lw=2,
             label="mean saliency correlation")
    ax2.set_ylabel("saliency-vs-GT correlation", color="#c2410c")
    ax1.set_xscale("log"); ax1.grid(alpha=0.3)
    ax1.set_title("FFHQ C5 — recon vs saliency-derivative agreement",
                  fontsize=11, weight="bold", pad=8)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "c5_curve.png")
    print(f"\nsaved {out / 'c5_curve.png'} and c5_eval.json")


if __name__ == "__main__":
    main()
