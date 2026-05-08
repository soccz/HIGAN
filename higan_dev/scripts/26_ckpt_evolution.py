"""How does the encoder's saliency quality evolve during training?

Take 5 checkpoints across the training trajectory (1k, 5k, 10k, 20k, 40k),
encode the same test image at each, then compute grad-saliency on the
encoded wp. Compare: does the saliency converge to the generator's
"truth" saliency (computed from a known-good wp) as training progresses?
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.utils import label_bar as _label
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.manipulate import load_boundary
from higan_dev.cam.grad_saliency import _layered_direction
from higan_dev.cam.diff_map import colorize_heat, overlay
from higan_dev.utils import load_image_tensor



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpts", nargs="+", default=[
        "out/encoder_train/ckpt/enc_001000.pt",
        "out/encoder_train/ckpt/enc_005000.pt",
        "out/encoder_train/ckpt/enc_010000.pt",
        "out/encoder_train/ckpt/enc_020000.pt",
        "out/encoder_train_resume/ckpt/enc_040000.pt",
    ])
    ap.add_argument("--testset", default="out/testset")
    ap.add_argument("--image-idx", type=int, default=2)
    ap.add_argument("--attr", default="indoor_lighting")
    ap.add_argument("--out", default="out/ckpt_evolution")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)

    img_paths = sorted(Path(args.testset).glob("img_*.png"))
    img_path = img_paths[args.image_idx]
    target = load_image_tensor(img_path, size=cfg.generator.resolution,
                               normalize="-1_1").to(G.device)
    target_u8 = ((target.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1)
    target_u8 = (target_u8.cpu().numpy() * 255).astype(np.uint8)[0]

    # ground-truth wp (from testset wp_gt)
    wp_gt_path = Path(args.testset) / "wp_gt.npy"
    wp_gt = torch.from_numpy(np.load(wp_gt_path)[args.image_idx:args.image_idx + 1]
                             ).to(G.device).float()

    # boundary
    boundary = load_boundary(cfg.paths.boundaries_dir, args.attr,
                             num_layers=G.num_layers).to(G.device)
    L, D = G.num_layers, G.w_dim
    b_layered = _layered_direction(boundary, L, D, G.device)

    def saliency_at(wp: torch.Tensor) -> np.ndarray:
        def f(alpha):
            return G.synthesize(wp + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
        _, dimg = jvp(
            f,
            (torch.zeros(1, device=G.device),),
            (torch.ones(1, device=G.device),),
        )
        sal = dimg.abs().mean(dim=1).squeeze(0).cpu().numpy()
        m = sal.max()
        return (sal / m).astype(np.float32) if m > 1e-8 else sal.astype(np.float32)

    # ground-truth saliency (from the actual wp that generated the image)
    sal_gt = saliency_at(wp_gt)

    rows = []
    metrics = []
    cell_w = target_u8.shape[1]

    # GT row first
    gt_recon = G.synthesize(wp_gt)
    gt_u8 = G.to_uint8(gt_recon)[0]
    sal_gt_rgb = colorize_heat(sal_gt)
    sal_gt_ovl = overlay(gt_u8, sal_gt, alpha=0.55)
    gt_label = _label("GT (wp_true)", cell_w)
    iter_label = _label("iter ∞ (true)", cell_w, fs=12)
    gt_row = np.concatenate([gt_u8, sal_gt_rgb, sal_gt_ovl], axis=1)
    gt_lbls = np.concatenate([_label("recon", cell_w), _label("saliency", cell_w),
                              _label("overlay", cell_w)], axis=1)
    rows.append(np.concatenate(
        [_label(f"━━ GROUND TRUTH (wp = wp_true) ━━", gt_row.shape[1], h=28, fs=16),
         gt_lbls, gt_row], axis=0
    ))

    for ckpt_path in args.ckpts:
        cp = Path(ckpt_path)
        if not cp.exists():
            print(f"[skip] {cp} not found")
            continue
        enc, w_avg = load_encoder(cp, device=G.device)
        res = encode_image(target, enc, G, w_avg=w_avg)
        wp_enc = res.wp
        recon_u8 = G.to_uint8(res.image)[0]
        sal = saliency_at(wp_enc)

        # quality metrics: pixel MSE between encoder recon and target
        with torch.no_grad():
            mse = float(F.mse_loss(res.image, target).item())
        # similarity between saliency and GT saliency (pixel correlation)
        sal_corr = float(np.corrcoef(sal.flatten(), sal_gt.flatten())[0, 1])

        # extract iter from filename "enc_NNNNNN.pt"
        iter_num = int(cp.stem.split("_")[-1])
        sal_rgb = colorize_heat(sal)
        sal_ovl = overlay(recon_u8, sal, alpha=0.55)
        row = np.concatenate([recon_u8, sal_rgb, sal_ovl], axis=1)
        lbls = np.concatenate([_label("recon", cell_w), _label("saliency", cell_w),
                                _label("overlay", cell_w)], axis=1)
        eyebrow = _label(
            f"━━ iter {iter_num:>6,d}  ·  recon MSE={mse:.4f}  ·  "
            f"saliency corr w/ GT={sal_corr:+.3f} ━━",
            row.shape[1], h=28, fs=15,
        )
        rows.append(np.concatenate([eyebrow, lbls, row], axis=0))
        metrics.append({"iter": iter_num, "mse": mse, "sal_corr": sal_corr})

    # also include target (top reference)
    target_row_label = _label(f"━━ TARGET IMAGE (img_{args.image_idx}.png) ━━",
                              target_u8.shape[1] * 3, h=28, fs=16)
    target_view = np.concatenate([target_u8] * 3, axis=1)  # repeat to match width
    rows.insert(0, np.concatenate([target_row_label, target_view], axis=0))

    final = np.concatenate(rows, axis=0)
    out_path = out_dir / f"evolution_img{args.image_idx}_{args.attr}.png"
    Image.fromarray(final).save(out_path)
    print(f"saved {out_path}")
    print("\nmetrics by iter:")
    print(f"  {'iter':>10s} {'recon_mse':>12s} {'saliency_corr':>14s}")
    for m in metrics:
        print(f"  {m['iter']:>10,d} {m['mse']:12.4f} {m['sal_corr']:+14.3f}")


if __name__ == "__main__":
    main()
