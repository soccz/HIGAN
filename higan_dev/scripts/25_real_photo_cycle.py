"""End-to-end pipeline on real LSUN bedroom photos.

For each real photo:
    1. Compute optim-based inversion (1000 step Adam) — strong reference.
    2. Compute encoder-based inversion (1 forward pass).
    3. Compute grad-saliency on the encoder's wp for selected attributes.
    4. Render edits at -d, 0, +d.

Outputs a single composite figure per photo + a side-by-side comparison
(synthetic vs real) of inversion difficulty.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
from PIL import Image, ImageDraw, ImageFont

from higan_dev.config import Config, resolve
from higan_dev.generator import HiGANGenerator
from higan_dev.inversion.encode import load_encoder, encode_image
from higan_dev.inversion.optim import invert_image
from higan_dev.manipulate import load_boundary, manipulate_wp
from higan_dev.cam.grad_saliency import _layered_direction
from higan_dev.cam.diff_map import colorize_heat, overlay
from higan_dev.losses import LPIPSLoss
from higan_dev.utils import load_image_tensor


def _label(text: str, w: int, h: int = 24, fs: int = 14) -> np.ndarray:
    img = Image.new("RGB", (w, h), (245, 245, 244))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(40, 40, 40), font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--photos-dir", default="data/real_bedrooms")
    ap.add_argument("--n-photos", type=int, default=4,
                    help="how many real photos to include")
    ap.add_argument("--attrs", nargs="+",
                    default=["indoor_lighting", "wood", "view"])
    ap.add_argument("--steps", type=int, default=800,
                    help="optim inversion steps")
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--out", default="out/real_photo")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = HiGANGenerator(higan_repo=cfg.paths.higan_repo,
                       model_name=cfg.generator.model_name,
                       device=cfg.train.device)
    enc, w_avg = load_encoder(args.ckpt, device=G.device)
    lpips_fn = LPIPSLoss().to(G.device)

    photos = sorted(Path(args.photos_dir).glob("*.png"))[: args.n_photos]
    if not photos:
        raise SystemExit(f"no photos in {args.photos_dir}")
    print(f"running on {len(photos)} real photos")

    cell_w = cfg.generator.resolution
    L, D = G.num_layers, G.w_dim

    table_rows = []
    photo_rows = []
    for pi, p in enumerate(photos):
        target = load_image_tensor(p, size=cell_w, normalize="-1_1").to(G.device)
        target_u8 = ((target.clamp(-1, 1) + 1) / 2)
        target_u8 = (target_u8.permute(0, 2, 3, 1).cpu().numpy() * 255
                     ).astype(np.uint8)[0]

        # optim inversion (no progress bar to keep log clean)
        print(f"  [{pi}] optim inversion ({args.steps} steps)...")
        opt_res = invert_image(
            target=target,
            generator=G,
            num_steps=args.steps,
            lr=0.1,
            num_inits=1,
            init_mode="w_avg",
            loss_weights={"pixel_l2": 1.0, "lpips": 0.8, "tv": 1e-4},
            log_every=10000,
            progress=False,
        )
        opt_recon_u8 = G.to_uint8(opt_res.image)[0]

        # encoder inversion
        enc_res = encode_image(target, enc, G, w_avg=w_avg)
        enc_recon_u8 = G.to_uint8(enc_res.image)[0]
        wp_enc = enc_res.wp

        # quality metrics
        with torch.no_grad():
            opt_lpips = float(lpips_fn(opt_res.image, target).item())
            enc_lpips = float(lpips_fn(enc_res.image, target).item())
            opt_mse = float(F.mse_loss(opt_res.image, target).item())
            enc_mse = float(F.mse_loss(enc_res.image, target).item())
        table_rows.append({
            "photo": p.name,
            "lpips_optim": opt_lpips, "lpips_enc": enc_lpips,
            "mse_optim": opt_mse, "mse_enc": enc_mse,
        })

        # grad-saliency on encoder's wp for each attribute
        sal_cells: list[tuple[str, np.ndarray]] = []
        for attr in args.attrs:
            b = load_boundary(cfg.paths.boundaries_dir, attr,
                              num_layers=L).to(G.device)
            b_layered = _layered_direction(b, L, D, G.device)
            def f(alpha):
                return G.synthesize(wp_enc + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))
            _, dimg = jvp(
                f,
                (torch.zeros(1, device=G.device),),
                (torch.ones(1, device=G.device),),
            )
            sal = dimg.abs().mean(dim=1).squeeze(0).cpu().numpy()
            sal = sal / max(sal.max(), 1e-8)
            sal_cells.append((attr, sal))

        # edits at +/- delta for first attribute, on encoder wp
        b0 = load_boundary(cfg.paths.boundaries_dir, args.attrs[0],
                           num_layers=L).to(G.device)
        manip = manipulate_wp(wp_enc, b0, distances=[-args.delta, args.delta])
        with torch.no_grad():
            edit_imgs = G.synthesize(manip.reshape(2, L, D))
        edit_neg = G.to_uint8(edit_imgs)[0]
        edit_pos = G.to_uint8(edit_imgs)[1]

        # build a row: target | optim recon | encoder recon | sal_a | sal_b | sal_c | edit-d | edit+d
        sal_rgbs = [colorize_heat(s) for _, s in sal_cells]
        row_cells = [target_u8, opt_recon_u8, enc_recon_u8, *sal_rgbs, edit_neg, edit_pos]
        row = np.concatenate(row_cells, axis=1)
        labels = ["real photo",
                  f"optim recon ({args.steps}step)",
                  "encoder recon",
                  *[f"sal({a})" for a, _ in sal_cells],
                  f"edit −{args.delta} ({args.attrs[0]})",
                  f"edit +{args.delta} ({args.attrs[0]})"]
        label_strip = np.concatenate([_label(l, cell_w) for l in labels], axis=1)
        eyebrow = _label(
            f"━━ {p.name}  ·  encoder LPIPS={enc_lpips:.3f}  ·  optim LPIPS={opt_lpips:.3f} ━━",
            row.shape[1], h=28, fs=16,
        )
        photo_rows.append(np.concatenate([eyebrow, label_strip, row], axis=0))

    final = np.concatenate(photo_rows, axis=0)
    out_path = out / "real_photo_cycle.png"
    Image.fromarray(final).save(out_path)
    print(f"\nsaved {out_path}  ({final.shape[1]} x {final.shape[0]})")

    # save metrics
    with open(out / "metrics.json", "w") as f:
        json.dump(table_rows, f, indent=2)
    print("\ninversion quality on real photos (lower = better):")
    print(f"  {'photo':18s} {'optim LPIPS':>12s} {'enc LPIPS':>12s} "
          f"{'optim MSE':>12s} {'enc MSE':>12s}")
    for r in table_rows:
        print(f"  {r['photo']:18s} "
              f"{r['lpips_optim']:12.4f} {r['lpips_enc']:12.4f} "
              f"{r['mse_optim']:12.4f} {r['mse_enc']:12.4f}")


if __name__ == "__main__":
    main()
