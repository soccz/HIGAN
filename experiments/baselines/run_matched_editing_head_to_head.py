"""Attribute-change-matched FFHQ editing head-to-head.

This is the stricter version of Track 2. It reuses the selected candidate
indices from `editing_head_to_head/metrics.json`, calibrates the edit strength
per direction on held-out latents to hit a common CLIP-attribute delta, then
compares identity preservation and perceptual drift at matched edit magnitude.

The script is GPU-only, but does not recompute curvature scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from lib.reproducibility import set_deterministic, run_metadata  # noqa: E402
from lib.experiment_io import execution_metadata  # noqa: E402
from domains.ffhq.generator import FFHQGenerator  # noqa: E402
from baselines.run_editing_head_to_head import (  # noqa: E402
    LAYERS_FOR,
    collect_pool,
    load_clip,
    clip_text_feature,
)


ATTR_PROMPTS = {
    "smile": ("a smiling face with teeth", "a face with closed mouth"),
    "age": ("an old face with wrinkles", "a young face, smooth skin"),
    "pose": ("a face in side profile", "a frontal face"),
    "gender": ("a male face with a beard", "a female face"),
    "eyeglasses": ("a face wearing glasses", "a face without glasses"),
}

GROUPS = ["curvature_low", "random", "curvature_high"]


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def normalize_img_for_clip(img: torch.Tensor,
                           mean: torch.Tensor,
                           std: torch.Tensor) -> torch.Tensor:
    x = (img.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - mean) / std


@torch.no_grad()
def clip_features(model, img: torch.Tensor,
                  mean: torch.Tensor,
                  std: torch.Tensor) -> torch.Tensor:
    x = normalize_img_for_clip(img, mean, std)
    f = model.encode_image(x)
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def precompute_originals(G: FFHQGenerator,
                         clip_model,
                         wp: torch.Tensor,
                         clip_mean: torch.Tensor,
                         clip_std: torch.Tensor,
                         batch: int,
                         keep_downsample: bool) -> dict[str, Any]:
    feats = []
    down = []
    for s in range(0, wp.shape[0], batch):
        cur = wp[s:s + batch]
        img = G.synthesize(cur).clamp(-1, 1)
        feats.append(clip_features(clip_model, img, clip_mean, clip_std).cpu())
        if keep_downsample:
            down.append(F.interpolate(img, size=(256, 256),
                                      mode="bilinear",
                                      align_corners=False).cpu())
        torch.cuda.empty_cache()
    return {
        "clip_features": torch.cat(feats, dim=0),
        "downsampled": torch.cat(down, dim=0) if keep_downsample else None,
    }


def layered_direction(G: FFHQGenerator, attr: str, v_np: np.ndarray) -> torch.Tensor:
    v = torch.from_numpy(v_np).to(G.device).float()
    v = v / v.norm().clamp_min(1e-8)
    bl = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for li in LAYERS_FOR[attr]:
        bl[li] = v
    return bl


@torch.no_grad()
def evaluate_direction(
    G: FFHQGenerator,
    clip_model,
    wp: torch.Tensor,
    orig_clip: torch.Tensor,
    orig_down: torch.Tensor | None,
    attr_vec: torch.Tensor,
    b_layered: torch.Tensor,
    alpha: float,
    clip_mean: torch.Tensor,
    clip_std: torch.Tensor,
    batch: int,
    lpips_fn=None,
    lpips_size: int = 256,
) -> dict[str, float]:
    attr_vec_cpu = attr_vec.detach().cpu()
    scores0 = (orig_clip @ attr_vec_cpu).numpy()
    scores1 = []
    id_cos = []
    drift = []
    lpips_true = []
    for s in range(0, wp.shape[0], batch):
        cur = wp[s:s + batch]
        img = G.synthesize(cur + alpha * b_layered.unsqueeze(0)).clamp(-1, 1)
        feat = clip_features(clip_model, img, clip_mean, clip_std).cpu()
        scores1.extend((feat @ attr_vec_cpu).numpy().tolist())
        id_cos.extend((orig_clip[s:s + batch] * feat).sum(dim=1).numpy().tolist())
        if orig_down is not None:
            edit_down = F.interpolate(img, size=(256, 256), mode="bilinear",
                                      align_corners=False).cpu()
            drift.extend(((edit_down - orig_down[s:s + batch]) ** 2)
                         .mean(dim=(1, 2, 3)).numpy().tolist())
            if lpips_fn is not None:
                lpips_orig = orig_down[s:s + batch].to(G.device)
                lpips_edit = edit_down.to(G.device)
                if lpips_size != 256:
                    lpips_orig = F.interpolate(
                        lpips_orig, size=(lpips_size, lpips_size),
                        mode="bilinear", align_corners=False)
                    lpips_edit = F.interpolate(
                        lpips_edit, size=(lpips_size, lpips_size),
                        mode="bilinear", align_corners=False)
                vals = lpips_fn(lpips_edit, lpips_orig).detach().view(-1).cpu()
                lpips_true.extend(vals.numpy().tolist())
        torch.cuda.empty_cache()
    scores1_np = np.asarray(scores1, dtype=float)
    scores0_np = np.asarray(scores0, dtype=float)
    delta = scores1_np - scores0_np
    out = {
        "mean_delta_attr": float(delta.mean()),
        "mean_abs_delta_attr": float(np.abs(delta).mean()),
        "mean_id_cos": float(np.mean(id_cos)),
        "mean_lpips_proxy": float(np.mean(drift)) if drift else float("nan"),
    }
    if lpips_true:
        out["mean_lpips_true"] = float(np.mean(lpips_true))
    return out


def calibrate_alpha(
    G: FFHQGenerator,
    clip_model,
    wp: torch.Tensor,
    orig_clip: torch.Tensor,
    attr_vec: torch.Tensor,
    b_layered: torch.Tensor,
    alpha_grid: list[float],
    clip_mean: torch.Tensor,
    clip_std: torch.Tensor,
    batch: int,
) -> dict[str, Any]:
    curves = []
    for sign in [1.0, -1.0]:
        vals = []
        signed_vals = []
        for mag in alpha_grid:
            r = evaluate_direction(
                G, clip_model, wp, orig_clip, None, attr_vec, b_layered,
                alpha=sign * mag, clip_mean=clip_mean, clip_std=clip_std,
                batch=batch,
            )
            vals.append(abs(r["mean_delta_attr"]))
            signed_vals.append(r["mean_delta_attr"])
        curves.append({"sign": sign, "abs_delta": vals, "signed_delta": signed_vals})
    best = max(curves, key=lambda c: c["abs_delta"][-1])
    return {
        "sign": best["sign"],
        "alpha_grid": alpha_grid,
        "abs_delta_grid": best["abs_delta"],
        "signed_delta_grid": best["signed_delta"],
        "max_abs_delta": float(best["abs_delta"][-1]),
    }


def alpha_for_target(calib: dict[str, Any], target: float) -> tuple[float, bool]:
    grid = calib["alpha_grid"]
    vals = calib["abs_delta_grid"]
    sign = float(calib["sign"])
    if target <= vals[0]:
        return sign * grid[0], True
    for i in range(1, len(grid)):
        if vals[i] >= target:
            lo_a, hi_a = grid[i - 1], grid[i]
            lo_v, hi_v = vals[i - 1], vals[i]
            if hi_v <= lo_v:
                return sign * hi_a, True
            t = (target - lo_v) / max(hi_v - lo_v, 1e-12)
            return sign * (lo_a + t * (hi_a - lo_a)), True
    return sign * grid[-1], False


def reconstruct_all_dirs(G: FFHQGenerator, source_config: dict[str, Any]) -> np.ndarray:
    latentclr = resolve_paper_path(
        source_config.get("latentclr_path", "experiments/out/latentclr_ffhq/directions.npy")
    )
    disco = resolve_paper_path(
        source_config.get("disco_path", "experiments/out/disco_ffhq/directions.npy")
    )
    pool = collect_pool(
        G,
        ["smile", "age", "pose", "gender", "eyeglasses"],
        latentclr_path=latentclr,
        disco_path=disco,
    )
    return np.concatenate([pool[m] for m in pool], axis=0)


def select_saved_indices(raw: dict[str, Any], attr: str, group: str,
                         k_per_group: int) -> list[dict[str, Any]]:
    rows = raw["per_attr"][attr][group]
    return rows[:k_per_group]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="experiments/out/editing_head_to_head/metrics.json")
    ap.add_argument("--out", default="experiments/out/matched_editing_head_to_head")
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--k-per-group", type=int, default=8)
    ap.add_argument("--max-alpha", type=float, default=4.0)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lod", type=float, default=2.0,
                    help="FFHQ lod override. 2.0 renders at 256px; use -1 for native 1024px.")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="direction_control")
    args = ap.parse_args()

    if not 0.0 <= args.target_quantile <= 1.0:
        raise ValueError("--target-quantile must be in [0, 1]")
    set_deterministic(args.seed)

    source_path = resolve_paper_path(args.source)
    out = resolve_paper_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads(source_path.read_text())

    lod_override = None if args.lod < 0 else args.lod
    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator lod={lod_override}...")
    G = FFHQGenerator(lod_override=lod_override)
    device = G.device

    print(f"[{time.strftime('%H:%M:%S')}] reconstructing candidate pool...")
    all_dirs = reconstruct_all_dirs(G, raw.get("config", {}))
    print(f"  all_dirs={all_dirs.shape}")

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP...")
    clip_model, clip_tok = load_clip(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=device).view(1, 3, 1, 1)
    attr_vecs = {}
    for attr, (pos, neg) in ATTR_PROMPTS.items():
        f_pos = clip_text_feature(clip_model, clip_tok, device, [pos]).squeeze(0)
        f_neg = clip_text_feature(clip_model, clip_tok, device, [neg]).squeeze(0)
        v = f_pos - f_neg
        attr_vecs[attr] = v / v.norm().clamp_min(1e-8)

    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 17)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 29)
    calib_wp = G.sample_wp(args.n_calib, generator=gen_calib)
    test_wp = G.sample_wp(args.n_test, generator=gen_test)

    print(f"[{time.strftime('%H:%M:%S')}] precomputing originals...")
    calib_orig = precompute_originals(
        G, clip_model, calib_wp, clip_mean, clip_std, args.batch,
        keep_downsample=False,
    )
    test_orig = precompute_originals(
        G, clip_model, test_wp, clip_mean, clip_std, args.batch,
        keep_downsample=True,
    )

    alpha_grid = np.linspace(0.0, args.max_alpha, args.alpha_steps).tolist()
    results: dict[str, Any] = {}

    for attr in args.attrs:
        print(f"\n=== {attr} ===")
        attr_vec = attr_vecs[attr]
        selected = []
        for group in GROUPS:
            for saved in select_saved_indices(raw, attr, group, args.k_per_group):
                idx = int(saved["k"])
                selected.append({
                    "group": group,
                    "index": idx,
                    "rho_this_dir": float(saved["rho_this_dir"]),
                    "direction": layered_direction(G, attr, all_dirs[idx]),
                })

        print(f"  calibrating {len(selected)} directions...")
        max_deltas = []
        for item in selected:
            calib = calibrate_alpha(
                G, clip_model, calib_wp, calib_orig["clip_features"],
                attr_vec, item["direction"], alpha_grid, clip_mean, clip_std,
                args.batch,
            )
            item["calibration"] = calib
            max_deltas.append(calib["max_abs_delta"])
        target = float(np.quantile(max_deltas, args.target_quantile))
        print(f"  target_abs_delta={target:.5f} "
              f"(q={args.target_quantile}, max_delta median={np.median(max_deltas):.5f})")

        attr_rows = {g: [] for g in GROUPS}
        for i, item in enumerate(selected, start=1):
            alpha, reached = alpha_for_target(item["calibration"], target)
            eval_row = evaluate_direction(
                G, clip_model, test_wp, test_orig["clip_features"],
                test_orig["downsampled"], attr_vec, item["direction"],
                alpha=alpha, clip_mean=clip_mean, clip_std=clip_std,
                batch=args.batch,
            )
            row = {
                "index": item["index"],
                "group": item["group"],
                "rho_this_dir": item["rho_this_dir"],
                "target_abs_delta_attr": target,
                "alpha": float(alpha),
                "target_reached_on_calib": bool(reached),
                "calib_max_abs_delta_attr": item["calibration"]["max_abs_delta"],
                **eval_row,
            }
            attr_rows[item["group"]].append(row)
            if i % 8 == 0:
                print(f"    evaluated {i}/{len(selected)}")

        summary = {}
        for group, rows in attr_rows.items():
            summary[group] = {
                "n": len(rows),
                "target_hit_rate_calib": float(np.mean([
                    r["target_reached_on_calib"] for r in rows
                ])),
                "mean_abs_delta_attr": float(np.mean([
                    r["mean_abs_delta_attr"] for r in rows
                ])),
                "mean_id_cos": float(np.mean([r["mean_id_cos"] for r in rows])),
                "mean_lpips_proxy": float(np.mean([
                    r["mean_lpips_proxy"] for r in rows
                ])),
                "mean_abs_alpha": float(np.mean([abs(r["alpha"]) for r in rows])),
                "mean_rho_this_dir": float(np.mean([
                    r["rho_this_dir"] for r in rows
                ])),
            }
            print(f"  {group:16s} |Δattr|={summary[group]['mean_abs_delta_attr']:.5f} "
                  f"ID={summary[group]['mean_id_cos']:.4f} "
                  f"LPIPS={summary[group]['mean_lpips_proxy']:.5f}")

        results[attr] = {
            "target_abs_delta_attr": target,
            "rows": attr_rows,
            "summary": summary,
        }
        payload_partial = {
            "per_attr": results,
            "config": vars(args),
            "_meta": run_metadata(seed=args.seed, extra={
                "script": "experiments/baselines/run_matched_editing_head_to_head.py",
                "source": str(source_path),
                "lod_override": lod_override,
                "execution": execution_metadata(
                    protocol=args.protocol,
                    protocol_key=args.protocol_key,
                ),
            }),
        }
        (out / "metrics_partial.json").write_text(json.dumps(payload_partial, indent=2))

    payload = {
        "per_attr": results,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/baselines/run_matched_editing_head_to_head.py",
            "source": str(source_path),
            "lod_override": lod_override,
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
