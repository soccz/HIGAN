"""Actual-edit composition guard for FFHQ.

The C4 track measured saliency superposition failure. This experiment measures
whether the same mixed-Hessian predictor forecasts failures in real composed
edits, then reports a simple guard: reject high-risk pairs before composition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from lib.reproducibility import set_deterministic, run_metadata  # noqa: E402
from lib.experiment_io import execution_metadata  # noqa: E402
from domains.ffhq.generator import FFHQGenerator  # noqa: E402
from baselines.run_matched_editing_head_to_head import (  # noqa: E402
    ATTR_PROMPTS,
    LAYERS_FOR,
    load_clip,
    clip_text_feature,
    clip_features,
    precompute_originals,
    calibrate_alpha,
    alpha_for_target,
)


ATTRS = ["smile", "age", "pose", "gender", "eyeglasses"]


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_boundary(attr: str, G: FFHQGenerator) -> torch.Tensor:
    bpath = (PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
             / f"stylegan_ffhq_{attr}_w_boundary.npy")
    vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
    b = torch.from_numpy(vec).to(G.device).float()
    b = b / b.norm().clamp_min(1e-8)
    out = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for li in LAYERS_FOR[attr]:
        out[li] = b
    return out


def latest_pixel_rhos(path: Path) -> dict[str, float]:
    metrics = load_json(path)
    table = metrics["bootstrap_ci"]
    latest_n = max(int(k) for k in table)
    return {a: float(v["mean"]) for a, v in table[str(latest_n)].items()}


def c4_predictors(path: Path) -> dict[tuple[str, str], float]:
    metrics = load_json(path)
    out = {}
    for row in metrics["pairs"]:
        key = tuple(sorted((row["a"], row["b"])))
        out[key] = float(row["predictor"])
    return out


def saliency_overlap(attr_a: str, attr_b: str, saliency_dir: Path) -> dict[str, float]:
    a = np.load(saliency_dir / f"{attr_a}_raw.npz")["mean_sal"].astype(np.float64)
    b = np.load(saliency_dir / f"{attr_b}_raw.npz")["mean_sal"].astype(np.float64)
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    cosine = float(np.dot(flat_a, flat_b) /
                   (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12))
    ma = flat_a >= np.quantile(flat_a, 0.80)
    mb = flat_b >= np.quantile(flat_b, 0.80)
    iou = float(np.logical_and(ma, mb).sum() /
                max(np.logical_or(ma, mb).sum(), 1))
    return {"saliency_cosine": cosine, "saliency_top20_iou": iou}


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += float(p > n) + 0.5 * float(p == n)
    return wins / (len(pos) * len(neg))


def safe_corr(fn, x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"r": None, "p": None}
    r, p = fn(x, y)
    return {"r": float(r), "p": float(p)}


@torch.no_grad()
def evaluate_pair(
    G: FFHQGenerator,
    clip_model,
    wp: torch.Tensor,
    orig_clip: torch.Tensor,
    orig_down: torch.Tensor,
    attr_vecs: dict[str, torch.Tensor],
    attr_a: str,
    attr_b: str,
    dir_a: torch.Tensor,
    dir_b: torch.Tensor,
    alpha_a: float,
    alpha_b: float,
    clip_mean: torch.Tensor,
    clip_std: torch.Tensor,
    batch: int,
) -> dict[str, float]:
    vec_a = attr_vecs[attr_a].detach().cpu()
    vec_b = attr_vecs[attr_b].detach().cpu()
    score0_a = (orig_clip @ vec_a).numpy()
    score0_b = (orig_clip @ vec_b).numpy()

    delta_a_self = []
    delta_b_self = []
    delta_ab_a = []
    delta_ab_b = []
    id_drop_ab = []
    lp_a = []
    lp_b = []
    lp_ab = []

    for s in range(0, wp.shape[0], batch):
        cur = wp[s:s + batch]
        img_a = G.synthesize(cur + alpha_a * dir_a.unsqueeze(0)).clamp(-1, 1)
        img_b = G.synthesize(cur + alpha_b * dir_b.unsqueeze(0)).clamp(-1, 1)
        img_ab = G.synthesize(
            cur + alpha_a * dir_a.unsqueeze(0) + alpha_b * dir_b.unsqueeze(0)
        ).clamp(-1, 1)

        feat_a = clip_features(clip_model, img_a, clip_mean, clip_std).cpu()
        feat_b = clip_features(clip_model, img_b, clip_mean, clip_std).cpu()
        feat_ab = clip_features(clip_model, img_ab, clip_mean, clip_std).cpu()
        sl = slice(s, s + cur.shape[0])
        delta_a_self.extend(((feat_a @ vec_a).numpy() - score0_a[sl]).tolist())
        delta_b_self.extend(((feat_b @ vec_b).numpy() - score0_b[sl]).tolist())
        delta_ab_a.extend(((feat_ab @ vec_a).numpy() - score0_a[sl]).tolist())
        delta_ab_b.extend(((feat_ab @ vec_b).numpy() - score0_b[sl]).tolist())
        id_drop_ab.extend((1.0 - (orig_clip[sl] * feat_ab).sum(dim=1).numpy()).tolist())

        down_a = F.interpolate(img_a, size=(256, 256), mode="bilinear",
                               align_corners=False).cpu()
        down_b = F.interpolate(img_b, size=(256, 256), mode="bilinear",
                               align_corners=False).cpu()
        down_ab = F.interpolate(img_ab, size=(256, 256), mode="bilinear",
                                align_corners=False).cpu()
        base = orig_down[sl]
        lp_a.extend(((down_a - base) ** 2).mean(dim=(1, 2, 3)).numpy().tolist())
        lp_b.extend(((down_b - base) ** 2).mean(dim=(1, 2, 3)).numpy().tolist())
        lp_ab.extend(((down_ab - base) ** 2).mean(dim=(1, 2, 3)).numpy().tolist())
        torch.cuda.empty_cache()

    da = np.asarray(delta_a_self, dtype=float)
    db = np.asarray(delta_b_self, dtype=float)
    dab_a = np.asarray(delta_ab_a, dtype=float)
    dab_b = np.asarray(delta_ab_b, dtype=float)
    eps = 1e-6
    add_err_a = np.abs(dab_a - da) / (np.abs(da) + eps)
    add_err_b = np.abs(dab_b - db) / (np.abs(db) + eps)
    target_additivity_error = 0.5 * (add_err_a + add_err_b)
    lp_a_np = np.asarray(lp_a, dtype=float)
    lp_b_np = np.asarray(lp_b, dtype=float)
    lp_ab_np = np.asarray(lp_ab, dtype=float)
    lp_nonadd = np.abs(lp_ab_np - (lp_a_np + lp_b_np))

    return {
        "mean_target_additivity_error": float(target_additivity_error.mean()),
        "median_target_additivity_error": float(np.median(target_additivity_error)),
        "mean_identity_drop_ab": float(np.mean(id_drop_ab)),
        "mean_lpips_nonadditivity": float(lp_nonadd.mean()),
        "mean_lpips_ab": float(lp_ab_np.mean()),
        "mean_abs_delta_a_self": float(np.abs(da).mean()),
        "mean_abs_delta_b_self": float(np.abs(db).mean()),
        "mean_abs_delta_ab_a": float(np.abs(dab_a).mean()),
        "mean_abs_delta_ab_b": float(np.abs(dab_b).mean()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [r["mean_target_additivity_error"] for r in rows]
    labels = [v >= np.quantile(primary, 0.50) for v in primary]
    metrics = {
        "mixed_hessian_predictor": [r["mixed_hessian_predictor"] for r in rows],
        "max_univariate_pixel_rho": [r["max_univariate_pixel_rho"] for r in rows],
        "sum_univariate_pixel_rho": [r["sum_univariate_pixel_rho"] for r in rows],
        "saliency_top20_iou": [r["saliency_top20_iou"] for r in rows],
        "saliency_cosine": [r["saliency_cosine"] for r in rows],
    }
    out = {
        "n_pairs": len(rows),
        "failure_threshold_median": float(np.quantile(primary, 0.50)),
        "predictors": {},
    }
    for name, vals in metrics.items():
        out["predictors"][name] = {
            "spearman": safe_corr(spearmanr, vals, primary),
            "pearson": safe_corr(pearsonr, vals, primary),
            "auroc_failure_top50": roc_auc(vals, labels),
        }

    pred = np.asarray(metrics["mixed_hessian_predictor"], dtype=float)
    guard_th = float(np.quantile(pred, 0.50))
    accepted = [r for r in rows if r["mixed_hessian_predictor"] < guard_th]
    rejected = [r for r in rows if r["mixed_hessian_predictor"] >= guard_th]
    out["guard_median_threshold"] = guard_th
    out["accepted_pairs"] = [r["pair"] for r in accepted]
    out["rejected_pairs"] = [r["pair"] for r in rejected]
    out["accepted_mean_failure"] = float(np.mean([
        r["mean_target_additivity_error"] for r in accepted
    ])) if accepted else None
    out["rejected_mean_failure"] = float(np.mean([
        r["mean_target_additivity_error"] for r in rejected
    ])) if rejected else None
    return out


def write_plot(payload: dict[str, Any], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["pairs"]
    x = [r["mixed_hessian_predictor"] for r in rows]
    y = [r["mean_target_additivity_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.scatter(x, y, s=76, color="#2563eb", alpha=0.85,
               edgecolors="white", linewidths=1.0)
    for r in rows:
        ax.annotate(r["pair"], (r["mixed_hessian_predictor"],
                    r["mean_target_additivity_error"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mixed-Hessian predictor")
    ax.set_ylabel("actual target additivity error")
    sp = payload["summary"]["predictors"]["mixed_hessian_predictor"]["spearman"]["r"]
    ax.set_title(f"Actual composition guard, Spearman={sp:+.3f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "composition_guard.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=ATTRS)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--max-alpha", type=float, default=4.0)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--attr-target-fraction", type=float, default=0.5,
                    help="Each single edit is set to this fraction of its own max calibration delta.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lod", type=float, default=2.0)
    ap.add_argument("--c4-source", default="experiments/out/ffhq_c4/metrics.json")
    ap.add_argument("--pixel-source", default="experiments/out/sample_scaling_ffhq_n512/metrics.json")
    ap.add_argument("--saliency-dir", default="experiments/out/ffhq_saliency")
    ap.add_argument("--out", default="experiments/out/control_composition_guard")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="composition_guard")
    args = ap.parse_args()

    set_deterministic(args.seed)
    out = resolve_paper_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c4_source = resolve_paper_path(args.c4_source)
    pixel_source = resolve_paper_path(args.pixel_source)
    saliency_dir = resolve_paper_path(args.saliency_dir)

    lod_override = None if args.lod < 0 else args.lod
    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator lod={lod_override}")
    G = FFHQGenerator(lod_override=lod_override)
    device = G.device

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP")
    clip_model, clip_tok = load_clip(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=device).view(1, 3, 1, 1)
    attr_vecs = {}
    for attr in args.attrs:
        pos, neg = ATTR_PROMPTS[attr]
        f_pos = clip_text_feature(clip_model, clip_tok, device, [pos]).squeeze(0)
        f_neg = clip_text_feature(clip_model, clip_tok, device, [neg]).squeeze(0)
        v = f_pos - f_neg
        attr_vecs[attr] = v / v.norm().clamp_min(1e-8)

    dirs = {attr: load_boundary(attr, G) for attr in args.attrs}
    mixed = c4_predictors(c4_source)
    pixel = latest_pixel_rhos(pixel_source)

    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 401)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 503)
    calib_wp = G.sample_wp(args.n_calib, generator=gen_calib)
    test_wp = G.sample_wp(args.n_test, generator=gen_test)
    calib_orig = precompute_originals(G, clip_model, calib_wp, clip_mean,
                                      clip_std, args.batch,
                                      keep_downsample=False)
    test_orig = precompute_originals(G, clip_model, test_wp, clip_mean,
                                     clip_std, args.batch,
                                     keep_downsample=True)

    alpha_grid = np.linspace(0.0, args.max_alpha, args.alpha_steps).tolist()
    attr_alpha = {}
    attr_calib = {}
    for attr in args.attrs:
        calib = calibrate_alpha(
            G, clip_model, calib_wp, calib_orig["clip_features"],
            attr_vecs[attr], dirs[attr], alpha_grid, clip_mean, clip_std,
            args.batch,
        )
        target = args.attr_target_fraction * calib["max_abs_delta"]
        alpha, reached = alpha_for_target(calib, target)
        attr_alpha[attr] = float(alpha)
        attr_calib[attr] = {
            "alpha": float(alpha),
            "target_abs_delta_attr": float(target),
            "target_reached_on_calib": bool(reached),
            "max_abs_delta_attr": calib["max_abs_delta"],
        }
        print(f"  {attr:12s} alpha={alpha:+.3f} target={target:.5f}")

    rows = []
    for attr_a, attr_b in combinations(args.attrs, 2):
        print(f"pair {attr_a}+{attr_b}")
        key = tuple(sorted((attr_a, attr_b)))
        overlap = saliency_overlap(attr_a, attr_b, saliency_dir)
        eval_row = evaluate_pair(
            G, clip_model, test_wp, test_orig["clip_features"],
            test_orig["downsampled"], attr_vecs, attr_a, attr_b,
            dirs[attr_a], dirs[attr_b], attr_alpha[attr_a], attr_alpha[attr_b],
            clip_mean, clip_std, args.batch,
        )
        row = {
            "pair": f"{attr_a}+{attr_b}",
            "a": attr_a,
            "b": attr_b,
            "mixed_hessian_predictor": mixed[key],
            "max_univariate_pixel_rho": max(pixel[attr_a], pixel[attr_b]),
            "sum_univariate_pixel_rho": pixel[attr_a] + pixel[attr_b],
            **overlap,
            **eval_row,
        }
        rows.append(row)
        print(f"  failure={row['mean_target_additivity_error']:.3f} "
              f"mixed={row['mixed_hessian_predictor']:.6f}")
        partial = {
            "pairs": rows,
            "attr_calibration": attr_calib,
            "config": vars(args),
            "_meta": run_metadata(seed=args.seed, extra={
                "script": "experiments/control/run_composition_guard.py",
                "c4_source": str(c4_source),
                "pixel_source": str(pixel_source),
                "lod_override": lod_override,
                "execution": execution_metadata(
                    protocol=args.protocol,
                    protocol_key=args.protocol_key,
                ),
            }),
        }
        (out / "metrics_partial.json").write_text(json.dumps(partial, indent=2))

    summary = summarize(rows)
    payload = {
        "pairs": rows,
        "summary": summary,
        "attr_calibration": attr_calib,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_composition_guard.py",
            "c4_source": str(c4_source),
            "pixel_source": str(pixel_source),
            "lod_override": lod_override,
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_plot(payload, out)
    sp = summary["predictors"]["mixed_hessian_predictor"]["spearman"]["r"]
    auc = summary["predictors"]["mixed_hessian_predictor"]["auroc_failure_top50"]
    print(f"\nactual composition guard: Spearman={sp:+.3f} AUROC={auc:.3f}")
    print(f"saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
