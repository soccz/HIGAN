"""Test whether curvature/risk predicts edit damage after controlling gain.

This is not a controller sweep.  It evaluates every predeclared candidate in a
domain/attribute universe at a matched semantic target, then asks whether rho
predicts identity/perceptual damage beyond semantic gain.  The experiment is
designed to address the main-paper concern that a controller may tie or lose to
gain-only while the risk signal can still be causally useful.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from control.run_cross_domain_risk_aware_controller import (  # noqa: E402
    alpha_for_target,
    build_candidate_pool,
    calibrate_alpha,
    clip_text_feature,
    curvature_ratio_for_direction,
    dino_features,
    evaluate_direction,
    layered_direction,
    load_clip,
    load_domain_config,
    precompute_originals,
    resolve_paper_path,
)
from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


def styled_prompt_pair(pos: str, neg: str, style: str) -> tuple[str, str]:
    if style == "default":
        return pos, neg
    if style == "photo":
        return f"a realistic photo of {pos}", f"a realistic photo of {neg}"
    if style == "caption":
        return f"an image captioned: {pos}", f"an image captioned: {neg}"
    raise ValueError(f"unknown prompt style: {style}")


def styled_prompt_pairs(pos: str, neg: str, style: str) -> list[tuple[str, str]]:
    if style == "ensemble":
        return [
            styled_prompt_pair(pos, neg, "default"),
            styled_prompt_pair(pos, neg, "photo"),
            styled_prompt_pair(pos, neg, "caption"),
        ]
    return [styled_prompt_pair(pos, neg, style)]


@torch.no_grad()
def curvature_ratio_for_direction_fd(G, wp: torch.Tensor,
                                     b_layered: torch.Tensor,
                                     eps: float) -> float:
    ratios = []
    for idx in range(wp.shape[0]):
        cur = wp[idx:idx + 1].detach()
        xp = G.synthesize(cur + eps * b_layered.unsqueeze(0))
        x0 = G.synthesize(cur)
        xn = G.synthesize(cur - eps * b_layered.unsqueeze(0))
        first = (xp - xn) / (2.0 * eps)
        second = (xp - 2.0 * x0 + xn) / (eps ** 2)
        first_map = first.abs().mean(dim=1).squeeze(0)
        second_map = second.abs().mean(dim=1).squeeze(0)
        ratios.append(float((second_map / (first_map + 1e-6)).mean().item()))
        torch.cuda.empty_cache()
    return float(np.mean(ratios))


def load_dino(device: torch.device, model_name: str,
              local_files_only: bool):
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).eval().to(device)


@torch.no_grad()
def precompute_dino_originals(G, dino_model, wp: torch.Tensor,
                              batch: int) -> torch.Tensor:
    feats = []
    for start in range(0, wp.shape[0], batch):
        img = G.synthesize(wp[start:start + batch]).clamp(-1, 1)
        feats.append(dino_features(dino_model, img).cpu())
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = rankdata(xa[mask])
    yr = rankdata(ya[mask])
    if float(xr.std()) == 0.0 or float(yr.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def standardized_beta(rows: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    x_keys = ["mean_abs_delta_attr", "probe_gain", "rho"]
    x = np.asarray([[float(r[k]) for k in x_keys] for r in rows], dtype=float)
    y = np.asarray([float(r[y_key]) for r in rows], dtype=float)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[mask]
    y = y[mask]
    if x.shape[0] <= len(x_keys):
        return {"n": int(x.shape[0]), "beta_rho": float("nan")}
    x_std = x.std(axis=0)
    y_std = float(y.std())
    keep = x_std > 1e-12
    if y_std <= 1e-12 or not keep[-1]:
        return {"n": int(x.shape[0]), "beta_rho": float("nan")}
    xz = (x[:, keep] - x[:, keep].mean(axis=0)) / x_std[keep]
    yz = (y - y.mean()) / y_std
    design = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)
    kept_keys = [k for k, ok in zip(x_keys, keep) if ok]
    beta_by_key = {k: float(v) for k, v in zip(kept_keys, coef[1:])}
    return {
        "n": int(x.shape[0]),
        "predictors": kept_keys,
        "beta_rho": beta_by_key.get("rho", float("nan")),
        "standardized_betas": beta_by_key,
    }


def summarize_values(values: list[float], orientation: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": std,
        "sem": sem,
        "ci95_normal": [
            float(arr.mean() - 1.96 * sem),
            float(arr.mean() + 1.96 * sem),
        ],
        "orientation": orientation,
    }
    if orientation == "higher":
        out["wins"] = int((arr > 0).sum())
    elif orientation == "lower":
        out["wins"] = int((arr < 0).sum())
    return out


def build_matched_pairs(rows: list[dict[str, Any]],
                        gain_match_rel: float) -> list[dict[str, Any]]:
    out = []
    for lhs, rhs in itertools.combinations(rows, 2):
        gain_scale = max(abs(float(lhs["probe_gain"])),
                         abs(float(rhs["probe_gain"])), 1e-8)
        if abs(float(lhs["probe_gain"]) - float(rhs["probe_gain"])) > (
                gain_match_rel * gain_scale):
            continue
        if float(lhs["rho"]) == float(rhs["rho"]):
            continue
        low, high = (lhs, rhs) if float(lhs["rho"]) < float(rhs["rho"]) else (rhs, lhs)
        row = {
            "attr": low["attr"],
            "low_candidate": low["candidate_index"],
            "high_candidate": high["candidate_index"],
            "low_source": low["source_group"],
            "high_source": high["source_group"],
            "probe_gain_abs_diff": abs(float(low["probe_gain"]) - float(high["probe_gain"])),
            "rho_diff_low_minus_high": float(low["rho"]) - float(high["rho"]),
            "mean_abs_delta_attr_diff_low_minus_high": (
                float(low["mean_abs_delta_attr"]) -
                float(high["mean_abs_delta_attr"])
            ),
            "mean_id_cos_diff_low_minus_high": (
                float(low["mean_id_cos"]) - float(high["mean_id_cos"])
            ),
            "mean_lpips_proxy_diff_low_minus_high": (
                float(low["mean_lpips_proxy"]) -
                float(high["mean_lpips_proxy"])
            ),
        }
        if "mean_lpips_true" in low and "mean_lpips_true" in high:
            row["mean_lpips_true_diff_low_minus_high"] = (
                float(low["mean_lpips_true"]) -
                float(high["mean_lpips_true"])
            )
        if "mean_dino_cos" in low and "mean_dino_cos" in high:
            row["mean_dino_cos_diff_low_minus_high"] = (
                float(low["mean_dino_cos"]) -
                float(high["mean_dino_cos"])
            )
        out.append(row)
    return out


def aggregate_rows(rows: list[dict[str, Any]],
                   matched_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    lpips_key = (
        "mean_lpips_true"
        if rows and all("mean_lpips_true" in r for r in rows)
        else "mean_lpips_proxy"
    )
    result = {
        "n_rows": len(rows),
        "n_matched_pairs": len(matched_pairs),
        "spearman": {
            "rho_vs_id": {
                "r": spearman([r["rho"] for r in rows],
                              [r["mean_id_cos"] for r in rows]),
                "orientation": "lower",
            },
            "rho_vs_lpips": {
                "r": spearman([r["rho"] for r in rows],
                              [r[lpips_key] for r in rows]),
                "orientation": "higher",
                "metric": lpips_key,
            },
        },
        "regression": {
            "id": {
                **standardized_beta(rows, "mean_id_cos"),
                "orientation": "lower",
            },
            "lpips": {
                **standardized_beta(rows, lpips_key),
                "orientation": "higher",
                "metric": lpips_key,
            },
        },
        "matched_pairs": {
            "mean_id_cos_diff_low_minus_high": summarize_values(
                [p["mean_id_cos_diff_low_minus_high"] for p in matched_pairs],
                "higher",
            ),
            f"{lpips_key}_diff_low_minus_high": summarize_values(
                [p[f"{lpips_key}_diff_low_minus_high"] for p in matched_pairs],
                "lower",
            ),
            "mean_abs_delta_attr_diff_low_minus_high": summarize_values(
                [p["mean_abs_delta_attr_diff_low_minus_high"] for p in matched_pairs],
                "matched",
            ),
        },
    }
    if rows and all("mean_dino_cos" in r for r in rows):
        result["spearman"]["rho_vs_dino"] = {
            "r": spearman([r["rho"] for r in rows],
                          [r["mean_dino_cos"] for r in rows]),
            "orientation": "lower",
            "metric": "mean_dino_cos",
        }
        result["regression"]["dino"] = {
            **standardized_beta(rows, "mean_dino_cos"),
            "orientation": "lower",
            "metric": "mean_dino_cos",
        }
        result["matched_pairs"]["mean_dino_cos_diff_low_minus_high"] = (
            summarize_values(
                [p["mean_dino_cos_diff_low_minus_high"]
                 for p in matched_pairs
                 if "mean_dino_cos_diff_low_minus_high" in p],
                "higher",
            )
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "church", "ffhq"], default="church")
    ap.add_argument("--attrs", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["ganspace", "sefa"])
    ap.add_argument("--candidate-k", type=int, default=6)
    ap.add_argument("--ganspace-samples", type=int, default=2048)
    ap.add_argument("--min-gain-quantile", type=float, default=0.5)
    ap.add_argument("--gain-match-rel", type=float, default=0.25)
    ap.add_argument("--n-risk", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--probe-alpha", type=float, default=1.0)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--max-alpha", type=float, default=6.0)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--true-lpips", action="store_true")
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--lpips-size", type=int, default=256)
    ap.add_argument("--risk-estimator", choices=["jvp", "fd"], default="jvp")
    ap.add_argument("--fd-eps", type=float, default=0.25)
    ap.add_argument("--dino-preservation", action="store_true")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--dino-local-files-only", action="store_true")
    ap.add_argument(
        "--prompt-style",
        choices=["default", "photo", "caption", "ensemble"],
        default="default",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2037)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="cross_domain_risk_predictive")
    args = ap.parse_args()

    if not 0.0 <= args.target_quantile <= 1.0:
        raise ValueError("--target-quantile must be in [0, 1]")
    if not 0.0 <= args.min_gain_quantile <= 1.0:
        raise ValueError("--min-gain-quantile must be in [0, 1]")
    if args.gain_match_rel < 0.0:
        raise ValueError("--gain-match-rel must be non-negative")
    if args.fd_eps <= 0.0:
        raise ValueError("--fd-eps must be positive")

    set_deterministic(args.seed)
    out = resolve_paper_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.domain} generator")
    G, layers_for, default_attrs, prompts = load_domain_config(args.domain)
    attrs = args.attrs or default_attrs
    unknown_attrs = sorted(set(attrs) - set(default_attrs))
    if unknown_attrs:
        raise ValueError(f"unknown attrs for domain {args.domain}: {unknown_attrs}")
    device = G.device

    print(f"[{time.strftime('%H:%M:%S')}] building candidate pool")
    base_pool = build_candidate_pool(
        G, args.methods, args.candidate_k, args.ganspace_samples, args.seed
    )
    print(f"  candidates={len(base_pool)} methods={args.methods}")

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP")
    clip_model, clip_tok = load_clip(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=device).view(1, 3, 1, 1)
    attr_vecs = {}
    for attr in attrs:
        pos, neg = prompts[attr]
        deltas = []
        for styled_pos, styled_neg in styled_prompt_pairs(pos, neg, args.prompt_style):
            f_pos = clip_text_feature(
                clip_model, clip_tok, device, [styled_pos]).squeeze(0)
            f_neg = clip_text_feature(
                clip_model, clip_tok, device, [styled_neg]).squeeze(0)
            deltas.append(f_pos - f_neg)
        v = torch.stack(deltas, dim=0).mean(dim=0)
        attr_vecs[attr] = v / v.norm().clamp_min(1e-8)

    lpips_fn = None
    if args.true_lpips:
        print(f"[{time.strftime('%H:%M:%S')}] loading LPIPS net={args.lpips_net}")
        import lpips

        lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).eval().to(device)
        for param in lpips_fn.parameters():
            param.requires_grad_(False)

    dino_model = None
    if args.dino_preservation:
        print(f"[{time.strftime('%H:%M:%S')}] loading DINO model={args.dino_model}")
        dino_model = load_dino(
            device,
            args.dino_model,
            args.dino_local_files_only,
        )
        for param in dino_model.parameters():
            param.requires_grad_(False)

    gen_risk = torch.Generator(device=device).manual_seed(args.seed + 503)
    gen_probe = torch.Generator(device=device).manual_seed(args.seed + 601)
    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 701)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 809)
    risk_wp = G.sample_wp(args.n_risk, generator=gen_risk)
    probe_wp = G.sample_wp(args.n_probe, generator=gen_probe)
    calib_wp = G.sample_wp(args.n_calib, generator=gen_calib)
    test_wp = G.sample_wp(args.n_test, generator=gen_test)

    print(f"[{time.strftime('%H:%M:%S')}] precomputing originals")
    probe_orig = precompute_originals(
        G, clip_model, probe_wp, clip_mean, clip_std, args.batch,
        keep_downsample=False,
    )
    calib_orig = precompute_originals(
        G, clip_model, calib_wp, clip_mean, clip_std, args.batch,
        keep_downsample=False,
    )
    test_orig = precompute_originals(
        G, clip_model, test_wp, clip_mean, clip_std, args.batch,
        keep_downsample=True,
    )
    test_dino_orig = None
    if dino_model is not None:
        print(f"[{time.strftime('%H:%M:%S')}] precomputing DINO originals")
        test_dino_orig = precompute_dino_originals(
            G, dino_model, test_wp, args.batch)

    alpha_grid = np.linspace(0.0, args.max_alpha, args.alpha_steps).tolist()
    probe_grid = [0.0, args.probe_alpha]
    all_rows: list[dict[str, Any]] = []
    all_matched_pairs: list[dict[str, Any]] = []
    per_attr: dict[str, Any] = {}

    for attr in attrs:
        print(f"\n=== {attr} ===")
        candidates = []
        for item in base_pool:
            direction = layered_direction(G, layers_for, attr, item["vector"])
            if args.risk_estimator == "jvp":
                rho = curvature_ratio_for_direction(G, risk_wp, direction)
            else:
                rho = curvature_ratio_for_direction_fd(
                    G, risk_wp, direction, args.fd_eps)
            probe_calib = calibrate_alpha(
                G, clip_model, probe_wp, probe_orig["clip_features"],
                attr_vecs[attr], direction, probe_grid, clip_mean, clip_std,
                args.batch,
            )
            candidates.append({
                "index": int(item["index"]),
                "source_group": item["source_group"],
                "source_rank": int(item["source_rank"]),
                "rho": rho,
                "probe_gain": probe_calib["max_abs_delta"],
                "probe_sign": probe_calib["sign"],
                "direction": direction,
            })

        calibrations = {}
        max_deltas = []
        for idx, cand in enumerate(candidates):
            calib = calibrate_alpha(
                G, clip_model, calib_wp, calib_orig["clip_features"],
                attr_vecs[attr], cand["direction"], alpha_grid, clip_mean,
                clip_std, args.batch,
            )
            calibrations[idx] = calib
            max_deltas.append(calib["max_abs_delta"])
        target = float(np.quantile(max_deltas, args.target_quantile))
        gain_floor = float(np.quantile(
            [c["probe_gain"] for c in candidates], args.min_gain_quantile))

        rows = []
        for idx, cand in enumerate(candidates):
            alpha, reached = alpha_for_target(calibrations[idx], target)
            row_eval = evaluate_direction(
                G, clip_model, test_wp, test_orig["clip_features"],
                test_orig["downsampled"], attr_vecs[attr], cand["direction"],
                alpha, clip_mean, clip_std, args.batch,
                lpips_fn=lpips_fn, lpips_size=args.lpips_size,
                dino_model=dino_model, orig_dino=test_dino_orig,
            )
            row = {
                "seed": args.seed,
                "domain": args.domain,
                "attr": attr,
                "candidate_index": int(cand["index"]),
                "source_group": cand["source_group"],
                "source_rank": int(cand["source_rank"]),
                "rho": float(cand["rho"]),
                "probe_gain": float(cand["probe_gain"]),
                "probe_sign": float(cand["probe_sign"]),
                "alpha": float(alpha),
                "target_abs_delta_attr": target,
                "target_reached_on_calib": bool(reached),
                "calib_max_abs_delta_attr": float(calibrations[idx]["max_abs_delta"]),
                "gain_eligible": bool(float(cand["probe_gain"]) >= gain_floor),
                **row_eval,
            }
            rows.append(row)
            all_rows.append(row)

        eligible_rows = [r for r in rows if r["gain_eligible"]]
        matched_pairs = build_matched_pairs(eligible_rows, args.gain_match_rel)
        all_matched_pairs.extend(matched_pairs)
        per_attr[attr] = {
            "target_abs_delta_attr": target,
            "gain_floor": gain_floor,
            "rows": rows,
            "matched_pairs": matched_pairs,
            "aggregate": aggregate_rows(eligible_rows, matched_pairs),
        }
        agg = per_attr[attr]["aggregate"]
        lpips_pair_key = next(k for k in agg["matched_pairs"] if "lpips" in k)
        print(f"  eligible={len(eligible_rows)} matched_pairs={len(matched_pairs)} "
              f"rho->LPIPS beta={agg['regression']['lpips']['beta_rho']:+.3f} "
              f"pair ΔID={agg['matched_pairs']['mean_id_cos_diff_low_minus_high'].get('mean', float('nan')):+.4f} "
              f"pair ΔLPIPS={agg['matched_pairs'][lpips_pair_key].get('mean', float('nan')):+.4f}")

        partial = {
            "per_attr": per_attr,
            "aggregate": aggregate_rows(
                [r for r in all_rows if r["gain_eligible"]],
                all_matched_pairs,
            ),
            "config": vars(args),
            "_meta": run_metadata(seed=args.seed, extra={
                "script": "experiments/control/run_cross_domain_risk_predictive.py",
                "execution": execution_metadata(
                    protocol=args.protocol,
                    protocol_key=args.protocol_key,
                ),
            }),
        }
        (out / "metrics_partial.json").write_text(json.dumps(partial, indent=2))

    payload = {
        "per_attr": per_attr,
        "aggregate": aggregate_rows(
            [r for r in all_rows if r["gain_eligible"]],
            all_matched_pairs,
        ),
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_cross_domain_risk_predictive.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    write_json_atomic(out / "metrics.json", payload)
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
