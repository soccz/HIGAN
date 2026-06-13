"""Curvature-guided layer control for FFHQ latent editing.

This experiment turns the interpretability signal into a control decision:
which W+ layers should receive an edit direction?

For each FFHQ attribute, the script:
  1. probes each single layer for target CLIP gain,
  2. combines that gain with precomputed per-layer curvature rho,
  3. selects a curvature-aware layer subset,
  4. compares it against gain-only, canonical, all-layer, random, and wrong
     layer subsets at matched semantic change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    precompute_originals,
    evaluate_direction,
    calibrate_alpha,
    alpha_for_target,
)


ALL_ATTRS = ["smile", "age", "pose", "gender", "eyeglasses"]
METHODS = [
    "curvature_control",
    "gain_control",
    "low_curvature_control",
    "canonical",
    "all_layers",
    "random_same_size",
    "wrong_layers",
]


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def load_boundary(attr: str, G: FFHQGenerator) -> torch.Tensor:
    bpath = (PAPER / "experiments" / "data" / "interfacegan" / "boundaries"
             / f"stylegan_ffhq_{attr}_w_boundary.npy")
    vec = np.load(bpath, allow_pickle=True).squeeze().astype(np.float32)
    b = torch.from_numpy(vec).to(G.device).float()
    return b / b.norm().clamp_min(1e-8)


def build_layered(G: FFHQGenerator, b_dir: torch.Tensor,
                  layers: list[int]) -> torch.Tensor:
    out = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for li in layers:
        out[li] = b_dir
    return out


def load_per_layer_rho(path: Path, attrs: list[str]) -> dict[str, list[float]]:
    metrics = json.loads(path.read_text())
    out = {}
    for attr in attrs:
        out[attr] = [float(x) for x in metrics["per_attr"][attr]["rho_per_layer"]]
    return out


def topk_desc(values: list[float], k: int) -> list[int]:
    return sorted(np.argsort(-np.asarray(values))[:k].astype(int).tolist())


def bottomk(values: list[float], k: int) -> list[int]:
    return sorted(np.argsort(np.asarray(values))[:k].astype(int).tolist())


def choose_subsets(gain: list[float], rho: list[float],
                   canonical: list[int], k: int, seed: int,
                   responsive_quantile: float) -> dict[str, list[int]]:
    gain_arr = np.asarray(gain, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    score = gain_arr / (rho_arr + 1e-6)
    responsive = np.where(gain_arr >= np.quantile(gain_arr, responsive_quantile))[0]
    if len(responsive) < k:
        responsive = np.argsort(-gain_arr)[:k]
    responsive_score = np.full_like(score, -np.inf)
    responsive_score[responsive] = score[responsive]

    if len(responsive) >= k:
        low_curv = responsive[np.argsort(rho_arr[responsive])[:k]]
        low_curv_layers = sorted(low_curv.astype(int).tolist())
    else:
        low_curv_layers = bottomk(rho, k)

    rng = np.random.default_rng(seed)
    random_layers = sorted(rng.choice(len(gain), size=k, replace=False).astype(int).tolist())

    return {
        "curvature_control": topk_desc(responsive_score.tolist(), k),
        "gain_control": topk_desc(gain, k),
        "low_curvature_control": low_curv_layers,
        "canonical": sorted(canonical),
        "all_layers": list(range(len(gain))),
        "random_same_size": random_layers,
        "wrong_layers": bottomk(score.tolist(), k),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(rows),
        "target_hit_rate_calib": float(np.mean([r["target_reached_on_calib"]
                                                for r in rows])),
        "mean_abs_delta_attr": float(np.mean([r["mean_abs_delta_attr"]
                                              for r in rows])),
        "mean_id_cos": float(np.mean([r["mean_id_cos"] for r in rows])),
        "mean_lpips_proxy": float(np.mean([r["mean_lpips_proxy"]
                                           for r in rows])),
        "mean_abs_alpha": float(np.mean([abs(r["alpha"]) for r in rows])),
    }


def write_plot(payload: dict[str, Any], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attrs = list(payload["per_attr"].keys())
    methods = ["curvature_control", "gain_control", "canonical",
               "random_same_size", "wrong_layers"]
    x = np.arange(len(attrs))
    width = 0.15
    colors = {
        "curvature_control": "#2563eb",
        "gain_control": "#0891b2",
        "canonical": "#16a34a",
        "random_same_size": "#6b7280",
        "wrong_layers": "#c2410c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][method]["mean_id_cos"] for a in attrs],
            width=width, color=colors[method], label=method,
        )
        axes[1].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][method]["mean_lpips_proxy"]
             for a in attrs],
            width=width, color=colors[method], label=method,
        )
    axes[0].set_title("Identity preservation")
    axes[0].set_ylabel("CLIP image cosine")
    axes[1].set_title("Perceptual drift proxy")
    axes[1].set_ylabel("downsampled L2")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "layer_control.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=ALL_ATTRS)
    ap.add_argument("--layer-k", type=int, default=4)
    ap.add_argument("--responsive-quantile", type=float, default=0.75,
                    help="Only layers above this target-gain quantile are eligible for curvature_control.")
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--probe-alpha", type=float, default=1.0)
    ap.add_argument("--max-alpha", type=float, default=6.0)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lod", type=float, default=2.0)
    ap.add_argument("--rho-source", default="experiments/out/per_layer_c1_ffhq/metrics.json")
    ap.add_argument("--out", default="experiments/out/control_layer_intervention")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="layer_control")
    args = ap.parse_args()

    set_deterministic(args.seed)
    out = resolve_paper_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rho_source = resolve_paper_path(args.rho_source)

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

    rho_per_attr = load_per_layer_rho(rho_source, args.attrs)
    boundaries = {attr: load_boundary(attr, G) for attr in args.attrs}

    gen_probe = torch.Generator(device=device).manual_seed(args.seed + 101)
    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 211)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 307)
    probe_wp = G.sample_wp(args.n_probe, generator=gen_probe)
    calib_wp = G.sample_wp(args.n_calib, generator=gen_calib)
    test_wp = G.sample_wp(args.n_test, generator=gen_test)

    print(f"[{time.strftime('%H:%M:%S')}] precomputing originals")
    probe_orig = precompute_originals(G, clip_model, probe_wp, clip_mean,
                                      clip_std, args.batch,
                                      keep_downsample=False)
    calib_orig = precompute_originals(G, clip_model, calib_wp, clip_mean,
                                      clip_std, args.batch,
                                      keep_downsample=False)
    test_orig = precompute_originals(G, clip_model, test_wp, clip_mean,
                                     clip_std, args.batch,
                                     keep_downsample=True)

    alpha_grid = np.linspace(0.0, args.max_alpha, args.alpha_steps).tolist()
    results: dict[str, Any] = {}
    for attr in args.attrs:
        print(f"\n=== {attr} ===")
        b_dir = boundaries[attr]
        gain = []
        for li in range(G.num_layers):
            bl = build_layered(G, b_dir, [li])
            row = evaluate_direction(
                G, clip_model, probe_wp, probe_orig["clip_features"], None,
                attr_vecs[attr], bl, args.probe_alpha, clip_mean, clip_std,
                args.batch,
            )
            gain.append(float(row["mean_abs_delta_attr"]))
            print(f"  probe layer {li:2d}: gain={gain[-1]:.5f} "
                  f"rho={rho_per_attr[attr][li]:.3f}")

        subsets = choose_subsets(
            gain, rho_per_attr[attr], LAYERS_FOR[attr], args.layer_k,
            seed=args.seed + sum((i + 1) * ord(c) for i, c in enumerate(attr)),
            responsive_quantile=args.responsive_quantile,
        )
        print("  subsets: " + " | ".join(
            f"{name}={layers}" for name, layers in subsets.items()
        ))

        calibrated = {}
        max_deltas_for_target = []
        for name, layers in subsets.items():
            bl = build_layered(G, b_dir, layers)
            calib = calibrate_alpha(
                G, clip_model, calib_wp, calib_orig["clip_features"],
                attr_vecs[attr], bl, alpha_grid, clip_mean, clip_std,
                args.batch,
            )
            calibrated[name] = {"layers": layers, "b_layered": bl,
                                "calibration": calib}
            if name != "wrong_layers":
                max_deltas_for_target.append(calib["max_abs_delta"])
        target = float(np.quantile(max_deltas_for_target, args.target_quantile))
        print(f"  target_abs_delta={target:.5f}")

        rows_by_method = {}
        summary = {}
        for name in METHODS:
            item = calibrated[name]
            alpha, reached = alpha_for_target(item["calibration"], target)
            eval_row = evaluate_direction(
                G, clip_model, test_wp, test_orig["clip_features"],
                test_orig["downsampled"], attr_vecs[attr], item["b_layered"],
                alpha, clip_mean, clip_std, args.batch,
            )
            row = {
                "method": name,
                "layers": item["layers"],
                "alpha": float(alpha),
                "target_abs_delta_attr": target,
                "target_reached_on_calib": bool(reached),
                "calib_max_abs_delta_attr": item["calibration"]["max_abs_delta"],
                **eval_row,
            }
            rows_by_method[name] = [row]
            summary[name] = summarize_rows([row])
            print(f"  {name:22s} layers={item['layers']} "
                  f"|delta|={summary[name]['mean_abs_delta_attr']:.5f} "
                  f"ID={summary[name]['mean_id_cos']:.4f} "
                  f"LPIPS={summary[name]['mean_lpips_proxy']:.5f}")

        results[attr] = {
            "layer_gain_probe": gain,
            "layer_rho": rho_per_attr[attr],
            "subsets": subsets,
            "target_abs_delta_attr": target,
            "rows": rows_by_method,
            "summary": summary,
        }
        partial = {
            "per_attr": results,
            "config": vars(args),
            "_meta": run_metadata(seed=args.seed, extra={
                "script": "experiments/control/run_layer_control.py",
                "rho_source": str(rho_source),
                "lod_override": lod_override,
                "execution": execution_metadata(
                    protocol=args.protocol,
                    protocol_key=args.protocol_key,
                ),
            }),
        }
        (out / "metrics_partial.json").write_text(json.dumps(partial, indent=2))

    payload = {
        "per_attr": results,
        "config": vars(args),
        "_meta": run_metadata(seed=args.seed, extra={
            "script": "experiments/control/run_layer_control.py",
            "rho_source": str(rho_source),
            "lod_override": lod_override,
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_plot(payload, out)
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
