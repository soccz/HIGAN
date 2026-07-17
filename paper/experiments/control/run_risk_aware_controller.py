"""Risk-aware latent edit controller.

This experiment is a control-method follow-up to the fixed low-rho selector.
For each attribute it uses a fixed candidate universe from the official
direction-control run, probes target response on a separate split, then selects
directions using a predeclared score:

    score = target_gain / (rho ** risk_power + eps)

All baselines are reported regardless of outcome.
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
    collect_pool,
    load_clip,
    clip_text_feature,
    precompute_originals,
    evaluate_direction,
    calibrate_alpha,
    alpha_for_target,
)


GROUPS = ["risk_aware", "gain_only", "low_risk", "random", "high_risk"]
ALL_ATTRS = ["smile", "age", "pose", "gender", "eyeglasses"]


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def reconstruct_all_dirs(G: FFHQGenerator) -> np.ndarray:
    pool = collect_pool(
        G,
        ALL_ATTRS,
        latentclr_path=PAPER / "experiments" / "out" / "latentclr_ffhq" / "directions.npy",
        disco_path=PAPER / "experiments" / "out" / "disco_ffhq" / "directions.npy",
    )
    return np.concatenate([pool[m] for m in pool], axis=0)


def layered_direction(G: FFHQGenerator, attr: str, v_np: np.ndarray) -> torch.Tensor:
    v = torch.from_numpy(v_np).to(G.device).float()
    v = v / v.norm().clamp_min(1e-8)
    out = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for li in LAYERS_FOR[attr]:
        out[li] = v
    return out


def candidate_universe(source: dict[str, Any], attr: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    attr_payload = source["per_attr"][attr]
    if "rows" in attr_payload:
        group_rows = attr_payload["rows"]
        index_key = "index"
    else:
        # Expanded universe from the fixed-alpha editing protocol.  This keeps
        # the source file explicit while avoiding a dependency on the narrower
        # matched-control output.
        group_rows = {
            group: vals
            for group, vals in attr_payload.items()
            if isinstance(vals, list)
        }
        index_key = "k"
    for group, vals in group_rows.items():
        for row in vals:
            idx = int(row[index_key])
            if idx in seen:
                continue
            seen.add(idx)
            rows.append({
                "index": idx,
                "source_group": group,
                "rho": float(row["rho_this_dir"]),
            })
    if not rows:
        raise ValueError(f"no candidate rows found for attr={attr}")
    return rows


def topk_indices(values: np.ndarray, k: int, reverse: bool = True) -> list[int]:
    order = np.argsort(values)
    if reverse:
        order = order[::-1]
    return order[:k].astype(int).tolist()


def transformed_selection_rho(rho: np.ndarray, mode: str, seed: int) -> np.ndarray:
    if mode == "actual":
        return rho.copy()
    if mode == "shuffled":
        rng = np.random.default_rng(seed + 17017)
        return rng.permutation(rho)
    if mode == "inverted":
        return float(rho.min() + rho.max()) - rho
    raise ValueError(f"unknown risk selection mode: {mode}")


def select_methods(candidates: list[dict[str, Any]], k: int,
                   min_gain_quantile: float, risk_power: float,
                   seed: int,
                   risk_selection_mode: str) -> tuple[dict[str, list[int]], np.ndarray]:
    gain = np.asarray([c["probe_gain"] for c in candidates], dtype=float)
    rho = np.asarray([c["rho"] for c in candidates], dtype=float)
    selection_rho = transformed_selection_rho(rho, risk_selection_mode, seed)
    eligible = gain >= np.quantile(gain, min_gain_quantile)
    if int(eligible.sum()) < k:
        eligible[topk_indices(gain, k, reverse=True)] = True

    score = np.full_like(gain, -np.inf)
    score[eligible] = gain[eligible] / (
        np.power(selection_rho[eligible] + 1e-8, risk_power) + 1e-8)

    rng = np.random.default_rng(seed)
    selected = {
        "risk_aware": topk_indices(score, k, reverse=True),
        "gain_only": topk_indices(gain, k, reverse=True),
        "low_risk": topk_indices(rho, k, reverse=False),
        "random": rng.choice(len(candidates), size=k, replace=False).astype(int).tolist(),
        "high_risk": topk_indices(rho, k, reverse=True),
    }
    return selected, selection_rho


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {
        "n": len(rows),
        "mean_probe_gain": float(np.mean([r["probe_gain"] for r in rows])),
        "mean_rho": float(np.mean([r["rho"] for r in rows])),
        "mean_selection_rho": float(np.mean([
            r.get("selection_rho", r["rho"]) for r in rows
        ])),
        "target_hit_rate_calib": float(np.mean([r["target_reached_on_calib"] for r in rows])),
        "mean_abs_delta_attr": float(np.mean([r["mean_abs_delta_attr"] for r in rows])),
        "mean_id_cos": float(np.mean([r["mean_id_cos"] for r in rows])),
        "mean_lpips_proxy": float(np.mean([r["mean_lpips_proxy"] for r in rows])),
        "mean_abs_alpha": float(np.mean([abs(r["alpha"]) for r in rows])),
    }
    if all("mean_lpips_true" in r for r in rows):
        out["mean_lpips_true"] = float(np.mean([
            r["mean_lpips_true"] for r in rows
        ]))
    return out


def write_plot(payload: dict[str, Any], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attrs = list(payload["per_attr"].keys())
    x = np.arange(len(attrs))
    width = 0.15
    colors = {
        "risk_aware": "#2563eb",
        "gain_only": "#0891b2",
        "low_risk": "#16a34a",
        "random": "#6b7280",
        "high_risk": "#c2410c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for i, group in enumerate(GROUPS):
        offset = (i - (len(GROUPS) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][group]["mean_id_cos"] for a in attrs],
            width=width, color=colors[group], label=group,
        )
        axes[1].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][group]["mean_lpips_proxy"] for a in attrs],
            width=width, color=colors[group], label=group,
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
    fig.savefig(out / "risk_aware_controller.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-source", default="experiments/out/control_direction_matched_full/metrics.json")
    ap.add_argument("--attrs", nargs="+", default=ALL_ATTRS)
    ap.add_argument("--k-select", type=int, default=8)
    ap.add_argument("--min-gain-quantile", type=float, default=0.50)
    ap.add_argument("--risk-power", type=float, default=1.0)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--probe-alpha", type=float, default=1.0)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--max-alpha", type=float, default=6.0)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--target-source", choices=["selected", "universe"],
                    default="selected",
                    help="Where to estimate the matched target magnitude from.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lod", type=float, default=2.0)
    ap.add_argument("--out", default="experiments/out/control_risk_aware_controller")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--true-lpips", action="store_true",
                    help="Also compute real LPIPS in addition to the cheap L2 proxy.")
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--lpips-size", type=int, default=256)
    ap.add_argument("--risk-selection-mode", choices=["actual", "shuffled", "inverted"],
                    default="actual")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="risk_aware_controller")
    args = ap.parse_args()

    set_deterministic(args.seed)
    out = resolve_paper_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidate_source_path = resolve_paper_path(args.candidate_source)
    candidate_source = json.loads(candidate_source_path.read_text())

    lod_override = None if args.lod < 0 else args.lod
    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator lod={lod_override}")
    G = FFHQGenerator(lod_override=lod_override)
    device = G.device

    print(f"[{time.strftime('%H:%M:%S')}] reconstructing fixed candidate pool")
    all_dirs = reconstruct_all_dirs(G)
    print(f"  all_dirs={all_dirs.shape}")

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP")
    clip_model, clip_tok = load_clip(device)
    lpips_fn = None
    if args.true_lpips:
        print(f"[{time.strftime('%H:%M:%S')}] loading LPIPS net={args.lpips_net}")
        import lpips
        lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).eval().to(device)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
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

    gen_probe = torch.Generator(device=device).manual_seed(args.seed + 601)
    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 701)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 809)
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
    probe_grid = [0.0, args.probe_alpha]
    results: dict[str, Any] = {}

    for attr in args.attrs:
        print(f"\n=== {attr} ===")
        candidates = candidate_universe(candidate_source, attr)
        for cand in candidates:
            cand["direction"] = layered_direction(G, attr, all_dirs[cand["index"]])
            probe_calib = calibrate_alpha(
                G, clip_model, probe_wp, probe_orig["clip_features"],
                attr_vecs[attr], cand["direction"], probe_grid, clip_mean,
                clip_std, args.batch,
            )
            cand["probe_gain"] = probe_calib["max_abs_delta"]
            cand["probe_sign"] = probe_calib["sign"]
        print(f"  fixed universe: {len(candidates)} candidates")

        selected, selection_rho = select_methods(
            candidates, args.k_select, args.min_gain_quantile,
            args.risk_power,
            seed=args.seed + sum((i + 1) * ord(c) for i, c in enumerate(attr)),
            risk_selection_mode=args.risk_selection_mode,
        )
        for j, cand in enumerate(candidates):
            cand["selection_rho"] = float(selection_rho[j])
        print("  selected: " + " | ".join(
            f"{name}={[candidates[i]['index'] for i in idxs]}"
            for name, idxs in selected.items()
        ))

        eval_candidates = {}
        for name, idxs in selected.items():
            for i in idxs:
                eval_candidates[i] = candidates[i]
        if args.target_source == "universe":
            target_candidates = {
                i: cand for i, cand in enumerate(candidates)
            }
        else:
            target_candidates = eval_candidates
        max_deltas = []
        calibrations = {}
        for i, cand in target_candidates.items():
            calib = calibrate_alpha(
                G, clip_model, calib_wp, calib_orig["clip_features"],
                attr_vecs[attr], cand["direction"], alpha_grid, clip_mean,
                clip_std, args.batch,
            )
            calibrations[i] = calib
            max_deltas.append(calib["max_abs_delta"])
        target = float(np.quantile(max_deltas, args.target_quantile))
        print(f"  target_abs_delta={target:.5f} source={args.target_source}")

        rows_by_group: dict[str, list[dict[str, Any]]] = {}
        summary = {}
        for name, idxs in selected.items():
            rows = []
            for i in idxs:
                cand = candidates[i]
                alpha, reached = alpha_for_target(calibrations[i], target)
                row_eval = evaluate_direction(
                    G, clip_model, test_wp, test_orig["clip_features"],
                    test_orig["downsampled"], attr_vecs[attr],
                    cand["direction"], alpha, clip_mean, clip_std, args.batch,
                    lpips_fn=lpips_fn, lpips_size=args.lpips_size,
                )
                rows.append({
                    "candidate_index": int(cand["index"]),
                    "source_group": cand["source_group"],
                    "rho": cand["rho"],
                    "selection_rho": cand["selection_rho"],
                    "probe_gain": cand["probe_gain"],
                    "probe_sign": cand["probe_sign"],
                    "alpha": float(alpha),
                    "target_abs_delta_attr": target,
                    "target_reached_on_calib": bool(reached),
                    "calib_max_abs_delta_attr": calibrations[i]["max_abs_delta"],
                    **row_eval,
                })
            rows_by_group[name] = rows
            summary[name] = summarize(rows)
            print(f"  {name:12s} gain={summary[name]['mean_probe_gain']:.5f} "
                  f"rho={summary[name]['mean_rho']:.3f} "
                  f"|delta|={summary[name]['mean_abs_delta_attr']:.5f} "
                  f"ID={summary[name]['mean_id_cos']:.4f} "
                  f"LPIPS={summary[name]['mean_lpips_proxy']:.5f}")

        results[attr] = {
            "candidate_universe": [
                {k: v for k, v in cand.items() if k != "direction"}
                for cand in candidates
            ],
            "selected_candidate_offsets": selected,
            "target_abs_delta_attr": target,
            "rows": rows_by_group,
            "summary": summary,
        }
        partial = {
            "per_attr": results,
            "config": vars(args),
            "_meta": run_metadata(seed=args.seed, extra={
                "script": "experiments/control/run_risk_aware_controller.py",
                "candidate_source": str(candidate_source_path),
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
            "script": "experiments/control/run_risk_aware_controller.py",
            "candidate_source": str(candidate_source_path),
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
