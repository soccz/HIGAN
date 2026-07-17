"""Cross-domain risk-aware latent edit controller.

This is the domain-general validation counterpart to the FFHQ controller.  It
keeps the same predeclared control rule,

    score = target_gain / (rho ** risk_power + eps)

but rebuilds the candidate universe and attribute prompts inside a non-FFHQ
domain.  The goal is to test whether the curvature/risk signal remains a
usable controller rather than a face-domain artifact.
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
import torch.nn.functional as F
from torch.func import jvp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from baselines.ganspace import ganspace_directions  # noqa: E402
from baselines.sefa import sefa_directions  # noqa: E402
from lib.experiment_io import execution_metadata, write_json_atomic  # noqa: E402
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


GROUPS = ["risk_aware", "gain_only", "low_risk", "random", "high_risk"]

BEDROOM_ATTRS = [
    "indoor_lighting",
    "wood",
    "view",
    "carpet",
    "cluttered_space",
    "glossy",
    "dirt",
    "scary",
]

BEDROOM_PROMPTS = {
    "indoor_lighting": ("a bright bedroom with warm lighting", "a dim bedroom"),
    "wood": ("a bedroom with wooden furniture", "a bedroom without wooden furniture"),
    "view": ("a bedroom with an outdoor view through a window", "a bedroom with no window view"),
    "carpet": ("a bedroom with carpeted floor", "a bedroom with bare hard floor"),
    "cluttered_space": ("a cluttered messy bedroom", "a tidy minimal bedroom"),
    "glossy": ("a bedroom with glossy shiny surfaces", "a bedroom with matte surfaces"),
    "dirt": ("a dirty bedroom", "a clean bedroom"),
    "scary": ("a scary dark bedroom", "a cozy ordinary bedroom"),
}

CHURCH_ATTRS = ["clouds", "sunny", "vegetation"]

CHURCH_PROMPTS = {
    "clouds": ("a church with cloudy sky", "a church under a clear sky"),
    "sunny": ("a sunny bright church exterior", "a dark overcast church exterior"),
    "vegetation": ("a church surrounded by trees and vegetation", "a church with little vegetation"),
}

CHURCH_LAYERS = {
    "clouds": list(range(0, 8)),
    "sunny": list(range(0, 8)),
    "vegetation": list(range(6, 12)),
}


def resolve_paper_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def load_bedroom_generator():
    from higan_dev.generator import HiGANGenerator  # noqa: E402

    return HiGANGenerator(
        higan_repo=str(PAPER.parent / "higan_dev" / "data" / "higan_repo")
    )


def bedroom_layers() -> dict[str, list[int]]:
    from higan_dev.manipulate import DEFAULT_MANIPULATE_LAYERS  # noqa: E402

    return {attr: list(DEFAULT_MANIPULATE_LAYERS[attr]) for attr in BEDROOM_ATTRS}


def load_church_generator():
    from domains.church.generator import ChurchGenerator  # noqa: E402

    return ChurchGenerator()


FFHQ_ATTRS = ["smile", "age", "pose", "gender", "eyeglasses"]

FFHQ_PROMPTS = {
    "smile": ("a smiling face with teeth showing", "a neutral face without smiling"),
    "age": ("an old face with wrinkles", "a young smooth face"),
    "pose": ("a face turned to the side, side profile", "a face looking straight at the camera"),
    "gender": ("a male face with a beard", "a female face"),
    "eyeglasses": ("a face wearing eyeglasses", "a face without eyeglasses"),
}

FFHQ_LAYERS = {
    "smile": list(range(4, 8)),
    "age": list(range(0, 8)),
    "pose": list(range(0, 4)),
    "gender": list(range(0, 8)),
    "eyeglasses": list(range(0, 8)),
}


def load_ffhq_generator():
    from domains.ffhq.generator import FFHQGenerator  # noqa: E402
    return FFHQGenerator(lod_override=2.0)


def load_domain_config(domain: str):
    if domain == "bedroom":
        return (
            load_bedroom_generator(),
            bedroom_layers(),
            BEDROOM_ATTRS,
            BEDROOM_PROMPTS,
        )
    if domain == "church":
        return (
            load_church_generator(),
            {attr: list(CHURCH_LAYERS[attr]) for attr in CHURCH_ATTRS},
            CHURCH_ATTRS,
            CHURCH_PROMPTS,
        )
    if domain == "ffhq":
        return (
            load_ffhq_generator(),
            {attr: list(FFHQ_LAYERS[attr]) for attr in FFHQ_ATTRS},
            FFHQ_ATTRS,
            FFHQ_PROMPTS,
        )
    raise ValueError(f"unknown domain: {domain}")


def load_clip(device: torch.device):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.eval().to(device), tokenizer


@torch.no_grad()
def clip_text_feature(model, tokenizer, device: torch.device,
                      texts: list[str]) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    feat = model.encode_text(tokens)
    return feat / feat.norm(dim=-1, keepdim=True)


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
    feat = model.encode_image(x)
    return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def dino_features(model, img: torch.Tensor) -> torch.Tensor:
    x = (img.clamp(-1, 1) + 1) / 2.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear",
                      align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(
        1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(
        1, 3, 1, 1)
    out = model(pixel_values=(x - mean) / std)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    else:
        feat = out.last_hidden_state.mean(dim=1)
    return feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def precompute_originals(G, clip_model, wp: torch.Tensor,
                         clip_mean: torch.Tensor, clip_std: torch.Tensor,
                         batch: int, keep_downsample: bool) -> dict[str, Any]:
    feats = []
    down = []
    for start in range(0, wp.shape[0], batch):
        cur = wp[start:start + batch]
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


def layered_direction(G, layers_for: dict[str, list[int]],
                      attr: str, vector: np.ndarray) -> torch.Tensor:
    v = torch.from_numpy(vector).to(G.device).float()
    v = v / v.norm().clamp_min(1e-8)
    out = torch.zeros(G.num_layers, G.w_dim, device=G.device)
    for layer_idx in layers_for[attr]:
        if 0 <= layer_idx < G.num_layers:
            out[layer_idx] = v
    return out


def build_candidate_pool(G, methods: list[str], candidate_k: int,
                         ganspace_samples: int, seed: int) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []

    def append(method: str, vectors: np.ndarray) -> None:
        for rank, vector in enumerate(vectors[:candidate_k]):
            pool.append({
                "index": len(pool),
                "source_group": method,
                "source_rank": rank,
                "vector": vector.astype(np.float32),
            })

    if "ganspace" in methods:
        result = ganspace_directions(
            G, n_samples=ganspace_samples, n_components=candidate_k, seed=seed
        )
        append("ganspace", result.components)
    if "sefa" in methods:
        result = sefa_directions(G, n_components=candidate_k)
        append("sefa", result.components)
    if "random" in methods:
        gen = torch.Generator(device=G.device).manual_seed(seed + 913)
        vectors = []
        for _ in range(candidate_k):
            v = torch.randn(G.w_dim, generator=gen, device=G.device)
            v = v / v.norm().clamp_min(1e-8)
            vectors.append(v.cpu().numpy().astype(np.float32))
        append("random", np.stack(vectors))

    if not pool:
        raise ValueError("empty candidate pool")
    return pool


def curvature_ratio_for_direction(G, wp: torch.Tensor,
                                  b_layered: torch.Tensor) -> float:
    ratios = []
    for idx in range(wp.shape[0]):
        cur = wp[idx:idx + 1].detach()

        def f(alpha: torch.Tensor) -> torch.Tensor:
            return G.synthesize(cur + alpha.view(1, 1, 1) * b_layered.unsqueeze(0))

        def df(alpha: torch.Tensor) -> torch.Tensor:
            _, first_ = jvp(f, (alpha,), (torch.ones_like(alpha),))
            return first_

        alpha0 = torch.zeros(1, device=G.device)
        tangent = torch.ones(1, device=G.device)
        _, first = jvp(f, (alpha0,), (tangent,))
        _, second = jvp(df, (alpha0,), (tangent,))
        first_map = first.abs().mean(dim=1).squeeze(0)
        second_map = second.abs().mean(dim=1).squeeze(0)
        ratios.append(float((second_map / (first_map + 1e-6)).mean().item()))
        torch.cuda.empty_cache()
    return float(np.mean(ratios))


@torch.no_grad()
def evaluate_direction(G, clip_model, wp: torch.Tensor,
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
                       dino_model=None,
                       orig_dino: torch.Tensor | None = None) -> dict[str, float]:
    attr_vec_cpu = attr_vec.detach().cpu()
    scores0 = (orig_clip @ attr_vec_cpu).numpy()
    scores1 = []
    id_cos = []
    drift = []
    lpips_true = []
    dino_cos = []
    for start in range(0, wp.shape[0], batch):
        cur = wp[start:start + batch]
        img = G.synthesize(cur + alpha * b_layered.unsqueeze(0)).clamp(-1, 1)
        feat = clip_features(clip_model, img, clip_mean, clip_std).cpu()
        scores1.extend((feat @ attr_vec_cpu).numpy().tolist())
        id_cos.extend((orig_clip[start:start + batch] * feat).sum(dim=1).numpy().tolist())
        if dino_model is not None and orig_dino is not None:
            dino_feat = dino_features(dino_model, img).cpu()
            dino_ref = orig_dino[start:start + batch]
            dino_cos.extend((dino_ref * dino_feat).sum(dim=1).numpy().tolist())
        if orig_down is not None:
            edit_down = F.interpolate(img, size=(256, 256), mode="bilinear",
                                      align_corners=False).cpu()
            drift.extend(((edit_down - orig_down[start:start + batch]) ** 2)
                         .mean(dim=(1, 2, 3)).numpy().tolist())
            if lpips_fn is not None:
                lpips_orig = orig_down[start:start + batch].to(G.device)
                lpips_edit = edit_down.to(G.device)
                if lpips_size != 256:
                    lpips_orig = F.interpolate(lpips_orig,
                                               size=(lpips_size, lpips_size),
                                               mode="bilinear",
                                               align_corners=False)
                    lpips_edit = F.interpolate(lpips_edit,
                                               size=(lpips_size, lpips_size),
                                               mode="bilinear",
                                               align_corners=False)
                vals = lpips_fn(lpips_edit, lpips_orig).detach().view(-1).cpu()
                lpips_true.extend(vals.numpy().tolist())
        torch.cuda.empty_cache()

    delta = np.asarray(scores1, dtype=float) - np.asarray(scores0, dtype=float)
    out = {
        "mean_delta_attr": float(delta.mean()),
        "mean_abs_delta_attr": float(np.abs(delta).mean()),
        "mean_id_cos": float(np.mean(id_cos)),
        "mean_lpips_proxy": float(np.mean(drift)) if drift else float("nan"),
    }
    if lpips_true:
        out["mean_lpips_true"] = float(np.mean(lpips_true))
    if dino_cos:
        out["mean_dino_cos"] = float(np.mean(dino_cos))
    return out


def calibrate_alpha(G, clip_model, wp: torch.Tensor, orig_clip: torch.Tensor,
                    attr_vec: torch.Tensor, b_layered: torch.Tensor,
                    alpha_grid: list[float], clip_mean: torch.Tensor,
                    clip_std: torch.Tensor, batch: int) -> dict[str, Any]:
    curves = []
    for sign in [1.0, -1.0]:
        vals = []
        signed_vals = []
        for mag in alpha_grid:
            row = evaluate_direction(
                G, clip_model, wp, orig_clip, None, attr_vec, b_layered,
                alpha=sign * mag, clip_mean=clip_mean, clip_std=clip_std,
                batch=batch,
            )
            vals.append(abs(row["mean_delta_attr"]))
            signed_vals.append(row["mean_delta_attr"])
        curves.append({"sign": sign, "abs_delta": vals,
                       "signed_delta": signed_vals})
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
    for idx in range(1, len(grid)):
        if vals[idx] >= target:
            lo_a, hi_a = grid[idx - 1], grid[idx]
            lo_v, hi_v = vals[idx - 1], vals[idx]
            if hi_v <= lo_v:
                return sign * hi_a, True
            t = (target - lo_v) / max(hi_v - lo_v, 1e-12)
            return sign * (lo_a + t * (hi_a - lo_a)), True
    return sign * grid[-1], False


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
                   seed: int, risk_selection_mode: str,
                   selection_rule: str = "ratio",
                   gain_pool_multiplier: float = 2.0,
                   ) -> tuple[dict[str, list[int]], np.ndarray]:
    gain = np.asarray([c["probe_gain"] for c in candidates], dtype=float)
    rho = np.asarray([c["rho"] for c in candidates], dtype=float)
    selection_rho = transformed_selection_rho(rho, risk_selection_mode, seed)
    eligible = gain >= np.quantile(gain, min_gain_quantile)
    if int(eligible.sum()) < k:
        eligible[topk_indices(gain, k, reverse=True)] = True

    if selection_rule == "ratio":
        score = np.full_like(gain, -np.inf)
        score[eligible] = gain[eligible] / (
            np.power(selection_rho[eligible] + 1e-8, risk_power) + 1e-8
        )
        risk_aware = topk_indices(score, k, reverse=True)
    elif selection_rule == "gain_topk_low_risk":
        eligible_idx = np.flatnonzero(eligible)
        pool_size = max(k, int(np.ceil(k * gain_pool_multiplier)))
        pool_size = min(pool_size, eligible_idx.size)
        gain_order = eligible_idx[np.argsort(gain[eligible_idx])[::-1]]
        pool = gain_order[:pool_size]
        risk_order = pool[np.argsort(selection_rho[pool])]
        risk_aware = risk_order[:k].astype(int).tolist()
    elif selection_rule == "gain_feasible_low_risk":
        eligible_idx = np.flatnonzero(eligible)
        risk_order = eligible_idx[np.argsort(selection_rho[eligible_idx])]
        risk_aware = risk_order[:k].astype(int).tolist()
    else:
        raise ValueError(f"unknown selection rule: {selection_rule}")

    rng = np.random.default_rng(seed)
    selected = {
        "risk_aware": risk_aware,
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
    metric_name = "mean_lpips_true"
    if not all(metric_name in payload["per_attr"][a]["summary"]["risk_aware"]
               for a in attrs):
        metric_name = "mean_lpips_proxy"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for idx, group in enumerate(GROUPS):
        offset = (idx - (len(GROUPS) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][group]["mean_id_cos"] for a in attrs],
            width=width, color=colors[group], label=group,
        )
        axes[1].bar(
            x + offset,
            [payload["per_attr"][a]["summary"][group][metric_name] for a in attrs],
            width=width, color=colors[group], label=group,
        )
    axes[0].set_title("CLIP image consistency")
    axes[0].set_ylabel("cosine")
    axes[1].set_title("Perceptual drift")
    axes[1].set_ylabel(metric_name)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cross_domain_risk_aware_controller.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bedroom", "church", "ffhq"], default="bedroom")
    ap.add_argument("--attrs", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["ganspace", "sefa", "random"])
    ap.add_argument("--candidate-k", type=int, default=8)
    ap.add_argument("--ganspace-samples", type=int, default=2048)
    ap.add_argument("--k-select", type=int, default=4)
    ap.add_argument("--min-gain-quantile", type=float, default=0.50)
    ap.add_argument("--risk-power", type=float, default=1.0)
    ap.add_argument("--selection-rule",
                    choices=["ratio", "gain_topk_low_risk",
                             "gain_feasible_low_risk"],
                    default="ratio")
    ap.add_argument("--gain-pool-multiplier", type=float, default=2.0)
    ap.add_argument("--n-risk", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--probe-alpha", type=float, default=1.0)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--max-alpha", type=float, default=6.0)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--target-source", choices=["selected", "universe"],
                    default="universe")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="experiments/out/control_bedroom_cross_domain")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--true-lpips", action="store_true")
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--lpips-size", type=int, default=256)
    ap.add_argument("--risk-selection-mode",
                    choices=["actual", "shuffled", "inverted"],
                    default="actual")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--protocol-key", default="bedroom_cross_domain_controller")
    args = ap.parse_args()

    if not 0.0 <= args.target_quantile <= 1.0:
        raise ValueError("--target-quantile must be in [0, 1]")
    if args.k_select > args.candidate_k * len(args.methods):
        raise ValueError("--k-select cannot exceed candidate universe size")

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
        f_pos = clip_text_feature(clip_model, clip_tok, device, [pos]).squeeze(0)
        f_neg = clip_text_feature(clip_model, clip_tok, device, [neg]).squeeze(0)
        v = f_pos - f_neg
        attr_vecs[attr] = v / v.norm().clamp_min(1e-8)

    lpips_fn = None
    if args.true_lpips:
        print(f"[{time.strftime('%H:%M:%S')}] loading LPIPS net={args.lpips_net}")
        import lpips

        lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).eval().to(device)
        for param in lpips_fn.parameters():
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

    alpha_grid = np.linspace(0.0, args.max_alpha, args.alpha_steps).tolist()
    probe_grid = [0.0, args.probe_alpha]
    results: dict[str, Any] = {}

    for attr in attrs:
        print(f"\n=== {attr} ===")
        candidates = []
        for item in base_pool:
            direction = layered_direction(G, layers_for, attr, item["vector"])
            rho = curvature_ratio_for_direction(G, risk_wp, direction)
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
        print(f"  measured rho/gain for {len(candidates)} candidates")

        selected, selection_rho = select_methods(
            candidates, args.k_select, args.min_gain_quantile,
            args.risk_power,
            seed=args.seed + sum((i + 1) * ord(c) for i, c in enumerate(attr)),
            risk_selection_mode=args.risk_selection_mode,
            selection_rule=args.selection_rule,
            gain_pool_multiplier=args.gain_pool_multiplier,
        )
        for idx, cand in enumerate(candidates):
            cand["selection_rho"] = float(selection_rho[idx])
        print("  selected: " + " | ".join(
            f"{name}={[candidates[i]['source_group'] + ':' + str(candidates[i]['source_rank']) for i in idxs]}"
            for name, idxs in selected.items()
        ))

        eval_candidates = {}
        for idxs in selected.values():
            for idx in idxs:
                eval_candidates[idx] = candidates[idx]
        target_candidates = (
            {i: cand for i, cand in enumerate(candidates)}
            if args.target_source == "universe" else eval_candidates
        )
        calibrations = {}
        max_deltas = []
        for idx, cand in target_candidates.items():
            calib = calibrate_alpha(
                G, clip_model, calib_wp, calib_orig["clip_features"],
                attr_vecs[attr], cand["direction"], alpha_grid, clip_mean,
                clip_std, args.batch,
            )
            calibrations[idx] = calib
            max_deltas.append(calib["max_abs_delta"])
        target = float(np.quantile(max_deltas, args.target_quantile))
        print(f"  target_abs_delta={target:.5f} source={args.target_source}")

        rows_by_group: dict[str, list[dict[str, Any]]] = {}
        summary = {}
        for name, idxs in selected.items():
            rows = []
            for idx in idxs:
                cand = candidates[idx]
                alpha, reached = alpha_for_target(calibrations[idx], target)
                row_eval = evaluate_direction(
                    G, clip_model, test_wp, test_orig["clip_features"],
                    test_orig["downsampled"], attr_vecs[attr],
                    cand["direction"], alpha, clip_mean, clip_std, args.batch,
                    lpips_fn=lpips_fn, lpips_size=args.lpips_size,
                )
                rows.append({
                    "candidate_index": int(cand["index"]),
                    "source_group": cand["source_group"],
                    "source_rank": int(cand["source_rank"]),
                    "rho": cand["rho"],
                    "selection_rho": cand["selection_rho"],
                    "probe_gain": cand["probe_gain"],
                    "probe_sign": cand["probe_sign"],
                    "alpha": float(alpha),
                    "target_abs_delta_attr": target,
                    "target_reached_on_calib": bool(reached),
                    "calib_max_abs_delta_attr": calibrations[idx]["max_abs_delta"],
                    **row_eval,
                })
            rows_by_group[name] = rows
            summary[name] = summarize(rows)
            lpips_key = "mean_lpips_true" if "mean_lpips_true" in summary[name] else "mean_lpips_proxy"
            print(f"  {name:12s} gain={summary[name]['mean_probe_gain']:.5f} "
                  f"rho={summary[name]['mean_rho']:.3f} "
                  f"|delta|={summary[name]['mean_abs_delta_attr']:.5f} "
                  f"ID={summary[name]['mean_id_cos']:.4f} "
                  f"{lpips_key}={summary[name][lpips_key]:.5f}")

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
                "script": "experiments/control/run_cross_domain_risk_aware_controller.py",
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
            "script": "experiments/control/run_cross_domain_risk_aware_controller.py",
            "execution": execution_metadata(
                protocol=args.protocol,
                protocol_key=args.protocol_key,
            ),
        }),
    }
    write_json_atomic(out / "metrics.json", payload)
    write_plot(payload, out)
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
