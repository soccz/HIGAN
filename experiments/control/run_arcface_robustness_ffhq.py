"""ArcFace robustness check for the prediction-vs-control gap (FFHQ).

The headline finding rests on identity damage measured by CLIP-image cosine.
A reviewer will object: for faces, identity should be a real face-recognition
embedding (ArcFace), not CLIP. This script re-tests BOTH halves of the claim
with a VGGFace2 InceptionResnetV1 (facenet-pytorch) identity metric:

  Prediction half  : does rho predict ArcFace identity damage RANK?  (Spearman)
  Control half     : does picking low-rho beat high-rho on ArcFace ID?
                     (matched-pair win-rate, vs the CLIP-cosine 56% flat result)

If both survive the metric swap, the prediction=strong / control=flat finding
is metric-robust, not an artifact of CLIP-cosine identity.  No production
pipeline is modified; only controller helpers are reused.
"""
from __future__ import annotations

import argparse
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
    clip_features,
    clip_text_feature,
    curvature_ratio_for_direction,
    layered_direction,
    load_clip,
    load_domain_config,
    precompute_originals,
    FFHQ_PROMPTS,
)
from lib.reproducibility import run_metadata, set_deterministic  # noqa: E402


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")

    def rank(a):
        order = a.argsort(); r = np.empty_like(order, float)
        r[order] = np.arange(len(a)); return r
    rx, ry = rank(x), rank(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


@torch.no_grad()
def arcface_embed(model, img: torch.Tensor) -> torch.Tensor:
    """img in [-1,1], (B,3,H,W). FFHQ is aligned/centered; resize to 160."""
    x = F.interpolate(img, size=(160, 160), mode="bilinear",
                      align_corners=False)
    feat = model(x)                                   # (B, 512)
    return feat / feat.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+",
                    default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--candidate-k", type=int, default=6)
    ap.add_argument("--ganspace-samples", type=int, default=2048)
    ap.add_argument("--methods", nargs="+", default=["ganspace", "sefa"])
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--max-alpha", type=float, default=6.0)
    ap.add_argument("--target-quantile", type=float, default=0.25)
    ap.add_argument("--min-gain-quantile", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2037)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading FFHQ generator")
    G, layers_for, default_attrs, prompts = load_domain_config("ffhq")

    print(f"[{time.strftime('%H:%M:%S')}] loading CLIP")
    clip_model, tokenizer = load_clip(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=device).view(1, 3, 1, 1)

    print(f"[{time.strftime('%H:%M:%S')}] loading ArcFace (VGGFace2)")
    from facenet_pytorch import InceptionResnetV1
    arc = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    pool = build_candidate_pool(G, args.methods, args.candidate_k,
                                args.ganspace_samples, args.seed)
    print(f"  candidates={len(pool)}")

    gen_calib = torch.Generator(device=device).manual_seed(args.seed + 701)
    gen_test = torch.Generator(device=device).manual_seed(args.seed + 809)
    calib_wp = G.sample_wp(args.n_calib, generator=gen_calib)
    test_wp = G.sample_wp(args.n_test, generator=gen_test)

    orig_calib = precompute_originals(
        G, clip_model, calib_wp, clip_mean, clip_std, args.batch, False)
    orig_clip_calib = orig_calib["clip_features"]
    orig_test = precompute_originals(
        G, clip_model, test_wp, clip_mean, clip_std, args.batch, False)
    orig_clip_test = orig_test["clip_features"]

    # ArcFace originals for test set
    arc_orig = []
    for s in range(0, test_wp.shape[0], args.batch):
        img = G.synthesize(test_wp[s:s + args.batch]).clamp(-1, 1)
        arc_orig.append(arcface_embed(arc, img).cpu())
    arc_orig = torch.cat(arc_orig, 0)

    alpha_grid = list(np.linspace(0.0, args.max_alpha, args.alpha_steps))

    all_rows = []
    for attr in args.attrs:
        pos, neg = FFHQ_PROMPTS[attr]
        tpos = clip_text_feature(clip_model, tokenizer, device, [pos])
        tneg = clip_text_feature(clip_model, tokenizer, device, [neg])
        attr_vec = (tpos - tneg).squeeze(0)
        attr_vec = attr_vec / attr_vec.norm()

        # measure rho + probe gain for every candidate; set matched target
        cand_rows = []
        for c in pool:
            b_layered = layered_direction(G, layers_for, attr,
                                          np.asarray(c["vector"], dtype=np.float32))
            rho = curvature_ratio_for_direction(G, calib_wp[:8], b_layered)
            calib = calibrate_alpha(G, clip_model, calib_wp, orig_clip_calib,
                                    attr_vec, b_layered, alpha_grid,
                                    clip_mean, clip_std, args.batch)
            cand_rows.append({"cand": c["index"], "rho": rho,
                              "b_layered": b_layered, "calib": calib,
                              "probe_gain": calib["max_abs_delta"]})

        gains = np.array([r["probe_gain"] for r in cand_rows])
        gain_floor = np.quantile(gains, args.min_gain_quantile)
        eligible = [r for r in cand_rows if r["probe_gain"] >= gain_floor]
        if len(eligible) < 2:
            print(f"  {attr}: <2 eligible, skip"); continue
        target = float(np.quantile(
            [r["probe_gain"] for r in eligible], args.target_quantile))

        # edit every eligible candidate to the matched target, measure both IDs
        for r in eligible:
            alpha, _ = alpha_for_target(r["calib"], target)
            clip_id, arc_id = [], []
            for s in range(0, test_wp.shape[0], args.batch):
                cur = test_wp[s:s + args.batch]
                img = G.synthesize(
                    cur + alpha * r["b_layered"].unsqueeze(0)).clamp(-1, 1)
                feat = clip_features(clip_model, img, clip_mean, clip_std).cpu()
                clip_id.extend((orig_clip_test[s:s + args.batch] * feat)
                               .sum(dim=1).numpy().tolist())
                af = arcface_embed(arc, img).cpu()
                arc_id.extend((arc_orig[s:s + args.batch] * af)
                              .sum(dim=1).numpy().tolist())
            all_rows.append({
                "attr": attr, "cand": r["cand"], "rho": r["rho"],
                "alpha": float(alpha),
                "clip_id_cos": float(np.mean(clip_id)),
                "arcface_id_cos": float(np.mean(arc_id)),
            })
        print(f"  {attr}: {len(eligible)} eligible edited "
              f"(target={target:.4f})")

    # ---- analysis: prediction (Spearman) + control (matched-pair win) ----
    # damage = 1 - id_cos ; higher rho should mean higher damage
    rho = [r["rho"] for r in all_rows]
    clip_dmg = [1 - r["clip_id_cos"] for r in all_rows]
    arc_dmg = [1 - r["arcface_id_cos"] for r in all_rows]

    res = {
        "prediction": {
            "spearman_rho_vs_clip_damage": spearman(rho, clip_dmg),
            "spearman_rho_vs_arcface_damage": spearman(rho, arc_dmg),
        },
        "n_candidates_edited": len(all_rows),
    }

    # matched pairs within attr: low-rho vs high-rho, does low win on each ID?
    from itertools import combinations
    by_attr: dict[str, list] = {}
    for r in all_rows:
        by_attr.setdefault(r["attr"], []).append(r)
    clip_wins = arc_wins = npairs = 0
    for attr, rows in by_attr.items():
        for a, b in combinations(rows, 2):
            lo, hi = (a, b) if a["rho"] < b["rho"] else (b, a)
            npairs += 1
            # low-rho "wins" if it preserved identity better (higher id_cos)
            if lo["clip_id_cos"] > hi["clip_id_cos"]:
                clip_wins += 1
            if lo["arcface_id_cos"] > hi["arcface_id_cos"]:
                arc_wins += 1
    res["control"] = {
        "n_pairs": npairs,
        "clip_id_low_rho_win_rate": clip_wins / max(1, npairs),
        "arcface_id_low_rho_win_rate": arc_wins / max(1, npairs),
    }

    res["_meta"] = run_metadata(seed=args.seed)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    print("\n=== RESULT ===")
    print(f"  PREDICTION (rho -> ID damage rank, Spearman):")
    print(f"    CLIP-cosine : {res['prediction']['spearman_rho_vs_clip_damage']:+.3f}")
    print(f"    ArcFace     : {res['prediction']['spearman_rho_vs_arcface_damage']:+.3f}")
    print(f"  CONTROL (low-rho matched-pair win-rate, n={npairs}):")
    print(f"    CLIP-cosine : {res['control']['clip_id_low_rho_win_rate']:.3f}")
    print(f"    ArcFace     : {res['control']['arcface_id_low_rho_win_rate']:.3f}")
    print(f"\nsaved {out}/metrics.json")


if __name__ == "__main__":
    main()
