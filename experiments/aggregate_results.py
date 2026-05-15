"""Aggregate all metrics.json files into a single summary table.

Run after the queued runner finishes to produce paper-ready numbers
and a per-track status dashboard.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "experiments" / "out"


def safe_load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    print("=" * 60)
    print("HIGAN paper — all-tracks aggregation")
    print("=" * 60)
    summary = {}

    # ---- Wave 1 ----
    summary["wave1"] = {}

    # Track 1 SD
    sd = safe_load(OUT / "sd_c1_c2" / "metrics.json")
    if sd:
        summary["wave1"]["track1_sd_c1c2"] = {
            "n_attrs": len(sd.get("per_attr", [])),
            "completed": all("per_t" in a for a in sd.get("per_attr", [])),
        }
        print(f"\n[Track 1: SD C1/C2]")
        for entry in sd.get("per_attr", []):
            attr = entry["attr"]
            for t_idx, m in entry.get("per_t", {}).items():
                print(f"  {attr:12s} t_idx={t_idx:>4s}  "
                      f"ρ={m.get('rho_mean', 'n/a'):.3f}  "
                      f"CLIP-path={m.get('clip_path_mean', 'n/a'):.3f}")

    # Track 3 sample scaling
    for d in ["bedroom", "ffhq", "church"]:
        p = OUT / f"sample_scaling_{d}" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 3 scaling, {d}]")
            for N, row in s.get("bootstrap_ci", {}).items():
                print(f"  N={N:>4s}  " + "  ".join(
                    f"{a}={row[a]['mean']:.3f}±{(row[a]['ci_hi']-row[a]['ci_lo'])/2:.3f}"
                    for a in row
                ))

    # Track 5 baselines
    for n in ["latentclr_ffhq", "disco_ffhq"]:
        p = OUT / n / "directions.npy"
        if p.exists():
            import numpy as np
            v = np.load(p)
            print(f"\n[Track 5 {n}]  K={v.shape[0]}  trained")

    # Track 4 C5 FFHQ
    ck = OUT / "ffhq_c5" / "eval"
    if (ck / "c5_eval.json").exists():
        c5 = safe_load(ck / "c5_eval.json")
        print(f"\n[Track 4 C5 FFHQ eval]")
        for r in c5:
            print(f"  iter={r['iter']:6d}  recon_mse={r['recon_mse']:.4f}  "
                  f"sal_corr_mean={r['sal_corr_mean']:+.3f}")

    # Track 2 editing head-to-head
    head = OUT / "editing_head_to_head" / "metrics.json"
    h = safe_load(head)
    if h and "per_attr" in h:
        print(f"\n[Track 2 editing head-to-head]")
        for attr, by_rank in h["per_attr"].items():
            for rank, dirs in by_rank.items():
                if not dirs: continue
                ids = [d["mean_id_cos"] for d in dirs]
                ds = [d["mean_delta_attr"] for d in dirs]
                import numpy as np
                print(f"  {attr:12s} {rank:20s}  "
                      f"ID-cos={np.mean(ids):.3f}  "
                      f"Δattr={np.mean(ds):+.3f}")

    # ---- Wave 2 ----
    summary["wave2"] = {}

    # Track 6 DAAM
    p = OUT / "sd_daam" / "daam_maps.json"
    if p.exists():
        print(f"\n[Track 6 DAAM] {p} ready")

    # Track 7 Park reproduction
    p = OUT / "sd_park_repro" / "metrics.json"
    s = safe_load(p)
    if s:
        agg = s.get("aggregate", {})
        print(f"\n[Track 7 Park reproduction]  "
              f"top-1 σ={agg.get('mean_top1_sigma', 'n/a')}  "
              f"top-1 ρ={agg.get('mean_top1_rho', 'n/a')}  "
              f"Spearman σ↔ρ={agg.get('mean_spearman_sigma_rho', 'n/a')}")

    # Track 8 truncation
    p = OUT / "ffhq_truncation" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 8 truncation-ψ]")
        for psi, attrs in s["per_psi"].items():
            print(f"  ψ={psi}  " + "  ".join(
                f"{a}={attrs[a]['mean']:.3f}" for a in attrs
            ))
        for pair, r in s.get("rank_stability", {}).items():
            print(f"  rank-stability {pair}: Spearman r={r['r']:+.3f}")

    # Track 9 multi-CLIP
    p = OUT / "multi_clip_c2" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 9 multi-CLIP C2]")
        for var, by_domain in s.items():
            for d, r in by_domain.items():
                print(f"  {var:10s} {d:8s}  "
                      f"Pearson={r['pearson']['r']:+.3f}  "
                      f"Spearman={r['spearman']['r']:+.3f}")

    # Track 10 per-layer
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"per_layer_c1_{d}" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 10 per-layer C1, {d}]  "
                  f"argmax-in-canonical rate: "
                  f"{s.get('argmax_canonical_hit_rate', 'n/a')}")

    # Track 11 walltime
    p = OUT / "walltime" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 11 wall-clock]  "
              f"JVP={s['jvp']['mean_ms']:.1f}ms  "
              f"FD={s['fd']['mean_ms']:.1f}ms  "
              f"vmap-vjp@{s['vmap_vjp_cap']['K']}px="
              f"{s['vmap_vjp_cap'].get('mean_ms', 'OOM')}")

    # Track 12 C6 scaling
    p = OUT / "c6_scaling_bedroom" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 12 C6 scaling, bedroom]")
        for N, r in s.items():
            ev = r.get("evaluation_topk", {}).get("K=3", {})
            print(f"  N={N:>4s}  P={ev.get('P', 'n/a'):.2f}  "
                  f"R={ev.get('R', 'n/a'):.2f}  F1={ev.get('F1', 'n/a'):.2f}")

    # Track 13 resolution invariance
    p = OUT / "ffhq_resolution" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 13 FFHQ resolution]")
        for lod, attrs in s.get("per_lod", {}).items():
            print(f"  lod={lod}  " + "  ".join(
                f"{a}={attrs[a]['mean']:.3f}" for a in attrs
            ))

    # ---- Wave 3 ----
    summary["wave3"] = {}

    # Track 14 noise robustness
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"noise_robustness_{d}" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 14 noise robustness, {d}]  "
                  f"mean pairwise Spearman = "
                  f"{s.get('mean_pairwise_r', 'n/a'):+.3f}")

    # Track 17 intrinsic dim
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"intrinsic_dim_{d}" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 17 intrinsic dim, {d}]  "
                  f"median rank = {s.get('median_effective_rank', 'n/a')}")

    # Track 18 FD validation
    p = OUT / "fd_validation" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 18 FD validation]")
        for attr, eps_data in s.items():
            print(f"  {attr}: " + "  ".join(
                f"ε={eps_data[e]['m2_rel_mean']:.0e}" for e in eps_data
            ))

    # ---- Wave 4 ----
    p = OUT / "dino_path_curvature" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 19 DINOv2 path curvature]")
        for d, r in s.items():
            print(f"  {d:8s}  Pearson={r['pearson']['r']:+.3f}  "
                  f"Spearman={r['spearman']['r']:+.3f}")

    p = OUT / "ffhq_alpha_scan" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 20 α magnitude scan]")
        for attr, r in s.get("results", {}).items():
            print(f"  {attr:12s} log-slope = {r.get('log_slope', 'n/a'):+.3f}")

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
