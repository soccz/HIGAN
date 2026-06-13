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

    # Track 3 sample scaling — both v1 (ratio-of-means) and v2 (mean-of-ratios)
    for d in ["bedroom", "ffhq", "church"]:
        for suffix in ["", "_v2"]:
            p = OUT / f"sample_scaling_{d}{suffix}" / "metrics.json"
            s = safe_load(p)
            if s:
                label = f"{d}{suffix or ' (v1 ratio-of-means)'}"
                if not suffix:
                    label = f"{d} (v1 ratio-of-means)"
                else:
                    label = f"{d} (v2 mean-of-ratios)"
                print(f"\n[Track 3 scaling, {label}]")
                for N, row in s.get("bootstrap_ci", {}).items():
                    print(f"  N={N:>4s}  " + "  ".join(
                        f"{a}={row[a]['mean']:.3f}±{(row[a]['ci_hi']-row[a]['ci_lo'])/2:.3f}"
                        for a in row
                    ))

    # Track 5 baselines — both K=20 (Round 1) and K=50 v2 (Round 2)
    for n in ["latentclr_ffhq", "latentclr_ffhq_v2",
              "disco_ffhq", "disco_ffhq_v2"]:
        p = OUT / n / "directions.npy"
        if p.exists():
            import numpy as np
            v = np.load(p)
            tlog = safe_load(OUT / n / "train_log.json")
            ep_done = tlog.get("epochs_done", "?") if tlog else "?"
            loss_final = (tlog.get("loss_per_epoch", [None])[-1]
                          if tlog and tlog.get("loss_per_epoch") else None)
            print(f"\n[Track 5 {n}]  K={v.shape[0]}  "
                  f"epochs_done={ep_done}  "
                  f"final_loss={loss_final:.4f}"
                  if loss_final is not None
                  else f"\n[Track 5 {n}]  K={v.shape[0]}  trained")

    # Track 4 C5 FFHQ eval — both v1 and v2
    import numpy as _np
    for suffix in ["", "_v2"]:
        ck = OUT / f"ffhq_c5{suffix}" / "eval"
        ev = ck / "c5_eval.json"
        if ev.exists():
            c5 = safe_load(ev)
            results = c5.get("results", c5) if isinstance(c5, dict) else c5
            label = "v1 (w_mse=0.1, 40k iter)" if not suffix \
                else "v2 (w_mse=2.0, 160k iter)"
            print(f"\n[Track 4 C5 FFHQ eval — {label}]")
            for r in results:
                it = r.get("iter", "?")
                rm = r.get("recon_mse", float("nan"))
                sc = r.get("sal_corr_mean", float("nan"))
                print(f"  iter={it:>6}  recon_mse={rm:.4f}  "
                      f"sal_corr_mean={sc:+.3f}")

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
            if var.startswith("_"):
                continue
            if not isinstance(by_domain, dict):
                continue
            for d, r in by_domain.items():
                if not isinstance(r, dict) or "pearson" not in r:
                    continue
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
            if N.startswith("_") or not isinstance(r, dict):
                continue
            ev = r.get("evaluation_topk", {}).get("K=3", {})
            P = ev.get('P'); R = ev.get('R'); F1 = ev.get('F1')
            if P is None or R is None or F1 is None:
                continue
            print(f"  N={N:>4s}  P={P:.2f}  R={R:.2f}  F1={F1:.2f}")

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
            if attr.startswith("_") or not isinstance(eps_data, dict):
                continue
            cells = []
            for e in eps_data:
                ed = eps_data[e]
                if isinstance(ed, dict) and "m2_rel_mean" in ed:
                    cells.append(f"ε={e}:m2_rel={ed['m2_rel_mean']:.0e}")
            if cells:
                print(f"  {attr}: " + "  ".join(cells))

    # ---- Wave 4 ----
    p = OUT / "dino_path_curvature" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 19 DINOv2 path curvature]")
        for d, r in s.items():
            if d.startswith("_") or not isinstance(r, dict):
                continue
            if "pearson" not in r:
                continue
            print(f"  {d:8s}  Pearson={r['pearson']['r']:+.3f}  "
                  f"Spearman={r['spearman']['r']:+.3f}")

    p = OUT / "ffhq_alpha_scan" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 20 α magnitude scan]")
        for attr, r in s.get("results", {}).items():
            print(f"  {attr:12s} log-slope = {r.get('log_slope', 'n/a'):+.3f}")

    # ---- Wave 5 / supplementary tracks (formerly silent in aggregator) ----

    # C2 path curvature (raw bedroom + ffhq)
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"{d}_c2_path" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 21 C2 path curvature, {d}]")
            for e in s.get("per_attr", [])[:8]:
                attr = e.get("attr", "?")
                mr = e.get("mean_ratio", float('nan'))
                med = e.get("median_ratio", float('nan'))
                print(f"  {attr:18s} mean_ratio={mr:.3f}  median={med:.3f}")
            vp = s.get("vs_pixel", {})
            if vp:
                pr = vp.get("pearson", {}).get("r", 'n/a')
                sr = vp.get("spearman", {}).get("r", 'n/a')
                if isinstance(pr, float) and isinstance(sr, float):
                    print(f"  vs_pixel  Pearson={pr:+.3f}  Spearman={sr:+.3f}")

    # C2 segmentation
    p = OUT / "bedroom_c2_seg" / "metrics.json"
    s = safe_load(p)
    if s:
        sp = s.get("spearman_saliency_vs_seg", {})
        if isinstance(sp, dict):
            r = sp.get("r")
            print(f"\n[Track 22 C2 saliency-vs-segmentation, bedroom]  "
                  f"Spearman r = "
                  + (f"{r:+.3f}" if isinstance(r, float) else f"{r}"))

    # C3 layer IOU bedroom + ffhq
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"{d}_c3_iou" / "metrics.json"
        s = safe_load(p)
        if s:
            mc = s.get("mean_c3", 'n/a')
            tf = s.get("top_frac", 'n/a')
            n = s.get("num_samples", '?')
            if isinstance(mc, float):
                print(f"\n[Track 23 C3 layer-IOU, {d}]  "
                      f"mean_c3={mc:.3f}  top_frac={tf:.3f}  N={n}")

    # C3 threshold sweep
    for d in ["bedroom", "ffhq"]:
        p = OUT / f"{d}_c3_threshold" / "metrics.json"
        s = safe_load(p)
        if s:
            pt = s.get("per_threshold", {})
            print(f"\n[Track 24 C3 threshold sweep, {d}]  "
                  f"thresholds={list(pt.keys())[:5]}")
            for th, val in list(pt.items())[:5]:
                if isinstance(val, dict):
                    mc = val.get("mean_c3", 'n/a')
                    if isinstance(mc, float):
                        print(f"  th={th}  mean_c3={mc:.3f}")

    # C4 composition (bedroom, ffhq, church)
    for d in ["bedroom", "ffhq", "church"]:
        p = OUT / f"{d}_c4" / "metrics.json"
        s = safe_load(p)
        if s:
            pr = s.get("pearson", {})
            sr = s.get("spearman", {})
            n_pairs = s.get("n_pairs", '?')
            if isinstance(pr, dict) and isinstance(sr, dict):
                prv = pr.get("r", 'n/a')
                srv = sr.get("r", 'n/a')
                if isinstance(prv, float) and isinstance(srv, float):
                    print(f"\n[Track 25 C4 composition, {d}]  "
                          f"Pearson={prv:+.3f}  Spearman={srv:+.3f}  "
                          f"n_pairs={n_pairs}")

    # C4 robustness — ablations: full / no_view / only_view
    p = OUT / "c4_robustness" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 26 C4 robustness (composition ablations)]")
        for d in ["bedroom", "ffhq"]:
            dd = s.get(d, {})
            if not isinstance(dd, dict):
                continue
            for split, vals in dd.items():
                if not isinstance(vals, dict):
                    continue
                sp = vals.get("spearman", {})
                sr = sp.get("r") if isinstance(sp, dict) else sp
                n = vals.get("n", "?")
                if isinstance(sr, float):
                    print(f"  {d:8s} {split:12s}  n={n}  Spearman={sr:+.3f}")

    # C6 precision/recall — list of (top_k, n_clusters, P, R, F1) per domain
    p = OUT / "c6_precision_recall" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 27 C6 precision/recall]")
        for d in ["bedroom", "ffhq"]:
            dd = s.get(d, [])
            if not isinstance(dd, list):
                continue
            for ev in dd[:6]:
                k = ev.get("top_k", "?")
                P = ev.get("precision"); R = ev.get("recall"); F1 = ev.get("f1")
                rl = ev.get("recall_lift"); pl = ev.get("precision_lift")
                if P is None or R is None:
                    continue
                lift_str = (f"  P_lift={pl:+.2f}  R_lift={rl:+.2f}"
                            if isinstance(pl, float) and isinstance(rl, float)
                            else "")
                print(f"  {d:8s} top-{k}  P={P:.2f}  R={R:.2f}  F1={F1:.2f}{lift_str}")

    # CLIP-GradCAM baseline (bedroom + ffhq) — corr (Pearson) + iou with JVP
    for d in ["clip_gradcam", "ffhq_clip_gradcam"]:
        p = OUT / d / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Track 28 CLIP-GradCAM baseline, {d}]")
            for attr, r in s.items():
                if attr.startswith("_") or not isinstance(r, dict):
                    continue
                corr = r.get("corr")
                iou = r.get("iou")
                if isinstance(corr, float):
                    print(f"  {attr:12s} corr={corr:+.3f}  IOU={iou:.3f}")

    # Spatial diversity (bedroom)
    p = OUT / "bedroom_spatial_diversity" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 29 spatial diversity, bedroom]")
        for method in ["ganspace", "sefa", "random", "higan_gt"]:
            m = s.get(method, {})
            if isinstance(m, dict):
                sd = m.get("spatial_diversity")
                mi = m.get("mean_pairwise_iou")
                nd = m.get("n_directions")
                if isinstance(sd, float):
                    print(f"  {method:12s} spatial_div={sd:.3f}  "
                          f"mean_IOU={mi:.3f}  n_dirs={nd}")

    # Baselines sweep (bedroom + ffhq)
    for d in ["bedroom_baselines_sweep", "ffhq_baselines_sweep"]:
        p = OUT / d / "metrics.json"
        s = safe_load(p)
        if s and "sweeps" in s:
            print(f"\n[Track 30 baselines sweep, {d}]")
            for method, runs in s["sweeps"].items():
                if not isinstance(runs, list):
                    continue
                best = max(runs, key=lambda r: r.get("n_coverage", 0))
                print(f"  {method:10s}  best K={best.get('K')}  "
                      f"n_coverage={best.get('n_coverage')}  "
                      f"diversity={best.get('diversity')}")

    # Crossdomain signature (Track 22, agreement_rate)
    p = OUT / "crossdomain_signature" / "metrics.json"
    s = safe_load(p)
    if s:
        ar = s.get("agreement_rate", 'n/a')
        n = len(s.get("names", []))
        if isinstance(ar, float):
            print(f"\n[Track 31 cross-domain k=2 signature]  "
                  f"agreement={ar:.3f} ({int(ar*n)}/{n})")

    # Geometric fingerprint (Track 23)
    p = OUT / "geometric_fingerprint" / "metrics.json"
    s = safe_load(p)
    if s:
        cov = s.get("coverage_used", 'n/a')
        partial_ari = s.get("partial_ari", s.get("ari", 'n/a'))
        if isinstance(partial_ari, float):
            print(f"\n[Track 32 geometric fingerprint]  "
                  f"partial-ARI={partial_ari:.3f}  coverage={cov}")

    # Church all (StyleGAN2) — higher_order is a list of entries
    p = OUT / "church_all" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 33 church (StyleGAN2) — higher-order ratio 2nd/1st]")
        ho = s.get("higher_order", [])
        if isinstance(ho, list):
            for r in ho:
                attr = r.get("attr", "?")
                rm = r.get("ratio_mean")
                rp95 = r.get("ratio_p95")
                if isinstance(rm, float):
                    print(f"  {attr:12s} ratio_mean={rm:.4f}  p95={rp95:.4f}")

    # FFHQ disentangle (Track 15 dev)
    p = OUT / "ffhq_disentangle" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 34 FFHQ disentangle]  "
              f"keys: {list(s.keys())[:5]}")

    # FFHQ higher_order — actual fields: ratio_mean, ratio_p95
    p = OUT / "ffhq_higher_order" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 35 FFHQ higher-order (∂²/∂¹ ratio)]")
        for r in s.get("results", []):
            if isinstance(r, dict):
                attr = r.get("attr", "?")
                rm = r.get("ratio_mean")
                rp95 = r.get("ratio_p95")
                if isinstance(rm, float):
                    print(f"  {attr:12s} ratio_mean={rm:.3f}  p95={rp95:.3f}")

    # FFHQ C6 discovery — cluster_labels is dict cid → top-K CLIP zeroshot labels
    p = OUT / "ffhq_c6" / "metrics.json"
    s = safe_load(p)
    if s:
        nc = s.get("num_clusters", "?")
        nd = s.get("num_directions_kept", "?")
        print(f"\n[Track 36 FFHQ C6 discovery]  "
              f"K={nc} clusters from {nd} candidate directions")
        cl = s.get("cluster_labels", {})
        if isinstance(cl, dict):
            for cid, topk in sorted(cl.items(), key=lambda x: int(x[0])):
                if not topk: continue
                top1 = topk[0]
                if isinstance(top1, (list, tuple)) and len(top1) >= 2:
                    print(f"  cluster {cid}: '{top1[0]}' (Δ={top1[1]:+.4f})")

    # SD C1/C2 at N=64 (extended)
    p = OUT / "sd_c1_c2_n64" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 37 SD C1/C2 at N=64]")
        for entry in s.get("per_attr", [])[:5]:
            attr = entry.get("attr", "?")
            for t_idx, m in entry.get("per_t", {}).items():
                rho = m.get('rho_mean')
                if isinstance(rho, float):
                    print(f"  {attr:12s} t={t_idx}  ρ={rho:.3f}")

    # ---- Round 3 new tracks ----

    # T38 8x14 attribute-layer matrix
    p = OUT / "full_matrix_bedroom" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 38 8x14 attribute-layer saliency matrix, bedroom]")
        intens = s.get("intensities_per_layer", {})
        peak = s.get("peak_layer_per_attr", {})
        canon = s.get("canonical_layers_per_attr", {})
        for attr, peak_layer in peak.items():
            cs = canon.get(attr, [])
            print(f"  {attr:18s} peak L={peak_layer}  canonical={cs}")

    # T39 CLIP cluster labels
    p = OUT / "clip_cluster_labels_bedroom" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 39 CLIP zero-shot cluster labels]")
        for cid, topk in s.get("clusters", {}).items():
            top1 = topk[0] if topk else {}
            print(f"  cluster {cid}:  "
                  f"top-1 = '{top1.get('label', '?')}' "
                  f"(Δ={top1.get('score', 0):+.3f})")

    # T41 InterFaceGAN supervised baseline (FFHQ)
    p = OUT / "ffhq_interfacegan_baseline" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Track 41 InterFaceGAN supervised baseline, FFHQ]")
        cov = s.get("coverage", [])
        nc = s.get("n_coverage", "?")
        div = s.get("diversity", "?")
        wt = s.get("wall_time_s", "?")
        print(f"  K={s.get('K')}  coverage={nc}/5 ({cov})  "
              f"diversity={div}  wall_time={wt:.2f}s"
              if isinstance(wt, float) else
              f"  K={s.get('K')}  coverage={nc}/5 ({cov})  "
              f"diversity={div}")
        print(f"  → SUPERVISED upper bound; unsupervised GANSpace matches at K=8")

    # T40 Real LSUN photo cycle
    p = OUT / "real_photo_cycle" / "metrics.json"
    s = safe_load(p)
    if s:
        summ = s.get("summary", {})
        if summ:
            print(f"\n[Track 40 Real LSUN photo encoder transfer]")
            print(f"  N photos: {summ.get('n_photos')}")
            print(f"  LPIPS optim-inv: {summ.get('lpips_optim_mean'):.3f}±"
                  f"{summ.get('lpips_optim_std'):.3f}")
            print(f"  LPIPS encoder:   {summ.get('lpips_enc_mean'):.3f}±"
                  f"{summ.get('lpips_enc_std'):.3f}")
            print(f"  encoder gap: +{summ.get('encoder_gap_pct'):.0f}% "
                  f"(synthetic-supervision limit)")

    # ---- Main-paper claim consolidation ----

    p = OUT / "main_edit_regime_prediction" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Main claim A edit-regime prediction]")
        feats = s.get("features", {})
        for name in ["log10_pixel_rho", "clip_path_ratio"]:
            r = feats.get(name, {})
            auc = r.get("auroc")
            lodo = r.get("mean_lodo_accuracy")
            n = r.get("n", "?")
            if isinstance(auc, float):
                print(f"  {name:20s} n={n}  AUROC={auc:.3f}  "
                      f"LODO-acc={lodo:.3f}")

    p = OUT / "main_composition_failure_prediction" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Main claim B composition-failure prediction]")
        for d, r in s.get("by_domain", {}).items():
            sp = r.get("spearman_mixed_hessian", {}).get("r")
            auc = r.get("auroc_mixed_hessian")
            base = r.get("auroc_max_univariate_pixel_rho")
            if isinstance(auc, float):
                print(f"  {d:8s} Spearman={sp:+.3f}  AUROC={auc:.3f}  "
                      f"baseline={base:.3f}")
        pooled = s.get("pooled_within_domain_percentiles", {})
        pa = pooled.get("auroc_mixed_hessian_percentile")
        if isinstance(pa, float):
            print(f"  pooled within-domain percentile AUROC={pa:.3f}")

    p = OUT / "main_curvature_guided_selection" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Main claim C fixed-alpha curvature-guided selection]")
        eff = s.get("paired_attr_effects", {})
        for cname in ["curvature_low_minus_random",
                      "curvature_low_minus_high"]:
            r = eff.get(cname, {})
            did = r.get("mean_id_cos", {}).get("mean")
            dlp = r.get("mean_lpips_proxy", {}).get("mean")
            dda = r.get("abs_delta_attr", {}).get("mean")
            if isinstance(did, float):
                print(f"  {cname:28s} ΔID={did:+.4f}  "
                      f"ΔLPIPS={dlp:+.4f}  Δ|attr|={dda:+.4f}")

    p = OUT / "matched_editing_head_to_head_pilot" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Main claim C2 matched-attribute edit pilot]")
        import numpy as np
        attrs = sorted(s.get("per_attr", {}))
        for attr in attrs:
            summ = s["per_attr"][attr].get("summary", {})
            cells = []
            for g in ["curvature_low", "random", "curvature_high"]:
                r = summ.get(g, {})
                if not r:
                    continue
                cells.append(f"{g}:ID={r.get('mean_id_cos'):.3f},"
                             f"LPIPS={r.get('mean_lpips_proxy'):.3f},"
                             f"|Δ|={r.get('mean_abs_delta_attr'):.3f}")
            print(f"  {attr:12s}  " + "  ".join(cells))
        for metric in ["mean_id_cos", "mean_lpips_proxy",
                       "mean_abs_delta_attr"]:
            diffs_r = []
            diffs_h = []
            for attr in attrs:
                summ = s["per_attr"][attr]["summary"]
                diffs_r.append(summ["curvature_low"][metric] -
                               summ["random"][metric])
                diffs_h.append(summ["curvature_low"][metric] -
                               summ["curvature_high"][metric])
            print(f"  pooled Δlow-random {metric}={np.mean(diffs_r):+.4f}  "
                  f"Δlow-high={np.mean(diffs_h):+.4f}")

    p = OUT / "control_direction_matched_full" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control A direction control, full matched semantic edit]")
        import numpy as np
        attrs = sorted(s.get("per_attr", {}))
        for metric in ["mean_id_cos", "mean_lpips_proxy",
                       "mean_abs_delta_attr"]:
            diffs_r = []
            diffs_h = []
            for attr in attrs:
                summ = s["per_attr"][attr]["summary"]
                diffs_r.append(summ["curvature_low"][metric] -
                               summ["random"][metric])
                diffs_h.append(summ["curvature_low"][metric] -
                               summ["curvature_high"][metric])
            print(f"  pooled Δlow-random {metric}={np.mean(diffs_r):+.4f}  "
                  f"Δlow-high={np.mean(diffs_h):+.4f}")

    p = OUT / "control_layer_intervention" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control B curvature-guided layer intervention]")
        import numpy as np
        attrs = sorted(s.get("per_attr", {}))
        for metric in ["mean_id_cos", "mean_lpips_proxy",
                       "mean_abs_delta_attr"]:
            diffs_rand = []
            diffs_wrong = []
            diffs_canon = []
            for attr in attrs:
                summ = s["per_attr"][attr]["summary"]
                diffs_rand.append(summ["curvature_control"][metric] -
                                  summ["random_same_size"][metric])
                diffs_wrong.append(summ["curvature_control"][metric] -
                                   summ["wrong_layers"][metric])
                diffs_canon.append(summ["curvature_control"][metric] -
                                   summ["canonical"][metric])
            print(f"  pooled Δcurv-random {metric}={np.mean(diffs_rand):+.4f}  "
                  f"Δcurv-wrong={np.mean(diffs_wrong):+.4f}  "
                  f"Δcurv-canonical={np.mean(diffs_canon):+.4f}")

    p = OUT / "control_composition_guard" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control C actual composition guard]")
        summ = s.get("summary", {})
        pred = summ.get("predictors", {}).get("mixed_hessian_predictor", {})
        sp = pred.get("spearman", {}).get("r")
        auc = pred.get("auroc_failure_top50")
        acc = summ.get("accepted_mean_failure")
        rej = summ.get("rejected_mean_failure")
        if isinstance(sp, float):
            print(f"  mixed-Hessian vs actual failure: "
                  f"Spearman={sp:+.3f}  AUROC={auc:.3f}")
            print(f"  guard accepted_mean_failure={acc:.3f}  "
                  f"rejected_mean_failure={rej:.3f}")

    p = OUT / "control_risk_aware_controller" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D risk-aware direction controller]")
        import numpy as np
        attrs = sorted(s.get("per_attr", {}))
        for metric in ["mean_id_cos", "mean_lpips_proxy",
                       "mean_abs_delta_attr"]:
            diffs_gain = []
            diffs_random = []
            diffs_high = []
            diffs_low = []
            for attr in attrs:
                summ = s["per_attr"][attr]["summary"]
                diffs_gain.append(summ["risk_aware"][metric] -
                                  summ["gain_only"][metric])
                diffs_random.append(summ["risk_aware"][metric] -
                                    summ["random"][metric])
                diffs_high.append(summ["risk_aware"][metric] -
                                  summ["high_risk"][metric])
                diffs_low.append(summ["risk_aware"][metric] -
                                 summ["low_risk"][metric])
            print(f"  pooled Δrisk-gain {metric}={np.mean(diffs_gain):+.4f}  "
                  f"Δrisk-random={np.mean(diffs_random):+.4f}  "
                  f"Δrisk-high={np.mean(diffs_high):+.4f}  "
                  f"Δrisk-low={np.mean(diffs_low):+.4f}")

    p = OUT / "control_risk_aware_robustness" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D2 risk-aware seed robustness]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_proxy", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"ΔLPIPS={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_risk_signal_negative_controls" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D3 risk-signal negative controls]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_proxy", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"ΔLPIPS={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label in ["risk_power_0p5", "risk_power_2p0"]:
        p = OUT / "control_risk_power_sensitivity" / label / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D4 risk-power sensitivity {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_proxy", {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"ΔLPIPS={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label, out_dir, metric_name in [
        ("true-LPIPS validation", "control_true_lpips_validation",
         "mean_lpips_true"),
        ("expanded candidate universe", "control_expanded_candidate_universe",
         "mean_lpips_proxy"),
    ]:
        p = OUT / out_dir / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D5/D6 {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get(metric_name, {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δ{metric_name}={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_fixed_target_negative_controls" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D7 fixed-target risk-signal negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_proxy", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"ΔLPIPS={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label, out_dir, metric_name in [
        ("fixed-target true-LPIPS", "control_fixed_target_true_lpips",
         "mean_lpips_true"),
        ("expanded universe true-LPIPS", "control_expanded_universe_true_lpips",
         "mean_lpips_true"),
        ("probe budget 8/16", "control_probe_budget_sensitivity/probe8_calib16",
         "mean_lpips_proxy"),
        ("probe budget 16/16", "control_probe_budget_sensitivity/probe16_calib16",
         "mean_lpips_proxy"),
        ("k sensitivity k=4", "control_k_sensitivity/k4",
         "mean_lpips_proxy"),
        ("k sensitivity k=12", "control_k_sensitivity/k12",
         "mean_lpips_proxy"),
        ("gain threshold q=.25", "control_gain_threshold_sensitivity/gain_q25",
         "mean_lpips_proxy"),
        ("gain threshold q=.75", "control_gain_threshold_sensitivity/gain_q75",
         "mean_lpips_proxy"),
        ("target magnitude q=.10", "control_target_magnitude_sensitivity/target_q10",
         "mean_lpips_proxy"),
        ("target magnitude q=.50", "control_target_magnitude_sensitivity/target_q50",
         "mean_lpips_proxy"),
        ("large held-out n=512", "control_large_testset_validation/ntest512",
         "mean_lpips_proxy"),
        ("probe alpha 0.5", "control_probe_alpha_sensitivity/probe_alpha_0p5",
         "mean_lpips_proxy"),
        ("probe alpha 2.0", "control_probe_alpha_sensitivity/probe_alpha_2p0",
         "mean_lpips_proxy"),
    ]:
        p = OUT / out_dir / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D8 extended validation {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get(metric_name, {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δ{metric_name}={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_true_lpips_fixed_target_negatives" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D9 true-LPIPS fixed-target risk-signal negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_bedroom_cross_domain_true_lpips" / "actual_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D10 bedroom cross-domain true-LPIPS controller]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_bedroom_cross_domain_true_lpips" / "negative_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D11 bedroom cross-domain risk-signal negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_cross_domain_true_lpips" / "actual_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D12 church StyleGAN2 cross-domain true-LPIPS controller]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_cross_domain_true_lpips" / "negative_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D13 church StyleGAN2 cross-domain risk-signal negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label in ["risk_power_0p5", "risk_power_2p0"]:
        p = OUT / "control_church_risk_power_sensitivity" / label / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D14 church StyleGAN2 risk-power sensitivity {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_true", {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label in ["structured_only", "high_gain_floor"]:
        p = OUT / "control_church_failure_diagnosis" / label / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D15 church StyleGAN2 failure diagnosis {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_true", {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_structured_negative_controls" / "negative_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D16 church StyleGAN2 structured-only risk-signal negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_structured_confirmatory" / "actual_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D17 church structured-only confirmatory replication]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_structured_confirmatory" / "negative_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D17b church structured-only confirmatory negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_structured_estimator_stability" / "summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D18 church structured-only estimator stability]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for label in ["ganspace_only", "sefa_only"]:
        p = OUT / "control_church_structured_source_ablation" / label / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D19 church structured-source ablation {label}]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_true", {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    for domain in ["church", "bedroom"]:
        p = OUT / "control_cross_domain_risk_predictive" / domain / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D20 {domain} risk predictive validity]")
            seed_level = s.get("seed_level", {})
            for key, label in [
                ("rho_vs_id_spearman", "rho vs ID Spearman"),
                ("rho_vs_lpips_spearman", "rho vs LPIPS Spearman"),
                ("id_beta_rho", "rho beta for ID"),
                ("lpips_beta_rho", "rho beta for LPIPS"),
                ("matched_pair_id_low_minus_high", "low-risk minus high-risk ID"),
                ("matched_pair_lpips_low_minus_high", "low-risk minus high-risk LPIPS"),
            ]:
                row = seed_level.get(key, {})
                if isinstance(row.get("mean"), float):
                    print(f"  {label:34s} "
                          f"mean={row['mean']:+.4f} "
                          f"wins={row.get('wins', '-')}/{row.get('n', '?')}")

    p = OUT / "control_church_gain_first_risk_tiebreak" / "actual_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D21 church gain-first/risk-tiebreak controller]")
        seed_level = s.get("seed_level", {})
        for comp in ["risk_minus_gain_only", "risk_minus_random",
                     "risk_minus_high_risk", "risk_minus_low_risk"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:22s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_church_gain_first_risk_tiebreak" / "negative_summary" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D21b church gain-first/risk-tiebreak negatives]")
        seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
        for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
            row = seed_level.get(comp, {})
            sid = row.get("mean_id_cos", {})
            slp = row.get("mean_lpips_true", {})
            sad = row.get("mean_abs_delta_attr", {})
            if isinstance(sid.get("mean"), float):
                print(f"  {comp:24s} "
                      f"seed-mean ΔID={sid['mean']:+.4f} "
                      f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                      f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                      f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                      f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_main_claim_readiness_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D22 main-claim readiness]")
        print(f"  readiness={s.get('readiness')}")
        print(f"  predictive={s.get('predictive_passes')}/{s.get('predictive_total')}  "
              f"controller={s.get('controller_passes')}/{s.get('controller_total')}")

    p = OUT / "control_main_evidence_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D23 main evidence table]")
        print(f"  readiness={s.get('readiness')}  "
              f"evidence_pass={s.get('pass_count')}/{s.get('total')}")
        print(f"  table={OUT / 'control_main_evidence_table_v1' / 'main_evidence_table.md'}")

    for domain in ["church", "bedroom"]:
        p = OUT / "control_feasible_risk_controller" / domain / "actual_summary" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D24 {domain} semantic-feasible minimum-risk controller]")
            seed_level = s.get("seed_level", {})
            for comp in ["risk_minus_gain_only", "risk_minus_random",
                         "risk_minus_high_risk", "risk_minus_low_risk"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_true", {})
                sad = row.get("mean_abs_delta_attr", {})
                spg = row.get("mean_probe_gain", {})
                saa = row.get("mean_abs_alpha", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:22s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}  "
                          f"Δprobe_gain={spg.get('mean', float('nan')):+.4f}  "
                          f"Δ|alpha|={saa.get('mean', float('nan')):+.4f}")

        p = OUT / "control_feasible_risk_controller" / domain / "negative_summary" / "metrics.json"
        s = safe_load(p)
        if s:
            print(f"\n[Control D25 {domain} feasible-controller risk-signal negatives]")
            seed_level = s.get("paired_controller_diffs", {}).get("seed_level", {})
            for comp in ["actual_minus_shuffled", "actual_minus_inverted"]:
                row = seed_level.get(comp, {})
                sid = row.get("mean_id_cos", {})
                slp = row.get("mean_lpips_true", {})
                sad = row.get("mean_abs_delta_attr", {})
                if isinstance(sid.get("mean"), float):
                    print(f"  {comp:24s} "
                          f"seed-mean ΔID={sid['mean']:+.4f} "
                          f"wins={sid.get('wins', '-')}/{sid.get('n', '?')}  "
                          f"Δmean_lpips_true={slp.get('mean', float('nan')):+.4f} "
                          f"wins={slp.get('wins', '-')}/{slp.get('n', '?')}  "
                          f"Δ|attr|={sad.get('mean', float('nan')):+.4f}")

    p = OUT / "control_feasible_readiness_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D26 feasible-control readiness]")
        print(f"  readiness={s.get('readiness')}")
        print(f"  predictive={s.get('predictive_passes')}/{s.get('predictive_total')}  "
              f"controller={s.get('controller_passes')}/{s.get('controller_total')}  "
              f"boundary={s.get('boundary_passes')}/{s.get('boundary_total')}")

    p = OUT / "control_feasible_evidence_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D27 feasible-control evidence table]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print(f"  table={OUT / 'control_feasible_evidence_table_v1' / 'feasible_control_evidence_table.md'}")

    for label, out_dir in [
        ("FD-risk-estimator", "control_cross_domain_risk_predictive_fd"),
        ("prompt-template", "control_cross_domain_risk_predictive_prompt_photo"),
        ("prompt-caption", "control_cross_domain_risk_predictive_prompt_caption"),
        ("prompt-ensemble", "control_cross_domain_risk_predictive_prompt_ensemble"),
        (
            "prompt-ensemble-high-ntest128",
            "control_cross_domain_risk_predictive_prompt_ensemble_high_ntest128",
        ),
        (
            "prompt-ensemble-strict-gain",
            "control_cross_domain_risk_predictive_prompt_ensemble_strict_gain_match",
        ),
        ("prompt-ensemble-FD", "control_cross_domain_risk_predictive_prompt_ensemble_fd"),
        (
            "prompt-ensemble-ganspace-only",
            "control_cross_domain_risk_predictive_prompt_ensemble_ganspace_only",
        ),
        (
            "prompt-ensemble-sefa-only",
            "control_cross_domain_risk_predictive_prompt_ensemble_sefa_only",
        ),
        (
            "ffhq-prompt-ensemble",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble",
        ),
        (
            "ffhq-prompt-ensemble-high-ntest128",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_high_ntest128",
        ),
        (
            "ffhq-prompt-ensemble-strict-gain",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_strict_gain_match",
        ),
        (
            "ffhq-prompt-ensemble-FD",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd",
        ),
        (
            "ffhq-prompt-ensemble-FD-n10",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd_n10",
        ),
        (
            "ffhq-prompt-ensemble-ganspace-only",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only",
        ),
        (
            "ffhq-prompt-ensemble-ganspace-only-n10",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only_n10",
        ),
        (
            "ffhq-prompt-ensemble-sefa-only",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only",
        ),
        (
            "ffhq-prompt-ensemble-sefa-only-n10",
            "control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only_n10",
        ),
        (
            "prompt-ensemble-random-universe",
            "control_cross_domain_risk_predictive_prompt_ensemble_random_universe",
        ),
        (
            "prompt-ensemble-random-universe-n10",
            "control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10",
        ),
        (
            "prompt-ensemble-random-high-budget",
            "control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget",
        ),
        (
            "prompt-ensemble-random-high-budget-n10",
            "control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_n10",
        ),
        (
            "prompt-ensemble-wide-universe",
            "control_cross_domain_risk_predictive_prompt_ensemble_wide_universe",
        ),
        (
            "prompt-ensemble-wide-universe-n10",
            "control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_n10",
        ),
        (
            "prompt-ensemble-target-q10",
            "control_cross_domain_risk_predictive_prompt_ensemble_target_q10",
        ),
        (
            "prompt-ensemble-target-q10-n10",
            "control_cross_domain_risk_predictive_prompt_ensemble_target_q10_n10",
        ),
        (
            "prompt-ensemble-target-q50",
            "control_cross_domain_risk_predictive_prompt_ensemble_target_q50",
        ),
        (
            "prompt-ensemble-target-q50-n10",
            "control_cross_domain_risk_predictive_prompt_ensemble_target_q50_n10",
        ),
        (
            "prompt-ensemble-DINO-preservation",
            "control_cross_domain_risk_predictive_dino_preservation",
        ),
        (
            "prompt-ensemble-DINO-preservation-n10",
            "control_cross_domain_risk_predictive_dino_preservation_n10",
        ),
        ("high-ntest128", "control_cross_domain_risk_predictive_high_ntest128"),
        ("strict-gain-match", "control_cross_domain_risk_predictive_strict_gain_match"),
    ]:
        for domain in ["church", "bedroom", "ffhq"]:
            p = OUT / out_dir / domain / "metrics.json"
            s = safe_load(p)
            if s:
                print(f"\n[Control D28 {domain} predictive stress: {label}]")
                seed_level = s.get("seed_level", {})
                for key, desc in [
                    ("rho_vs_id_spearman", "rho vs ID Spearman"),
                    ("rho_vs_lpips_spearman", "rho vs LPIPS Spearman"),
                    ("id_beta_rho", "rho beta for ID"),
                    ("lpips_beta_rho", "rho beta for LPIPS"),
                    ("matched_pair_id_low_minus_high", "low-risk minus high-risk ID"),
                    ("matched_pair_lpips_low_minus_high", "low-risk minus high-risk LPIPS"),
                    ("rho_vs_dino_spearman", "rho vs DINO Spearman"),
                    ("dino_beta_rho", "rho beta for DINO"),
                    ("matched_pair_dino_low_minus_high", "low-risk minus high-risk DINO"),
                ]:
                    row = seed_level.get(key, {})
                    if isinstance(row.get("mean"), float):
                        print(f"  {desc:34s} "
                              f"mean={row['mean']:+.4f} "
                              f"wins={row.get('wins', '-')}/{row.get('n', '?')}")

    p = OUT / "control_predictive_assumption_readiness_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D29 predictive assumption-stress readiness]")
        print(f"  readiness={s.get('readiness')}  "
              f"passes={s.get('passes')}/{s.get('total')}")

    p = OUT / "control_predictive_stress_evidence_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D30 extended predictive stress evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v1' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v2" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D31 extended predictive stress evidence + prompt ensemble]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v2' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v3" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D32 extended predictive stress evidence + ensemble follow-up]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v3' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v4" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D33 prompt-ensemble estimator/source evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v4' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v5" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D34 main-grade extension stress evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v5' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_dino_evidence_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D35 DINO preservation evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_predictive_dino_evidence_table_v1' / 'predictive_dino_evidence_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v6_n10" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D36 seed-scaled extension evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v6_n10' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_dino_evidence_table_v2_n10" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D37 DINO seed-scaled evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_predictive_dino_evidence_table_v2_n10' / 'predictive_dino_evidence_table.md'}")

    p = OUT / "control_rho_incremental_value_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D38 rho incremental-value evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_rho_incremental_value_table_v1' / 'predictive_incremental_value_table.md'}")

    p = OUT / "control_failure_boundary_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D39 failure-boundary table]")
        print(f"  fail={s.get('fail_rows')}/{s.get('total_rows')}  "
              f"required_fail={s.get('required_fail_rows')}/{s.get('required_rows')}")
        print("  table="
              f"{OUT / 'control_failure_boundary_table_v1' / 'failure_boundary_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v7_exhaustive" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D40 exhaustive top-tier defense evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v7_exhaustive' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_permutation_null_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D41 predictive permutation-null evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_predictive_permutation_null_table_v1' / 'predictive_permutation_null_table.md'}")

    p = OUT / "control_failure_boundary_table_v2" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D42 failure-boundary table v2]")
        print(f"  fail={s.get('fail_rows')}/{s.get('total_rows')}  "
              f"required_fail={s.get('required_fail_rows')}/{s.get('required_rows')}")
        print("  table="
              f"{OUT / 'control_failure_boundary_table_v2' / 'failure_boundary_table.md'}")

    p = OUT / "control_predictive_stress_evidence_table_v8_exhaustive_n10" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D43 exhaustive n=10 defense evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('all_pass_count')}/{s.get('all_total')}")
        print("  table="
              f"{OUT / 'control_predictive_stress_evidence_table_v8_exhaustive_n10' / 'predictive_stress_evidence_table.md'}")

    p = OUT / "control_predictive_permutation_null_table_v2_n10" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D44 n=10 predictive permutation-null evidence]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_predictive_permutation_null_table_v2_n10' / 'predictive_permutation_null_table.md'}")

    p = OUT / "control_failure_boundary_table_v3" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D45 failure-boundary table v3]")
        print(f"  fail={s.get('fail_rows')}/{s.get('total_rows')}  "
              f"required_fail={s.get('required_fail_rows')}/{s.get('required_rows')}")
        print("  table="
              f"{OUT / 'control_failure_boundary_table_v3' / 'failure_boundary_table.md'}")

    p = OUT / "control_predictive_diagnostic_utility_table_v1" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D46 predictive diagnostic utility]")
        print(f"  readiness={s.get('readiness')}  "
              f"required_pass={s.get('required_pass_count')}/{s.get('required_total')}  "
              f"all_pass={s.get('pass_count')}/{s.get('total')}")
        print("  table="
              f"{OUT / 'control_predictive_diagnostic_utility_table_v1' / 'predictive_diagnostic_utility_table.md'}")

    p = OUT / "control_failure_boundary_table_v4" / "metrics.json"
    s = safe_load(p)
    if s:
        print(f"\n[Control D47 failure-boundary table v4]")
        print(f"  fail={s.get('fail_rows')}/{s.get('total_rows')}  "
              f"required_fail={s.get('required_fail_rows')}/{s.get('required_rows')}")
        print("  table="
              f"{OUT / 'control_failure_boundary_table_v4' / 'failure_boundary_table.md'}")

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
